# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 32 of 100

## 目录 (Table of Contents)

1. [A Generalist Hanabi Agent](#A-Generalist-Hanabi-Agent)
2. [Rethinking Fair Representation Learning for Performance-Sensitive Tasks](#Rethinking-Fair-Representation-Learning-for-Performance-Sensitive-Tasks)
3. [Generative Flows on Synthetic Pathway for Drug Design](#Generative-Flows-on-Synthetic-Pathway-for-Drug-Design)
4. [Revisit Micro-batch Clipping: Adaptive Data Pruning via Gradient Manipulation](#Revisit-Micro-batch-Clipping-Adaptive-Data-Pruning-via-Gradient-Manipulation)
5. [FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models](#FakeShield-Explainable-Image-Forgery-Detection-and-Localization-via-Multi-modal-Large-Language-Models)
6. [XLand-100B: A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning](#XLand-100B-A-Large-Scale-Multi-Task-Dataset-for-In-Context-Reinforcement-Learning)
7. [Varying Shades of Wrong: Aligning LLMs with Wrong Answers Only](#Varying-Shades-of-Wrong-Aligning-LLMs-with-Wrong-Answers-Only)
8. [Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling](#Task-Adaptive-Pretrained-Language-Models-via-Clustered-Importance-Sampling)
9. [NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation](#NExT-Mol-3D-Diffusion-Meets-1D-Language-Modeling-for-3D-Molecule-Generation)
10. [Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions](#Content-Style-Learning-from-Unaligned-Domains-Identifiability-under-Unknown-Latent-Dimensions)
11. [A Benchmark for Semantic Sensitive Information in LLMs Outputs](#A-Benchmark-for-Semantic-Sensitive-Information-in-LLMs-Outputs)
12. [Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models](#Analyzing-and-Boosting-the-Power-of-Fine-Grained-Visual-Recognition-for-Multi-modal-Large-Language-Models)
13. [Spreading Out-of-Distribution Detection on Graphs](#Spreading-Out-of-Distribution-Detection-on-Graphs)
14. [Stealthy Shield Defense: A Conditional Mutual Information-Based Approach against Black-Box Model Inversion Attacks](#Stealthy-Shield-Defense-A-Conditional-Mutual-Information-Based-Approach-against-Black-Box-Model-Inversion-Attacks)
15. [Solving New Tasks by Adapting Internet Video Knowledge](#Solving-New-Tasks-by-Adapting-Internet-Video-Knowledge)
16. [Learning Causal Alignment for Reliable Disease Diagnosis](#Learning-Causal-Alignment-for-Reliable-Disease-Diagnosis)
17. [Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning](#Facilitating-Multi-turn-Function-Calling-for-LLMs-via-Compositional-Instruction-Tuning)
18. [GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning](#GlycanML-A-Multi-Task-and-Multi-Structure-Benchmark-for-Glycan-Machine-Learning)
19. [Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models](#Hypothetical-Minds-Scaffolding-Theory-of-Mind-for-Multi-Agent-Tasks-with-Large-Language-Models)
20. [Stable Segment Anything Model](#Stable-Segment-Anything-Model)
21. [Equivariant Denoisers Cannot Copy Graphs: Align Your Graph Diffusion Models](#Equivariant-Denoisers-Cannot-Copy-Graphs-Align-Your-Graph-Diffusion-Models)
22. [Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization](#Achieving-Dimension-Free-Communication-in-Federated-Learning-via-Zeroth-Order-Optimization)
23. [Large (Vision) Language Models are Unsupervised In-Context Learners](#Large-Vision-Language-Models-are-Unsupervised-In-Context-Learners)
24. [ColPali: Efficient Document Retrieval with Vision Language Models](#ColPali-Efficient-Document-Retrieval-with-Vision-Language-Models)
25. [Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning](#Conflict-Averse-Gradient-Aggregation-for-Constrained-Multi-Objective-Reinforcement-Learning)
26. [SpinQuant: LLM Quantization with Learned Rotations](#SpinQuant-LLM-Quantization-with-Learned-Rotations)
27. [Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escape, and Network Embedding](#Loss-Landscape-of-Shallow-ReLU-like-Neural-Networks-Stationary-Points-Saddle-Escape-and-Network-Embedding)
28. [On the Byzantine-Resilience of Distillation-Based Federated Learning](#On-the-Byzantine-Resilience-of-Distillation-Based-Federated-Learning)
29. [OMG: Opacity Matters in Material Modeling with Gaussian Splatting](#OMG-Opacity-Matters-in-Material-Modeling-with-Gaussian-Splatting)
30. [Counterfactual Generative Modeling with Variational Causal Inference](#Counterfactual-Generative-Modeling-with-Variational-Causal-Inference)
31. [Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning](#Towards-Scalable-Exact-Machine-Unlearning-Using-Parameter-Efficient-Fine-Tuning)
32. [Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models](#Promptriever-Instruction-Trained-Retrievers-Can-Be-Prompted-Like-Language-Models)
33. [SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes](#SSLAM-Enhancing-Self-Supervised-Models-with-Audio-Mixtures-for-Polyphonic-Soundscapes)
34. [Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift](#Minimax-Optimal-Two-Stage-Algorithm-For-Moment-Estimation-Under-Covariate-Shift)
35. [Support is All You Need for Certified VAE Training](#Support-is-All-You-Need-for-Certified-VAE-Training)
36. [Do Mice Grok? Glimpses of Hidden Progress in Sensory Cortex](#Do-Mice-Grok-Glimpses-of-Hidden-Progress-in-Sensory-Cortex)

---


## A Generalist Hanabi Agent

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

## A Generalist Hanabi Agent


### Images

![3bcf4a7b250c233919d03edaa805f786bbf8c3a623fc2b3246c816520ef016e8.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/3bcf4a7b250c233919d03edaa805f786bbf8c3a623fc2b3246c816520ef016e8.jpg)

![4440779d53bb50eb521499273c66ffa6b547d3ebe1d275244e8f10a368837047.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/4440779d53bb50eb521499273c66ffa6b547d3ebe1d275244e8f10a368837047.jpg)

![4fe869a9432546c99ec557b996f8a31b5e22e0ff577c15290eb05582c08ba4fa.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/4fe869a9432546c99ec557b996f8a31b5e22e0ff577c15290eb05582c08ba4fa.jpg)

![5ce614eff7dc6d3e3026f5d05186377d08900b1bfc1f5e4de27917e50598999a.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/5ce614eff7dc6d3e3026f5d05186377d08900b1bfc1f5e4de27917e50598999a.jpg)

![607d4662c94fb088e4329a1a4d0cb7637bb088b28bdf4642411f80f48e2378a4.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/607d4662c94fb088e4329a1a4d0cb7637bb088b28bdf4642411f80f48e2378a4.jpg)

![6f5bfba07c258cac02d204ce71718a53b4ee46eb8a98cd357d46f087ae8baeac.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/6f5bfba07c258cac02d204ce71718a53b4ee46eb8a98cd357d46f087ae8baeac.jpg)

![78b4ddb385e3400f3bfe63a6818111dca01e25208e1711439482a956ecb0a4aa.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/78b4ddb385e3400f3bfe63a6818111dca01e25208e1711439482a956ecb0a4aa.jpg)

![9a3f39d2e9ccc84a22c419a9b1ccdc80632d623741eab713c2fdb82fa30b4609.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/9a3f39d2e9ccc84a22c419a9b1ccdc80632d623741eab713c2fdb82fa30b4609.jpg)

![c94f3b8bd88636296f209f6f15a90d97768cc6ceac9087bb7242ec1aaa573da0.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/c94f3b8bd88636296f209f6f15a90d97768cc6ceac9087bb7242ec1aaa573da0.jpg)

![cb31eb1628e85376372a826ee04fba973cd0c7f137edfe5b301ab4a02df0dff7.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/cb31eb1628e85376372a826ee04fba973cd0c7f137edfe5b301ab4a02df0dff7.jpg)

![d3274a4b6b8455a120cb103c0e57205af416146550d5fe71e94f18905a2037e5.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/d3274a4b6b8455a120cb103c0e57205af416146550d5fe71e94f18905a2037e5.jpg)

![e8d79a09cc3412e480ce0886a52c33ebb400dd85ca5e856346bf832808d1a0e8.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/e8d79a09cc3412e480ce0886a52c33ebb400dd85ca5e856346bf832808d1a0e8.jpg)

![eb257e1278f659a7c54b350a23f03203334385179712d020609ed5a361068f8d.jpg](../iclr_results/1166_A Generalist Hanabi Agent/images/eb257e1278f659a7c54b350a23f03203334385179712d020609ed5a361068f8d.jpg)

### Tables

![d77d063f6f5e61c27a12d86c2c230cb5e210cc61bd02a78ef766c0918e42ea71.jpg](../iclr_results/1166_A Generalist Hanabi Agent/tables/d77d063f6f5e61c27a12d86c2c230cb5e210cc61bd02a78ef766c0918e42ea71.jpg)

![e85d7a091736c6fb80117ca5665443f2febdf4ff9db6ff8c916d20d15812839e.jpg](../iclr_results/1166_A Generalist Hanabi Agent/tables/e85d7a091736c6fb80117ca5665443f2febdf4ff9db6ff8c916d20d15812839e.jpg)

## Rethinking Fair Representation Learning for Performance-Sensitive Tasks


### Images

![4f79acb527ab88900a288fe975b7b8b572f975e8e3f2b2e8fda80b269d1486b8.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/4f79acb527ab88900a288fe975b7b8b572f975e8e3f2b2e8fda80b269d1486b8.jpg)

![6dd14e8943d92c76b413a2bfe605d93a05b0336a31b994be9f6eaadcbe194058.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/6dd14e8943d92c76b413a2bfe605d93a05b0336a31b994be9f6eaadcbe194058.jpg)

![77cd0cfa270456f05fbf48d9bdec34a0ae67f2a2893cbb3213315ab2675bbbc1.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/77cd0cfa270456f05fbf48d9bdec34a0ae67f2a2893cbb3213315ab2675bbbc1.jpg)

![8b5025b407209e1b43f1b1bd854a624c79e1de33e69057c953dc5f47ab407b3d.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/8b5025b407209e1b43f1b1bd854a624c79e1de33e69057c953dc5f47ab407b3d.jpg)

![ce857b52ebe801fd37bb68e94f7ca363aacd8947b52050842aaa8244514b4d8b.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/ce857b52ebe801fd37bb68e94f7ca363aacd8947b52050842aaa8244514b4d8b.jpg)

![d2fe901604e2313499fec05b8b469b2c5d7bc0fab989d363df35a03371babf1f.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/d2fe901604e2313499fec05b8b469b2c5d7bc0fab989d363df35a03371babf1f.jpg)

![fca963d1bfe0a6bf8fb13dfbba3d32955240ec60dd82357f88e16805368db42e.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/fca963d1bfe0a6bf8fb13dfbba3d32955240ec60dd82357f88e16805368db42e.jpg)

![fcf9237055bb75df69d1515d4c15eaa9c6880e32b8fb08894aa9121131cd4508.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/images/fcf9237055bb75df69d1515d4c15eaa9c6880e32b8fb08894aa9121131cd4508.jpg)

### Tables

![01ebd7a0deee0cfcbeb8df3fd3ce11be8a2e4965224d242045af3e883a0fc718.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/tables/01ebd7a0deee0cfcbeb8df3fd3ce11be8a2e4965224d242045af3e883a0fc718.jpg)

![7d4bc12db8a14972192a77f77fca5591184f9237788e3dd0fdd3092685f8782c.jpg](../iclr_results/1167_Rethinking Fair Representation Learning for Performance-Sensitive Tasks/tables/7d4bc12db8a14972192a77f77fca5591184f9237788e3dd0fdd3092685f8782c.jpg)

## Generative Flows on Synthetic Pathway for Drug Design


### Images

![097a4d74932c25b9ed2d350f9ad75e7f4ca09fc57557f3410262b6cf3e495084.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/097a4d74932c25b9ed2d350f9ad75e7f4ca09fc57557f3410262b6cf3e495084.jpg)

![4e4a7c5760c55c6f66a9204af2a7dbf51c5edf44d2dc0ec1d39ffa66c75f226f.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/4e4a7c5760c55c6f66a9204af2a7dbf51c5edf44d2dc0ec1d39ffa66c75f226f.jpg)

![5957e68f98ec5647a9e195755e43b5da028afdb17bb3042b71f941c53f2c12af.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/5957e68f98ec5647a9e195755e43b5da028afdb17bb3042b71f941c53f2c12af.jpg)

![65dc348e5cbbbf200ca3a081d9917910bb0204efd51f19e671a479a48086dfeb.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/65dc348e5cbbbf200ca3a081d9917910bb0204efd51f19e671a479a48086dfeb.jpg)

![85ab0bbb19992992321ff0cae19a1cb87d10abbedfa9a6b3f2661d48d87941b1.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/85ab0bbb19992992321ff0cae19a1cb87d10abbedfa9a6b3f2661d48d87941b1.jpg)

![8833c667b69acb21e31153db4ce58847895fc4cd7fa003c349e0cc14486ed5b0.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/8833c667b69acb21e31153db4ce58847895fc4cd7fa003c349e0cc14486ed5b0.jpg)

![95f1414b176201600b7f01b549af39858da2e5f5b3c12882408e6777b3e80e88.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/95f1414b176201600b7f01b549af39858da2e5f5b3c12882408e6777b3e80e88.jpg)

![9bd178156c184a0d298f43efe6a22809222028d7aad6b0b86b21460f38dc6751.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/9bd178156c184a0d298f43efe6a22809222028d7aad6b0b86b21460f38dc6751.jpg)

![9f1fabee88b33f6f89bdf6a645881fb21c1b0f53e62468601fb1719bdb56caff.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/9f1fabee88b33f6f89bdf6a645881fb21c1b0f53e62468601fb1719bdb56caff.jpg)

![a109d14c6b1e2b36d560de4fecb45e538b7fc221e973ae34a75b667047f1980b.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/a109d14c6b1e2b36d560de4fecb45e538b7fc221e973ae34a75b667047f1980b.jpg)

![a30cc7e9c193eb506bbc9f6ad19bbe16863ee20d0248589a2859cf3e7f0b3bbd.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/a30cc7e9c193eb506bbc9f6ad19bbe16863ee20d0248589a2859cf3e7f0b3bbd.jpg)

![a32403261f553f691f34a01e83716a033d3a6c15569f06016f142644d723e21e.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/a32403261f553f691f34a01e83716a033d3a6c15569f06016f142644d723e21e.jpg)

![b4b2fb3d7fda32417d09de9c08fa2fea7371166b7e97c6e05305356e95143234.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/b4b2fb3d7fda32417d09de9c08fa2fea7371166b7e97c6e05305356e95143234.jpg)

![c115869a0187a978af5ea51c30dc94c83f8cb2bb927b2a85532633c500a19093.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/c115869a0187a978af5ea51c30dc94c83f8cb2bb927b2a85532633c500a19093.jpg)

![c32eacd74031d5bdb3e62b55369078dec9ab6efca91d658880113d4dff6f346e.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/c32eacd74031d5bdb3e62b55369078dec9ab6efca91d658880113d4dff6f346e.jpg)

![e03dce181d7b30244d43a60df39d29800b088bdd16efc7bf34b8c73ec63586af.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/e03dce181d7b30244d43a60df39d29800b088bdd16efc7bf34b8c73ec63586af.jpg)

![e4cdb57c09a08181a38563ddf1bd6127f4a5ec30ffcc1dd37d87cbf2aa1ed154.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/images/e4cdb57c09a08181a38563ddf1bd6127f4a5ec30ffcc1dd37d87cbf2aa1ed154.jpg)

### Tables

![368b9491a5a6583a3ca047cabeb41932491d4abb9f65100a6b666cc7858139e3.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/368b9491a5a6583a3ca047cabeb41932491d4abb9f65100a6b666cc7858139e3.jpg)

![41458d577f3f44d3d55a3f619578667b4b46c5663510798f0eff7e1af9450a67.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/41458d577f3f44d3d55a3f619578667b4b46c5663510798f0eff7e1af9450a67.jpg)

![432d72e3e6a7f63b0b1b50f21b7ec98de921dca2b1190e5307ac3eed833aa535.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/432d72e3e6a7f63b0b1b50f21b7ec98de921dca2b1190e5307ac3eed833aa535.jpg)

![53d2e129537758c7ba83b295143389ea97a7d1fd2ddda4463445040f6757bc63.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/53d2e129537758c7ba83b295143389ea97a7d1fd2ddda4463445040f6757bc63.jpg)

![804f1f73b890fdc53c878787492010c30689effa7c579c7af15739f1335927fd.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/804f1f73b890fdc53c878787492010c30689effa7c579c7af15739f1335927fd.jpg)

![96e185aa6197f374119450b8b61dab8be16f111f2df0c1a9d503aab9139e5d05.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/96e185aa6197f374119450b8b61dab8be16f111f2df0c1a9d503aab9139e5d05.jpg)

![af663585b9514e738578f04532c3414d01756567ee082531c6e180f8e202f522.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/af663585b9514e738578f04532c3414d01756567ee082531c6e180f8e202f522.jpg)

![babfd15d91604e23bcb9dab851c5a13fcde139e8b307f24fa29112c9b2c7c3ee.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/babfd15d91604e23bcb9dab851c5a13fcde139e8b307f24fa29112c9b2c7c3ee.jpg)

![be03a64ddb551378429d655d10e1a092f527399748e8e49b0a14d36a5b4d74e3.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/be03a64ddb551378429d655d10e1a092f527399748e8e49b0a14d36a5b4d74e3.jpg)

![cd12276944cf2de773c31423513b6b808facd36f9a93b9df129fe4fb170b1772.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/cd12276944cf2de773c31423513b6b808facd36f9a93b9df129fe4fb170b1772.jpg)

![d1b5bca52bc30fcb91ad7a69cfcba04a0d2ba1e2b2d1df2a08a0382d4263bd02.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/d1b5bca52bc30fcb91ad7a69cfcba04a0d2ba1e2b2d1df2a08a0382d4263bd02.jpg)

![f06fd4b32bcf214ebffb9d5b27784482110b77b38d306a09e367cc7419433145.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/f06fd4b32bcf214ebffb9d5b27784482110b77b38d306a09e367cc7419433145.jpg)

![f3156e5c1be317226cd7e4911c72b64d1e18e533ac8c7f49ccfe6ed6f2d20b81.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/f3156e5c1be317226cd7e4911c72b64d1e18e533ac8c7f49ccfe6ed6f2d20b81.jpg)

![f7c2bb195f189fbc85bfb59e80f99c47dd8e14057bd7359b72541d4c6349a2ec.jpg](../iclr_results/1168_Generative Flows on Synthetic Pathway for Drug Design/tables/f7c2bb195f189fbc85bfb59e80f99c47dd8e14057bd7359b72541d4c6349a2ec.jpg)

## Revisit Micro-batch Clipping: Adaptive Data Pruning via Gradient Manipulation


### Images

![8ad70321ba245744cc33d59bc12d237429a8f1c35ad1865c119267ab3f3565f6.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/images/8ad70321ba245744cc33d59bc12d237429a8f1c35ad1865c119267ab3f3565f6.jpg)

![aa9749c4f950b9169ef719a1d6d59145d4d5080d2cfd0fb64feb51922de4ca8f.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/images/aa9749c4f950b9169ef719a1d6d59145d4d5080d2cfd0fb64feb51922de4ca8f.jpg)

### Tables

![15138c9b299838a32cbf0cb3abd89aaae43ab4b7b6e26a357cd2b6be5fc9b5d7.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/tables/15138c9b299838a32cbf0cb3abd89aaae43ab4b7b6e26a357cd2b6be5fc9b5d7.jpg)

![16656259c8382ce09d7b2bef1a3b6e3fd15d6c44347bc222b8db7e078d31c423.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/tables/16656259c8382ce09d7b2bef1a3b6e3fd15d6c44347bc222b8db7e078d31c423.jpg)

![42a2fb0c6d804edae842701b8b882585624e05243247730b9801a1d2f32aa5f6.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/tables/42a2fb0c6d804edae842701b8b882585624e05243247730b9801a1d2f32aa5f6.jpg)

![bbc5f3951d57b9e3aea5b77b09a6942728d648a93bb1ae1e3f6cd251f98b19b1.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/tables/bbc5f3951d57b9e3aea5b77b09a6942728d648a93bb1ae1e3f6cd251f98b19b1.jpg)

![bc4045ee42aeb1036119b682f78c8f095d3844d13f79d1b6fd2ccba4b0c0823d.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/tables/bc4045ee42aeb1036119b682f78c8f095d3844d13f79d1b6fd2ccba4b0c0823d.jpg)

![e5ff86e7717127fe5340199d58bf74ba101aae793563b13218544b22733fdfd3.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/tables/e5ff86e7717127fe5340199d58bf74ba101aae793563b13218544b22733fdfd3.jpg)

![eed28190caed483295014d1c8b489f6e5c69a4fc6fca7845e07c9ed7053c00c9.jpg](../iclr_results/1169_Revisit Micro-batch Clipping_ Adaptive Data Pruning via Gradient Manipulation/tables/eed28190caed483295014d1c8b489f6e5c69a4fc6fca7845e07c9ed7053c00c9.jpg)

## FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models


### Images

![09ebfacd09952bd45a7dc1444c03802efb774884e95cbf021874a24ead8b348b.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/09ebfacd09952bd45a7dc1444c03802efb774884e95cbf021874a24ead8b348b.jpg)

![0ce0217822d0a1709447c2cb89829f86f76d7d9f9dd035281cd5894128f1d4cc.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/0ce0217822d0a1709447c2cb89829f86f76d7d9f9dd035281cd5894128f1d4cc.jpg)

![17eb581a2072af279acc5f57dc913f03c037eec7f599ad8aa8641030c90ec0d8.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/17eb581a2072af279acc5f57dc913f03c037eec7f599ad8aa8641030c90ec0d8.jpg)

![2f241a1ee41b73e1f0950d4e21624d655bef2e54ae5649690e7c58af3cf1d98f.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/2f241a1ee41b73e1f0950d4e21624d655bef2e54ae5649690e7c58af3cf1d98f.jpg)

![413de365d1f7b9437104d76c3e160417ccb59e57969f16022fcdb05536b87108.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/413de365d1f7b9437104d76c3e160417ccb59e57969f16022fcdb05536b87108.jpg)

![4796bf2eed4661db846d56f4d39e514641b0ca836d236be32bcdbe41c3a4af46.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/4796bf2eed4661db846d56f4d39e514641b0ca836d236be32bcdbe41c3a4af46.jpg)

![49f916198efa17ee6a4a3f34dcee85df2cf9d57e951d8a2074611176a0bcd9b0.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/49f916198efa17ee6a4a3f34dcee85df2cf9d57e951d8a2074611176a0bcd9b0.jpg)

![4ed73241fe34eec9cf1d547c68fc07417f48e13d29de0a093dba697df00bd027.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/4ed73241fe34eec9cf1d547c68fc07417f48e13d29de0a093dba697df00bd027.jpg)

![7138cbeb10834431475341ce30aaeda10abe2ee245601b95bdecd00c8bfe44cc.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/7138cbeb10834431475341ce30aaeda10abe2ee245601b95bdecd00c8bfe44cc.jpg)

![78f77a7615b4a819b53574762b321d65d71b286de44b8c24223c7d6b372e8687.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/78f77a7615b4a819b53574762b321d65d71b286de44b8c24223c7d6b372e8687.jpg)

![82560dce10683a38b1e241075c3f4a3ed178b9a25d21f48f168afd3adfd313d6.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/82560dce10683a38b1e241075c3f4a3ed178b9a25d21f48f168afd3adfd313d6.jpg)

![865378b69423f141822ad114716301d8aeb7ea76e008cd7ef6d41f9294f0caa6.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/865378b69423f141822ad114716301d8aeb7ea76e008cd7ef6d41f9294f0caa6.jpg)

![874c7c829eaf5fc35bb995ade612132c604a9a56c28a93c164d7b4f3e9966a81.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/874c7c829eaf5fc35bb995ade612132c604a9a56c28a93c164d7b4f3e9966a81.jpg)

![89d552ae03d4a57918f25aa7479220478404b0b87e3557cbff01252d41a33a23.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/89d552ae03d4a57918f25aa7479220478404b0b87e3557cbff01252d41a33a23.jpg)

![8d56de94b4db51969392794a3079b0d800ca335ae165555c0961460d1ad65a33.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/8d56de94b4db51969392794a3079b0d800ca335ae165555c0961460d1ad65a33.jpg)

![8d6b1e91cd32e5aac9c4816e23ebe9a2f1ad8916ff3e4a25833c2c7a98968d14.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/8d6b1e91cd32e5aac9c4816e23ebe9a2f1ad8916ff3e4a25833c2c7a98968d14.jpg)

![941cec7eaca8f08ad250f47ec3957a7dc1360b3ab28c3880454db02589e8ad14.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/941cec7eaca8f08ad250f47ec3957a7dc1360b3ab28c3880454db02589e8ad14.jpg)

![b60afc790a1f1a17ce08e2dacb79426148bfda39b73457d96d97241213baa104.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/b60afc790a1f1a17ce08e2dacb79426148bfda39b73457d96d97241213baa104.jpg)

![b8b0c81523bcb430ee9247a4be15c1768f34672560e35395f9267e9d8e58d0b9.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/b8b0c81523bcb430ee9247a4be15c1768f34672560e35395f9267e9d8e58d0b9.jpg)

![b8ffb087ec7d88699b383fdea9d94e9537ca422b86e832faab6bf2593cb572bf.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/b8ffb087ec7d88699b383fdea9d94e9537ca422b86e832faab6bf2593cb572bf.jpg)

![b9167055a17997252d3201ff5d2fd64e0d57a417418fea4aee8e08201c692898.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/b9167055a17997252d3201ff5d2fd64e0d57a417418fea4aee8e08201c692898.jpg)

![bd7b52644af1d7ecf26b8efe3afefc9bf1840f614f19720aa3ea80e67cfcd5c1.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/bd7b52644af1d7ecf26b8efe3afefc9bf1840f614f19720aa3ea80e67cfcd5c1.jpg)

![c2fabd36a9dd1bdf39b042e47c37ec0d6f56ef0d9001b1bfbc28a68fb277c00c.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/c2fabd36a9dd1bdf39b042e47c37ec0d6f56ef0d9001b1bfbc28a68fb277c00c.jpg)

![ca27c24ddbe071aa6c1b0a2461d4e11fde278c3ba3a6e087a57533f471f788b4.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/ca27c24ddbe071aa6c1b0a2461d4e11fde278c3ba3a6e087a57533f471f788b4.jpg)

![ccae82a4d0e711aed809851a3c7ed3c6c9473893917aba8807204edb581e6941.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/ccae82a4d0e711aed809851a3c7ed3c6c9473893917aba8807204edb581e6941.jpg)

![d69849923ff5573ed8cb5cfc103781b542fd97fe4218c9e40aa30820d4bfb839.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/d69849923ff5573ed8cb5cfc103781b542fd97fe4218c9e40aa30820d4bfb839.jpg)

![d96fea740ce665994a8018e65a04ed99cadaca6dfa2b047f83490f243215dd74.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/d96fea740ce665994a8018e65a04ed99cadaca6dfa2b047f83490f243215dd74.jpg)

![dc54fddd20fff6c7ac3f6d3b0a4733ae17eb2c7ccbfd8e33acdada5366722547.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/dc54fddd20fff6c7ac3f6d3b0a4733ae17eb2c7ccbfd8e33acdada5366722547.jpg)

![e2dc28b3932f699f4a3b58b86ccb1edcc089279b89f305705e38ffb278fc20e6.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/e2dc28b3932f699f4a3b58b86ccb1edcc089279b89f305705e38ffb278fc20e6.jpg)

![e754191a020fa5a8fa38a8331a4213f965f913697f4d82ebf2f12fd3d0351ba5.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/e754191a020fa5a8fa38a8331a4213f965f913697f4d82ebf2f12fd3d0351ba5.jpg)

![e915fd993d00c5fac62f2f392beb6f90d4035df3fc22b45f8e70092ae6092839.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/e915fd993d00c5fac62f2f392beb6f90d4035df3fc22b45f8e70092ae6092839.jpg)

![eaa91784e5ecfc40e60c217cda0b32a7bd300ff2a5d6247233847864e7efb675.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/eaa91784e5ecfc40e60c217cda0b32a7bd300ff2a5d6247233847864e7efb675.jpg)

![ec419194e0a07ef8ebea259aa82cee8889b66fc50d811158553c1cb672f50871.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/ec419194e0a07ef8ebea259aa82cee8889b66fc50d811158553c1cb672f50871.jpg)

![ef99947fdad3161e63bc0989697bfd05319423019409740b9a13ccc66b10bd12.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/ef99947fdad3161e63bc0989697bfd05319423019409740b9a13ccc66b10bd12.jpg)

![f1737b35cd743172bdc0a26b383958a910fe5843a6299d2623df57973a9cfcd2.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/f1737b35cd743172bdc0a26b383958a910fe5843a6299d2623df57973a9cfcd2.jpg)

![f72a03d4397c9780d44ec43ba4f0c0444f9a46ea77ca06102f4de1bfc6a12004.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/images/f72a03d4397c9780d44ec43ba4f0c0444f9a46ea77ca06102f4de1bfc6a12004.jpg)

### Tables

![1973ad711c566cea91262388d034e1b6c47b71d98e237362e0e960b57099e4d9.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/1973ad711c566cea91262388d034e1b6c47b71d98e237362e0e960b57099e4d9.jpg)

![39456e57401f40113de3445c4ac4c498104b9e32158b3a03cb6de5b9b790518c.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/39456e57401f40113de3445c4ac4c498104b9e32158b3a03cb6de5b9b790518c.jpg)

![4407d832ae200b4b63b3dc3c2995edcd1e9e2d6527ff63cf070e35274044f3f4.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/4407d832ae200b4b63b3dc3c2995edcd1e9e2d6527ff63cf070e35274044f3f4.jpg)

![725e1998793e5a96657373a5ef3d84756b2c04b5997192e114c2f8a3a03d6104.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/725e1998793e5a96657373a5ef3d84756b2c04b5997192e114c2f8a3a03d6104.jpg)

![adb190ee0281fca55ba91b875e136b9a55e037568193bf2280db089b5536a275.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/adb190ee0281fca55ba91b875e136b9a55e037568193bf2280db089b5536a275.jpg)

![aee550f19a3012bc0f7cc2638140c8ffac448c8a311e6df58fd18754dca2920c.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/aee550f19a3012bc0f7cc2638140c8ffac448c8a311e6df58fd18754dca2920c.jpg)

![b886f64cef0483cee7a25c9c3d99f60e56b0ad51cf94e47a3678d1e1b185923b.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/b886f64cef0483cee7a25c9c3d99f60e56b0ad51cf94e47a3678d1e1b185923b.jpg)

![b9561922a20fd95ce42b57ab372905d71da16c822a1eb2699ec7a7e87cae7054.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/b9561922a20fd95ce42b57ab372905d71da16c822a1eb2699ec7a7e87cae7054.jpg)

![f348fdb0cd43f525b8a6f20558ca62b1a26315529f05d52830123e7c9cb9082b.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/f348fdb0cd43f525b8a6f20558ca62b1a26315529f05d52830123e7c9cb9082b.jpg)

![f8d2a0beed37afcd77bbf67617e88d9f45bfa0fe2ea334e40528b306aa504b04.jpg](../iclr_results/1170_FakeShield_ Explainable Image Forgery Detection and Localization via Multi-modal Large Language Mode/tables/f8d2a0beed37afcd77bbf67617e88d9f45bfa0fe2ea334e40528b306aa504b04.jpg)

## XLand-100B: A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning


### Images

![070f5472e10ced694afbeef5654c20ac252d1187e35e8e836555becdd308de1c.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/070f5472e10ced694afbeef5654c20ac252d1187e35e8e836555becdd308de1c.jpg)

![0cd2e2bc29e643add4da0d77e778de5a819a4ef25fdb1d1d284a3bb7d237c827.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/0cd2e2bc29e643add4da0d77e778de5a819a4ef25fdb1d1d284a3bb7d237c827.jpg)

![1bf7a71ffddbdb0ddb2c80ce05aaf2b26ca44f203b2e749837320cb7722e36f8.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/1bf7a71ffddbdb0ddb2c80ce05aaf2b26ca44f203b2e749837320cb7722e36f8.jpg)

![253c264f4b85e4086bfc2928e73e350a245e49663871dd3b5db2dfdfc8c520d8.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/253c264f4b85e4086bfc2928e73e350a245e49663871dd3b5db2dfdfc8c520d8.jpg)

![3ac5efa90b7cdc0e6d733d5558c85a2ebc9205e620bc78e705798f5b24adcfbb.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/3ac5efa90b7cdc0e6d733d5558c85a2ebc9205e620bc78e705798f5b24adcfbb.jpg)

![471701dae1cee1ba72a9a27282a1f6c5dffb2f07bebbf527e53af141e3ac3f88.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/471701dae1cee1ba72a9a27282a1f6c5dffb2f07bebbf527e53af141e3ac3f88.jpg)

![4c6b7795dca0e386a73e824f2f00d439f455b4c28e8535f0d3e6c54a95addf32.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/4c6b7795dca0e386a73e824f2f00d439f455b4c28e8535f0d3e6c54a95addf32.jpg)

![6437dfeffe5b4cbfa6bbb253c4541cab120579c659839fe8e23fe115ad45754c.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/6437dfeffe5b4cbfa6bbb253c4541cab120579c659839fe8e23fe115ad45754c.jpg)

![72da6f2901a84e1f8e25c0cf7d770133bc6d814257ddb02b72114ea070540589.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/72da6f2901a84e1f8e25c0cf7d770133bc6d814257ddb02b72114ea070540589.jpg)

![79304ef9ab7b6ea3924dd3ee4c2fef1558ace9aad48fca854e8133d81c71d8f7.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/79304ef9ab7b6ea3924dd3ee4c2fef1558ace9aad48fca854e8133d81c71d8f7.jpg)

![7c0f641abe451d9ab03a3518b2765aa22de957e17e727dc3fb6ab862cdb6f445.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/7c0f641abe451d9ab03a3518b2765aa22de957e17e727dc3fb6ab862cdb6f445.jpg)

![88fe25d12734f9caaf2fde27ad304e1ac3f7b46706b4212da64466cff5831930.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/88fe25d12734f9caaf2fde27ad304e1ac3f7b46706b4212da64466cff5831930.jpg)

![8dc712d253b9754cec4cdf85cb38d02bc0d76708656e93b412ed38872bc17ffb.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/8dc712d253b9754cec4cdf85cb38d02bc0d76708656e93b412ed38872bc17ffb.jpg)

![994dc25b6a97a339609a600c932ac27a7b2a89b7baf7940f1d28d4eb99fa95e7.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/994dc25b6a97a339609a600c932ac27a7b2a89b7baf7940f1d28d4eb99fa95e7.jpg)

![996b416eac829879189b2acf944471182567c570bc2b8d34011663e61c90b3d2.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/996b416eac829879189b2acf944471182567c570bc2b8d34011663e61c90b3d2.jpg)

![a8ebeb42a1b24810dd204b95b0ea342955a8a2171e40832538dc871c7dfd8f5f.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/a8ebeb42a1b24810dd204b95b0ea342955a8a2171e40832538dc871c7dfd8f5f.jpg)

![ac67a272fe7f375ce2bac53272daeb4bacc857c69de4d4cdfc0ee2377849288d.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/ac67a272fe7f375ce2bac53272daeb4bacc857c69de4d4cdfc0ee2377849288d.jpg)

![afd6094344b4fb05368224931f2a2b0e1dbe9dfbfbfd3033cb8b47227d313b7f.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/afd6094344b4fb05368224931f2a2b0e1dbe9dfbfbfd3033cb8b47227d313b7f.jpg)

![ba983c5e325912609590cdd8d1beee15627a81cdcb9bd0482a6754495de75ba7.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/ba983c5e325912609590cdd8d1beee15627a81cdcb9bd0482a6754495de75ba7.jpg)

![bcc3426a38acd2ab956b489d7d9ee28eb300587869d556202266854df04ff02e.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/bcc3426a38acd2ab956b489d7d9ee28eb300587869d556202266854df04ff02e.jpg)

![c0e6e5126bec33a9ed37a328a817cd45b8d4c3bad7adac798102e50cba4ba1d2.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/c0e6e5126bec33a9ed37a328a817cd45b8d4c3bad7adac798102e50cba4ba1d2.jpg)

![cea05a24e04e0c4e21c365c6d71910975174bed0e0d09b74d33df012b0dcedeb.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/cea05a24e04e0c4e21c365c6d71910975174bed0e0d09b74d33df012b0dcedeb.jpg)

![fee6c779c51e7645f2b65803d32f8a805233f1684e211791b1e48a3a24034265.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/images/fee6c779c51e7645f2b65803d32f8a805233f1684e211791b1e48a3a24034265.jpg)

### Tables

![147ac12ecfaac149196291bcc74ed15d7a2cec5ffab2608eea655c2e422d3727.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/147ac12ecfaac149196291bcc74ed15d7a2cec5ffab2608eea655c2e422d3727.jpg)

![293d27767f9127ea3d8cc7ba9e32d22f3e429fc1ba9ce08bcafad2cf76cbec84.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/293d27767f9127ea3d8cc7ba9e32d22f3e429fc1ba9ce08bcafad2cf76cbec84.jpg)

![534cbc719e2af987c0ffe58d358cb00875df8827316ba48165d86f8b2e3701f0.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/534cbc719e2af987c0ffe58d358cb00875df8827316ba48165d86f8b2e3701f0.jpg)

![66b9060809d6664a1b8e229187ebb757bfca691ad5e8c92712bf99d31d317863.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/66b9060809d6664a1b8e229187ebb757bfca691ad5e8c92712bf99d31d317863.jpg)

![7985eac94813bedaf3d4810f887f2aefc3dae200d1cc35851ac3bdb63915bb62.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/7985eac94813bedaf3d4810f887f2aefc3dae200d1cc35851ac3bdb63915bb62.jpg)

![7a1a6c673e5df55a2bd41bf7f927f8767d5e16e9316b8e869621194bba7022c1.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/7a1a6c673e5df55a2bd41bf7f927f8767d5e16e9316b8e869621194bba7022c1.jpg)

![ae49d24ddece11a076d33dda0a1275bcf3f8aec3a36e0f777cb3bc0099443e3e.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/ae49d24ddece11a076d33dda0a1275bcf3f8aec3a36e0f777cb3bc0099443e3e.jpg)

![b929c11a0d999a8deea7cb503007a4fed0e2cdf592509b1d3ae9ce46e7bf44b5.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/b929c11a0d999a8deea7cb503007a4fed0e2cdf592509b1d3ae9ce46e7bf44b5.jpg)

![c65219a1758fa59d0950b8d6e1e35fca847a58e4586b1b508e842587bb315a2f.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/c65219a1758fa59d0950b8d6e1e35fca847a58e4586b1b508e842587bb315a2f.jpg)

![cd27566666383e48710097d594bcef155482aa14e0333e8df8959cebfacd400e.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/cd27566666383e48710097d594bcef155482aa14e0333e8df8959cebfacd400e.jpg)

![fe715c2b15f8da9e226f40e113417998a1d1594bdcaaa0e6cd395fbfe48aca80.jpg](../iclr_results/1171_XLand-100B_ A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning/tables/fe715c2b15f8da9e226f40e113417998a1d1594bdcaaa0e6cd395fbfe48aca80.jpg)

## Varying Shades of Wrong: Aligning LLMs with Wrong Answers Only


### Images

![546059372be163469e9e59792d7ffd1f6e925c3d21e9e2c855b4e479d3b6ee77.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/images/546059372be163469e9e59792d7ffd1f6e925c3d21e9e2c855b4e479d3b6ee77.jpg)

![5839530ca0c0e0192b14ea1ad553602d65eb5dfd537830581083305b76eddca6.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/images/5839530ca0c0e0192b14ea1ad553602d65eb5dfd537830581083305b76eddca6.jpg)

![762aa0e51792850aae01e81775071f5022cfc71f0dd9a797a33b16dd80ce4b5b.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/images/762aa0e51792850aae01e81775071f5022cfc71f0dd9a797a33b16dd80ce4b5b.jpg)

![c0d1a1b0bd726afa0fa79d4dcafa3ad8eb9a03e3bed26b03db64e119fb024bfc.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/images/c0d1a1b0bd726afa0fa79d4dcafa3ad8eb9a03e3bed26b03db64e119fb024bfc.jpg)

![cc123bc970a6979779d07a726beaa75c833692ccf0d074d50a2462c5920ed28c.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/images/cc123bc970a6979779d07a726beaa75c833692ccf0d074d50a2462c5920ed28c.jpg)

![d6440a941697990a8d638a9b58f4153658410e84ea13f63f68e598b4a05e5005.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/images/d6440a941697990a8d638a9b58f4153658410e84ea13f63f68e598b4a05e5005.jpg)

### Tables

![2f98cb5c220be0391d9b9520763a6defc82023aaa930583c3e972b8fdbac2448.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/2f98cb5c220be0391d9b9520763a6defc82023aaa930583c3e972b8fdbac2448.jpg)

![4ca53c591d61605e89e089a6f2070b4b009c1154f6bbfdc1843494ba2a8770eb.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/4ca53c591d61605e89e089a6f2070b4b009c1154f6bbfdc1843494ba2a8770eb.jpg)

![53b70d65ba78dc8e1ed2a01b74ee9351abd53773bc98520bc62479caa6ab9c16.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/53b70d65ba78dc8e1ed2a01b74ee9351abd53773bc98520bc62479caa6ab9c16.jpg)

![5681f7e346a1da83120b4c0247f33f875d9d34cda5ddc09a895e9b57e2805aca.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/5681f7e346a1da83120b4c0247f33f875d9d34cda5ddc09a895e9b57e2805aca.jpg)

![684c9445592240a2531d0ac64f035fba91d865654c219a79fd5879f4bcbda6a4.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/684c9445592240a2531d0ac64f035fba91d865654c219a79fd5879f4bcbda6a4.jpg)

![714a2ffc23607c973ef8ea6805a03b82ed3df9478671c4f15227abef83abea3b.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/714a2ffc23607c973ef8ea6805a03b82ed3df9478671c4f15227abef83abea3b.jpg)

![7559c492e8c20b692cda8fbf95fd673b0d7e5522a1c3d70a1981d5fe1317b831.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/7559c492e8c20b692cda8fbf95fd673b0d7e5522a1c3d70a1981d5fe1317b831.jpg)

![8487a6b06327b697935335e64d5e1defde02fa842f3a318f95ebe5b8c4693229.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/8487a6b06327b697935335e64d5e1defde02fa842f3a318f95ebe5b8c4693229.jpg)

![9c188860e691e75326f775b05c85609a2eea66140348357f0550e1f03a6d7a02.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/9c188860e691e75326f775b05c85609a2eea66140348357f0550e1f03a6d7a02.jpg)

![d83bc1f4e10ad5fdec097341c76c26ae4926bd86d6e7c39853862f0f341691f1.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/d83bc1f4e10ad5fdec097341c76c26ae4926bd86d6e7c39853862f0f341691f1.jpg)

![dbca6d526c13c66c085dc5357cab3f1847ba25a9f81de58c4b2653887fd35335.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/dbca6d526c13c66c085dc5357cab3f1847ba25a9f81de58c4b2653887fd35335.jpg)

![fd27f6f5db1cb37b2ef70a3a593fa7ef5ca08fb3a4b939c1ee9e4d275300b056.jpg](../iclr_results/1172_Varying Shades of Wrong_ Aligning LLMs with Wrong Answers Only/tables/fd27f6f5db1cb37b2ef70a3a593fa7ef5ca08fb3a4b939c1ee9e4d275300b056.jpg)

## Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling


### Images

![04192c220cf8f4eb98799dfd1e876fa123ce6e97efe5a52e82b258f961b8d46a.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/04192c220cf8f4eb98799dfd1e876fa123ce6e97efe5a52e82b258f961b8d46a.jpg)

![0d3bf8d05ab71f62ba16676baff45fd74a23e0606c5cc4b96201c2f9f89beb11.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/0d3bf8d05ab71f62ba16676baff45fd74a23e0606c5cc4b96201c2f9f89beb11.jpg)

![2164dcd73c1bc74de1992e2f4d4d112b9d6fb175dc37ebbdaca15722d5fb1c4b.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/2164dcd73c1bc74de1992e2f4d4d112b9d6fb175dc37ebbdaca15722d5fb1c4b.jpg)

![396ed30745ac4abbf1319ebc9afcf027c0baeefe36ce2c7b9a5e955b292bf833.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/396ed30745ac4abbf1319ebc9afcf027c0baeefe36ce2c7b9a5e955b292bf833.jpg)

![4b30cd06d7bad8c2f49919d22953766a1a20e52e3d123159bc02fe238d23f550.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/4b30cd06d7bad8c2f49919d22953766a1a20e52e3d123159bc02fe238d23f550.jpg)

![65fc953c0e7196b1394a56fb96b7b52570b9a6150b67bdb653beb4394fab2532.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/65fc953c0e7196b1394a56fb96b7b52570b9a6150b67bdb653beb4394fab2532.jpg)

![75694527a68f9193da43a74ac1bfee36c4d4fd1fe3983c48b36e6e32b8dd34a6.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/75694527a68f9193da43a74ac1bfee36c4d4fd1fe3983c48b36e6e32b8dd34a6.jpg)

![8144a3bcd1458e8e061751abee866543c213995ee843c6a6deab2c374f10afa1.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/8144a3bcd1458e8e061751abee866543c213995ee843c6a6deab2c374f10afa1.jpg)

![a22f1c8995e2031658a91e07fa58be24885049fbb071147ad61cf9ae15de1482.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/a22f1c8995e2031658a91e07fa58be24885049fbb071147ad61cf9ae15de1482.jpg)

![a52017eb3d9b808a6f15f8aea505851d7cbd9fb28abe49383aefd835e262b8fc.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/a52017eb3d9b808a6f15f8aea505851d7cbd9fb28abe49383aefd835e262b8fc.jpg)

![df878998c8e2f7389936e4fdf84a9748edc1267b74307e287603eb4a5c9fdf00.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/df878998c8e2f7389936e4fdf84a9748edc1267b74307e287603eb4a5c9fdf00.jpg)

![f8fc1ea064f3892de6d676ce9bb33fc90b240ec1a1f5dfe878dd80c428dafd8b.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/f8fc1ea064f3892de6d676ce9bb33fc90b240ec1a1f5dfe878dd80c428dafd8b.jpg)

![fae0a73de5a129a47c9fb508207657c81d66b52afaeed295c42b112633907f73.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/images/fae0a73de5a129a47c9fb508207657c81d66b52afaeed295c42b112633907f73.jpg)

### Tables

![0e0863255c7f67e0f1de47458341581b6b276389c4643438ca5fba9e77527de0.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/0e0863255c7f67e0f1de47458341581b6b276389c4643438ca5fba9e77527de0.jpg)

![1366b81349375745566800ddffe0aa6fb1bbaf507224fab466aa2a6dba646999.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/1366b81349375745566800ddffe0aa6fb1bbaf507224fab466aa2a6dba646999.jpg)

![4bf239a72f7f7174e1058895930b590208c6ff6473ed69b72e3f3bb38f15025f.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/4bf239a72f7f7174e1058895930b590208c6ff6473ed69b72e3f3bb38f15025f.jpg)

![51126781b838f8c2e162676a29cec849e8fc84bd73d98dc1f83b394c7bc069f4.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/51126781b838f8c2e162676a29cec849e8fc84bd73d98dc1f83b394c7bc069f4.jpg)

![52e4f85e30f90148853af4164bae37473d3b24028fc8787ee8a512a24fb7bc24.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/52e4f85e30f90148853af4164bae37473d3b24028fc8787ee8a512a24fb7bc24.jpg)

![55ea4ec7e4a595df3057aab2262bd9acd7259f71e39185bb5a70c25b63e0919d.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/55ea4ec7e4a595df3057aab2262bd9acd7259f71e39185bb5a70c25b63e0919d.jpg)

![57dc6b47f1a8c3ff5bfcfc072cc8f93cf8719a01f8c39b4865fc73c2eb0f5c25.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/57dc6b47f1a8c3ff5bfcfc072cc8f93cf8719a01f8c39b4865fc73c2eb0f5c25.jpg)

![7ee26bc6a1f3ffe3c28849bbbd0055b64fbfa5fe19b76b3552a30c8cfe2013a7.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/7ee26bc6a1f3ffe3c28849bbbd0055b64fbfa5fe19b76b3552a30c8cfe2013a7.jpg)

![9a913bac92b9daa3d07e6fbb2525c9288a9be8871fa2d27e3c698f43d7159f97.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/9a913bac92b9daa3d07e6fbb2525c9288a9be8871fa2d27e3c698f43d7159f97.jpg)

![a163519cd7161e2a3d61470e6fc6d03307bafb33dd8abff25b9821f159aba30a.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/a163519cd7161e2a3d61470e6fc6d03307bafb33dd8abff25b9821f159aba30a.jpg)

![ace402ad269d007a5b5d834d9962e30a91433e38dad3dcb6ba2f2776662b8934.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/ace402ad269d007a5b5d834d9962e30a91433e38dad3dcb6ba2f2776662b8934.jpg)

![c74d9e9cb2063e70a5ef8c176585fbc05e32bcfb57f416f81efc73857787e003.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/c74d9e9cb2063e70a5ef8c176585fbc05e32bcfb57f416f81efc73857787e003.jpg)

![e7325d7ea9ce989f2c5b92a9bef5bdeb20782472ba06279f3eea3e09e3cc6cbc.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/e7325d7ea9ce989f2c5b92a9bef5bdeb20782472ba06279f3eea3e09e3cc6cbc.jpg)

![ea2abfc917fff787cdfead0ec08b01f46dba0379d3bbaa7a59e026617d2548f4.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/ea2abfc917fff787cdfead0ec08b01f46dba0379d3bbaa7a59e026617d2548f4.jpg)

![fe21bb9604cfdba89b6b414c85313e364caacfee47199ea7155946b348eaec38.jpg](../iclr_results/1173_Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling/tables/fe21bb9604cfdba89b6b414c85313e364caacfee47199ea7155946b348eaec38.jpg)

## NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation


### Images

![05207c5b74173e97377b61ac72cf4b5ae2f2fe21ea1bcc335e6ae3afca3c2d0a.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/05207c5b74173e97377b61ac72cf4b5ae2f2fe21ea1bcc335e6ae3afca3c2d0a.jpg)

![36953fd22ca44e38e44e8cab6dfd391831fe0015d8019005729789fa1851f499.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/36953fd22ca44e38e44e8cab6dfd391831fe0015d8019005729789fa1851f499.jpg)

![434cff6ceab87b5343088f5c28b3b63f2fae90d0d029e57a056f7cbf27a0eb49.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/434cff6ceab87b5343088f5c28b3b63f2fae90d0d029e57a056f7cbf27a0eb49.jpg)

![73d13417bf539a70807803d9a4dcb0e60a44d9b1502b45ed3d9bd54c9f009055.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/73d13417bf539a70807803d9a4dcb0e60a44d9b1502b45ed3d9bd54c9f009055.jpg)

![7ca7e781853844157449e9b19a0d00aaac71842377997bd48d8ed2219b1b1be7.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/7ca7e781853844157449e9b19a0d00aaac71842377997bd48d8ed2219b1b1be7.jpg)

![baad736f477c8c32462775269092d152888b933bd3a90220649be70e3c24d419.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/baad736f477c8c32462775269092d152888b933bd3a90220649be70e3c24d419.jpg)

![bc120552f128ada3a72a4082800a35a8adf78f1d02372656555513071f895f3b.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/bc120552f128ada3a72a4082800a35a8adf78f1d02372656555513071f895f3b.jpg)

![d1545a8dfce3e1a161f54ed0466f9775148ad28db7e622da29c0119d52da6779.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/d1545a8dfce3e1a161f54ed0466f9775148ad28db7e622da29c0119d52da6779.jpg)

![d1f177107256dc60bb90a96ec6947be01214cfbb07818922c4a2ddc12a746f40.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/d1f177107256dc60bb90a96ec6947be01214cfbb07818922c4a2ddc12a746f40.jpg)

![eeef15139efe9307d7660ba13f43d1f40d6f472f2eb3fa504722ee9160295599.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/eeef15139efe9307d7660ba13f43d1f40d6f472f2eb3fa504722ee9160295599.jpg)

![fc1b4caae94090db39f960f041e65b11ab9d208fd1474b6260f86194c21a9569.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/images/fc1b4caae94090db39f960f041e65b11ab9d208fd1474b6260f86194c21a9569.jpg)

### Tables

![00c175ab84a576f991567f8c6753088df742128eedfae6a1789071bbb8d266e0.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/00c175ab84a576f991567f8c6753088df742128eedfae6a1789071bbb8d266e0.jpg)

![321355f258f62706d8d8377d403b3deceede8a68eee60b1b9afe4ec3ca93b125.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/321355f258f62706d8d8377d403b3deceede8a68eee60b1b9afe4ec3ca93b125.jpg)

![3467c22e65030310dd5c1d13d651b36ce1afa471f2b6578a57d74e235ba49aa0.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/3467c22e65030310dd5c1d13d651b36ce1afa471f2b6578a57d74e235ba49aa0.jpg)

![3fd2514823139e41f371c38a7a1440a558dcb92b84591271cf6e6ae282da2a0e.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/3fd2514823139e41f371c38a7a1440a558dcb92b84591271cf6e6ae282da2a0e.jpg)

![43ab69fd6c5e392354f17658c49b3218864fa999a2e24c536aa83593848706a1.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/43ab69fd6c5e392354f17658c49b3218864fa999a2e24c536aa83593848706a1.jpg)

![4aa065d25507de33f3dec0f1560fd3f8313f949972ce6f603c7865f4cb97d831.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/4aa065d25507de33f3dec0f1560fd3f8313f949972ce6f603c7865f4cb97d831.jpg)

![5e3c16fe2316e8b242d14ace205a79599e01ad4ce4eb8b6805b38fcd2e17192a.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/5e3c16fe2316e8b242d14ace205a79599e01ad4ce4eb8b6805b38fcd2e17192a.jpg)

![5e4ac09d6a30c22a67eedb25c24ebb3b329f81892bdfda59012c7e8c40df6e14.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/5e4ac09d6a30c22a67eedb25c24ebb3b329f81892bdfda59012c7e8c40df6e14.jpg)

![6290fe4ca12a1f15a02157273caf727c4f76932fd918e3b4c9a8cf3127b2e5ec.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/6290fe4ca12a1f15a02157273caf727c4f76932fd918e3b4c9a8cf3127b2e5ec.jpg)

![8d64a54907c66ac6c0bcac5a8d07fbf97fccbf4c90894a1348187018db4e6d1d.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/8d64a54907c66ac6c0bcac5a8d07fbf97fccbf4c90894a1348187018db4e6d1d.jpg)

![95c12860b058101f4376f78ab12ae629294db0658002e30df095c75a887766d9.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/95c12860b058101f4376f78ab12ae629294db0658002e30df095c75a887766d9.jpg)

![9abfc08b5403e7b4a7d91d0f10c3933dffcdfbbe4043fcfdc5dedb8160639adb.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/9abfc08b5403e7b4a7d91d0f10c3933dffcdfbbe4043fcfdc5dedb8160639adb.jpg)

![ad46e95acd4f19afea7db953369c7295a938ad0f55276ec2dd12b422da8605c9.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/ad46e95acd4f19afea7db953369c7295a938ad0f55276ec2dd12b422da8605c9.jpg)

![aed771fbbd33d45ff4f2a99671a79368cbaf38378d69eea18951982d9267f7d5.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/aed771fbbd33d45ff4f2a99671a79368cbaf38378d69eea18951982d9267f7d5.jpg)

![c0ca6950e8949bd770473cbb0ebd91f6532284a33fdcb60745cdea5e708b187b.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/c0ca6950e8949bd770473cbb0ebd91f6532284a33fdcb60745cdea5e708b187b.jpg)

![cb8c7ffed684ebe2caed307fa5f75c1e4f817126a5854ac39031bf90f0db947e.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/cb8c7ffed684ebe2caed307fa5f75c1e4f817126a5854ac39031bf90f0db947e.jpg)

![d40c9ebc7d8ebc6f92c48a375f5bec5c52927838049c955f867ee3d19eeee5d3.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/d40c9ebc7d8ebc6f92c48a375f5bec5c52927838049c955f867ee3d19eeee5d3.jpg)

![d952220c7f981544cad8c489fd23b76ab9db895ebc247b8aec86b77cc1c891ce.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/d952220c7f981544cad8c489fd23b76ab9db895ebc247b8aec86b77cc1c891ce.jpg)

![e1dc4cfb147172bf2ce22f765d33fb8a8ebaf1cdfc5dbfed1108672fb8d28a0d.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/e1dc4cfb147172bf2ce22f765d33fb8a8ebaf1cdfc5dbfed1108672fb8d28a0d.jpg)

![f9358853f4e92c4ee3ae9376f1d45a69531d5433a1d729df6033070e855aceb8.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/f9358853f4e92c4ee3ae9376f1d45a69531d5433a1d729df6033070e855aceb8.jpg)

![fb89ed3e40340601eaecf6d6052da74ce126f52110a00390cc0ca196b5959ccf.jpg](../iclr_results/1174_NExT-Mol_ 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation/tables/fb89ed3e40340601eaecf6d6052da74ce126f52110a00390cc0ca196b5959ccf.jpg)

## Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions


### Images

![05442535d02647979ebe75d257dc4af9a56bd093ff68d5ec29e1485c10c4c8dc.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/05442535d02647979ebe75d257dc4af9a56bd093ff68d5ec29e1485c10c4c8dc.jpg)

![0775a021767480cd6abfefcd0eec92208b96de270be4eeeb0ab0f0fb573ed2d3.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/0775a021767480cd6abfefcd0eec92208b96de270be4eeeb0ab0f0fb573ed2d3.jpg)

![0c31b0a24ecd883bb1956d0997ffa915376bf6a9d238f1dd3d45030ac0f821ed.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/0c31b0a24ecd883bb1956d0997ffa915376bf6a9d238f1dd3d45030ac0f821ed.jpg)

![0dd575b763200c3c9741ba503756a10dfd4652919656d2c0492e8345bd77fe70.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/0dd575b763200c3c9741ba503756a10dfd4652919656d2c0492e8345bd77fe70.jpg)

![3c3633d5da32790379d0745f89baed719ad73fcab70ad5b4cb384c003ab3092d.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/3c3633d5da32790379d0745f89baed719ad73fcab70ad5b4cb384c003ab3092d.jpg)

![4d635309f54a9959460af07a201249e8e8d70ca5a1ee66cd3cf58cf261266a67.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/4d635309f54a9959460af07a201249e8e8d70ca5a1ee66cd3cf58cf261266a67.jpg)

![62aa5625ba6828b462b3d5e86eaf65932e150efc3576c98e3c4f25f899cdc663.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/62aa5625ba6828b462b3d5e86eaf65932e150efc3576c98e3c4f25f899cdc663.jpg)

![6cc31530fa7699febe2a81c25ac3908dcfd3cc2c9e9252dfe659414484b24a6a.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/6cc31530fa7699febe2a81c25ac3908dcfd3cc2c9e9252dfe659414484b24a6a.jpg)

![6df8fa748a409464f4161e57807f70b2ebbc35f31230395e91d7263db684e1c1.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/6df8fa748a409464f4161e57807f70b2ebbc35f31230395e91d7263db684e1c1.jpg)

![80838cb09a4f26236ce0c3ee77ecbcbff8cc196af35601104c9800f09ada1868.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/80838cb09a4f26236ce0c3ee77ecbcbff8cc196af35601104c9800f09ada1868.jpg)

![94a7e59125ad9f9cd124ceeb0ac0117e40020239d68d1eb258da32efc5ce076d.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/94a7e59125ad9f9cd124ceeb0ac0117e40020239d68d1eb258da32efc5ce076d.jpg)

![c41a0072a6ac4f8bcdbd46ce72a250eec39dd2bb91b40e49bbeb2f51d195255d.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/c41a0072a6ac4f8bcdbd46ce72a250eec39dd2bb91b40e49bbeb2f51d195255d.jpg)

![c4a132abe0f7e0fa327abc91c20ccb5b2edd968ba6c7dd04c2cd0f45d5db8257.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/c4a132abe0f7e0fa327abc91c20ccb5b2edd968ba6c7dd04c2cd0f45d5db8257.jpg)

![cf665f2a236826d688244cf5a3bf64d7184ca3ac82d3c672a86915c17d6f34e5.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/cf665f2a236826d688244cf5a3bf64d7184ca3ac82d3c672a86915c17d6f34e5.jpg)

![e05708653438628967757719ed9ad29232cff90e6acafb0475da84c1d0e2c02b.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/e05708653438628967757719ed9ad29232cff90e6acafb0475da84c1d0e2c02b.jpg)

![e4a5565d2aaec6f1aa97611affc1ef596a60d74a45d65914b1d728ac48e0b328.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/e4a5565d2aaec6f1aa97611affc1ef596a60d74a45d65914b1d728ac48e0b328.jpg)

![e4ccc9314a792a90acd7c468fde2d51fd890d0573d5b6983dd172a55cf52cb2c.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/e4ccc9314a792a90acd7c468fde2d51fd890d0573d5b6983dd172a55cf52cb2c.jpg)

![e5fcd790e78c6f5e250347b67d14e704173b9b982b2da487c343631d461e0b59.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/e5fcd790e78c6f5e250347b67d14e704173b9b982b2da487c343631d461e0b59.jpg)

![f106a5e5f1f8b6d928072ff38a2e9597e99af284a195ed6e8345d192e5542b08.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/images/f106a5e5f1f8b6d928072ff38a2e9597e99af284a195ed6e8345d192e5542b08.jpg)

### Tables

![07e03657bb4fdc924ff7b1c90a9c7afa6f6943240fa69e83162898ad55c43c20.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/tables/07e03657bb4fdc924ff7b1c90a9c7afa6f6943240fa69e83162898ad55c43c20.jpg)

![9fdec04c542ad82fba215273bd387e08ad1d7c2a40d4d66ffe2c3a5786064cd7.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/tables/9fdec04c542ad82fba215273bd387e08ad1d7c2a40d4d66ffe2c3a5786064cd7.jpg)

![ae50e00f9645a6ecdf973a663ec5379041be7e1a3a51db5abe70bd5640fcbb0c.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/tables/ae50e00f9645a6ecdf973a663ec5379041be7e1a3a51db5abe70bd5640fcbb0c.jpg)

![cc6661ec77371d9e0f8e37d5c84e5ee66c5634baf9f790f439553b0f5e9730be.jpg](../iclr_results/1175_Content-Style Learning from Unaligned Domains_ Identifiability under Unknown Latent Dimensions/tables/cc6661ec77371d9e0f8e37d5c84e5ee66c5634baf9f790f439553b0f5e9730be.jpg)

## A Benchmark for Semantic Sensitive Information in LLMs Outputs


### Images

![2f382eef5a9ea387c9d55f9980d30e3b7ed58886f489b7a61b618941433e0772.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/2f382eef5a9ea387c9d55f9980d30e3b7ed58886f489b7a61b618941433e0772.jpg)

![4fdfd222f17e49bcf26c684146a6cb18db621d50ff79dbe2157ada815e91075a.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/4fdfd222f17e49bcf26c684146a6cb18db621d50ff79dbe2157ada815e91075a.jpg)

![60ecbf6d1c46d33aefd75b7d9d541b4b44e22d7f80a3f47a74a1c2d2619124e9.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/60ecbf6d1c46d33aefd75b7d9d541b4b44e22d7f80a3f47a74a1c2d2619124e9.jpg)

![842878943f7cce8139b3b15c8a183c3d6f3675c1301d181eb8006eb542c88bcb.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/842878943f7cce8139b3b15c8a183c3d6f3675c1301d181eb8006eb542c88bcb.jpg)

![8cf10e8bb27d787244f2d95381b950ac5a0b6684098dd70e3f0f89b668cb35a3.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/8cf10e8bb27d787244f2d95381b950ac5a0b6684098dd70e3f0f89b668cb35a3.jpg)

![b44572c40a7987ce8382495559e6f2bb53d7fe89ae3eafa46311dc8417225e1a.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/b44572c40a7987ce8382495559e6f2bb53d7fe89ae3eafa46311dc8417225e1a.jpg)

![bbb9679098f4b8db981a035dd091934ecb375a71fb6e7798e15dcebeca39e0f4.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/bbb9679098f4b8db981a035dd091934ecb375a71fb6e7798e15dcebeca39e0f4.jpg)

![c0c690f20e3f3245be49bc8026946c7ba4f30a66cf7f881ea11b511cdc47317d.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/c0c690f20e3f3245be49bc8026946c7ba4f30a66cf7f881ea11b511cdc47317d.jpg)

![de6df32d5e5df65b1a2ea32bd4f748f0738ec6951ad2fe8d157180f82ae69f78.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/images/de6df32d5e5df65b1a2ea32bd4f748f0738ec6951ad2fe8d157180f82ae69f78.jpg)

### Tables

![0111e69cf3b3026f3610fdffccd2747f0a509c2eb732f1445855f11b9dc98207.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/0111e69cf3b3026f3610fdffccd2747f0a509c2eb732f1445855f11b9dc98207.jpg)

![0796d24506532ad2dc5bfc153ac46a6d6405d652819e28c43abe9e7cbcc473eb.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/0796d24506532ad2dc5bfc153ac46a6d6405d652819e28c43abe9e7cbcc473eb.jpg)

![1209d92917e15cd534048f590a2d38d371268a4a79f7bdf78820674bee742205.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/1209d92917e15cd534048f590a2d38d371268a4a79f7bdf78820674bee742205.jpg)

![2ea27b1c96e4cdf2db68c30fc6aabf6c0870b70ddf4c20572c5c8dda561374ac.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/2ea27b1c96e4cdf2db68c30fc6aabf6c0870b70ddf4c20572c5c8dda561374ac.jpg)

![45d88b040de14114f27400e08db754d97c498f0647d34c5528697720df508d25.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/45d88b040de14114f27400e08db754d97c498f0647d34c5528697720df508d25.jpg)

![aee5aae8ae77b557c008315bd67150c9646a9fa18e82caeb88d3ecad922b32e6.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/aee5aae8ae77b557c008315bd67150c9646a9fa18e82caeb88d3ecad922b32e6.jpg)

![cebc5a3d666359ddcdc7af1f35741052fb58125d1277e3cd031a5f04fd518375.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/cebc5a3d666359ddcdc7af1f35741052fb58125d1277e3cd031a5f04fd518375.jpg)

![e93241c5a7b9bacce3521c42b5963ef510d7e4b471ab290099215eb6024c2678.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/e93241c5a7b9bacce3521c42b5963ef510d7e4b471ab290099215eb6024c2678.jpg)

![f4719cb9333da825a73249092a8bc82568a9a758be8123c94f450790e6611614.jpg](../iclr_results/1176_A Benchmark for Semantic Sensitive Information in LLMs Outputs/tables/f4719cb9333da825a73249092a8bc82568a9a758be8123c94f450790e6611614.jpg)

## Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models


### Images

![017505ee256a4621d98afb2c22378a229a7aef9cf5ce3c7a468cf4838921c119.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/images/017505ee256a4621d98afb2c22378a229a7aef9cf5ce3c7a468cf4838921c119.jpg)

![4cd1a86857ecfdf5541aa808e217a8046910695a92aa22621917a953a57dc36a.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/images/4cd1a86857ecfdf5541aa808e217a8046910695a92aa22621917a953a57dc36a.jpg)

![5f9af6ac04083518ec3f8783a6029c413af41f8067aee1dbab86142ccfc64820.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/images/5f9af6ac04083518ec3f8783a6029c413af41f8067aee1dbab86142ccfc64820.jpg)

![a3cc180637318cbf1e1de53feef7d5556c36f664e296658ead4507afc98015ff.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/images/a3cc180637318cbf1e1de53feef7d5556c36f664e296658ead4507afc98015ff.jpg)

![ab8e9fc0290efea89383fd40a53904b7de63825ed4b93b55b44ca491b1dff58b.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/images/ab8e9fc0290efea89383fd40a53904b7de63825ed4b93b55b44ca491b1dff58b.jpg)

![b2308dcce5a37c7b60337d55ff17fe9d1d63528577a958929e19b25bc1a7f99d.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/images/b2308dcce5a37c7b60337d55ff17fe9d1d63528577a958929e19b25bc1a7f99d.jpg)

![f387d3a85c9f17e45065ed6f4f556a04c85a8687119d2cbd04a0e2c808e873ea.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/images/f387d3a85c9f17e45065ed6f4f556a04c85a8687119d2cbd04a0e2c808e873ea.jpg)

### Tables

![17c44130b763cde151234d064a72c32bbd19c777ef53571c15e0b548c5163622.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/17c44130b763cde151234d064a72c32bbd19c777ef53571c15e0b548c5163622.jpg)

![18c7b610d1018021b7886c06f3db7f5f7dc76d538d4b73d885dab8258f06fc7f.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/18c7b610d1018021b7886c06f3db7f5f7dc76d538d4b73d885dab8258f06fc7f.jpg)

![18f7f88b3a18e3036442ebfe77fc9d29752558950d2bef0aa98f5f66f03ebb86.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/18f7f88b3a18e3036442ebfe77fc9d29752558950d2bef0aa98f5f66f03ebb86.jpg)

![2a68cd3141c8e2573e100a4796c5fefd17900d01bb55f56aab510746752f16e7.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/2a68cd3141c8e2573e100a4796c5fefd17900d01bb55f56aab510746752f16e7.jpg)

![3e2bbc41c475a6071d5bbe0f845ee020e6c16de4e48fef86b91bff17a0b27678.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/3e2bbc41c475a6071d5bbe0f845ee020e6c16de4e48fef86b91bff17a0b27678.jpg)

![422f63659acbaa004b3214fab5c947b47912b11f38d2f0bb8caf7188613d94b4.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/422f63659acbaa004b3214fab5c947b47912b11f38d2f0bb8caf7188613d94b4.jpg)

![430b81da7196c4906e8944a134f9cb1378bd4fd1f401b61d8e5e58ba0ad2dd70.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/430b81da7196c4906e8944a134f9cb1378bd4fd1f401b61d8e5e58ba0ad2dd70.jpg)

![5899bf536f60f89346b24226ccb7002c6ae34d553a78e27454357661d9e27173.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/5899bf536f60f89346b24226ccb7002c6ae34d553a78e27454357661d9e27173.jpg)

![677e63656c41b651dffa7f82c7e3e4ee282738085c7d06234a17d939bf8ec1b2.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/677e63656c41b651dffa7f82c7e3e4ee282738085c7d06234a17d939bf8ec1b2.jpg)

![6c6ef89f55c90c5efc2167379d7c59d49d44df885f8f35d00a4e2f912b2630c8.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/6c6ef89f55c90c5efc2167379d7c59d49d44df885f8f35d00a4e2f912b2630c8.jpg)

![751776ae2499603d0b3863566cc8a17aa63a57e7028383a49a09913458a2108e.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/751776ae2499603d0b3863566cc8a17aa63a57e7028383a49a09913458a2108e.jpg)

![757e9b4c8fe81fe3ff2efeb2b6524e6050e98a9e4d4027093c0b7081277d98b6.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/757e9b4c8fe81fe3ff2efeb2b6524e6050e98a9e4d4027093c0b7081277d98b6.jpg)

![89510cf759e2f3692d1c7c29f32e7cd7d281a98223c12023e7e45a4a9c69c995.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/89510cf759e2f3692d1c7c29f32e7cd7d281a98223c12023e7e45a4a9c69c995.jpg)

![a84e577718639cb2d3b2c746246234670b164ec9c698af0f68599803e9d1c2fe.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/a84e577718639cb2d3b2c746246234670b164ec9c698af0f68599803e9d1c2fe.jpg)

![b6177438491f41568488ee19ba107157422fc945363aa4dddcf50d80001d6582.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/b6177438491f41568488ee19ba107157422fc945363aa4dddcf50d80001d6582.jpg)

![d5083dd84a66341fa0d4e6a79c8c778a62cc79b22790a405964b1f2a7a3fcef4.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/d5083dd84a66341fa0d4e6a79c8c778a62cc79b22790a405964b1f2a7a3fcef4.jpg)

![e1650bd7ebdca16889aa07e162359b73604bd2b636f1198f36573e6d1779475c.jpg](../iclr_results/1177_Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language M/tables/e1650bd7ebdca16889aa07e162359b73604bd2b636f1198f36573e6d1779475c.jpg)

## Spreading Out-of-Distribution Detection on Graphs


### Images

![035334eb1bd0aa7afd9f38442c80e1915b51d5e77a17100882944599d0c5d618.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/035334eb1bd0aa7afd9f38442c80e1915b51d5e77a17100882944599d0c5d618.jpg)

![1dc343ea58e10155ecf134a15b977bc10080720cb6a21715f633869c70e58701.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/1dc343ea58e10155ecf134a15b977bc10080720cb6a21715f633869c70e58701.jpg)

![31810db404da623f65ad48e1bbb66e52bfbdc9b0a8e194ad32127e98537ac888.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/31810db404da623f65ad48e1bbb66e52bfbdc9b0a8e194ad32127e98537ac888.jpg)

![61e88570efe50c5f0c9b702e7f7185573471a233824733a1f2f3d3cde9ca01e9.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/61e88570efe50c5f0c9b702e7f7185573471a233824733a1f2f3d3cde9ca01e9.jpg)

![692cabf5711686b5af2619afe7b371c5896fde6abf21c0fd69d70b7f111380f3.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/692cabf5711686b5af2619afe7b371c5896fde6abf21c0fd69d70b7f111380f3.jpg)

![7fdacce1f62e55529f0c256890737c62f8d37faa0071012ebda7173df99b22bd.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/7fdacce1f62e55529f0c256890737c62f8d37faa0071012ebda7173df99b22bd.jpg)

![87f7922b169502b59398e53e81cd971570b113c53a6761d2b385b99f355d532e.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/87f7922b169502b59398e53e81cd971570b113c53a6761d2b385b99f355d532e.jpg)

![92c0c7a57db2cd3d47ab4269f5b098316af8097225b555bd1c5580a92067e21d.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/92c0c7a57db2cd3d47ab4269f5b098316af8097225b555bd1c5580a92067e21d.jpg)

![dbb07303c510670b530274b5a6d2a0f08cf4eb1c9552e416a876579c474ea7b4.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/images/dbb07303c510670b530274b5a6d2a0f08cf4eb1c9552e416a876579c474ea7b4.jpg)

### Tables

![02684b749ab37793543499500b2b350e34ec630384ee96ce96fee3264bbca213.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/02684b749ab37793543499500b2b350e34ec630384ee96ce96fee3264bbca213.jpg)

![0c854d2af14513c0dbcf0dbe329c54aa844b9b2e4b091d71485c87bd180dc218.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/0c854d2af14513c0dbcf0dbe329c54aa844b9b2e4b091d71485c87bd180dc218.jpg)

![2a8154116a7a5a9efa314aabfdb1ab01fb9482eb282a480a890b5df0c76fe9e6.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/2a8154116a7a5a9efa314aabfdb1ab01fb9482eb282a480a890b5df0c76fe9e6.jpg)

![310085eb739a323bd46fbf538fec0423ad47a9c4acf1eabf25f060a1afa9db8c.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/310085eb739a323bd46fbf538fec0423ad47a9c4acf1eabf25f060a1afa9db8c.jpg)

![3b90299530b7b676ef49ac36f72a6c5b489af3ccec94db8d5abf647ee4a30984.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/3b90299530b7b676ef49ac36f72a6c5b489af3ccec94db8d5abf647ee4a30984.jpg)

![48cc80832e04407aa37bb04ffb902fca51842990b2f044775f754c2d15eaa8dc.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/48cc80832e04407aa37bb04ffb902fca51842990b2f044775f754c2d15eaa8dc.jpg)

![493e272d4170f2d88e5572f5666bc7d343835fc3e561755e88b4ad6ed5446b72.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/493e272d4170f2d88e5572f5666bc7d343835fc3e561755e88b4ad6ed5446b72.jpg)

![4ee9afe3aaf48084efb4a98d613f6c319c03b5d3bca86a0927132ed7d969b055.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/4ee9afe3aaf48084efb4a98d613f6c319c03b5d3bca86a0927132ed7d969b055.jpg)

![5685cbaa38b3b472eab3007956df7c1ebb80f561474bb21b8c27686180d80b16.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/5685cbaa38b3b472eab3007956df7c1ebb80f561474bb21b8c27686180d80b16.jpg)

![736cdc9caf62c5d424fcc549b6351efc40cbcb78ef5416b3b7612c4da2593036.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/736cdc9caf62c5d424fcc549b6351efc40cbcb78ef5416b3b7612c4da2593036.jpg)

![7e0d2c3d459f7ae436bea1d3c913f9962622d07ede2f8c28afb30a4417b576cd.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/7e0d2c3d459f7ae436bea1d3c913f9962622d07ede2f8c28afb30a4417b576cd.jpg)

![8d5605d6f6aeb82fef80278e71bfce6b5dd39279a67e4dd24a12df037778534d.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/8d5605d6f6aeb82fef80278e71bfce6b5dd39279a67e4dd24a12df037778534d.jpg)

![a5993301dc9dbd8a7905683864479f9a5539100c5f6eff4cc6f9f8c1538ab7a5.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/a5993301dc9dbd8a7905683864479f9a5539100c5f6eff4cc6f9f8c1538ab7a5.jpg)

![abf5195e03d8d3b3dde130b2504da12eafc186d85fe1c366b24f522586444259.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/abf5195e03d8d3b3dde130b2504da12eafc186d85fe1c366b24f522586444259.jpg)

![c3db4c8b25a63aba77a79f77b57854312d7fada173aa5c199ff38d69a35b21f2.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/c3db4c8b25a63aba77a79f77b57854312d7fada173aa5c199ff38d69a35b21f2.jpg)

![f16fc8a4ce2991e547874a0b4a7b1d2cc44ae1330ddf498952593bdd62f3506d.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/f16fc8a4ce2991e547874a0b4a7b1d2cc44ae1330ddf498952593bdd62f3506d.jpg)

![fa8cf995750e3f3651eb13f4c142e554db53bb866a0948db54ff3f543bfefe39.jpg](../iclr_results/1178_Spreading Out-of-Distribution Detection on Graphs/tables/fa8cf995750e3f3651eb13f4c142e554db53bb866a0948db54ff3f543bfefe39.jpg)

## Stealthy Shield Defense: A Conditional Mutual Information-Based Approach against Black-Box Model Inversion Attacks


### Images

![14d2a80247a08e6a4d9888aa534710c40c60cfb2b4f9610582ec7d22145a7dea.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/images/14d2a80247a08e6a4d9888aa534710c40c60cfb2b4f9610582ec7d22145a7dea.jpg)

![2a5511ce8dd9bbfe9ff9b88787e36cbf8f84d1540063137cd23b4194c0e13b57.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/images/2a5511ce8dd9bbfe9ff9b88787e36cbf8f84d1540063137cd23b4194c0e13b57.jpg)

![62f57a79353039429b5c7add753932855932c1163358c4f38071db7c2dd7c0df.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/images/62f57a79353039429b5c7add753932855932c1163358c4f38071db7c2dd7c0df.jpg)

![6488a59c65baab3996ad71e3de64644f7a40622466f4b65c86108d06d09c4af0.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/images/6488a59c65baab3996ad71e3de64644f7a40622466f4b65c86108d06d09c4af0.jpg)

### Tables

![0b3e704c084541e8bc008af37aab08d4285991ebc90c9502b72cdf3db10ded1d.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/0b3e704c084541e8bc008af37aab08d4285991ebc90c9502b72cdf3db10ded1d.jpg)

![20a81166a16c4cb42fa82b82b00046bf7fdee63d1923a8c8596b2554e0ab1bdf.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/20a81166a16c4cb42fa82b82b00046bf7fdee63d1923a8c8596b2554e0ab1bdf.jpg)

![40fc4dc6238a378e07b5adc36835fed6d4a7089bce6a1b70d7d35e5a144e5363.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/40fc4dc6238a378e07b5adc36835fed6d4a7089bce6a1b70d7d35e5a144e5363.jpg)

![50e895ec4775dec129c98406d0152cbb1f1cd660c899495aeaea8336bda9e445.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/50e895ec4775dec129c98406d0152cbb1f1cd660c899495aeaea8336bda9e445.jpg)

![7a1568a07a6bf083320c0eda36e20aff66fc08698adc30bfc4d6e16ddac0cd55.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/7a1568a07a6bf083320c0eda36e20aff66fc08698adc30bfc4d6e16ddac0cd55.jpg)

![90392ec25b8d39cfc09773b2da24600c7151a0e0ff09b4943338d8d37fbd22d6.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/90392ec25b8d39cfc09773b2da24600c7151a0e0ff09b4943338d8d37fbd22d6.jpg)

![a2a834d1022e7718b0a9793a9fdb3598e664d2da990847b4a82ea1f2a473ed44.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/a2a834d1022e7718b0a9793a9fdb3598e664d2da990847b4a82ea1f2a473ed44.jpg)

![b72bca26ac1f95531e05468867f063c7682994c467a4cfe8f9035368f731a27c.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/b72bca26ac1f95531e05468867f063c7682994c467a4cfe8f9035368f731a27c.jpg)

![e4f60c4526d3d757d6784bbd2f269a16e27b388a78abdf13a1309dd54811dc7f.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/e4f60c4526d3d757d6784bbd2f269a16e27b388a78abdf13a1309dd54811dc7f.jpg)

![f6b58f3c922c37b826a350fbd96a3e2c4f53a69e0888040fa17ffa461d12236e.jpg](../iclr_results/1179_Stealthy Shield Defense_ A Conditional Mutual Information-Based Approach against Black-Box Model Inv/tables/f6b58f3c922c37b826a350fbd96a3e2c4f53a69e0888040fa17ffa461d12236e.jpg)

## Solving New Tasks by Adapting Internet Video Knowledge


### Images

![17391194b46de2fdc6eae4df05a5af9d99653ddd324b87686e6b64bff907935f.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/images/17391194b46de2fdc6eae4df05a5af9d99653ddd324b87686e6b64bff907935f.jpg)

![820511e57af1bc7c0ed660ab4c1b3dac26b6aa12ae2012f4d9f099ec4ac5b3fe.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/images/820511e57af1bc7c0ed660ab4c1b3dac26b6aa12ae2012f4d9f099ec4ac5b3fe.jpg)

![cd830841638aeb4af04f317fb30cfc1a15b05ae536fcc6d532fa5f6f6eb25655.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/images/cd830841638aeb4af04f317fb30cfc1a15b05ae536fcc6d532fa5f6f6eb25655.jpg)

![ec4013e2328b3f067b3b806ebfb5f706077799c8731aef3c2a78cf55722365e3.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/images/ec4013e2328b3f067b3b806ebfb5f706077799c8731aef3c2a78cf55722365e3.jpg)

![f068c9566be76554c63fedd3a544c4f523d2ee502e5ebcad5ba47db85d95e59b.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/images/f068c9566be76554c63fedd3a544c4f523d2ee502e5ebcad5ba47db85d95e59b.jpg)

### Tables

![006a98d29a61637cbadfac4fd3ceed2e47c8fd10470f7d2c115eaa4f51c69bb0.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/006a98d29a61637cbadfac4fd3ceed2e47c8fd10470f7d2c115eaa4f51c69bb0.jpg)

![080c2319c548d9c1e0c2bdec189a535359d0c7e45017e97d4b8e3629ad4e882c.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/080c2319c548d9c1e0c2bdec189a535359d0c7e45017e97d4b8e3629ad4e882c.jpg)

![169743ab1f7573631915cbf2f95c21f80a9b1539016ad91caba8c12a225d5a6e.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/169743ab1f7573631915cbf2f95c21f80a9b1539016ad91caba8c12a225d5a6e.jpg)

![196d59f0fc3002e0592f8b9af9c067eea35da19d22e4d2754f3f1cc1fa42c40c.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/196d59f0fc3002e0592f8b9af9c067eea35da19d22e4d2754f3f1cc1fa42c40c.jpg)

![30e3cdb1f737797633a7794b15145a270ce5de148310525c62293394d7850877.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/30e3cdb1f737797633a7794b15145a270ce5de148310525c62293394d7850877.jpg)

![414e63ccf56a17520b55aa1bd6df220a78a095eb95ad7e33f05be6fd12081ed4.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/414e63ccf56a17520b55aa1bd6df220a78a095eb95ad7e33f05be6fd12081ed4.jpg)

![658696b828639566411fdd59eaeaab44b36e257e6c3284bb1d2996870a3bce52.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/658696b828639566411fdd59eaeaab44b36e257e6c3284bb1d2996870a3bce52.jpg)

![70861cc071a5ffc240e507a24cf4fea6c758d89a0c1de0c2f1a342bc71b159ca.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/70861cc071a5ffc240e507a24cf4fea6c758d89a0c1de0c2f1a342bc71b159ca.jpg)

![711b85d22a42ddbcdcabe9da360efc0fe75b8e0ed0e98091b8b51230803c58e3.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/711b85d22a42ddbcdcabe9da360efc0fe75b8e0ed0e98091b8b51230803c58e3.jpg)

![7286821af58262f1f0a312cd4008f9754741a549c0d425c8f403786936edd9f2.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/7286821af58262f1f0a312cd4008f9754741a549c0d425c8f403786936edd9f2.jpg)

![817a97b75c3f0553bfbbbbc1b75e1191a3cbcd36384f30f3a48edb426a416080.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/817a97b75c3f0553bfbbbbc1b75e1191a3cbcd36384f30f3a48edb426a416080.jpg)

![8a03ea23db054bac9621a0789bef79148e9be2f70ca65767f727be4d5b8bb616.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/8a03ea23db054bac9621a0789bef79148e9be2f70ca65767f727be4d5b8bb616.jpg)

![a1137aa62465b7af5af2c10eaa9235be2a78db8a274bcbd8401c546bb212080a.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/a1137aa62465b7af5af2c10eaa9235be2a78db8a274bcbd8401c546bb212080a.jpg)

![ac52b57c62a5d890fb3e212e2cdeec53682efd65c4a965b4a3a3f51a4df203e0.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/ac52b57c62a5d890fb3e212e2cdeec53682efd65c4a965b4a3a3f51a4df203e0.jpg)

![bd4faf031f931a710e631e49b46b02f4c17062422caea41220fe01b2f3f84a1e.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/bd4faf031f931a710e631e49b46b02f4c17062422caea41220fe01b2f3f84a1e.jpg)

![c15739d3d4988826ce865affc9e6b9c891ad5d4d36ca9162b0eaaf5ae448bb2f.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/c15739d3d4988826ce865affc9e6b9c891ad5d4d36ca9162b0eaaf5ae448bb2f.jpg)

![c96fd97477e70168b42b5a53e8530bc9eb04a01c4e8548ebee127fa1efa2ecdd.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/c96fd97477e70168b42b5a53e8530bc9eb04a01c4e8548ebee127fa1efa2ecdd.jpg)

![d99a457f641aef1bce43072e13a83a0942d2de5cd897cfaf8c7231a1144fbedd.jpg](../iclr_results/1180_Solving New Tasks by Adapting Internet Video Knowledge/tables/d99a457f641aef1bce43072e13a83a0942d2de5cd897cfaf8c7231a1144fbedd.jpg)

## Learning Causal Alignment for Reliable Disease Diagnosis


### Images

![3531ac96e559832d7b32c3d545a12d86cacc41a6aeff90aeb2fcf14cba3ec9ac.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/3531ac96e559832d7b32c3d545a12d86cacc41a6aeff90aeb2fcf14cba3ec9ac.jpg)

![3c2382032a9ecf565e8875cb3054b6f4d7df4cbb6418294368d8e5e2220d53ec.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/3c2382032a9ecf565e8875cb3054b6f4d7df4cbb6418294368d8e5e2220d53ec.jpg)

![5a7da0764bd2bb5807c399b2093c203a2dbb1fcdb3539191502568a8fcae5bf6.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/5a7da0764bd2bb5807c399b2093c203a2dbb1fcdb3539191502568a8fcae5bf6.jpg)

![5af4f10f1c9d2a0d0da3fbeb22ceb9d7ac5fdc47027a0fe6232dc039fe5846d4.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/5af4f10f1c9d2a0d0da3fbeb22ceb9d7ac5fdc47027a0fe6232dc039fe5846d4.jpg)

![6e7fd8b29219a42dbaa2472c29e2deca8f1545f4fa9203765b309a6cfb5d093e.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/6e7fd8b29219a42dbaa2472c29e2deca8f1545f4fa9203765b309a6cfb5d093e.jpg)

![751f07721b169daa60dd96454de18a3c9b003c12d7db913c5b973742a2505452.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/751f07721b169daa60dd96454de18a3c9b003c12d7db913c5b973742a2505452.jpg)

![88aaadc8ba40805c630a8e14193ed484ba73ce0528becdd8fc0532cb3a1ea6ab.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/88aaadc8ba40805c630a8e14193ed484ba73ce0528becdd8fc0532cb3a1ea6ab.jpg)

![c51d70e806b2cad8964473fef4327eed5f0ed81635c3a837fd09f9f43529bd9b.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/c51d70e806b2cad8964473fef4327eed5f0ed81635c3a837fd09f9f43529bd9b.jpg)

![f56e05209173253a781ff2acc954818a1d136b679380d23e8da39259dc13fae8.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/images/f56e05209173253a781ff2acc954818a1d136b679380d23e8da39259dc13fae8.jpg)

### Tables

![100ca8235c6c2b63566899beab7094655612d36aa5049572e9a09f696213f0d3.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/tables/100ca8235c6c2b63566899beab7094655612d36aa5049572e9a09f696213f0d3.jpg)

![413cf5911f8a5273acc9bb88a672176dff36e8916fe9a66bdbaa0b4e46502c98.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/tables/413cf5911f8a5273acc9bb88a672176dff36e8916fe9a66bdbaa0b4e46502c98.jpg)

![a5688cf53c41f7d403786fc12265995ae443323e9591dc162506b4d88d9e604a.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/tables/a5688cf53c41f7d403786fc12265995ae443323e9591dc162506b4d88d9e604a.jpg)

![c690a940a10d23a853b66f8aad618660ae92d69039d9b33af5b6902e5c06e7c6.jpg](../iclr_results/1181_Learning Causal Alignment for Reliable Disease Diagnosis/tables/c690a940a10d23a853b66f8aad618660ae92d69039d9b33af5b6902e5c06e7c6.jpg)

## Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning


### Images

![3e96484e694880aef46a82a0657de6c8991b85916663e029b4f9e8c1c7bc5c1e.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/images/3e96484e694880aef46a82a0657de6c8991b85916663e029b4f9e8c1c7bc5c1e.jpg)

![6c9209d8cb6a5c253603d5a5992e5d002f92b7140cc27c3892d741609124146a.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/images/6c9209d8cb6a5c253603d5a5992e5d002f92b7140cc27c3892d741609124146a.jpg)

![b3f486b60c5adb1fe7e501d1abbac427d63c148894d84f2ed2fc7b635d9c201f.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/images/b3f486b60c5adb1fe7e501d1abbac427d63c148894d84f2ed2fc7b635d9c201f.jpg)

![c3d133b3a656d0a62568c2c7bdefeb17cf7245efe6b3bcf243dc0e344c05bd33.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/images/c3d133b3a656d0a62568c2c7bdefeb17cf7245efe6b3bcf243dc0e344c05bd33.jpg)

### Tables

![15337dd57b0a6fb62147968057f102af4af928aa93b330a0be78df1e87a83985.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/15337dd57b0a6fb62147968057f102af4af928aa93b330a0be78df1e87a83985.jpg)

![1e92b3e901c0b3fa866c33ab82741463f63539e4b59baee517fd655bd2a75107.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/1e92b3e901c0b3fa866c33ab82741463f63539e4b59baee517fd655bd2a75107.jpg)

![5dd6ddd85da69d6ad8bd1eaac44e3bd3ebb81f9379ed31af7885d3e05d12070e.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/5dd6ddd85da69d6ad8bd1eaac44e3bd3ebb81f9379ed31af7885d3e05d12070e.jpg)

![65f751284cdd8df34483ce9e3c83e80d6857edf477cd1ee7cb836bae7a92fb34.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/65f751284cdd8df34483ce9e3c83e80d6857edf477cd1ee7cb836bae7a92fb34.jpg)

![b92fc5686cc927b9642d90ff34a4da1f0cfb0a0f2a4ead2af8379e379551dc53.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/b92fc5686cc927b9642d90ff34a4da1f0cfb0a0f2a4ead2af8379e379551dc53.jpg)

![c6e5e7e6cbdaa510befe8cc35cdc6900f9f5d1abd3f5463fa263d08fc668628c.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/c6e5e7e6cbdaa510befe8cc35cdc6900f9f5d1abd3f5463fa263d08fc668628c.jpg)

![e156f5d36c18d53d350c849d929b6e86947c0311714c41db5e013416902f3dfb.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/e156f5d36c18d53d350c849d929b6e86947c0311714c41db5e013416902f3dfb.jpg)

![ee414894ccc16a513d7db57fd90931f97a77f6fa1aee58ec7581a38c08275fa2.jpg](../iclr_results/1182_Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning/tables/ee414894ccc16a513d7db57fd90931f97a77f6fa1aee58ec7581a38c08275fa2.jpg)

## GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning


### Images

![ae0e964d77e2ce1ccb71fc7e290bf18a506e0ff01365b18463264f1299111504.jpg](../iclr_results/1183_GlycanML_ A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning/images/ae0e964d77e2ce1ccb71fc7e290bf18a506e0ff01365b18463264f1299111504.jpg)

![e085c4390989f9b0bdf892545f04eeba93ce93e05b83470a81bfc9bfcff1a5d0.jpg](../iclr_results/1183_GlycanML_ A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning/images/e085c4390989f9b0bdf892545f04eeba93ce93e05b83470a81bfc9bfcff1a5d0.jpg)

### Tables

![25ff13aeb4b3d2051c97da0e567aedc0801a4dcc66e6f10d88ed32cdf50b62fc.jpg](../iclr_results/1183_GlycanML_ A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning/tables/25ff13aeb4b3d2051c97da0e567aedc0801a4dcc66e6f10d88ed32cdf50b62fc.jpg)

![64685f19dc7cc103f982c6e2ad56107deac660a49d2f13ea8852ae5aad883263.jpg](../iclr_results/1183_GlycanML_ A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning/tables/64685f19dc7cc103f982c6e2ad56107deac660a49d2f13ea8852ae5aad883263.jpg)

![a89070ba942d11d812a2595209f7b7dba7897543092011c2434a6a30720c62d7.jpg](../iclr_results/1183_GlycanML_ A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning/tables/a89070ba942d11d812a2595209f7b7dba7897543092011c2434a6a30720c62d7.jpg)

![cc3df5ecdc8f846b65a87537efcf1d403a90fead78171633cc7ce5b412bcea22.jpg](../iclr_results/1183_GlycanML_ A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning/tables/cc3df5ecdc8f846b65a87537efcf1d403a90fead78171633cc7ce5b412bcea22.jpg)

![cedbc16bf9a5971f4409a8bd5bb0bb2f3abac2b434bc7634729c46419ebf7226.jpg](../iclr_results/1183_GlycanML_ A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning/tables/cedbc16bf9a5971f4409a8bd5bb0bb2f3abac2b434bc7634729c46419ebf7226.jpg)

## Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models


### Images

![3954e893b7650e9c9e483cbe757e81976b83c7fd8f8ad2eb12ecf8d14e6dcc40.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/3954e893b7650e9c9e483cbe757e81976b83c7fd8f8ad2eb12ecf8d14e6dcc40.jpg)

![3efc1479a3b709f5fc99bd161aedfc8ac1ba6c2364a8702c57249d75c0974e29.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/3efc1479a3b709f5fc99bd161aedfc8ac1ba6c2364a8702c57249d75c0974e29.jpg)

![3efda5251b9cb19a1d193cf021bf0b77317ac4d4d8ce49c1f0211daffa1e9643.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/3efda5251b9cb19a1d193cf021bf0b77317ac4d4d8ce49c1f0211daffa1e9643.jpg)

![445956ec7a8e0225adaf8a29c7d80ce0aeddec3b1a909f345e353909bfdc802f.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/445956ec7a8e0225adaf8a29c7d80ce0aeddec3b1a909f345e353909bfdc802f.jpg)

![4a8bd4571e9a89fe0bf08bb05b33569e471a662c7df1142126a8e0bebfabca12.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/4a8bd4571e9a89fe0bf08bb05b33569e471a662c7df1142126a8e0bebfabca12.jpg)

![62e40f0fdb95d6a989e931465cd271becece7c29f8a4bec221c0483113c598de.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/62e40f0fdb95d6a989e931465cd271becece7c29f8a4bec221c0483113c598de.jpg)

![71756d7daba5b8b7442df30fcbeeea828a848673bda2b4b0201bedf472b25e58.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/71756d7daba5b8b7442df30fcbeeea828a848673bda2b4b0201bedf472b25e58.jpg)

![7641ec94a427b8a874ced23399d0204c8c98c27219c252e5f4a284070b843d2e.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/7641ec94a427b8a874ced23399d0204c8c98c27219c252e5f4a284070b843d2e.jpg)

![78dc403ef9630d84751ae1af101f03aaf662d0a38f8bae14572f29f91263dff1.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/78dc403ef9630d84751ae1af101f03aaf662d0a38f8bae14572f29f91263dff1.jpg)

![941c99f10463be6e3dbb09c75f51ac74efefe88fff308d04f6e7fb6664b98f05.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/941c99f10463be6e3dbb09c75f51ac74efefe88fff308d04f6e7fb6664b98f05.jpg)

![a4cd83a132ffb25e0b0db4862a6140d9fdf84d6de904994e12186f83ebe50f1f.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/a4cd83a132ffb25e0b0db4862a6140d9fdf84d6de904994e12186f83ebe50f1f.jpg)

![a8201250ab2287aad7d9a573e29248f31d5c7ac7e8a1ef5bcfe70c99cbaff24a.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/a8201250ab2287aad7d9a573e29248f31d5c7ac7e8a1ef5bcfe70c99cbaff24a.jpg)

![a9bb4bc994793666b04dc33de7c30d37b176fa07674f83a71be3e7be6d126d38.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/a9bb4bc994793666b04dc33de7c30d37b176fa07674f83a71be3e7be6d126d38.jpg)

![c3d3415fb80df4cdeba1ed11e22d59186d824c05dbc68d6222dabd855f69a0e6.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/c3d3415fb80df4cdeba1ed11e22d59186d824c05dbc68d6222dabd855f69a0e6.jpg)

![cb49f5ec5b62cab29fcab1a32d08f0aadd3b389a1314a0df5b541d3955b3f818.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/cb49f5ec5b62cab29fcab1a32d08f0aadd3b389a1314a0df5b541d3955b3f818.jpg)

![e8715c9a07974a8be8c8b9c0a9bf76c8d5c7a895485fa4489621d9689a9ec1ad.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/images/e8715c9a07974a8be8c8b9c0a9bf76c8d5c7a895485fa4489621d9689a9ec1ad.jpg)

### Tables

![8d6959f9694a7fe4baf07aeea1f147844727257e0d83718238049844409896a1.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/tables/8d6959f9694a7fe4baf07aeea1f147844727257e0d83718238049844409896a1.jpg)

![99b9dcdf52d49185e5da4471c48b28779db7836652bc22e1d9808852961c9354.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/tables/99b9dcdf52d49185e5da4471c48b28779db7836652bc22e1d9808852961c9354.jpg)

![a492e46b2bc9b5f29b866e2afb500b508d0baa68bf22d359e23c1fad82e3d53d.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/tables/a492e46b2bc9b5f29b866e2afb500b508d0baa68bf22d359e23c1fad82e3d53d.jpg)

![d39f12c0dd4d0478cb601742b1591a60066e5cb79d6461b7081e1abb0f1d7569.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/tables/d39f12c0dd4d0478cb601742b1591a60066e5cb79d6461b7081e1abb0f1d7569.jpg)

![d7618ab779a2253cbc77ea432dd7ef0a68e5d6668667af541eaa19a94f88f6c2.jpg](../iclr_results/1184_Hypothetical Minds_ Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models/tables/d7618ab779a2253cbc77ea432dd7ef0a68e5d6668667af541eaa19a94f88f6c2.jpg)

## Stable Segment Anything Model


### Images

![07d5223a634701e5e377384293be66544555b8414e7f04b1adc66f0e4a3df0c0.jpg](../iclr_results/1185_Stable Segment Anything Model/images/07d5223a634701e5e377384293be66544555b8414e7f04b1adc66f0e4a3df0c0.jpg)

![184eb0c4248265a48c56494d2474b6bacaa56c64cd7ed60be1fc58e3d71d862a.jpg](../iclr_results/1185_Stable Segment Anything Model/images/184eb0c4248265a48c56494d2474b6bacaa56c64cd7ed60be1fc58e3d71d862a.jpg)

![27748102bbb300956590786e0e7ffa5bd357ee2f6f1e4a05ecee8660a5aac774.jpg](../iclr_results/1185_Stable Segment Anything Model/images/27748102bbb300956590786e0e7ffa5bd357ee2f6f1e4a05ecee8660a5aac774.jpg)

![bb839f098853b1028efe8937751e574375b11bdd288e1ecc97ab952e97a7ee53.jpg](../iclr_results/1185_Stable Segment Anything Model/images/bb839f098853b1028efe8937751e574375b11bdd288e1ecc97ab952e97a7ee53.jpg)

### Tables

![0dac3970bd517b8f7c63adb219d53317fa78a376895e28bde8756f8770a332f4.jpg](../iclr_results/1185_Stable Segment Anything Model/tables/0dac3970bd517b8f7c63adb219d53317fa78a376895e28bde8756f8770a332f4.jpg)

![49b14811564e22f97ac9168efd6b83f193db3db39e4dfa24c2880ecdfc62c6dd.jpg](../iclr_results/1185_Stable Segment Anything Model/tables/49b14811564e22f97ac9168efd6b83f193db3db39e4dfa24c2880ecdfc62c6dd.jpg)

![81310d8a0b4e2228ad056b783af70848ee0048d3226b6b4e7db1b7ac038c613d.jpg](../iclr_results/1185_Stable Segment Anything Model/tables/81310d8a0b4e2228ad056b783af70848ee0048d3226b6b4e7db1b7ac038c613d.jpg)

![9159c6072be7f12f5d7e3e03404e4fdfc4893a60995fa6f954e46a1622090139.jpg](../iclr_results/1185_Stable Segment Anything Model/tables/9159c6072be7f12f5d7e3e03404e4fdfc4893a60995fa6f954e46a1622090139.jpg)

![9e09f372636fb265b33cdf3c0246da77b8c8db2dc559c511f69bb402ad60adb7.jpg](../iclr_results/1185_Stable Segment Anything Model/tables/9e09f372636fb265b33cdf3c0246da77b8c8db2dc559c511f69bb402ad60adb7.jpg)

![a10554fbb284d0ea254e35f0301e86a704dddc447f9c8c9cdc076ff3ff7396a5.jpg](../iclr_results/1185_Stable Segment Anything Model/tables/a10554fbb284d0ea254e35f0301e86a704dddc447f9c8c9cdc076ff3ff7396a5.jpg)

![c1ce624122be42c075b95849c5bac1f5ef548d11397d9be51c2880aa789cf71e.jpg](../iclr_results/1185_Stable Segment Anything Model/tables/c1ce624122be42c075b95849c5bac1f5ef548d11397d9be51c2880aa789cf71e.jpg)

## Equivariant Denoisers Cannot Copy Graphs: Align Your Graph Diffusion Models


### Images

![0e24a75f272f3898dcce8dcb4dff9c0005fd406821d7c5e53b2f2d16801f12c8.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/0e24a75f272f3898dcce8dcb4dff9c0005fd406821d7c5e53b2f2d16801f12c8.jpg)

![0f424f3710185b3d613364256e610eef2498abc7511db9204466bf939455deef.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/0f424f3710185b3d613364256e610eef2498abc7511db9204466bf939455deef.jpg)

![24cc7e3aa4d4ef719f1087fef745f2decb3ae2664c71ae4f0ad954360c5103b8.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/24cc7e3aa4d4ef719f1087fef745f2decb3ae2664c71ae4f0ad954360c5103b8.jpg)

![2bd44f6de2ac8aa10500222e9125305b38e036f3e42439cc6d78484dcc736f70.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/2bd44f6de2ac8aa10500222e9125305b38e036f3e42439cc6d78484dcc736f70.jpg)

![38792b70a1ffd3150c5c1cdb0c1af7e6b9388b2eeb82d8ccb20e518b6647a5e8.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/38792b70a1ffd3150c5c1cdb0c1af7e6b9388b2eeb82d8ccb20e518b6647a5e8.jpg)

![61e18c38e352e81a07e164a78b7a2a4bb0a9e0a30a528b9cada2db81e0539b15.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/61e18c38e352e81a07e164a78b7a2a4bb0a9e0a30a528b9cada2db81e0539b15.jpg)

![771ed7fd1ba3ae05f8e4eb7bb8fecbce0427c84c61c1a7137f077a8c56f48f61.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/771ed7fd1ba3ae05f8e4eb7bb8fecbce0427c84c61c1a7137f077a8c56f48f61.jpg)

![7d3ef4343bdb869add634d91758d4ea96bf63f46dcb14b138914750f6387a322.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/7d3ef4343bdb869add634d91758d4ea96bf63f46dcb14b138914750f6387a322.jpg)

![87530584801439cc95931d99b0674644a8bd9846b530e588f6d55af24c55746f.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/87530584801439cc95931d99b0674644a8bd9846b530e588f6d55af24c55746f.jpg)

![9cb6738abc166be919a4e7a7a4d840bfeb065a4b575b87372aeec48bb3042032.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/9cb6738abc166be919a4e7a7a4d840bfeb065a4b575b87372aeec48bb3042032.jpg)

![9cf817497550e27acf6ffe38315b7141b425a8cdaba70e7b8762af41b913e58c.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/9cf817497550e27acf6ffe38315b7141b425a8cdaba70e7b8762af41b913e58c.jpg)

![9df9d8c6a17cd45de27ec4d08479794c45d388a8dc15ab2b7d2e4dff43351778.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/9df9d8c6a17cd45de27ec4d08479794c45d388a8dc15ab2b7d2e4dff43351778.jpg)

![be42f8ba39662e27d9de4fcab5826c3a18ed49f84d902a61d7ee0d31e015158b.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/be42f8ba39662e27d9de4fcab5826c3a18ed49f84d902a61d7ee0d31e015158b.jpg)

![c13a58d29aab79746aa3bc6ef64b7492323f3e32899e0dc2c4822cc16c122db1.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/c13a58d29aab79746aa3bc6ef64b7492323f3e32899e0dc2c4822cc16c122db1.jpg)

![e7045e56bcd4212ddbbf8cf01bfb49837f2c6928fc85a6b488babef2b90b7363.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/e7045e56bcd4212ddbbf8cf01bfb49837f2c6928fc85a6b488babef2b90b7363.jpg)

![fbd6c9f9e7748de1f51c5f7a5174ec2c8a5a2b5eb455d664f6bb196709628d38.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/images/fbd6c9f9e7748de1f51c5f7a5174ec2c8a5a2b5eb455d664f6bb196709628d38.jpg)

### Tables

![33c8391e3e3cbb51578c0f6024d12fcf6700fdd8955f6a34a90bc1e4a1fe2a8a.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/33c8391e3e3cbb51578c0f6024d12fcf6700fdd8955f6a34a90bc1e4a1fe2a8a.jpg)

![3c9d6c54f6fd0526a46dddd35c9ab515348189e6fdc46ed008e795307f72d682.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/3c9d6c54f6fd0526a46dddd35c9ab515348189e6fdc46ed008e795307f72d682.jpg)

![433976d4594519244d1de33be94d6551b9f4fdbde8884f97c3d8cf7944991e38.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/433976d4594519244d1de33be94d6551b9f4fdbde8884f97c3d8cf7944991e38.jpg)

![53e36e0d0cfd4616bb6d62ddee1c54550ad90dc7266f9535ff7a4b3ddeb21edb.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/53e36e0d0cfd4616bb6d62ddee1c54550ad90dc7266f9535ff7a4b3ddeb21edb.jpg)

![712e898a4ce07a9cae77c41f683a358cc96e2b21b08bf4c33a187513870c64dc.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/712e898a4ce07a9cae77c41f683a358cc96e2b21b08bf4c33a187513870c64dc.jpg)

![837290dabcfc82ed9ac82170d09ef045b641dce02cc4c54b0a1b4bef263f12f7.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/837290dabcfc82ed9ac82170d09ef045b641dce02cc4c54b0a1b4bef263f12f7.jpg)

![887ad8594287fde2c68aa6dbb97fbcf29967d5d5a0bf4faa32df424923549fe3.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/887ad8594287fde2c68aa6dbb97fbcf29967d5d5a0bf4faa32df424923549fe3.jpg)

![8e30108a134d1205ee734671de07308bbd025750b553329bdf8eb03c68e89d60.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/8e30108a134d1205ee734671de07308bbd025750b553329bdf8eb03c68e89d60.jpg)

![a86be392c44c9ed5c51b6a59edcfa1c9788b70aea8b5dcf87e64ec999fe5ccd8.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/a86be392c44c9ed5c51b6a59edcfa1c9788b70aea8b5dcf87e64ec999fe5ccd8.jpg)

![b33ad90bf6fc1da4153b9d0be27405864a7be489edb8b5edfd20489a129cce5a.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/b33ad90bf6fc1da4153b9d0be27405864a7be489edb8b5edfd20489a129cce5a.jpg)

![c4a43b00ea359c161ab549ec24aa4d460ba471a1040206e7a1f768a3517c0a6d.jpg](../iclr_results/1186_Equivariant Denoisers Cannot Copy Graphs_ Align Your Graph Diffusion Models/tables/c4a43b00ea359c161ab549ec24aa4d460ba471a1040206e7a1f768a3517c0a6d.jpg)

## Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization


### Images

![7a7a5dfcc8edb7384ff6e0d2e884a0bd1c1504af0d2fec3081a8f81d7fd5bccc.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/images/7a7a5dfcc8edb7384ff6e0d2e884a0bd1c1504af0d2fec3081a8f81d7fd5bccc.jpg)

![8b18667696ff4f61da89cf2ab0b9aa444903af1dc7e3787876997a47db06d0be.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/images/8b18667696ff4f61da89cf2ab0b9aa444903af1dc7e3787876997a47db06d0be.jpg)

![c116e2f5ca9d85857754efc940901a5d02f915aef85b7147ff6e7909358a0ad6.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/images/c116e2f5ca9d85857754efc940901a5d02f915aef85b7147ff6e7909358a0ad6.jpg)

![cf22d833fde96713c53ffaf3c7c79a67c86cabb8f5cc2c69d01ad73146dc0a6c.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/images/cf22d833fde96713c53ffaf3c7c79a67c86cabb8f5cc2c69d01ad73146dc0a6c.jpg)

![d921302a9e5c8e28f8067f0314f426870d9e2ce65b927681e85fe8c6913aadc6.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/images/d921302a9e5c8e28f8067f0314f426870d9e2ce65b927681e85fe8c6913aadc6.jpg)

![f71e90125ca7488b5b779ed6652d742884a483f3f4ece732e94edd1c8b32d894.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/images/f71e90125ca7488b5b779ed6652d742884a483f3f4ece732e94edd1c8b32d894.jpg)

### Tables

![2201d1b68283882699a81e9beeb1c849f00dcf8f811b295968bc61e78fb7230d.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/tables/2201d1b68283882699a81e9beeb1c849f00dcf8f811b295968bc61e78fb7230d.jpg)

![3cfd9f1fa5da9c1a9398856f4321d0a6f878dd7f83a8b7cb9d3cf87e99cb10bf.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/tables/3cfd9f1fa5da9c1a9398856f4321d0a6f878dd7f83a8b7cb9d3cf87e99cb10bf.jpg)

![56746187d8244433a2785883ccd7b7bb3ec99016a5a83cadf21e5abcc4f23284.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/tables/56746187d8244433a2785883ccd7b7bb3ec99016a5a83cadf21e5abcc4f23284.jpg)

![754438f4622f1e798b700e4c6baa31bb549cea2f23ef9f9aef7bcbc38f7c382f.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/tables/754438f4622f1e798b700e4c6baa31bb549cea2f23ef9f9aef7bcbc38f7c382f.jpg)

![a1f16d1bce71e32214fc1c38ff1b078d2328db74268edb99ba0ffe7fbb67d498.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/tables/a1f16d1bce71e32214fc1c38ff1b078d2328db74268edb99ba0ffe7fbb67d498.jpg)

![b9a14110094fbd6d53b8502fb25a2418bc901cf505aac1b6b0ff5b9bfbb81ba2.jpg](../iclr_results/1187_Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization/tables/b9a14110094fbd6d53b8502fb25a2418bc901cf505aac1b6b0ff5b9bfbb81ba2.jpg)

## Large (Vision) Language Models are Unsupervised In-Context Learners


### Images

![2216edda9005f9b2c8a359b6ceed0883353c1636485741c875e0729bc3e417b6.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/2216edda9005f9b2c8a359b6ceed0883353c1636485741c875e0729bc3e417b6.jpg)

![4b885fb356ee97c2964139754848ec0e0a4f1011cd8e961fe638fc78f779971b.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/4b885fb356ee97c2964139754848ec0e0a4f1011cd8e961fe638fc78f779971b.jpg)

![554b306ff30d7e4b480dbca811166619efb2dfa80e79c89e7e76c2f2d0e7f94d.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/554b306ff30d7e4b480dbca811166619efb2dfa80e79c89e7e76c2f2d0e7f94d.jpg)

![5ea705245e885ede2090ce091b2cd47261c20edb21dc565a958bcbc4c83200c2.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/5ea705245e885ede2090ce091b2cd47261c20edb21dc565a958bcbc4c83200c2.jpg)

![793a205fa19cb4ab9d12017c36bb7c9681d7b51e6b879cad18efb41b73607a22.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/793a205fa19cb4ab9d12017c36bb7c9681d7b51e6b879cad18efb41b73607a22.jpg)

![881bcff38a595ee224f9621a45165f195dd040b71138946f47d8de8e902d34ff.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/881bcff38a595ee224f9621a45165f195dd040b71138946f47d8de8e902d34ff.jpg)

![aee2bbafbac91db9e66ccafc5c68d183881ddc9b5bbe0f629451e14a700f0ef5.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/aee2bbafbac91db9e66ccafc5c68d183881ddc9b5bbe0f629451e14a700f0ef5.jpg)

![ccc550efca9cb3e96d4117a7d0e7f30ce7ce58fa307c8d6b92a40364bc85ed3c.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/ccc550efca9cb3e96d4117a7d0e7f30ce7ce58fa307c8d6b92a40364bc85ed3c.jpg)

![d792aa78195f7730051404d8475b2036269d9bc88ff97c82cbba5b74d29bfc9a.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/images/d792aa78195f7730051404d8475b2036269d9bc88ff97c82cbba5b74d29bfc9a.jpg)

### Tables

![22204a7a4402270e1f1cd39d509ac44bfddb10b6aec9ee6a89ca651fb4898a42.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/22204a7a4402270e1f1cd39d509ac44bfddb10b6aec9ee6a89ca651fb4898a42.jpg)

![396fe3d6a61f37302097b9f50f0f82683d4dd77d4772b8705cb9ed663533968c.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/396fe3d6a61f37302097b9f50f0f82683d4dd77d4772b8705cb9ed663533968c.jpg)

![42e91e9c896ccf0c0f1464f274259e8865dea8059d791df0019096060b701e2f.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/42e91e9c896ccf0c0f1464f274259e8865dea8059d791df0019096060b701e2f.jpg)

![bac1faba9f30a1fe111e09cc3f550914b3580f40af8a3d7536933edefe0ef0e6.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/bac1faba9f30a1fe111e09cc3f550914b3580f40af8a3d7536933edefe0ef0e6.jpg)

![e070768ef5a3e1e044311c731b0da8e259f526e469c279fe9d50301a40d04bb8.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/e070768ef5a3e1e044311c731b0da8e259f526e469c279fe9d50301a40d04bb8.jpg)

![ec46c47825955cd03438e768fb6aaedf2300337743640be1173c863c1f617e28.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/ec46c47825955cd03438e768fb6aaedf2300337743640be1173c863c1f617e28.jpg)

![eeeb07ed478b0bd0ecf7fdb075d496fe5faf46df4a026c0cee574b29ae8ab638.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/eeeb07ed478b0bd0ecf7fdb075d496fe5faf46df4a026c0cee574b29ae8ab638.jpg)

![f94f631e263cef41db87f27f4bd794323964c7022ca8de59419f958aef1f54c9.jpg](../iclr_results/1188_Large (Vision) Language Models are Unsupervised In-Context Learners/tables/f94f631e263cef41db87f27f4bd794323964c7022ca8de59419f958aef1f54c9.jpg)

## ColPali: Efficient Document Retrieval with Vision Language Models


### Images

![0db404643be475fef014fac7bbfe0ca8ff6efed6bf2bf85c10695b13009adbc5.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/0db404643be475fef014fac7bbfe0ca8ff6efed6bf2bf85c10695b13009adbc5.jpg)

![121871eb80f4f0e6e6464dd50ada13aabc4a672b04e8c38c14a3b1618f0b242e.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/121871eb80f4f0e6e6464dd50ada13aabc4a672b04e8c38c14a3b1618f0b242e.jpg)

![19896ca83aa7cd9e455c1eb99ab4dae73c583c8cd57e0c2534769181b0ad6cbf.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/19896ca83aa7cd9e455c1eb99ab4dae73c583c8cd57e0c2534769181b0ad6cbf.jpg)

![27294b7871ce69614321fcb6328370143c9514b8c2ea56b960ae3bb7e4118ad0.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/27294b7871ce69614321fcb6328370143c9514b8c2ea56b960ae3bb7e4118ad0.jpg)

![2ba4e19a74f5b1fe9bd4993843be60002ce003ad1e4dbd236f4692503ec9375e.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/2ba4e19a74f5b1fe9bd4993843be60002ce003ad1e4dbd236f4692503ec9375e.jpg)

![4f92b77ddd308107d40598e709b6278850a2082f835f75fbb8f4ed1f34d4adb0.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/4f92b77ddd308107d40598e709b6278850a2082f835f75fbb8f4ed1f34d4adb0.jpg)

![55b4822d3982ef28035a115df0786ba38400dff38e8c4f6ee2d4211a721e630f.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/55b4822d3982ef28035a115df0786ba38400dff38e8c4f6ee2d4211a721e630f.jpg)

![66ff1f0c58862d72fdd1f287690183232df679d010ef7eb0236d966049eed4b7.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/66ff1f0c58862d72fdd1f287690183232df679d010ef7eb0236d966049eed4b7.jpg)

![70524deefe04b731230e177f1c08b71fdcfb493166592eb13b43de599c9f1b6f.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/70524deefe04b731230e177f1c08b71fdcfb493166592eb13b43de599c9f1b6f.jpg)

![747d82dd7f41809306167e9b21f8973c5dd5ad7dc34a72ccb5d7c4e451d9159f.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/747d82dd7f41809306167e9b21f8973c5dd5ad7dc34a72ccb5d7c4e451d9159f.jpg)

![7736de564c8b26e5bc5fe400bf4c2efe63fefc5e63f910e7e7bec06c43dbaec2.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/7736de564c8b26e5bc5fe400bf4c2efe63fefc5e63f910e7e7bec06c43dbaec2.jpg)

![a5caa45790763f90e34b941d155dcbab5176283ff0e08478a477784078626d63.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/a5caa45790763f90e34b941d155dcbab5176283ff0e08478a477784078626d63.jpg)

![b61b4edf10ab7eb927e8337f8b66e9c16ddac7243bfcc22a9a7c62bc437a1c0a.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/b61b4edf10ab7eb927e8337f8b66e9c16ddac7243bfcc22a9a7c62bc437a1c0a.jpg)

![c0ded7af0381e5076d593cfa33409fdbdd6a064c98aacdc2cf5b484cbbbcdcf8.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/c0ded7af0381e5076d593cfa33409fdbdd6a064c98aacdc2cf5b484cbbbcdcf8.jpg)

![deba9c5531dfdb54a887a4e46c7932b8b44f48424b9639fccc17bcc285afa018.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/deba9c5531dfdb54a887a4e46c7932b8b44f48424b9639fccc17bcc285afa018.jpg)

![ee8e873bdfca140593ea907f61c0ff09b7d221ece9bec40edf8b7271ef924080.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/ee8e873bdfca140593ea907f61c0ff09b7d221ece9bec40edf8b7271ef924080.jpg)

![f2bb55fb433bb41f5edaf7665f716a31c5df9a901e7e9920a296c555ee5cf8c6.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/f2bb55fb433bb41f5edaf7665f716a31c5df9a901e7e9920a296c555ee5cf8c6.jpg)

![f3798f3aa6c4f9b4f8c25dcc4af55fdc5a1c28d6df5b335d411964f6e00eb4dd.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/f3798f3aa6c4f9b4f8c25dcc4af55fdc5a1c28d6df5b335d411964f6e00eb4dd.jpg)

![f5e9420e132b8bc8c7ac18fc15215155ebd07df04d582813a6fd6937e1ad59b7.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/f5e9420e132b8bc8c7ac18fc15215155ebd07df04d582813a6fd6937e1ad59b7.jpg)

![fd42d3b8016784fea3478da81a37d8a6c266c94aca1bd51fae6e8d8d00adaad6.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/fd42d3b8016784fea3478da81a37d8a6c266c94aca1bd51fae6e8d8d00adaad6.jpg)

![ff3c37d21ec6f651f0ba88a18705aa3ccf15821577e5b04a64f2812503e8f788.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/images/ff3c37d21ec6f651f0ba88a18705aa3ccf15821577e5b04a64f2812503e8f788.jpg)

### Tables

![0ceef7ba121e883b10e6c388b607e5f169344b1a514a4cd31a314b1177a64a5a.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/tables/0ceef7ba121e883b10e6c388b607e5f169344b1a514a4cd31a314b1177a64a5a.jpg)

![16329f6736d3a9fe4af99cc045aba70cee135da0b2b31cf8da78e4b5f911f2d9.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/tables/16329f6736d3a9fe4af99cc045aba70cee135da0b2b31cf8da78e4b5f911f2d9.jpg)

![2a871e6af4b945ee0a8782bf7a3b7b666018f596bb8a9c2e08713a13048bfb56.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/tables/2a871e6af4b945ee0a8782bf7a3b7b666018f596bb8a9c2e08713a13048bfb56.jpg)

![6e85a6cfa34de0bb4aae81176229c8d34fd2f248fc2900575506aacf7b2922e7.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/tables/6e85a6cfa34de0bb4aae81176229c8d34fd2f248fc2900575506aacf7b2922e7.jpg)

![9a24caeebf5d003294db06523c74e28299ce32dc652084a9512ff8c6e11dc0e1.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/tables/9a24caeebf5d003294db06523c74e28299ce32dc652084a9512ff8c6e11dc0e1.jpg)

![b90617e3de7febb3ad2248a66195ca5103cbb24a30215a4e8d59c604cb532b8c.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/tables/b90617e3de7febb3ad2248a66195ca5103cbb24a30215a4e8d59c604cb532b8c.jpg)

![c32a95b37c0d29a4406ed3982659cca5b6d12cce7e06ed8aa83d369fdc835b5e.jpg](../iclr_results/1189_ColPali_ Efficient Document Retrieval with Vision Language Models/tables/c32a95b37c0d29a4406ed3982659cca5b6d12cce7e06ed8aa83d369fdc835b5e.jpg)

## Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning


### Images

![0ec6bdc35b8327550c9cf7dfb24e69a49b7938e6727730b2bc1e6af249911e72.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/0ec6bdc35b8327550c9cf7dfb24e69a49b7938e6727730b2bc1e6af249911e72.jpg)

![342548457569683a9c3295e5110db82cf5786da24145af882b07f490e2b6aebe.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/342548457569683a9c3295e5110db82cf5786da24145af882b07f490e2b6aebe.jpg)

![43b293e25c5eb3f0ce56c6b3adb3d88a32168b6a6484209d66fe26834de98e91.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/43b293e25c5eb3f0ce56c6b3adb3d88a32168b6a6484209d66fe26834de98e91.jpg)

![499463925f28c83de76348f477a33d1817ebdcd3c6f4abd5690d466737d08e8a.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/499463925f28c83de76348f477a33d1817ebdcd3c6f4abd5690d466737d08e8a.jpg)

![4bc2fdb8dd42af2c1f994238d835c48b3422e92461a6c7d5a67b91cf66c44c6a.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/4bc2fdb8dd42af2c1f994238d835c48b3422e92461a6c7d5a67b91cf66c44c6a.jpg)

![4e94fb9d116fc953b8624cb23a9df2a335415f75be1927091e0ffe68f65a8588.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/4e94fb9d116fc953b8624cb23a9df2a335415f75be1927091e0ffe68f65a8588.jpg)

![4f3f94851586764acf8395ab03bdf098e893de0e49c201da3603547aa808b93f.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/4f3f94851586764acf8395ab03bdf098e893de0e49c201da3603547aa808b93f.jpg)

![5481ddb4d46ef73b31fca776222824568215bcdb705edf463e030e19c8787f8a.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/5481ddb4d46ef73b31fca776222824568215bcdb705edf463e030e19c8787f8a.jpg)

![617831d796df0936a915379c52dd5e8f2f7a7c62676b17687535d8d84be3ec47.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/617831d796df0936a915379c52dd5e8f2f7a7c62676b17687535d8d84be3ec47.jpg)

![65d06533f8d9386af0636638173f4e60a204677229613024ce1a6a1cc6c3b140.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/65d06533f8d9386af0636638173f4e60a204677229613024ce1a6a1cc6c3b140.jpg)

![66a6276b27fe127d67fd295f4075d56b82e8385769ef59b14e9bf3d08e7d6551.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/66a6276b27fe127d67fd295f4075d56b82e8385769ef59b14e9bf3d08e7d6551.jpg)

![7277b001683b35be9a71d1194bf2e6d94ac29d324d43239cff124fb57bd7c54f.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/7277b001683b35be9a71d1194bf2e6d94ac29d324d43239cff124fb57bd7c54f.jpg)

![7fdb52176e498f0aa2aaf94286e69256a93987304845429f2210feff6dba4706.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/7fdb52176e498f0aa2aaf94286e69256a93987304845429f2210feff6dba4706.jpg)

![8e19e6ee8e595dd23da68bfdf32ed40afdf0474b86ebf9fc2b6c05b130fc2531.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/8e19e6ee8e595dd23da68bfdf32ed40afdf0474b86ebf9fc2b6c05b130fc2531.jpg)

![955a63fbd25f8f8a48004335c88a3a9c19a93d6410210d3cbcf398b94c687c05.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/955a63fbd25f8f8a48004335c88a3a9c19a93d6410210d3cbcf398b94c687c05.jpg)

![a2fdf395471b6f0911949c70e6a5c26882b16f276dfaaa378a388260f6da636c.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/a2fdf395471b6f0911949c70e6a5c26882b16f276dfaaa378a388260f6da636c.jpg)

![ac1e93ebc3897c2662df4bde88aa9649b64d2b7210a5a3c621e2fe20c67b759f.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/ac1e93ebc3897c2662df4bde88aa9649b64d2b7210a5a3c621e2fe20c67b759f.jpg)

![ce56b12b8400b2e29d79420f9d6a9c5c2397a368fc3e3aa81c7dab86ec86650e.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/images/ce56b12b8400b2e29d79420f9d6a9c5c2397a368fc3e3aa81c7dab86ec86650e.jpg)

### Tables

![092706d3221e1fd5dd6c05d3e1e11293f3b9ad721625a114e8e5bde0c97d9ebf.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/tables/092706d3221e1fd5dd6c05d3e1e11293f3b9ad721625a114e8e5bde0c97d9ebf.jpg)

![1d9a6d0e719bf252c6565e5900f092e56784948917b2e4948ff2f3f0230eedc2.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/tables/1d9a6d0e719bf252c6565e5900f092e56784948917b2e4948ff2f3f0230eedc2.jpg)

![7179af7c8f57c9a1a3eaa8ae8470a7d1d5166edb3a97bafff7bdf7ce01d5fad5.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/tables/7179af7c8f57c9a1a3eaa8ae8470a7d1d5166edb3a97bafff7bdf7ce01d5fad5.jpg)

![90d053851e35a55fab97cc33e30d05484de5020b462dae9910f46254fdf7784e.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/tables/90d053851e35a55fab97cc33e30d05484de5020b462dae9910f46254fdf7784e.jpg)

![ca16623368449f93e7fdeedea75cc62b5390015086485a098e0073797f4b4945.jpg](../iclr_results/1190_Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning/tables/ca16623368449f93e7fdeedea75cc62b5390015086485a098e0073797f4b4945.jpg)

## SpinQuant: LLM Quantization with Learned Rotations


### Images

![017d214aae012dc602a5bf0ade16f2cae42148142a1a10efef778ece557a21e6.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/017d214aae012dc602a5bf0ade16f2cae42148142a1a10efef778ece557a21e6.jpg)

![2b70115204522c8c818d06792295798d67d6355f1ffeb6f7c1ca305b98164d91.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/2b70115204522c8c818d06792295798d67d6355f1ffeb6f7c1ca305b98164d91.jpg)

![2f9213a5e0ab8b59fc07a3c2f9bea0511be9b7b81814ceb969f9f258bd8395a5.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/2f9213a5e0ab8b59fc07a3c2f9bea0511be9b7b81814ceb969f9f258bd8395a5.jpg)

![449467df46d832a360ffbbf5e5a94b177d5b0a58b1dd103a137e079efe721550.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/449467df46d832a360ffbbf5e5a94b177d5b0a58b1dd103a137e079efe721550.jpg)

![5b0bc7b862da0aaabe8add09a2f572c0d5c9174354c8ce95ba9ce6a7866f8650.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/5b0bc7b862da0aaabe8add09a2f572c0d5c9174354c8ce95ba9ce6a7866f8650.jpg)

![6ecd4279e63942e730a47f94d4319eb3b494084039530c2ef20be2c13edd1b9c.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/6ecd4279e63942e730a47f94d4319eb3b494084039530c2ef20be2c13edd1b9c.jpg)

![78c5ab05f184772555681203b41f0380c9edb9269b2d1cc90e1a2124ba84c0aa.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/78c5ab05f184772555681203b41f0380c9edb9269b2d1cc90e1a2124ba84c0aa.jpg)

![799fa53423181b9c600e4a3d7173029fcab822f8c3c97a110ddb4b264ba6fbd7.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/799fa53423181b9c600e4a3d7173029fcab822f8c3c97a110ddb4b264ba6fbd7.jpg)

![7b9d97a71667bf7f9ee268568ef5dd8d585c7856db940c60f7809384ad9c82e8.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/7b9d97a71667bf7f9ee268568ef5dd8d585c7856db940c60f7809384ad9c82e8.jpg)

![c374b1ae8f4f5b4727280758b916452e1ff79f7aa30cd17139d237dffd7d1b14.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/c374b1ae8f4f5b4727280758b916452e1ff79f7aa30cd17139d237dffd7d1b14.jpg)

![ffe5a2552aa641b627c17f88902bc4d4734cade842ea4cdcd419583729a84068.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/images/ffe5a2552aa641b627c17f88902bc4d4734cade842ea4cdcd419583729a84068.jpg)

### Tables

![0f0bde9473a38726611fc1a81ce5cf2e7726e86082432aa3eb356681e311aca6.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/0f0bde9473a38726611fc1a81ce5cf2e7726e86082432aa3eb356681e311aca6.jpg)

![17bdacc7842ea036de6e81d44b26ff9f805eeb730780d8704b1ac8c8846fe205.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/17bdacc7842ea036de6e81d44b26ff9f805eeb730780d8704b1ac8c8846fe205.jpg)

![18c0f86855cb981f800a2ce944d35399a0a3c01938396c11327c92bae086a00b.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/18c0f86855cb981f800a2ce944d35399a0a3c01938396c11327c92bae086a00b.jpg)

![2b12b54e1028457b311f0a8b18a3b49afaad3cd7d7ae5d394681f93195a636ed.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/2b12b54e1028457b311f0a8b18a3b49afaad3cd7d7ae5d394681f93195a636ed.jpg)

![312671bfbfaf5a5130cbb16bcefbbdc10da31c07f6fd8de36052555dd090375b.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/312671bfbfaf5a5130cbb16bcefbbdc10da31c07f6fd8de36052555dd090375b.jpg)

![3efe8131433d2f1ede1d91736260dda05406bb84d2f5ce68dcac25376417ac5a.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/3efe8131433d2f1ede1d91736260dda05406bb84d2f5ce68dcac25376417ac5a.jpg)

![60a8a59f26ef2f64f4461c3d29734a39212bd45bc61665aa5cbc4246ab8d6482.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/60a8a59f26ef2f64f4461c3d29734a39212bd45bc61665aa5cbc4246ab8d6482.jpg)

![7b596a2dcb38b3f4270da8e219d7a5138c2c16a780b29778fbc1eb7e1d00b791.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/7b596a2dcb38b3f4270da8e219d7a5138c2c16a780b29778fbc1eb7e1d00b791.jpg)

![83714d09143ec32358763981c5540668c37419b86617f68280d14271f67225f5.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/83714d09143ec32358763981c5540668c37419b86617f68280d14271f67225f5.jpg)

![8765591b2a9a2f769798449df151a23115ba588f4dc9ebdf943814a1e36d91c1.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/8765591b2a9a2f769798449df151a23115ba588f4dc9ebdf943814a1e36d91c1.jpg)

![886e3d208b93c410f330b9490c158b616502e9fa65a844f42f12003193e99ef8.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/886e3d208b93c410f330b9490c158b616502e9fa65a844f42f12003193e99ef8.jpg)

![a3b872ac600fcde257cb89b807aa9a7cc731abc5a751d8568a5f6b502418dad3.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/a3b872ac600fcde257cb89b807aa9a7cc731abc5a751d8568a5f6b502418dad3.jpg)

![c251ee844f9a8cde93d40b884b0569d24f3cdc26e0db663f4fe02cb8ebc1e744.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/c251ee844f9a8cde93d40b884b0569d24f3cdc26e0db663f4fe02cb8ebc1e744.jpg)

![c3ebc0a6407822bb8b0564871c976e734b66f9385329c931c333d278d559276f.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/c3ebc0a6407822bb8b0564871c976e734b66f9385329c931c333d278d559276f.jpg)

![cc3d09ca406eb0277d87a5f1d11e94cf295441a173ec65d09dc1bebfe5a1ab6f.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/cc3d09ca406eb0277d87a5f1d11e94cf295441a173ec65d09dc1bebfe5a1ab6f.jpg)

![cf3701dfae33e882434abac1dea3412e1bfaa9e403b873db3f54d86728570a4e.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/cf3701dfae33e882434abac1dea3412e1bfaa9e403b873db3f54d86728570a4e.jpg)

![d9e0e0571b2524970585001ff6fe7313af3398a9a2972a3bba1ffac583d19276.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/d9e0e0571b2524970585001ff6fe7313af3398a9a2972a3bba1ffac583d19276.jpg)

![e7a6fcbd581bf74bc5785b67e5480232e91aeb3be06f514bd788759de05bbab8.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/e7a6fcbd581bf74bc5785b67e5480232e91aeb3be06f514bd788759de05bbab8.jpg)

![ffe757f2dfdca64ddf5a40d0688ff6816e032167297f042b72a4a8a29686b6cc.jpg](../iclr_results/1191_SpinQuant_ LLM Quantization with Learned Rotations/tables/ffe757f2dfdca64ddf5a40d0688ff6816e032167297f042b72a4a8a29686b6cc.jpg)

## Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escape, and Network Embedding


### Images

![03e030d5407c61cc50dbcd235aea1be8649a00e8d77f0e4c484f83193a483cf4.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/03e030d5407c61cc50dbcd235aea1be8649a00e8d77f0e4c484f83193a483cf4.jpg)

![2760ad5faa31d6a6f3aa3859b2a67a078dc57d7e421943165fc63e0f2a3bbe11.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/2760ad5faa31d6a6f3aa3859b2a67a078dc57d7e421943165fc63e0f2a3bbe11.jpg)

![278b6457eceb48f8503c174dac32801befea0a2db02d12a66647f03905ec1c10.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/278b6457eceb48f8503c174dac32801befea0a2db02d12a66647f03905ec1c10.jpg)

![2973910747150dba640f2f9be92f6c4315cd001a2a0e826a97a1f9108b9e7fa8.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/2973910747150dba640f2f9be92f6c4315cd001a2a0e826a97a1f9108b9e7fa8.jpg)

![333ac93b3537c7944b6f34ef611402a12523de55b8a91c21ed2b5e7ccf2018f0.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/333ac93b3537c7944b6f34ef611402a12523de55b8a91c21ed2b5e7ccf2018f0.jpg)

![4d8a023cd46123d0b1fc7684d432f3d8fb940354764e735acaf31d0a36b6f581.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/4d8a023cd46123d0b1fc7684d432f3d8fb940354764e735acaf31d0a36b6f581.jpg)

![4f1b8a92057b090c06978271b6dd00362c512d1b66a05b9b4430297ed05fb1e4.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/4f1b8a92057b090c06978271b6dd00362c512d1b66a05b9b4430297ed05fb1e4.jpg)

![5b603824f4ba29406ed973dc957fde88fab53bd5e436a1916081f5f1c90a4474.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/5b603824f4ba29406ed973dc957fde88fab53bd5e436a1916081f5f1c90a4474.jpg)

![5c0d0a6ff296774a004c75c91e404e544f0493338757666bec283e63f2b165b1.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/5c0d0a6ff296774a004c75c91e404e544f0493338757666bec283e63f2b165b1.jpg)

![5fdcae99a198e18d01d7fa8944554750dc2bb1026d47817d20bcb042d3a455dd.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/5fdcae99a198e18d01d7fa8944554750dc2bb1026d47817d20bcb042d3a455dd.jpg)

![6ed198eaa9f32e5ab1e134fd3579c096cfd44410402ed70982933a8104dcac05.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/6ed198eaa9f32e5ab1e134fd3579c096cfd44410402ed70982933a8104dcac05.jpg)

![7ab39a288843234c69fe414e099bb630087f3703f9bcb57712f60d10c5ef7193.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/7ab39a288843234c69fe414e099bb630087f3703f9bcb57712f60d10c5ef7193.jpg)

![7dccfecfa981dc21c8366f8cd083b2ed8d244a3ceef851691b72e3bb05ba964c.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/7dccfecfa981dc21c8366f8cd083b2ed8d244a3ceef851691b72e3bb05ba964c.jpg)

![9b6fa4e6e50ed838353c1bd6f280c02f027a59ff4b68340c6a0fe1d9106415de.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/9b6fa4e6e50ed838353c1bd6f280c02f027a59ff4b68340c6a0fe1d9106415de.jpg)

![9dc9487f245df723c056ec20ee549e6277584742956e58550c62d6bcbca624c7.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/9dc9487f245df723c056ec20ee549e6277584742956e58550c62d6bcbca624c7.jpg)

![a63dd7605bdddff7bc33db02daa39fb8bdd913c280036898e658954a4a157011.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/a63dd7605bdddff7bc33db02daa39fb8bdd913c280036898e658954a4a157011.jpg)

![abcedc0c79a3553925b9a8a6e43e6bee40aafc8dbc08acb826a84cfc1dc92fba.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/abcedc0c79a3553925b9a8a6e43e6bee40aafc8dbc08acb826a84cfc1dc92fba.jpg)

![b73b8a6c4d22b8d653976db08714ab218006bbe3577f6e3ccffbf8f550d1e79f.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/b73b8a6c4d22b8d653976db08714ab218006bbe3577f6e3ccffbf8f550d1e79f.jpg)

![ecc64b9e2461fbdb728463b6cc73f7824c6d60eda976b7abd210f2a9dcdaf522.jpg](../iclr_results/1192_Loss Landscape of Shallow ReLU-like Neural Networks_ Stationary Points, Saddle Escape, and Network E/images/ecc64b9e2461fbdb728463b6cc73f7824c6d60eda976b7abd210f2a9dcdaf522.jpg)

## On the Byzantine-Resilience of Distillation-Based Federated Learning


### Images

![49015dfd512c20019c890b1456ba0e8199f8ae60a30943b201061c21fb12a15c.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/images/49015dfd512c20019c890b1456ba0e8199f8ae60a30943b201061c21fb12a15c.jpg)

![6d57890b9241a34be5f959c9b5a1df38d5bd74794b37c74da272789cf169bcdd.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/images/6d57890b9241a34be5f959c9b5a1df38d5bd74794b37c74da272789cf169bcdd.jpg)

![835d26d31d1ba8077ddf460fc5502234326e6b947e3c98d4e6605bbaee927856.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/images/835d26d31d1ba8077ddf460fc5502234326e6b947e3c98d4e6605bbaee927856.jpg)

![858bc725c8d76d28941961ef1c5fb0f990c7e6e6d0fa25df2233bc194843ec03.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/images/858bc725c8d76d28941961ef1c5fb0f990c7e6e6d0fa25df2233bc194843ec03.jpg)

![8a7a07d3681ff785ecde6d62c31221a6e8be1d555589125e639fc43fa445981c.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/images/8a7a07d3681ff785ecde6d62c31221a6e8be1d555589125e639fc43fa445981c.jpg)

### Tables

![08cc169b7c8e235bc53deef1dfbdee022a71c71755c845ab29100d42c6be25e9.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/08cc169b7c8e235bc53deef1dfbdee022a71c71755c845ab29100d42c6be25e9.jpg)

![0d19cc45d727fba3247513ae5fafd7995aeb784759c0e36a1c06675052b1fcef.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/0d19cc45d727fba3247513ae5fafd7995aeb784759c0e36a1c06675052b1fcef.jpg)

![133c716ae26ef6e9affd2fc3b0289b1929cf10597446fd453f38ce6961344d03.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/133c716ae26ef6e9affd2fc3b0289b1929cf10597446fd453f38ce6961344d03.jpg)

![18e26e708c92ab2cde3f796820da5d55fb0aeed8baccef4de727da875da614f0.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/18e26e708c92ab2cde3f796820da5d55fb0aeed8baccef4de727da875da614f0.jpg)

![205c2dab55c35105bb470036a0562d5b653ad210be2a8662d782f83d153d6f56.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/205c2dab55c35105bb470036a0562d5b653ad210be2a8662d782f83d153d6f56.jpg)

![3d3892719a8d964c972517f7e8cd2a7294a1cbcd82b90cc72692f41add2778a3.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/3d3892719a8d964c972517f7e8cd2a7294a1cbcd82b90cc72692f41add2778a3.jpg)

![634b69331b51dadb00368d3399dec8122d2133cb73e2b36aab3874d027b5257a.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/634b69331b51dadb00368d3399dec8122d2133cb73e2b36aab3874d027b5257a.jpg)

![64cb7d1763e9a1e95cdb36435e5ccd4c4b50548a895511ecab0fa013397d4d50.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/64cb7d1763e9a1e95cdb36435e5ccd4c4b50548a895511ecab0fa013397d4d50.jpg)

![970fd5ecfa64e1dd8017643a536982ffac64e804d30dfa88c1c24a3f554648ac.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/970fd5ecfa64e1dd8017643a536982ffac64e804d30dfa88c1c24a3f554648ac.jpg)

![c59440e3e28159e35f4201b3b18635f6aa5c7dc20f2a0db0978101191bb36cd6.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/c59440e3e28159e35f4201b3b18635f6aa5c7dc20f2a0db0978101191bb36cd6.jpg)

![caa29d63762fbfde29b66f50f2d09f5f5532dd0ee0cb01a232584330eaff39f0.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/caa29d63762fbfde29b66f50f2d09f5f5532dd0ee0cb01a232584330eaff39f0.jpg)

![ea4741693e2b8f41d4b382580f071140b5f214ee300023f7a857d53ef7830712.jpg](../iclr_results/1193_On the Byzantine-Resilience of Distillation-Based Federated Learning/tables/ea4741693e2b8f41d4b382580f071140b5f214ee300023f7a857d53ef7830712.jpg)

## OMG: Opacity Matters in Material Modeling with Gaussian Splatting


### Images

![00c89a2a752249ff2bd01ab2750556d7bcffc6a44e4be5bc6b24c958f4b63d97.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/00c89a2a752249ff2bd01ab2750556d7bcffc6a44e4be5bc6b24c958f4b63d97.jpg)

![41b894088e23d3f3544fb7e149cc5d135f1a185489c9e347cc10942d48e63f57.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/41b894088e23d3f3544fb7e149cc5d135f1a185489c9e347cc10942d48e63f57.jpg)

![49da7501e11edbd437e5d20eeccf8ec32483fb04bdcfe51e86d0f7c213edac3a.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/49da7501e11edbd437e5d20eeccf8ec32483fb04bdcfe51e86d0f7c213edac3a.jpg)

![5e19224fae0166bc6a14f911cca2bff5d895e0ddbed27b2ccc19f3535b55bb27.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/5e19224fae0166bc6a14f911cca2bff5d895e0ddbed27b2ccc19f3535b55bb27.jpg)

![75bf4fae7e86d39b0495e4592879798136ab1ea1bb5052134afca68569fa4411.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/75bf4fae7e86d39b0495e4592879798136ab1ea1bb5052134afca68569fa4411.jpg)

![bcb8077c31bae31553c689daf48c0f29781e8fc2eda920a7c6723eb93a5aac03.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/bcb8077c31bae31553c689daf48c0f29781e8fc2eda920a7c6723eb93a5aac03.jpg)

![c732a33891f48ce97b356b3d20036ded58d45de6b68e5562c1a1c6ff8076e14c.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/c732a33891f48ce97b356b3d20036ded58d45de6b68e5562c1a1c6ff8076e14c.jpg)

![d8d3912f4dcf93b7ba60aa00210fa839d0fdff3ba65ae9fd8c36edbde0f92517.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/d8d3912f4dcf93b7ba60aa00210fa839d0fdff3ba65ae9fd8c36edbde0f92517.jpg)

![e93c5c47fa7765b019a3a6596dfbacb8b598360915986ba481a49a36ade2bd6a.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/e93c5c47fa7765b019a3a6596dfbacb8b598360915986ba481a49a36ade2bd6a.jpg)

![f63b451635d66a30c891c3b0d90c6dd56c5046b91cc82e1dd55d352adb403ee5.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/f63b451635d66a30c891c3b0d90c6dd56c5046b91cc82e1dd55d352adb403ee5.jpg)

![f7a906af9724cff412de076921157eda85f0025c0a73177a18be98966854e9d9.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/f7a906af9724cff412de076921157eda85f0025c0a73177a18be98966854e9d9.jpg)

![f7fa3b989bcbad7e37e2f19678fc5fc855c1155620486762d05ddd83c05bc6d3.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/images/f7fa3b989bcbad7e37e2f19678fc5fc855c1155620486762d05ddd83c05bc6d3.jpg)

### Tables

![32b028203ab77823621071be43ba16e2c2287d8762eb290623e87c262d8c9c84.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/tables/32b028203ab77823621071be43ba16e2c2287d8762eb290623e87c262d8c9c84.jpg)

![7ae8968e0706b5b26f69c3e4241e3085ea2bdb9208b198f2299f0e7e24207635.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/tables/7ae8968e0706b5b26f69c3e4241e3085ea2bdb9208b198f2299f0e7e24207635.jpg)

![a4f5bbfb7c1de59adbb11158a3378828d54248ad779cf47cfda25b9c617448e0.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/tables/a4f5bbfb7c1de59adbb11158a3378828d54248ad779cf47cfda25b9c617448e0.jpg)

![caeb605b6c72a1e737f9170736fef20c3c192583a73e1f94e18e34ee3837f689.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/tables/caeb605b6c72a1e737f9170736fef20c3c192583a73e1f94e18e34ee3837f689.jpg)

![d21e9ef87157ed37605e204e077334945295c12ed4e9d6306ac0836e3facbe66.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/tables/d21e9ef87157ed37605e204e077334945295c12ed4e9d6306ac0836e3facbe66.jpg)

![e7175568dae902a8385d3a5691ebd94a5d70f1bc389f8e3ac4fd856494387591.jpg](../iclr_results/1194_OMG_ Opacity Matters in Material Modeling with Gaussian Splatting/tables/e7175568dae902a8385d3a5691ebd94a5d70f1bc389f8e3ac4fd856494387591.jpg)

## Counterfactual Generative Modeling with Variational Causal Inference


### Images

![007f0c6cc50956cc9228e364d2fce8583ddc11af402f8826414ccf8a86f413ea.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/007f0c6cc50956cc9228e364d2fce8583ddc11af402f8826414ccf8a86f413ea.jpg)

![094d11bc94f2c7e6c3463d71d6ed5ddda8953977574a5ef57ea9a19c53606521.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/094d11bc94f2c7e6c3463d71d6ed5ddda8953977574a5ef57ea9a19c53606521.jpg)

![19490eb87154c32311be8613bb6d866b56be27c42deeda18af9b76c5e64092a2.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/19490eb87154c32311be8613bb6d866b56be27c42deeda18af9b76c5e64092a2.jpg)

![40626e172f937154bb44930f6b9b2d133672324ef68c62795b510509e39997bf.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/40626e172f937154bb44930f6b9b2d133672324ef68c62795b510509e39997bf.jpg)

![417edf726c81d2811aca9a921463e93f9ee2c757fccdfe54e8d398f61a5ea472.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/417edf726c81d2811aca9a921463e93f9ee2c757fccdfe54e8d398f61a5ea472.jpg)

![4d2c25e583abc814b9052ffe05882ffb015fc6aeb31adfe761cd3f805ef0fb25.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/4d2c25e583abc814b9052ffe05882ffb015fc6aeb31adfe761cd3f805ef0fb25.jpg)

![76a027d5af8c1c9e0ba8100b5f4c34d9d05311f0680cb6c74015f14f16417669.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/76a027d5af8c1c9e0ba8100b5f4c34d9d05311f0680cb6c74015f14f16417669.jpg)

![78993f0330ff1b247017e15c369235b68b02e6daf78d12e44efecfac46296337.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/78993f0330ff1b247017e15c369235b68b02e6daf78d12e44efecfac46296337.jpg)

![7e6586146ad77069b126e33e616dae9569075206bf3d227b7e3f4aa17a77b78c.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/7e6586146ad77069b126e33e616dae9569075206bf3d227b7e3f4aa17a77b78c.jpg)

![7f6182c52d7958c566c6d6f1b964697602529d073c9daeac9acf66f146e3489a.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/7f6182c52d7958c566c6d6f1b964697602529d073c9daeac9acf66f146e3489a.jpg)

![8b51928670590b69ac8e525d820ff9d1a742ff05004656d32e878129c8574d37.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/8b51928670590b69ac8e525d820ff9d1a742ff05004656d32e878129c8574d37.jpg)

![8ced16c12e37c1f0dc9b8c10371920af69c0e5649d2470f3089f77364f7e4f0f.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/8ced16c12e37c1f0dc9b8c10371920af69c0e5649d2470f3089f77364f7e4f0f.jpg)

![c545afc798ba5cfbd119bfd6267d75b9ccc0e999fa58592c55b42ae19c347368.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/c545afc798ba5cfbd119bfd6267d75b9ccc0e999fa58592c55b42ae19c347368.jpg)

![c97153133b0dedc8e459a768bf0c6aaa5d7dec95c3c13888c88cfc53a433a332.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/c97153133b0dedc8e459a768bf0c6aaa5d7dec95c3c13888c88cfc53a433a332.jpg)

![ce800bee6f2b85d84f1be15820bd2515027d0e222a01d9e0ccc0d897427041e6.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/ce800bee6f2b85d84f1be15820bd2515027d0e222a01d9e0ccc0d897427041e6.jpg)

![fafbf7c80617fe5f44766fc4f49f3e3f59a00d5192a8e76825f865e151261ebd.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/images/fafbf7c80617fe5f44766fc4f49f3e3f59a00d5192a8e76825f865e151261ebd.jpg)

### Tables

![055f6edc73c5fab18c8d18db0e5653eec122479e559f3c9c6f08df55f7b971e3.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/055f6edc73c5fab18c8d18db0e5653eec122479e559f3c9c6f08df55f7b971e3.jpg)

![0aa701efd7537caef39e25120ca6dfd053f54a00ab177dfcfb9b049bbe754f81.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/0aa701efd7537caef39e25120ca6dfd053f54a00ab177dfcfb9b049bbe754f81.jpg)

![66fe5eb7682ceb5afcdc82effeb20b7b446d5a6536c81c9ed46fd996b2dd7826.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/66fe5eb7682ceb5afcdc82effeb20b7b446d5a6536c81c9ed46fd996b2dd7826.jpg)

![75582453a7fbd7acdee368aba4681ffc7f53c6f7196b6368f6a2d8b421a1d72d.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/75582453a7fbd7acdee368aba4681ffc7f53c6f7196b6368f6a2d8b421a1d72d.jpg)

![7f614637b5df051cea4a76beee0a1282c08472b6b9072962723462fa8a62b213.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/7f614637b5df051cea4a76beee0a1282c08472b6b9072962723462fa8a62b213.jpg)

![c45081c56247bfb0e372f779c3b3833c6c1d3b42d6af389045b89c69fab4e4ae.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/c45081c56247bfb0e372f779c3b3833c6c1d3b42d6af389045b89c69fab4e4ae.jpg)

![d04bf6e3ae542aad9e84576cdd65d1b93e469854b06ed857da6cfc0bfff6212d.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/d04bf6e3ae542aad9e84576cdd65d1b93e469854b06ed857da6cfc0bfff6212d.jpg)

![fa243c18fb6b5dd59e14943971e6c8d0a950cb9e2430f0ed45c90edaf4cad904.jpg](../iclr_results/1195_Counterfactual Generative Modeling with Variational Causal Inference/tables/fa243c18fb6b5dd59e14943971e6c8d0a950cb9e2430f0ed45c90edaf4cad904.jpg)

## Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning


### Images

![0248cc1e2835cae23994838a51c4a632e0e135cf93e838630e6526292c00fae0.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/0248cc1e2835cae23994838a51c4a632e0e135cf93e838630e6526292c00fae0.jpg)

![0a29a57e84ef088b5197f5620601e4358d7199c44c4f782795c28fd32d5b33a7.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/0a29a57e84ef088b5197f5620601e4358d7199c44c4f782795c28fd32d5b33a7.jpg)

![19215d840e3d0436adf195de77d1bca985486125003c62d5a1ab06c0c537bfc9.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/19215d840e3d0436adf195de77d1bca985486125003c62d5a1ab06c0c537bfc9.jpg)

![28c7589c3da226a37a14dfd93a59706332f195a95b2a04cfde518bd02e661a41.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/28c7589c3da226a37a14dfd93a59706332f195a95b2a04cfde518bd02e661a41.jpg)

![2f6beff8d3648ab75a0dafd8fbed4e2534a6d77f42d07a5ceb58cb57530ba95e.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/2f6beff8d3648ab75a0dafd8fbed4e2534a6d77f42d07a5ceb58cb57530ba95e.jpg)

![36811a2ee6151411c2020e15b9b5546ed9186cea70fb15a59d32c70e3872f996.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/36811a2ee6151411c2020e15b9b5546ed9186cea70fb15a59d32c70e3872f996.jpg)

![3eb447c3ec533b7befe5dc850efe980c7adf153230e34ad24340b5567b8eaacf.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/3eb447c3ec533b7befe5dc850efe980c7adf153230e34ad24340b5567b8eaacf.jpg)

![47a5b4cb2ac99626f4207f04fbdec66da41f50711683ebd45c4f7af7672d1ddc.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/47a5b4cb2ac99626f4207f04fbdec66da41f50711683ebd45c4f7af7672d1ddc.jpg)

![4a54bb5a58ae1643b70732ea29376087934117b37139ebeb393fc1190e781b40.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/4a54bb5a58ae1643b70732ea29376087934117b37139ebeb393fc1190e781b40.jpg)

![662a1489a1d9e4a2650f6949eb87c4b7359b96c22291e40bd64fab49d11d8438.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/662a1489a1d9e4a2650f6949eb87c4b7359b96c22291e40bd64fab49d11d8438.jpg)

![7ae203bad1f81f735f73b683e49f2084274634cd9ef2bb497d79964e12227a89.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/7ae203bad1f81f735f73b683e49f2084274634cd9ef2bb497d79964e12227a89.jpg)

![924cf6eafc47de012ddea0162b59a48ab9daf6064c3889534613f93e9b17f21b.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/924cf6eafc47de012ddea0162b59a48ab9daf6064c3889534613f93e9b17f21b.jpg)

![92deff0c36503ae0769f8e8f88d3d5738a1b6522781259c7ac41dd676d2761d2.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/92deff0c36503ae0769f8e8f88d3d5738a1b6522781259c7ac41dd676d2761d2.jpg)

![b73807b7bdfec815e0d7686965800a606b5f431020efda71cfe6efc5e2af608d.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/b73807b7bdfec815e0d7686965800a606b5f431020efda71cfe6efc5e2af608d.jpg)

![c6b5785e213b6ec5f85d11461b970b77f8d76f3229351e44e78b7212907c5af8.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/c6b5785e213b6ec5f85d11461b970b77f8d76f3229351e44e78b7212907c5af8.jpg)

![d3dd442b0280ea4ba7f066cdd2f738bb788e999c8c5eb888d96a3f3a721b113c.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/images/d3dd442b0280ea4ba7f066cdd2f738bb788e999c8c5eb888d96a3f3a721b113c.jpg)

### Tables

![65d48bce423710bbf20dc20a96358316e65681a7aed4447b40aa9042d6be1981.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/tables/65d48bce423710bbf20dc20a96358316e65681a7aed4447b40aa9042d6be1981.jpg)

![7165f8750bed66b7501fcc14d0b213084c76512a158c8d82d1a2ff2396612f99.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/tables/7165f8750bed66b7501fcc14d0b213084c76512a158c8d82d1a2ff2396612f99.jpg)

![8872273b35fec245df56bd46855488e6deda42b16a0b7d8cc288ca71079386b1.jpg](../iclr_results/1196_Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning/tables/8872273b35fec245df56bd46855488e6deda42b16a0b7d8cc288ca71079386b1.jpg)

## Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models


### Images

![b29b12c0b7c9307d9abdb722ba6af23ef64fc09e31e99644976a0b3accacd6c4.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/images/b29b12c0b7c9307d9abdb722ba6af23ef64fc09e31e99644976a0b3accacd6c4.jpg)

![db95a629a68826116e9c9f5c4a94136075519463d2d2d5c1c3307aa2689de3bf.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/images/db95a629a68826116e9c9f5c4a94136075519463d2d2d5c1c3307aa2689de3bf.jpg)

![dcbafff3f1e9b80bbec1ebbc5db0fab208b059836a3c675ce0db10f4fa9ca7c9.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/images/dcbafff3f1e9b80bbec1ebbc5db0fab208b059836a3c675ce0db10f4fa9ca7c9.jpg)

### Tables

![0d22e4df1ada98db4f3d63024556ee177cde4b9f3b95dbf2d9c85f9e7cd8e3e7.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/0d22e4df1ada98db4f3d63024556ee177cde4b9f3b95dbf2d9c85f9e7cd8e3e7.jpg)

![13f7515b1d353c68f2379c747a4f56c595afbea141bc2d775082b11f337896de.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/13f7515b1d353c68f2379c747a4f56c595afbea141bc2d775082b11f337896de.jpg)

![3632cb34eabe5355ccd3f7c08767ee490580f7888519c3c0a49043302da4b98a.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/3632cb34eabe5355ccd3f7c08767ee490580f7888519c3c0a49043302da4b98a.jpg)

![3956eb53ecdaa1125364188db91967e720148aa4b281d64240ca30f838182858.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/3956eb53ecdaa1125364188db91967e720148aa4b281d64240ca30f838182858.jpg)

![53539ada5db31da903134cc13a1fee0dab4d2e19819c4825bfbca1d2c8941289.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/53539ada5db31da903134cc13a1fee0dab4d2e19819c4825bfbca1d2c8941289.jpg)

![6ada166802a2e437adf2036262bb5138bb8f95bc2e68be3a8309504b2b3d15b0.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/6ada166802a2e437adf2036262bb5138bb8f95bc2e68be3a8309504b2b3d15b0.jpg)

![83d105e92f26b4bd5d71a3374df584a2f37d53810943868aa83d49566fe0449d.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/83d105e92f26b4bd5d71a3374df584a2f37d53810943868aa83d49566fe0449d.jpg)

![bc23f4ff5a3252c9623e83790fbef82255bb4dc8fe30bdffc66c5b18a5a431d2.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/bc23f4ff5a3252c9623e83790fbef82255bb4dc8fe30bdffc66c5b18a5a431d2.jpg)

![bd87586ff59d8e19cee9251e803f76efd22352b9f59e5e5b871afc23b7916164.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/bd87586ff59d8e19cee9251e803f76efd22352b9f59e5e5b871afc23b7916164.jpg)

![cce37942bae0db254c51b1e91564ce3d123b1c653634924b0aba415ff89e8664.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/cce37942bae0db254c51b1e91564ce3d123b1c653634924b0aba415ff89e8664.jpg)

![d82d135e4c17a5d3bbc98aac94556d47944f5dc5224374a4a0d6ac9445a89ff4.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/d82d135e4c17a5d3bbc98aac94556d47944f5dc5224374a4a0d6ac9445a89ff4.jpg)

![ddfceb6f9fd7eeb3ebd7a7345a9545761ddd2211ad7f3e4e7c38fcf4c8e75da8.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/ddfceb6f9fd7eeb3ebd7a7345a9545761ddd2211ad7f3e4e7c38fcf4c8e75da8.jpg)

![f621eccd38f044559886bfbd07f768c1596296cfa69e60cd38cde45bfac9983e.jpg](../iclr_results/1197_Promptriever_ Instruction-Trained Retrievers Can Be Prompted Like Language Models/tables/f621eccd38f044559886bfbd07f768c1596296cfa69e60cd38cde45bfac9983e.jpg)

## SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes


### Images

![1ba447e87fd25843b37f0d21318553c6146d13f215fbfe708467204dd86fa831.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/images/1ba447e87fd25843b37f0d21318553c6146d13f215fbfe708467204dd86fa831.jpg)

![253989a8d60a70a43a5deae7f00cd4a47545e6c3f40e55ebb67885ba82a4a432.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/images/253989a8d60a70a43a5deae7f00cd4a47545e6c3f40e55ebb67885ba82a4a432.jpg)

![427625bea8b90abb77fa704218c0a3735f56538f102ad0262a0404dcafeda782.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/images/427625bea8b90abb77fa704218c0a3735f56538f102ad0262a0404dcafeda782.jpg)

![4c485c2c51d92c88a6bbf2fbe609ece586e080fe9651e0309e416eb6891d870d.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/images/4c485c2c51d92c88a6bbf2fbe609ece586e080fe9651e0309e416eb6891d870d.jpg)

![dd64539b3a4a0a4226f1262bf1f10b9b402d58c8d9d22df571969c7d5856233c.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/images/dd64539b3a4a0a4226f1262bf1f10b9b402d58c8d9d22df571969c7d5856233c.jpg)

### Tables

![03ab9db33b15d19e8748b0cc939c8d6b5024ea5bd435987b339b583f0d8a09e4.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/03ab9db33b15d19e8748b0cc939c8d6b5024ea5bd435987b339b583f0d8a09e4.jpg)

![38bbea1db2cf253d6dc213c85816088b49fc1f097bff0022d6d1598b9a102df4.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/38bbea1db2cf253d6dc213c85816088b49fc1f097bff0022d6d1598b9a102df4.jpg)

![43b4259f205bbf9aed23802b1e4bf643fdff57351c0758c25cb354dd4e83508e.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/43b4259f205bbf9aed23802b1e4bf643fdff57351c0758c25cb354dd4e83508e.jpg)

![45c92e52d07dcb87c31bd9508e9b054db35239e55ee9cb2e7b1bf90f5d459bc8.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/45c92e52d07dcb87c31bd9508e9b054db35239e55ee9cb2e7b1bf90f5d459bc8.jpg)

![50753654bd028e5c2e7f789993b2ab81a2e7466aa0f0b0ccc8351faaaa0d5668.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/50753654bd028e5c2e7f789993b2ab81a2e7466aa0f0b0ccc8351faaaa0d5668.jpg)

![534678dae5b7a246d8d25cd21557c78be6d7ad6d2514bb95ba91a3dbfedd22c1.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/534678dae5b7a246d8d25cd21557c78be6d7ad6d2514bb95ba91a3dbfedd22c1.jpg)

![6c30e6d08e7bb170b4e8bd061f53578fb129f117363bb4bf03bf4c450975f714.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/6c30e6d08e7bb170b4e8bd061f53578fb129f117363bb4bf03bf4c450975f714.jpg)

![a2ef6967bc665b1537af25c9f0609aac97e93280f64f1e2ac70d82fcec01c472.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/a2ef6967bc665b1537af25c9f0609aac97e93280f64f1e2ac70d82fcec01c472.jpg)

![c3f93df74e8ed1764c83877ab66933133bd9fad1b1190b245209fd8ee065f8e4.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/c3f93df74e8ed1764c83877ab66933133bd9fad1b1190b245209fd8ee065f8e4.jpg)

![d38c93fe6a1a27cbc727cd7c4bf07494a1662953271c42da753dd8e2835f7024.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/d38c93fe6a1a27cbc727cd7c4bf07494a1662953271c42da753dd8e2835f7024.jpg)

![e3dc70be531a61ac1d79ae6853da5a0db4a1b43a651cd207aa9f917f466626cd.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/e3dc70be531a61ac1d79ae6853da5a0db4a1b43a651cd207aa9f917f466626cd.jpg)

![fb522dd93d06e9633e642eb5ef1ae2ef3ac85535585775afbd3ba3e2bbecc58d.jpg](../iclr_results/1198_SSLAM_ Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes/tables/fb522dd93d06e9633e642eb5ef1ae2ef3ac85535585775afbd3ba3e2bbecc58d.jpg)

## Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift


### Images

![13659fbe21f288d59421f3cc35edc203b9c3169710098f64e7f912aefcf64c5b.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/13659fbe21f288d59421f3cc35edc203b9c3169710098f64e7f912aefcf64c5b.jpg)

![29dfff9b1fcaba494b2b88dd2b48af36cae15d61e3f469eab0f1e9395382fecd.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/29dfff9b1fcaba494b2b88dd2b48af36cae15d61e3f469eab0f1e9395382fecd.jpg)

![3e90b1dcc1f837ac3b61133b2746d5ed1ea9c031f8090c16505d675148fac86c.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/3e90b1dcc1f837ac3b61133b2746d5ed1ea9c031f8090c16505d675148fac86c.jpg)

![8a9480e8e3c6a145beb334cb843d72cd685e1865f9580ff2bd5f6db21312939f.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/8a9480e8e3c6a145beb334cb843d72cd685e1865f9580ff2bd5f6db21312939f.jpg)

![9efa6b63453f037d7c0c92d76959622c6f35d0b54820d3a7e38b2d18e62f1cb4.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/9efa6b63453f037d7c0c92d76959622c6f35d0b54820d3a7e38b2d18e62f1cb4.jpg)

![af433be5c5901f17e87fa7a37d786c09928775e9a8e24ff61192a44222ef14bd.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/af433be5c5901f17e87fa7a37d786c09928775e9a8e24ff61192a44222ef14bd.jpg)

![ca6134f51d417dcd1b53222161c91ec2fbec76d1ef29239f7174a2f72e0063bd.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/ca6134f51d417dcd1b53222161c91ec2fbec76d1ef29239f7174a2f72e0063bd.jpg)

![f86d0549039cd218daec793c1fad0dc9fae7f1fda32abbe9165ea9e04fba2a9b.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/f86d0549039cd218daec793c1fad0dc9fae7f1fda32abbe9165ea9e04fba2a9b.jpg)

![fec8dce815ecd39f990f4b394a8725a25b6bdda349d7f44f059dd85803636b32.jpg](../iclr_results/1199_Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift/images/fec8dce815ecd39f990f4b394a8725a25b6bdda349d7f44f059dd85803636b32.jpg)

## Support is All You Need for Certified VAE Training


### Images

![10c8e86afb090370dc5c133c8bc602fec43e63a9a7feed505db27c3e8a850a22.jpg](../iclr_results/1200_Support is All You Need for Certified VAE Training/images/10c8e86afb090370dc5c133c8bc602fec43e63a9a7feed505db27c3e8a850a22.jpg)

![11984d5c3ef096e31bc982ba87344fbebbfbcdf7b86a75310ed2fca116da82f2.jpg](../iclr_results/1200_Support is All You Need for Certified VAE Training/images/11984d5c3ef096e31bc982ba87344fbebbfbcdf7b86a75310ed2fca116da82f2.jpg)

![e3a55027964ae473700e3557ea912a297f13154fc32600ecc4f56c3fa86e26a6.jpg](../iclr_results/1200_Support is All You Need for Certified VAE Training/images/e3a55027964ae473700e3557ea912a297f13154fc32600ecc4f56c3fa86e26a6.jpg)

### Tables

![7315cdca86935ceb6732c3ee1bd6857d164464188a1e349f2c7a3e0b20f55b24.jpg](../iclr_results/1200_Support is All You Need for Certified VAE Training/tables/7315cdca86935ceb6732c3ee1bd6857d164464188a1e349f2c7a3e0b20f55b24.jpg)

![8a4df7639da6346b313b9d81640e406203a7acee73c250bf8073ab3734f3bc89.jpg](../iclr_results/1200_Support is All You Need for Certified VAE Training/tables/8a4df7639da6346b313b9d81640e406203a7acee73c250bf8073ab3734f3bc89.jpg)

![ae4f7a0a6fb80a5f8bb20cb008cb6bf80a60ffea410177d04ea5093fd73ffa46.jpg](../iclr_results/1200_Support is All You Need for Certified VAE Training/tables/ae4f7a0a6fb80a5f8bb20cb008cb6bf80a60ffea410177d04ea5093fd73ffa46.jpg)

![de80e265e69ac2dcd1b213dfbb19e2ed9cf1e8a91c439227b86faa71ab986f4b.jpg](../iclr_results/1200_Support is All You Need for Certified VAE Training/tables/de80e265e69ac2dcd1b213dfbb19e2ed9cf1e8a91c439227b86faa71ab986f4b.jpg)

## Do Mice Grok? Glimpses of Hidden Progress in Sensory Cortex

### Images

![0770aac517ca5f2ee3330e19fc40203216172c1ee44ce198ee82fd801d6be274.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/0770aac517ca5f2ee3330e19fc40203216172c1ee44ce198ee82fd801d6be274.jpg)

![45e9e7517eae2e77c7b4956a847ab07ad412c20b10adc18032bba6d536e8046f.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/45e9e7517eae2e77c7b4956a847ab07ad412c20b10adc18032bba6d536e8046f.jpg)

![491d69d114c11d79b7c731305c8c390a28ee504b2e69fc981ed28a57dcdbe6c3.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/491d69d114c11d79b7c731305c8c390a28ee504b2e69fc981ed28a57dcdbe6c3.jpg)

![5306fa4cde02c57319d11af6bd237e5ff0bea10ec50c4eca25629db27e79600f.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/5306fa4cde02c57319d11af6bd237e5ff0bea10ec50c4eca25629db27e79600f.jpg)

![5ba63f896e36b0c016aa0cc60bfc34cedcecb9ec43ec66c53684751778bd8eae.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/5ba63f896e36b0c016aa0cc60bfc34cedcecb9ec43ec66c53684751778bd8eae.jpg)

![b46bb584a50a3ebfda0d5799d286f11fa3da3d0fb0b9635469e7e042c1741450.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/b46bb584a50a3ebfda0d5799d286f11fa3da3d0fb0b9635469e7e042c1741450.jpg)

![c711f8b331e06d39bddb9e66fb197ff5961d8459b5ad4272a9d3aa5b685cd78f.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/c711f8b331e06d39bddb9e66fb197ff5961d8459b5ad4272a9d3aa5b685cd78f.jpg)

![d1c033609b40adb73851aeac30a127c90f9e2be7ffe69a34d2fff12d2f536a23.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/d1c033609b40adb73851aeac30a127c90f9e2be7ffe69a34d2fff12d2f536a23.jpg)

![d589ed2b5fb70712c94536e454268d5b1296c809b481539c07da51f818c1a784.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/d589ed2b5fb70712c94536e454268d5b1296c809b481539c07da51f818c1a784.jpg)

![daf44a763a04a7d2253eb5afee07d9c48d274bee266a743e23fd10e6ee5e9709.jpg](../iclr_results/1201_Do Mice Grok_ Glimpses of Hidden Progress in Sensory Cortex/images/daf44a763a04a7d2253eb5afee07d9c48d274bee266a743e23fd10e6ee5e9709.jpg)
