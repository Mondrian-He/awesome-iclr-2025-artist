# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 10 of 100

## 目录 (Table of Contents)

1. [DRoP: Distributionally Robust Data Pruning](#DRoP-Distributionally-Robust-Data-Pruning)
2. [Vision Language Models are In-Context Value Learners](#Vision-Language-Models-are-In-Context-Value-Learners)
3. [Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data](#Linear-Spherical-Sliced-Optimal-Transport-A-Fast-Metric-for-Comparing-Spherical-Data)
4. [Conformal Prediction Sets Can Cause Disparate Impact](#Conformal-Prediction-Sets-Can-Cause-Disparate-Impact)
5. [PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems](#PhyMPGN-Physics-encoded-Message-Passing-Graph-Network-for-spatiotemporal-PDE-systems)
6. [Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors](#Generalization-Guarantees-for-Representation-Learning-via-Data-Dependent-Gaussian-Mixture-Priors)
7. [Weak-to-Strong Preference Optimization: Stealing Reward from Weak Aligned Model](#Weak-to-Strong-Preference-Optimization-Stealing-Reward-from-Weak-Aligned-Model)
8. [Strong Model Collapse](#Strong-Model-Collapse)
9. [CBQ: Cross-Block Quantization for Large Language Models](#CBQ-Cross-Block-Quantization-for-Large-Language-Models)
10. [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](#Time-MoE-Billion-Scale-Time-Series-Foundation-Models-with-Mixture-of-Experts)
11. [Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics](#Reinforcement-Learning-for-Control-of-Non-Markovian-Cellular-Population-Dynamics)
12. [BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics](#BirdSet-A-Large-Scale-Dataset-for-Audio-Classification-in-Avian-Bioacoustics)
13. [Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling](#Budgeted-Online-Continual-Learning-by-Adaptive-Layer-Freezing-and-Frequency-based-Sampling)
14. [Training-Free Activation Sparsity in Large Language Models](#Training-Free-Activation-Sparsity-in-Large-Language-Models)
15. [How Feature Learning Can Improve Neural Scaling Laws](#How-Feature-Learning-Can-Improve-Neural-Scaling-Laws)
16. [Beyond Next Token Prediction: Patch-Level Training for Large Language Models](#Beyond-Next-Token-Prediction-Patch-Level-Training-for-Large-Language-Models)
17. [Exact Certification of (Graph) Neural Networks Against Label Poisoning](#Exact-Certification-of-Graph-Neural-Networks-Against-Label-Poisoning)
18. [Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees](#Decentralized-Sporadic-Federated-Learning-A-Unified-Algorithmic-Framework-with-Convergence-Guarantees)
19. [Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification](#Credal-Wrapper-of-Model-Averaging-for-Uncertainty-Estimation-in-Classification)
20. [X-ALMA: Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale](#X-ALMA-Plug-Play-Modules-and-Adaptive-Rejection-for-Quality-Translation-at-Scale)
21. [TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models](#TAID-Temporally-Adaptive-Interpolated-Distillation-for-Efficient-Knowledge-Transfer-in-Language-Models)
22. [Wasserstein Distances, Neuronal Entanglement, and Sparsity](#Wasserstein-Distances-Neuronal-Entanglement-and-Sparsity)
23. [Geometric Inductive Biases of Deep Networks: The Role of Data and Architecture](#Geometric-Inductive-Biases-of-Deep-Networks-The-Role-of-Data-and-Architecture)
24. [Online Preference Alignment for Language Models via Count-based Exploration](#Online-Preference-Alignment-for-Language-Models-via-Count-based-Exploration)
25. [Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution](#Knowledge-Distillation-with-Multi-granularity-Mixture-of-Priors-for-Image-Super-Resolution)
26. [BodyGen: Advancing Towards Efficient Embodiment Co-Design](#BodyGen-Advancing-Towards-Efficient-Embodiment-Co-Design)
27. [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models](#Self-play-with-Execution-Feedback-Improving-Instruction-following-Capabilities-of-Large-Language-Models)
28. [One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt](#One-Prompt-One-Story-Free-Lunch-Consistent-Text-to-Image-Generation-Using-a-Single-Prompt)
29. [Linear SCM Identification in the Presence of Confounders and Gaussian Noise](#Linear-SCM-Identification-in-the-Presence-of-Confounders-and-Gaussian-Noise)
30. [AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs](#AutoDAN-Turbo-A-Lifelong-Agent-for-Strategy-Self-Exploration-to-Jailbreak-LLMs)
31. [NetFormer: An interpretable model for recovering dynamical connectivity in neuronal population dynamics](#NetFormer-An-interpretable-model-for-recovering-dynamical-connectivity-in-neuronal-population-dynamics)
32. [MotionAura: Generating High-Quality and Motion Consistent Videos using Discrete Diffusion](#MotionAura-Generating-High-Quality-and-Motion-Consistent-Videos-using-Discrete-Diffusion)
33. [Harnessing Diversity for Important Data Selection in Pretraining Large Language Models](#Harnessing-Diversity-for-Important-Data-Selection-in-Pretraining-Large-Language-Models)
34. [Generating Freeform Endoskeletal Robots](#Generating-Freeform-Endoskeletal-Robots)
35. [DARE the Extreme: Revisiting Delta-Parameter Pruning For Fine-Tuned Models](#DARE-the-Extreme-Revisiting-Delta-Parameter-Pruning-For-Fine-Tuned-Models)
36. [Dense Video Object Captioning from Disjoint Supervision](#Dense-Video-Object-Captioning-from-Disjoint-Supervision)
37. [Targeted Attack Improves Protection against Unauthorized Diffusion Customization](#Targeted-Attack-Improves-Protection-against-Unauthorized-Diffusion-Customization)

---


## DRoP: Distributionally Robust Data Pruning

### Images

![0424fac39962036b3e05930919f3691fcaf266c160a49ec0154456bce5395b91.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/0424fac39962036b3e05930919f3691fcaf266c160a49ec0154456bce5395b91.jpg)

![0daa7005c927f60cab58ceb87d22014ad92ee071324f5f34b5aa6e6ab18c80c2.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/0daa7005c927f60cab58ceb87d22014ad92ee071324f5f34b5aa6e6ab18c80c2.jpg)

![11dcf8729a98f174c392cd050d0ecf2ad30851052c4aa27c196c564797df33b0.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/11dcf8729a98f174c392cd050d0ecf2ad30851052c4aa27c196c564797df33b0.jpg)

![1567ab991a7fb24405e74d69efc46750145fda422fec59f99904859b25ef3642.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/1567ab991a7fb24405e74d69efc46750145fda422fec59f99904859b25ef3642.jpg)

![1ade4fd6bb1ead77f42d5c68f3eec0f586f0720ecc70699e9c36463483938e22.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/1ade4fd6bb1ead77f42d5c68f3eec0f586f0720ecc70699e9c36463483938e22.jpg)

![1e8f5a80bbabb92f700b9da40f1baa6fe8e5633d7ef45909e7372af86e78a8d5.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/1e8f5a80bbabb92f700b9da40f1baa6fe8e5633d7ef45909e7372af86e78a8d5.jpg)

![1ef4594a53c9e49297451aa3054647b27293188662c9b3e9d75cd8feb03aaeb5.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/1ef4594a53c9e49297451aa3054647b27293188662c9b3e9d75cd8feb03aaeb5.jpg)

![2216cf5279b844d31d4e3dc3087bcd08ea21979066f6d40a391b8e71067f7d6a.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/2216cf5279b844d31d4e3dc3087bcd08ea21979066f6d40a391b8e71067f7d6a.jpg)

![2bce6dfaebe87a4336c165a1e239bad984fa1262cfdc4d0b789b22bf49149d4d.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/2bce6dfaebe87a4336c165a1e239bad984fa1262cfdc4d0b789b22bf49149d4d.jpg)

![2f92ee0572cbf8e9526edb2241459a7f1505130e65add936916e555dc8d19c3f.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/2f92ee0572cbf8e9526edb2241459a7f1505130e65add936916e555dc8d19c3f.jpg)

![33fa8e89a863b9d0222c6f11ac07bed42cef25527260f8c2d59ee092c642359e.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/33fa8e89a863b9d0222c6f11ac07bed42cef25527260f8c2d59ee092c642359e.jpg)

![3dfab0d7c03eee31d171ca4ff9e756c4ce630062b612ad0cae68b898f4e290b0.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/3dfab0d7c03eee31d171ca4ff9e756c4ce630062b612ad0cae68b898f4e290b0.jpg)

![48e7f26bf56d4c9019da0bf7428c66d174f94bde7739e1bcbcd33835c42e9b08.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/48e7f26bf56d4c9019da0bf7428c66d174f94bde7739e1bcbcd33835c42e9b08.jpg)

![593f2f07d68fa922b4febdb96188818848346cfe4126b98f76fbf235376cf663.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/593f2f07d68fa922b4febdb96188818848346cfe4126b98f76fbf235376cf663.jpg)

![6012efc70c28dcf6ea7fc96596ba14b32d804c95cb0c9ff837c987b3c9e6d0b9.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/6012efc70c28dcf6ea7fc96596ba14b32d804c95cb0c9ff837c987b3c9e6d0b9.jpg)

![72532063e528ded43acea942c67c220e9a1d79dd017c7ccf16ceb0d6e247cacf.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/72532063e528ded43acea942c67c220e9a1d79dd017c7ccf16ceb0d6e247cacf.jpg)

![7f6c777cd42111efe28c714abeddb4e086c63be41f6801ec92ae8955b2822f44.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/7f6c777cd42111efe28c714abeddb4e086c63be41f6801ec92ae8955b2822f44.jpg)

![844c7162a8c9cab6486ce50397ec9e52e1926cf44dbc11b91a44c6868bdd7786.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/844c7162a8c9cab6486ce50397ec9e52e1926cf44dbc11b91a44c6868bdd7786.jpg)

![8f9c09ea95feeeedd5bc45946f13d802ec9a352f808b5983264133bf63b125ae.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/8f9c09ea95feeeedd5bc45946f13d802ec9a352f808b5983264133bf63b125ae.jpg)

![94b59c9ef5a0413df959a815813338400af3c49ec671fdbd6f4db8e22372174a.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/94b59c9ef5a0413df959a815813338400af3c49ec671fdbd6f4db8e22372174a.jpg)

![96da2d0afa149ee1fe4891dc8b98ca4f2bd440c92d65ca76f4a3bbc0b4bc9ae8.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/96da2d0afa149ee1fe4891dc8b98ca4f2bd440c92d65ca76f4a3bbc0b4bc9ae8.jpg)

![979c995e0fc767fda27d4616083274702bec3c4a4c278f1826fb0a898e7b7ed5.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/979c995e0fc767fda27d4616083274702bec3c4a4c278f1826fb0a898e7b7ed5.jpg)

![a0dfd961c5cd7cf11d5f4b8d99764d9028c557d5833e5035886e5012e0abdd76.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/a0dfd961c5cd7cf11d5f4b8d99764d9028c557d5833e5035886e5012e0abdd76.jpg)

![a465acea3332a9ce92476498456d19148c53bcc6395b361f38b624d3f7dc1ed7.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/a465acea3332a9ce92476498456d19148c53bcc6395b361f38b624d3f7dc1ed7.jpg)

![ac27c0d72c39fd94ccd0d3fd9b65cf79ee6903197ccd10f8ea18de25cf668559.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/ac27c0d72c39fd94ccd0d3fd9b65cf79ee6903197ccd10f8ea18de25cf668559.jpg)

![b84b7b883907548a1b2b4fd8193378226b4ec958022c04943def42274bf8076b.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/b84b7b883907548a1b2b4fd8193378226b4ec958022c04943def42274bf8076b.jpg)

![d189cac3169966b044d93f8a1170346c60b96e79e713122280fde864a03564d9.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/d189cac3169966b044d93f8a1170346c60b96e79e713122280fde864a03564d9.jpg)

![d31d13375151d1a516bed4df3476290697815ea21cb094cf7f79c7df454735c8.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/d31d13375151d1a516bed4df3476290697815ea21cb094cf7f79c7df454735c8.jpg)

![f18f05e68724eefce0afcfd2f1494d595fba9c7c6a0a2882b1f150a26d1bfd6c.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/f18f05e68724eefce0afcfd2f1494d595fba9c7c6a0a2882b1f150a26d1bfd6c.jpg)

![f6f4f9c878a2e5a344a25932330a9236be001cabc2e6926af34a86cf6f9c8318.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/images/f6f4f9c878a2e5a344a25932330a9236be001cabc2e6926af34a86cf6f9c8318.jpg)

### Tables

![09018a92a0433773429c32453574231b766aa0a3ebf5da3df8716dc2ad6492f0.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/tables/09018a92a0433773429c32453574231b766aa0a3ebf5da3df8716dc2ad6492f0.jpg)

![592c7487a43b5bf6cc0edca5381fc79f2d8e835cc34b9dc06e6a7e8b852dbd97.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/tables/592c7487a43b5bf6cc0edca5381fc79f2d8e835cc34b9dc06e6a7e8b852dbd97.jpg)

![6a5446b4cd3f6362cbe3b03669ec278a5e98720755f9109b156efcd7b6556b38.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/tables/6a5446b4cd3f6362cbe3b03669ec278a5e98720755f9109b156efcd7b6556b38.jpg)

![8a6a4e800fc6643698144cf1372bb19ecc3e15d4b467a3f3497391e659f20729.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/tables/8a6a4e800fc6643698144cf1372bb19ecc3e15d4b467a3f3497391e659f20729.jpg)

![977afe3ef9ddc409f00dbb1faa8243be5d51bd3876e410793b29f062dd46d75d.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/tables/977afe3ef9ddc409f00dbb1faa8243be5d51bd3876e410793b29f062dd46d75d.jpg)

![aaa71b7f5447119f366585d6ce83f6a59794bf193fbbaada0686d8a8b7a3a95c.jpg](../iclr_results/337_Bilinear MLPs enable weight-based mechanistic interpretability/tables/aaa71b7f5447119f366585d6ce83f6a59794bf193fbbaada0686d8a8b7a3a95c.jpg)

## DRoP: Distributionally Robust Data Pruning


### Images

![032fb6cb4894d0e148d4ade384676a6b7037a5c07339635ccd93d2daecf36418.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/032fb6cb4894d0e148d4ade384676a6b7037a5c07339635ccd93d2daecf36418.jpg)

![0d1894ae0db1130469dadcdf3a90ff38f95bdbf76c79f9154a3f7ee502707cb4.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/0d1894ae0db1130469dadcdf3a90ff38f95bdbf76c79f9154a3f7ee502707cb4.jpg)

![25ee7f210c209dd8c91a3ff4046603b6003b3d23378d1c8683e48d87faae2937.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/25ee7f210c209dd8c91a3ff4046603b6003b3d23378d1c8683e48d87faae2937.jpg)

![29b1836b0b88cb16459f8e444ccf264a222ef9b5c6ef676dd732009b5e82c294.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/29b1836b0b88cb16459f8e444ccf264a222ef9b5c6ef676dd732009b5e82c294.jpg)

![3c14555056efdb5b18d1f2b764b0f5a36fc67b88b222ca5f12f7781aee421b6e.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/3c14555056efdb5b18d1f2b764b0f5a36fc67b88b222ca5f12f7781aee421b6e.jpg)

![4ee403af5e922d5b1ea4d964fe2c42fb62de766baa3b9baeb1a0f33f34043328.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/4ee403af5e922d5b1ea4d964fe2c42fb62de766baa3b9baeb1a0f33f34043328.jpg)

![599a41407ca208dbfdad17fded8144e548cd27b023add1f4b0b179f4587dd6e5.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/599a41407ca208dbfdad17fded8144e548cd27b023add1f4b0b179f4587dd6e5.jpg)

![66bf47ca0d9cadad61e0784147686470b8623402c01e37781e08a848f2b1c87b.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/66bf47ca0d9cadad61e0784147686470b8623402c01e37781e08a848f2b1c87b.jpg)

![6d2d4772113ba563717da9db14eab8130c92aa055ce0bf490f4dfd7ae5dcb34f.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/6d2d4772113ba563717da9db14eab8130c92aa055ce0bf490f4dfd7ae5dcb34f.jpg)

![71f784d6248a6045ecd80be018502f38b1809d155f6ce8adbcbaa68a3a8cb6cd.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/71f784d6248a6045ecd80be018502f38b1809d155f6ce8adbcbaa68a3a8cb6cd.jpg)

![73dd7ba8fa44fe06fb9d025a525d8db3935b1cf2a5db1235a676ee724757f1c7.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/73dd7ba8fa44fe06fb9d025a525d8db3935b1cf2a5db1235a676ee724757f1c7.jpg)

![7f25abf8e721684a012500ca463575d01d5851e46cdbabf3fc2ca81f680c68e0.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/7f25abf8e721684a012500ca463575d01d5851e46cdbabf3fc2ca81f680c68e0.jpg)

![800d1b6c1ef44291a4e3d52a4e9c3e77e990f7ea43e351adb98d570d3301b28e.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/800d1b6c1ef44291a4e3d52a4e9c3e77e990f7ea43e351adb98d570d3301b28e.jpg)

![91439f138e17bdcfc41c17b29b1c85eecb89ff7858fab1f653fedbd874074888.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/91439f138e17bdcfc41c17b29b1c85eecb89ff7858fab1f653fedbd874074888.jpg)

![dc05c43a01fc3bda71337e4e99da7969f38586f11bdde398e59281635101a148.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/dc05c43a01fc3bda71337e4e99da7969f38586f11bdde398e59281635101a148.jpg)

![ebb3631bd11886d0cafd73961a9e6980c4f292f36272d72a23239438f5ace5e2.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/ebb3631bd11886d0cafd73961a9e6980c4f292f36272d72a23239438f5ace5e2.jpg)

![f31dc2779c5312805e24af416c399f8934978a40a4fd3d1a4a128f8c75ce88bf.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/images/f31dc2779c5312805e24af416c399f8934978a40a4fd3d1a4a128f8c75ce88bf.jpg)

### Tables

![084ec6e9b4eaada853b823d80a558a72cf38d32a7cb3ceaa7f22a6ed70d8f254.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/tables/084ec6e9b4eaada853b823d80a558a72cf38d32a7cb3ceaa7f22a6ed70d8f254.jpg)

![fb3d6b9b2faf6c20f643cba2a0c87298c75b025f0ac5802c4d85d8049ab6a0d4.jpg](../iclr_results/338_DRoP_ Distributionally Robust Data Pruning/tables/fb3d6b9b2faf6c20f643cba2a0c87298c75b025f0ac5802c4d85d8049ab6a0d4.jpg)

## Vision Language Models are In-Context Value Learners


### Images

![0691f65e7a79c8d6e2eec75f6674ff6407c7468766ba18f55acc2669844c1899.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/0691f65e7a79c8d6e2eec75f6674ff6407c7468766ba18f55acc2669844c1899.jpg)

![07cb00a0ba6f4fc9f4e7964393e0f35ff3013e9845d8078470cdad3bec9e4e55.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/07cb00a0ba6f4fc9f4e7964393e0f35ff3013e9845d8078470cdad3bec9e4e55.jpg)

![0d0f5eec221e96bb0ae7160e1fbee86c726bd9ca0cb234fe9bf3b2d95daf8cb8.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/0d0f5eec221e96bb0ae7160e1fbee86c726bd9ca0cb234fe9bf3b2d95daf8cb8.jpg)

![12a80f97d180ffff7236e04da7c68107f096de8840f09758289e072cdbd1b766.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/12a80f97d180ffff7236e04da7c68107f096de8840f09758289e072cdbd1b766.jpg)

![1589cf0469fc1aa65b0dd18880de3b36eb6579a3b7e4fc9f767cd21f15d2e515.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/1589cf0469fc1aa65b0dd18880de3b36eb6579a3b7e4fc9f767cd21f15d2e515.jpg)

![21f44cb26003618a3726103436d88e2a20bc729925cfdedf92d7376c08f95e2a.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/21f44cb26003618a3726103436d88e2a20bc729925cfdedf92d7376c08f95e2a.jpg)

![33a094ac90261afa9716f400a353961aec9cb1905555509b5ee51bc5f892f998.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/33a094ac90261afa9716f400a353961aec9cb1905555509b5ee51bc5f892f998.jpg)

![6aa9dc33f28d1218d5900b82faa5a03b318a4151d4d906a8ac8a8f040e127a76.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/6aa9dc33f28d1218d5900b82faa5a03b318a4151d4d906a8ac8a8f040e127a76.jpg)

![8d5db07e20c460a14b6b528d8af320ad25c7a4590e737f0949e1939b76101b96.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/8d5db07e20c460a14b6b528d8af320ad25c7a4590e737f0949e1939b76101b96.jpg)

![8e9c7a3b3233bf17cf6f87da66894fd19ea4281b8cfbdf988ebfffc70efdad5b.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/8e9c7a3b3233bf17cf6f87da66894fd19ea4281b8cfbdf988ebfffc70efdad5b.jpg)

![918737d246b1f90890c682ac7a2ead413fa11abf46fda26060e17de420652d2e.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/918737d246b1f90890c682ac7a2ead413fa11abf46fda26060e17de420652d2e.jpg)

![9dfd17638cf01cf838e92762ea8e1193ef6bd5490254fcac0c4a4c01c77d94a7.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/9dfd17638cf01cf838e92762ea8e1193ef6bd5490254fcac0c4a4c01c77d94a7.jpg)

![b66b600c64ebc26ae54be3179be84538d4e1efa80a3df925e7138110b625b7b2.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/b66b600c64ebc26ae54be3179be84538d4e1efa80a3df925e7138110b625b7b2.jpg)

![bcdf4578959ea60dad0340951698903f69cc7e2953b0cb22b1c36aab0bb42fd5.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/bcdf4578959ea60dad0340951698903f69cc7e2953b0cb22b1c36aab0bb42fd5.jpg)

![bd24837209ba3e746e125901d0965b69c6c7cd644d9046b636a3d40c70e01434.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/bd24837209ba3e746e125901d0965b69c6c7cd644d9046b636a3d40c70e01434.jpg)

![dc79860144f49fd13bb20ce073c0b7270a2110030d72ddadaf45e0f31ea12338.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/dc79860144f49fd13bb20ce073c0b7270a2110030d72ddadaf45e0f31ea12338.jpg)

![e66e41f5bf07cefac0397898a3d74609ea138eb60aa97f42919ee857c3ca4238.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/e66e41f5bf07cefac0397898a3d74609ea138eb60aa97f42919ee857c3ca4238.jpg)

![e6e75193f95e6022cd55ead07bc77d54f638b9e3c431f1db2df08608000944d1.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/e6e75193f95e6022cd55ead07bc77d54f638b9e3c431f1db2df08608000944d1.jpg)

![e99b91de4af35d54b5e3e6f3f0a2f5e1aa99fbac8e461b6ee2c7dbb2dc05ca17.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/e99b91de4af35d54b5e3e6f3f0a2f5e1aa99fbac8e461b6ee2c7dbb2dc05ca17.jpg)

![f3b078e374cd4c6543b6c637f2b9793df1a86afc20f89eb3a661158786e835fe.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/images/f3b078e374cd4c6543b6c637f2b9793df1a86afc20f89eb3a661158786e835fe.jpg)

### Tables

![0cc954c735c7671c266d8beb35e1c2d01b5a90e0019398a9a58b9951549af75a.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/tables/0cc954c735c7671c266d8beb35e1c2d01b5a90e0019398a9a58b9951549af75a.jpg)

![0e64b7e3dc9cf876e7d25d7c2b0f61c4e47c0395c327e747ce99f01b1fde41b9.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/tables/0e64b7e3dc9cf876e7d25d7c2b0f61c4e47c0395c327e747ce99f01b1fde41b9.jpg)

![41b36d6fc7a797d559535e2d9dc5d01a6e6c2824c052bf9007f1f43642ad2d5d.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/tables/41b36d6fc7a797d559535e2d9dc5d01a6e6c2824c052bf9007f1f43642ad2d5d.jpg)

![dc996a0cf6b88e10109a94bd77ae99511c04ff148993093f2518bab85b53afc1.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/tables/dc996a0cf6b88e10109a94bd77ae99511c04ff148993093f2518bab85b53afc1.jpg)

![e73bcfb868fb8e828bb7e3bf16b954641f6a63aec3331c648c293e7745f1a727.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/tables/e73bcfb868fb8e828bb7e3bf16b954641f6a63aec3331c648c293e7745f1a727.jpg)

![eaa740cd6235c874362f9d1cddb2202313411b24eb5f9948599b08e6b0711946.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/tables/eaa740cd6235c874362f9d1cddb2202313411b24eb5f9948599b08e6b0711946.jpg)

![ef7452c820d1bf1a720a0fc13f2c796d566868bc4419af05f9fe6daee5b155f1.jpg](../iclr_results/339_Vision Language Models are In-Context Value Learners/tables/ef7452c820d1bf1a720a0fc13f2c796d566868bc4419af05f9fe6daee5b155f1.jpg)

## Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data


### Images

![063380c59293f198e7e6d8d48ffa95ddf3e36d7920e5a592e0544d1e44d667d5.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/063380c59293f198e7e6d8d48ffa95ddf3e36d7920e5a592e0544d1e44d667d5.jpg)

![0c060695721dced49943018563790e04e06563bef34efc3c91238bd94251b444.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/0c060695721dced49943018563790e04e06563bef34efc3c91238bd94251b444.jpg)

![0fad61a0d3867d5d81950071a602937d06c68903a2fc08f093d0b9ec6f1f074d.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/0fad61a0d3867d5d81950071a602937d06c68903a2fc08f093d0b9ec6f1f074d.jpg)

![1088f62578d038144d572cffad6c5e1881a297dfbe67d8c8fc774f31b2ea0c0a.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/1088f62578d038144d572cffad6c5e1881a297dfbe67d8c8fc774f31b2ea0c0a.jpg)

![140acd62f170e240f1af13413c95589c169c81ba025ed95a50f6c947e00690d0.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/140acd62f170e240f1af13413c95589c169c81ba025ed95a50f6c947e00690d0.jpg)

![20a197c92b61d56c011366b1cb5deefbb4cb694b1647dd59889a022ff74d0e2d.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/20a197c92b61d56c011366b1cb5deefbb4cb694b1647dd59889a022ff74d0e2d.jpg)

![3500dadda5e6d17ca9ba8bbbe09e3ed50d38a77161c9e064f2b21a5f485e4da2.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/3500dadda5e6d17ca9ba8bbbe09e3ed50d38a77161c9e064f2b21a5f485e4da2.jpg)

![42da035e64972e8cc04827cf7b8c355ffb52b26dafc1264c1bd8114fdb792dd4.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/42da035e64972e8cc04827cf7b8c355ffb52b26dafc1264c1bd8114fdb792dd4.jpg)

![4a4a91ba6e4f334d7c5622f6062397c4a92e1e7eb55993f56187237d7baf16b9.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/4a4a91ba6e4f334d7c5622f6062397c4a92e1e7eb55993f56187237d7baf16b9.jpg)

![4b2eeb121551303c21d1c1723d4548d1eb070e1f641c23431d606a9630106a9a.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/4b2eeb121551303c21d1c1723d4548d1eb070e1f641c23431d606a9630106a9a.jpg)

![4e4951ed64f24a0d383424235943f03d52d0ff40930ed58bbc858c314864fc53.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/4e4951ed64f24a0d383424235943f03d52d0ff40930ed58bbc858c314864fc53.jpg)

![4eea6971f8706aef3d04c118b413f27c178fd6c07517913a3a03145c37fa08f8.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/4eea6971f8706aef3d04c118b413f27c178fd6c07517913a3a03145c37fa08f8.jpg)

![52155a0bc2cf889ffffbe366393ee4ff5a5be896ae83672cda90e5c050b8a7f4.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/52155a0bc2cf889ffffbe366393ee4ff5a5be896ae83672cda90e5c050b8a7f4.jpg)

![64b9a0f760d6834b7ea74e08e109eebc835b9e54b17488252207fd543be04af6.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/64b9a0f760d6834b7ea74e08e109eebc835b9e54b17488252207fd543be04af6.jpg)

![6c627e44803852c083c6b13151f94cd7868ac0d71d22702db7db4ff0e0784c5c.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/6c627e44803852c083c6b13151f94cd7868ac0d71d22702db7db4ff0e0784c5c.jpg)

![7642c24d48f9471e528d548d8687fad61a517d24b587819e7c9a2e321d468b1d.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/7642c24d48f9471e528d548d8687fad61a517d24b587819e7c9a2e321d468b1d.jpg)

![76b050f9ca2d590e2989b5e4a28d572cb05ce94aad6a3dc9ff79f9e0824e0f64.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/76b050f9ca2d590e2989b5e4a28d572cb05ce94aad6a3dc9ff79f9e0824e0f64.jpg)

![7ac71a6184567b96d0e36d78b263bcfff309ba98c23e01670afc6314207bc946.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/7ac71a6184567b96d0e36d78b263bcfff309ba98c23e01670afc6314207bc946.jpg)

![7b45f9f15b36ddeb44c189de22ad9d65a229135114c03fb5b9d352bda523b09a.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/7b45f9f15b36ddeb44c189de22ad9d65a229135114c03fb5b9d352bda523b09a.jpg)

![7bb2f4e6c5c717e8319a6e775c0f5397bc02d0b4ac6afa840cfcf737ce021433.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/7bb2f4e6c5c717e8319a6e775c0f5397bc02d0b4ac6afa840cfcf737ce021433.jpg)

![7c0395cf6f4ddaa2a5a6f172aaa9435ef182609f1fb2decbd2fc77363ce1fa9e.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/7c0395cf6f4ddaa2a5a6f172aaa9435ef182609f1fb2decbd2fc77363ce1fa9e.jpg)

![7d28ebcb19c0f3da35a685c6742f3b4efec4b027b462acd4b5b7fbbfc501cbe4.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/7d28ebcb19c0f3da35a685c6742f3b4efec4b027b462acd4b5b7fbbfc501cbe4.jpg)

![881a92c360e73924234fc0538e54522197395ba2044f3fa76f2966a138557ee9.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/881a92c360e73924234fc0538e54522197395ba2044f3fa76f2966a138557ee9.jpg)

![926e74de0045441418bfc93c7773c71a4cd5e60c0db1150fb3f252e824d85393.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/926e74de0045441418bfc93c7773c71a4cd5e60c0db1150fb3f252e824d85393.jpg)

![95cb56e86aba82eb56a93ae582aba01575c146e2d18bb349b5e9c0517192b63f.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/95cb56e86aba82eb56a93ae582aba01575c146e2d18bb349b5e9c0517192b63f.jpg)

![96002b17bcc8d9cee5938379dda8787d8ab837f743721c17ee5be563563576b3.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/96002b17bcc8d9cee5938379dda8787d8ab837f743721c17ee5be563563576b3.jpg)

![b02380958ce8665d7a79a8d45f4d1a8092e06d072539c21716f5e6002bc74ddd.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/b02380958ce8665d7a79a8d45f4d1a8092e06d072539c21716f5e6002bc74ddd.jpg)

![b9d5862306846db6381bfa962c4731c57232c396048c33bf86c4f63c86912f65.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/b9d5862306846db6381bfa962c4731c57232c396048c33bf86c4f63c86912f65.jpg)

![c0b4ea9f50c868af845b161ac1dcd94c66c0aebaab2e0a6ac8b7ed8ae8af8255.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/c0b4ea9f50c868af845b161ac1dcd94c66c0aebaab2e0a6ac8b7ed8ae8af8255.jpg)

![d8b7ab111da94e7c36c1f9819b7db5d4a35281b10b3b5fa5cbf995130d26dc04.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/d8b7ab111da94e7c36c1f9819b7db5d4a35281b10b3b5fa5cbf995130d26dc04.jpg)

![d8cd96b13c46f8e98ff608aa8075cc327f6e52d6d2353ded54e6ebb36d3ffb27.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/d8cd96b13c46f8e98ff608aa8075cc327f6e52d6d2353ded54e6ebb36d3ffb27.jpg)

![ef1fd41495adc5394232aaf63953539717f30707d993efbce69d37b65215b53a.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/ef1fd41495adc5394232aaf63953539717f30707d993efbce69d37b65215b53a.jpg)

![fcab871af14e29707b585ad9a196368734069ebd12a1bbbe2fd74e73c1632312.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/fcab871af14e29707b585ad9a196368734069ebd12a1bbbe2fd74e73c1632312.jpg)

![fe8301f79b0ee44914c4dc65e3eab6684c9dd3bcbfcb19ae5ee795f4e1e636a9.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/images/fe8301f79b0ee44914c4dc65e3eab6684c9dd3bcbfcb19ae5ee795f4e1e636a9.jpg)

### Tables

![132fa6a40af3f15c950c7c39b559064312f11ca32e9599ae5c69a7c49fd2a1a3.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/tables/132fa6a40af3f15c950c7c39b559064312f11ca32e9599ae5c69a7c49fd2a1a3.jpg)

![ad6f2ec45cbd9d24f79029c78b063fb1dff996cdb507ae1a32028fad7820356e.jpg](../iclr_results/340_Linear Spherical Sliced Optimal Transport_ A Fast Metric for Comparing Spherical Data/tables/ad6f2ec45cbd9d24f79029c78b063fb1dff996cdb507ae1a32028fad7820356e.jpg)

## Conformal Prediction Sets Can Cause Disparate Impact


### Images

![4236236c721a15bd8058a4eca1f6b9061a3197455c05829cad3583af1c18b8ed.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/images/4236236c721a15bd8058a4eca1f6b9061a3197455c05829cad3583af1c18b8ed.jpg)

![61cf3068b876c5ba997309988c18f21fe33b65366fa33805900394b591ff2086.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/images/61cf3068b876c5ba997309988c18f21fe33b65366fa33805900394b591ff2086.jpg)

![7847418016d10a262e1f8e8c7f1d8501b16555ff8c8d9c02483e2c04fd691395.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/images/7847418016d10a262e1f8e8c7f1d8501b16555ff8c8d9c02483e2c04fd691395.jpg)

![b5bc470676a5484a81c39010ec9c8bf7f6211359846f5e378bfc5f9be9f95345.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/images/b5bc470676a5484a81c39010ec9c8bf7f6211359846f5e378bfc5f9be9f95345.jpg)

![f93d3cd82fb1b4ec6825d80ee2af070804490a9f6c04f0ea609b14422de7f0b2.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/images/f93d3cd82fb1b4ec6825d80ee2af070804490a9f6c04f0ea609b14422de7f0b2.jpg)

![fef0ad1a88e69f77ff2d3dfef1043dcec0e34924188f369b0ca883e62f74666d.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/images/fef0ad1a88e69f77ff2d3dfef1043dcec0e34924188f369b0ca883e62f74666d.jpg)

### Tables

![171b9611156dcf25f3d13a846ecc27d4db3b6d5a075ceef32faaa8b2eb2df8d5.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/171b9611156dcf25f3d13a846ecc27d4db3b6d5a075ceef32faaa8b2eb2df8d5.jpg)

![535c493f305d6527c2a6a3ba774d0c8dbc9091ad7891eb67c9e0bc489306468b.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/535c493f305d6527c2a6a3ba774d0c8dbc9091ad7891eb67c9e0bc489306468b.jpg)

![5bf73fb659d2c161c2f5eff5fc7a489357f2c3707b583684098fab50497e4b24.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/5bf73fb659d2c161c2f5eff5fc7a489357f2c3707b583684098fab50497e4b24.jpg)

![769cc7e2b6c6b214cff766bb042235b10869b895e0b3308726e797b38335dc93.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/769cc7e2b6c6b214cff766bb042235b10869b895e0b3308726e797b38335dc93.jpg)

![7c6de8c27d0f402c648bfd6509a3f296e596a2f2f1e6ab929c5ac2fa18f70935.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/7c6de8c27d0f402c648bfd6509a3f296e596a2f2f1e6ab929c5ac2fa18f70935.jpg)

![877dcac2a6fb030513bb0e78bccee98dbb11078967788af518037aec46987b8d.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/877dcac2a6fb030513bb0e78bccee98dbb11078967788af518037aec46987b8d.jpg)

![b46794e2b06e8ae09386d3ef9a9d2420dfce8970dcdd7fb5aad1699857bf4adf.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/b46794e2b06e8ae09386d3ef9a9d2420dfce8970dcdd7fb5aad1699857bf4adf.jpg)

![dec018ff757feec47d90adeba5a1339679588a217529f6348d434ad0da1259f8.jpg](../iclr_results/341_Conformal Prediction Sets Can Cause Disparate Impact/tables/dec018ff757feec47d90adeba5a1339679588a217529f6348d434ad0da1259f8.jpg)

## PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems


### Images

![01a7a19f1c2363ba0ed3b57c95df64270e3ffc53551d1d57dbea7460d742004e.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/01a7a19f1c2363ba0ed3b57c95df64270e3ffc53551d1d57dbea7460d742004e.jpg)

![1f2aa38a1538a70962ba3de0b281451f17a7dea9e2ca421b5acbe2cd0c4088ad.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/1f2aa38a1538a70962ba3de0b281451f17a7dea9e2ca421b5acbe2cd0c4088ad.jpg)

![241b0b841ab700acce28ca866ced194f968bd8f910d695e3385eaad39856d703.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/241b0b841ab700acce28ca866ced194f968bd8f910d695e3385eaad39856d703.jpg)

![2b6d0937ce282be01e9cd17d194199259af943329978eecf79f0e4c1e3f8a4b0.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/2b6d0937ce282be01e9cd17d194199259af943329978eecf79f0e4c1e3f8a4b0.jpg)

![500d7f9ac01707d139501854b6a7d4d3c43c8cc1c18ca2c6b18df79764e8a47e.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/500d7f9ac01707d139501854b6a7d4d3c43c8cc1c18ca2c6b18df79764e8a47e.jpg)

![729007c95841039ccdab55b1986cd3421ab426e4ca76fe0b8ee04055ee2d5b37.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/729007c95841039ccdab55b1986cd3421ab426e4ca76fe0b8ee04055ee2d5b37.jpg)

![8afaece61af2937a5e1f42edb2471768ec88c7fcd30e00e0134702a7bda29ef7.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/8afaece61af2937a5e1f42edb2471768ec88c7fcd30e00e0134702a7bda29ef7.jpg)

![8fb7017dc2688875ce6a4d4a20d3506e498213d3ac81e1dd4e99864ce7e4abd8.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/8fb7017dc2688875ce6a4d4a20d3506e498213d3ac81e1dd4e99864ce7e4abd8.jpg)

![a15b2137e9f2f085ee55996c944790d3591dd24936686f37c4347627666abd4d.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/a15b2137e9f2f085ee55996c944790d3591dd24936686f37c4347627666abd4d.jpg)

![adc38be24f5885056b76c5a2ed5a22a4f317f814b431a6fb62d0b484499a9504.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/adc38be24f5885056b76c5a2ed5a22a4f317f814b431a6fb62d0b484499a9504.jpg)

![b689bc95abf0102ed98fc6dcf1123b87fbb673fd4ec0039754b9f1b93b74d621.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/b689bc95abf0102ed98fc6dcf1123b87fbb673fd4ec0039754b9f1b93b74d621.jpg)

![baaf43b4b9b8291de4a163484fd91a143a0a656445271ed91160ea436c753802.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/baaf43b4b9b8291de4a163484fd91a143a0a656445271ed91160ea436c753802.jpg)

![c16793dda44d5db6bf7b6b76be40ae9857b75096a7391a5214fe9fcccd43d114.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/c16793dda44d5db6bf7b6b76be40ae9857b75096a7391a5214fe9fcccd43d114.jpg)

![eb4b4d69bb1bc9714e8dc567ab65e4590beab9f81119eb594a950d6611c8f358.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/images/eb4b4d69bb1bc9714e8dc567ab65e4590beab9f81119eb594a950d6611c8f358.jpg)

### Tables

![116b392754a2913452c0cf11daae4f684abe58c8b081cfbaed7829bab53c3533.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/116b392754a2913452c0cf11daae4f684abe58c8b081cfbaed7829bab53c3533.jpg)

![1db6b1ea02bd6d1c13988cd5486d45eb330d1d406ba3e25e97fe68c352721b7c.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/1db6b1ea02bd6d1c13988cd5486d45eb330d1d406ba3e25e97fe68c352721b7c.jpg)

![260a262e51d1ba2f6881076b6917f6e5af5a835ce1b399b23fc5e87e43d7c64d.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/260a262e51d1ba2f6881076b6917f6e5af5a835ce1b399b23fc5e87e43d7c64d.jpg)

![2f4067b583e39b9fe278130969afca7e28cbb322484d835a30b4b5eaa4355d99.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/2f4067b583e39b9fe278130969afca7e28cbb322484d835a30b4b5eaa4355d99.jpg)

![2fb686a74f86784868020a7fdfb40943e8668ed66b060b8c521d0e03ec7965ec.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/2fb686a74f86784868020a7fdfb40943e8668ed66b060b8c521d0e03ec7965ec.jpg)

![49867ca0c03ec45e82f17a50a15cdcbd876025995a1338d52f99e64c996b5ae2.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/49867ca0c03ec45e82f17a50a15cdcbd876025995a1338d52f99e64c996b5ae2.jpg)

![78117ef2078972b0a38f0062df389ed4de38db6bf3c3c3bab3c1c7b5ef5a82bb.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/78117ef2078972b0a38f0062df389ed4de38db6bf3c3c3bab3c1c7b5ef5a82bb.jpg)

![859eba7a634f43ca27d2194b73980f7233b56618a5dfedaf19cfd64c36f0177d.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/859eba7a634f43ca27d2194b73980f7233b56618a5dfedaf19cfd64c36f0177d.jpg)

![86b63801a2d4727aa59bfc406849d99ddb86022615905831ee09304b3d8a9d66.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/86b63801a2d4727aa59bfc406849d99ddb86022615905831ee09304b3d8a9d66.jpg)

![8a9346800b29e98021d7cf249b6f1121a1dd45797a3aa2b4e7f649da392c871c.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/8a9346800b29e98021d7cf249b6f1121a1dd45797a3aa2b4e7f649da392c871c.jpg)

![9c02a0c4a96bbf12e82b1a5b4357f4ab2935c0612f0456efa4c474cea10f74ce.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/9c02a0c4a96bbf12e82b1a5b4357f4ab2935c0612f0456efa4c474cea10f74ce.jpg)

![bfa4cf799df195efdfbbf66fba80b1ca431427ed0ea6d03c10ef6a17d31d4133.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/bfa4cf799df195efdfbbf66fba80b1ca431427ed0ea6d03c10ef6a17d31d4133.jpg)

![c5dbef58045ac7c48e2dccc672eb0544567a0fb64959048f2b31c5faeb98fc5f.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/c5dbef58045ac7c48e2dccc672eb0544567a0fb64959048f2b31c5faeb98fc5f.jpg)

![c83585d88746ff8de1a0c307d6eb1fc74b795181df519b5f436323bdbac0cdfd.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/c83585d88746ff8de1a0c307d6eb1fc74b795181df519b5f436323bdbac0cdfd.jpg)

![d539ccee9187787f428c520076b49f7c5a9831a5adb6c60c0e529e88fbc6702a.jpg](../iclr_results/342_PhyMPGN_ Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems/tables/d539ccee9187787f428c520076b49f7c5a9831a5adb6c60c0e529e88fbc6702a.jpg)

## Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors


### Images

![a555dd18ae1048eefeaea4a69902e149652231c9522de650e3d60ff751079672.jpg](../iclr_results/343_Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors/images/a555dd18ae1048eefeaea4a69902e149652231c9522de650e3d60ff751079672.jpg)

![c4efc46b62b81d93eee7ca69a12192369aaaac5cdc289dd172e01a609c618db2.jpg](../iclr_results/343_Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors/images/c4efc46b62b81d93eee7ca69a12192369aaaac5cdc289dd172e01a609c618db2.jpg)

![d0e82520905a2460bbc16427e13ec44dd30bb321f64bdc3b1187d744efbe6725.jpg](../iclr_results/343_Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors/images/d0e82520905a2460bbc16427e13ec44dd30bb321f64bdc3b1187d744efbe6725.jpg)

![e09987a4074ab9428491b27df9df50ce238528209999435f6f1e697ab076b4a0.jpg](../iclr_results/343_Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors/images/e09987a4074ab9428491b27df9df50ce238528209999435f6f1e697ab076b4a0.jpg)

### Tables

![641bb89fea8a97143659b943c3e0926e0bdbbf79b27b69fafa75a0147efda529.jpg](../iclr_results/343_Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors/tables/641bb89fea8a97143659b943c3e0926e0bdbbf79b27b69fafa75a0147efda529.jpg)

![796081ff09076c0cc301021bc3d802b34a7743184a9e404a8c0224250488002b.jpg](../iclr_results/343_Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors/tables/796081ff09076c0cc301021bc3d802b34a7743184a9e404a8c0224250488002b.jpg)

## Weak-to-Strong Preference Optimization: Stealing Reward from Weak Aligned Model


### Images

![095caeb4ec07793cd1de9459eadaf72e1b4bc6b947625480aeff69573fa7ce8b.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/095caeb4ec07793cd1de9459eadaf72e1b4bc6b947625480aeff69573fa7ce8b.jpg)

![10c9c654720b2d49001f86f19a5a22db1d1cccb9fa1497b1b1d84ef3a06acb45.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/10c9c654720b2d49001f86f19a5a22db1d1cccb9fa1497b1b1d84ef3a06acb45.jpg)

![22b92ea20755e11b5dce7bade7e988c9d26c07d7b006d0501f078dc76a63061f.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/22b92ea20755e11b5dce7bade7e988c9d26c07d7b006d0501f078dc76a63061f.jpg)

![237bce39c291cb110a7526c53ad59e5b27826597f8b0a58c658bcc1fd6ee51cf.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/237bce39c291cb110a7526c53ad59e5b27826597f8b0a58c658bcc1fd6ee51cf.jpg)

![3170084790d3a4b082a85e609b414d9bb4b649fa0c99a8adc450b9567e58e4ea.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/3170084790d3a4b082a85e609b414d9bb4b649fa0c99a8adc450b9567e58e4ea.jpg)

![326bd750b1a70b9d70e551c01c36b8b56e986d853f7969b4b706e01ffd1d42fd.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/326bd750b1a70b9d70e551c01c36b8b56e986d853f7969b4b706e01ffd1d42fd.jpg)

![57e442523bd9131f26fcdf43d1307cde5ccb085b2403b3983b031e984cafefea.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/57e442523bd9131f26fcdf43d1307cde5ccb085b2403b3983b031e984cafefea.jpg)

![5897d3dba7a038aa7c878ac44b38e9e44c0f33c1ff5ad54cfb9e2ab6577521e5.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/5897d3dba7a038aa7c878ac44b38e9e44c0f33c1ff5ad54cfb9e2ab6577521e5.jpg)

![7866ea07055056db8586e2e253226c61529b608915a359727dbf7ffb62b6891d.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/7866ea07055056db8586e2e253226c61529b608915a359727dbf7ffb62b6891d.jpg)

![97851a85408cbdc3898362ecd86dbe83a7f17f476e0a5c7d6d33abfe5e0f73b3.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/97851a85408cbdc3898362ecd86dbe83a7f17f476e0a5c7d6d33abfe5e0f73b3.jpg)

![dd01893a9791f5e0aa592c61471fdae46b309c45f4c24c246cf71f1a736e392e.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/dd01893a9791f5e0aa592c61471fdae46b309c45f4c24c246cf71f1a736e392e.jpg)

![e28ad27ee722c0db885dad2ac411edd8f88f09c5e2aee01818d210904949bf41.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/images/e28ad27ee722c0db885dad2ac411edd8f88f09c5e2aee01818d210904949bf41.jpg)

### Tables

![15c0166a5ebe4a719a110ca7b888b079fa61a78527fc0c25c01f911759bd6518.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/15c0166a5ebe4a719a110ca7b888b079fa61a78527fc0c25c01f911759bd6518.jpg)

![222004b90372a0d826530c6c7124ccea1e7ddecb8921a0ed46007f5e9f383021.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/222004b90372a0d826530c6c7124ccea1e7ddecb8921a0ed46007f5e9f383021.jpg)

![3587a278ddd0b7936a6cb6e52ff6a69e73ecbd2587a0b3c8f73eb5d5de5fbcc0.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/3587a278ddd0b7936a6cb6e52ff6a69e73ecbd2587a0b3c8f73eb5d5de5fbcc0.jpg)

![3c412ada7922f854684e9e753d13301255aa5a5608930cb407c288aaba3474ce.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/3c412ada7922f854684e9e753d13301255aa5a5608930cb407c288aaba3474ce.jpg)

![414553308ebc16c8c4cefc8c688588b1ece225e73074e948bf09485babd34780.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/414553308ebc16c8c4cefc8c688588b1ece225e73074e948bf09485babd34780.jpg)

![46761e669a16844e4f49194e34037c53821a8918fe06e39c475c7e5d078e239f.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/46761e669a16844e4f49194e34037c53821a8918fe06e39c475c7e5d078e239f.jpg)

![5c635ab0b2cdcc9a387e4ca7230ccfcfd0043afceddf187f52ebd7283f7cf222.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/5c635ab0b2cdcc9a387e4ca7230ccfcfd0043afceddf187f52ebd7283f7cf222.jpg)

![75e9104accf35bdd379edcddafc6235b3b1366596833ac56a6fe353bc87d2ebc.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/75e9104accf35bdd379edcddafc6235b3b1366596833ac56a6fe353bc87d2ebc.jpg)

![84e02787d270a4362d47fcf354772086cbf4981a051c4a4b34cf08c2aeee4014.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/84e02787d270a4362d47fcf354772086cbf4981a051c4a4b34cf08c2aeee4014.jpg)

![87dcd2fad0a29f444a1e7158ba1c25bbe6b43ed18dbb732cefeb1ddd258a42a7.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/87dcd2fad0a29f444a1e7158ba1c25bbe6b43ed18dbb732cefeb1ddd258a42a7.jpg)

![b047741a33f35532918e291ec20d9cef6cbc6c41d66b8b7b372670a315d04a28.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/b047741a33f35532918e291ec20d9cef6cbc6c41d66b8b7b372670a315d04a28.jpg)

![b1bc8f7e70d3ceea9c5581cd07d1bb87748e9692d34d8f96e827aebdb8341a80.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/b1bc8f7e70d3ceea9c5581cd07d1bb87748e9692d34d8f96e827aebdb8341a80.jpg)

![b23dc8480fe487ec8469e0812dcf0864912c3c7ac2f4b3b079136a25cdff1c5c.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/b23dc8480fe487ec8469e0812dcf0864912c3c7ac2f4b3b079136a25cdff1c5c.jpg)

![bbfd3bb2166be1c0d1adb694be8deb275e208edebf5582ce41705a969cf5b985.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/bbfd3bb2166be1c0d1adb694be8deb275e208edebf5582ce41705a969cf5b985.jpg)

![bec591c4ee96a448d51124b307599a201765e726bf4b8fa2f59dd1ce44515746.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/bec591c4ee96a448d51124b307599a201765e726bf4b8fa2f59dd1ce44515746.jpg)

![bf0de9fc8d952f0aade2f9fa28be2f8753e94fe467041797246d0f0aabab070d.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/bf0de9fc8d952f0aade2f9fa28be2f8753e94fe467041797246d0f0aabab070d.jpg)

![c7454067e07beaa15ad5be6c58c4240c1db7a802deae2cf4ccbaf99a5c25501d.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/c7454067e07beaa15ad5be6c58c4240c1db7a802deae2cf4ccbaf99a5c25501d.jpg)

![df4607889e96c0527dabd1cced869042f75a4bedd740e8890655b39a5c2c07a3.jpg](../iclr_results/344_Weak-to-Strong Preference Optimization_ Stealing Reward from Weak Aligned Model/tables/df4607889e96c0527dabd1cced869042f75a4bedd740e8890655b39a5c2c07a3.jpg)

## Strong Model Collapse


### Images

![0dea40edb4cb1540dab673d34d4167bcbdd0256a852c67dde402df42e601b896.jpg](../iclr_results/345_Strong Model Collapse/images/0dea40edb4cb1540dab673d34d4167bcbdd0256a852c67dde402df42e601b896.jpg)

![3d688d6cc7ea5340e21b4b9b90762299cb13a12bc8e85f9734ea7aa6e1ed0223.jpg](../iclr_results/345_Strong Model Collapse/images/3d688d6cc7ea5340e21b4b9b90762299cb13a12bc8e85f9734ea7aa6e1ed0223.jpg)

![4df9618eeec175f8c86832dfbe03b7efbb206303bbdfed496c5ca43a9990815d.jpg](../iclr_results/345_Strong Model Collapse/images/4df9618eeec175f8c86832dfbe03b7efbb206303bbdfed496c5ca43a9990815d.jpg)

![56b10ea9b9debe9c5762451cc4506788c41bb9ee58fd8cdc43d5ea804981fffc.jpg](../iclr_results/345_Strong Model Collapse/images/56b10ea9b9debe9c5762451cc4506788c41bb9ee58fd8cdc43d5ea804981fffc.jpg)

![641270548291355eea09910b79d788512b021c607a9ac409d8fde63313caa8e9.jpg](../iclr_results/345_Strong Model Collapse/images/641270548291355eea09910b79d788512b021c607a9ac409d8fde63313caa8e9.jpg)

![814501da2a3680bce587f31b62d80a68a81fd4cae1d16580df3aa40a1d96474a.jpg](../iclr_results/345_Strong Model Collapse/images/814501da2a3680bce587f31b62d80a68a81fd4cae1d16580df3aa40a1d96474a.jpg)

![8d79d5afa4e91b5ecd3f3e521dfc192443766bf26cb1991e54d1f6b6a667b6c8.jpg](../iclr_results/345_Strong Model Collapse/images/8d79d5afa4e91b5ecd3f3e521dfc192443766bf26cb1991e54d1f6b6a667b6c8.jpg)

![9e7faf826a5c3e2aaf63e9481b72d340471ab5770e14aabd8972759b5a617c7d.jpg](../iclr_results/345_Strong Model Collapse/images/9e7faf826a5c3e2aaf63e9481b72d340471ab5770e14aabd8972759b5a617c7d.jpg)

![a4387af651ae7b6eb728c3115f67539e9853e06a336d3e6bb9354125168d7e2e.jpg](../iclr_results/345_Strong Model Collapse/images/a4387af651ae7b6eb728c3115f67539e9853e06a336d3e6bb9354125168d7e2e.jpg)

![a8e961ab65b27e6e44c160c00d7776974873cf79fe734d767bf4a11743db15ed.jpg](../iclr_results/345_Strong Model Collapse/images/a8e961ab65b27e6e44c160c00d7776974873cf79fe734d767bf4a11743db15ed.jpg)

![befbd49c66dbe96f1ad78ef8fef3e24b7c97942ccff26f8a4793bfd10de90516.jpg](../iclr_results/345_Strong Model Collapse/images/befbd49c66dbe96f1ad78ef8fef3e24b7c97942ccff26f8a4793bfd10de90516.jpg)

![c2a9eb5ffbc937129828acaa969c8339f96715b436262734fe2a5aaa89d660f4.jpg](../iclr_results/345_Strong Model Collapse/images/c2a9eb5ffbc937129828acaa969c8339f96715b436262734fe2a5aaa89d660f4.jpg)

## CBQ: Cross-Block Quantization for Large Language Models


### Images

![22df2f518b3189fe0f2c861fa47947677b6d3a5f94cabf72a6e1e7776f916baf.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/images/22df2f518b3189fe0f2c861fa47947677b6d3a5f94cabf72a6e1e7776f916baf.jpg)

![9001bc875370fd34476de890d1b9d6a37a06a462ec463fe12088b0f2a42aed57.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/images/9001bc875370fd34476de890d1b9d6a37a06a462ec463fe12088b0f2a42aed57.jpg)

![b8124cbeb05bfa04763d9ad82fd7e57a51dd758f808628633e1a788430f49d41.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/images/b8124cbeb05bfa04763d9ad82fd7e57a51dd758f808628633e1a788430f49d41.jpg)

### Tables

![0f596a6c0a47e392ddcfcf544ea7eb0ca0c84c4714679c9f0e6252cd271e977c.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/0f596a6c0a47e392ddcfcf544ea7eb0ca0c84c4714679c9f0e6252cd271e977c.jpg)

![1ae2e238f6033986f3f4e2e207fcb0695af69819356f8f58761c26e599ffebc0.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/1ae2e238f6033986f3f4e2e207fcb0695af69819356f8f58761c26e599ffebc0.jpg)

![3ff3c1f68e7302449b321682646244724f7666a96875a53ff8dd2cff2588e90b.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/3ff3c1f68e7302449b321682646244724f7666a96875a53ff8dd2cff2588e90b.jpg)

![74019b4c62653c3a475f8bdcd47a9d6a31bb5eb3b87b3b30174aa8b5dc4e455c.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/74019b4c62653c3a475f8bdcd47a9d6a31bb5eb3b87b3b30174aa8b5dc4e455c.jpg)

![7df2f1864e3edb512c932a2169c55ee5fa72331ba6d08cdd18be8a2ef3b48b4e.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/7df2f1864e3edb512c932a2169c55ee5fa72331ba6d08cdd18be8a2ef3b48b4e.jpg)

![81258973bb52743674a290af5cf5e0b57d9a4e5b6f48fee8ebc9f56a1f733f7e.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/81258973bb52743674a290af5cf5e0b57d9a4e5b6f48fee8ebc9f56a1f733f7e.jpg)

![8219bec2503d1973edf96be0014eede70f17bd16d2c9ec401900b807d07ad381.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/8219bec2503d1973edf96be0014eede70f17bd16d2c9ec401900b807d07ad381.jpg)

![863552e44a53104f9a48d4c5f74a12d6e68319cf8f6f3e4b484999f36adb2c43.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/863552e44a53104f9a48d4c5f74a12d6e68319cf8f6f3e4b484999f36adb2c43.jpg)

![92e1cd7a74cded9ca5604f728a3decde324470d4f7dd51e9411a66733c2be363.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/92e1cd7a74cded9ca5604f728a3decde324470d4f7dd51e9411a66733c2be363.jpg)

![9613985d1ca3d843403edd1a3792b5f3fcf2d3e65e2c69d2403ea0a7ae2a80b0.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/9613985d1ca3d843403edd1a3792b5f3fcf2d3e65e2c69d2403ea0a7ae2a80b0.jpg)

![a71982501a279411253a956f746d93e13107f2e92d754f40615c658f97a70647.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/a71982501a279411253a956f746d93e13107f2e92d754f40615c658f97a70647.jpg)

![ab762cc059cdb2e66f81c2414f44928f3b1f9e1570a45ed96a596bf387d2e438.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/ab762cc059cdb2e66f81c2414f44928f3b1f9e1570a45ed96a596bf387d2e438.jpg)

![c0ac9db04b8cb1d5bb0da1122859537e06c68769590e7aa5e6423d24067c76a7.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/c0ac9db04b8cb1d5bb0da1122859537e06c68769590e7aa5e6423d24067c76a7.jpg)

![df6c853264a887a76c21f051666910282decfd82c674aafafc23354d7deba078.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/df6c853264a887a76c21f051666910282decfd82c674aafafc23354d7deba078.jpg)

![e75f0057f592092321bde0e986d754c9d35f074718dcbeee9c9e60b0b64f6a4c.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/e75f0057f592092321bde0e986d754c9d35f074718dcbeee9c9e60b0b64f6a4c.jpg)

![fe0f4977e34ee20916092f56f506c800bbeebd79c6e9e98fcf8103907e9b5bb8.jpg](../iclr_results/346_CBQ_ Cross-Block Quantization for Large Language Models/tables/fe0f4977e34ee20916092f56f506c800bbeebd79c6e9e98fcf8103907e9b5bb8.jpg)

## Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts


### Images

![2baa1bfc15a975b62d74d7fc8913996d37d85240ddef8319076c17c446faba72.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/2baa1bfc15a975b62d74d7fc8913996d37d85240ddef8319076c17c446faba72.jpg)

![2efccf869da3caffeb8d4461b4cdbe385a81da21437889ac3a45bbf91e8718a8.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/2efccf869da3caffeb8d4461b4cdbe385a81da21437889ac3a45bbf91e8718a8.jpg)

![302721e384902b734e3fc01aa2c0a00c7baafc125b8fe56d65a95aff3012ff23.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/302721e384902b734e3fc01aa2c0a00c7baafc125b8fe56d65a95aff3012ff23.jpg)

![77e52ca113527994349396152d374675105e149d61dd049764f165a91ec98ae8.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/77e52ca113527994349396152d374675105e149d61dd049764f165a91ec98ae8.jpg)

![8647fd82559dee0deec46f24adfda86521752cf40ecfc62688107dcce639b0b9.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/8647fd82559dee0deec46f24adfda86521752cf40ecfc62688107dcce639b0b9.jpg)

![b43d59c4672efa2797df4f8f1a90a711dd3d599af56ed2eb9bf8220126c2d129.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/b43d59c4672efa2797df4f8f1a90a711dd3d599af56ed2eb9bf8220126c2d129.jpg)

![b53b3d949fde8675300528609dbbf9619051e0e5ee2c40a1c4495ed2f87b4504.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/b53b3d949fde8675300528609dbbf9619051e0e5ee2c40a1c4495ed2f87b4504.jpg)

![c1776efc25fe88e06ba87f8c0649422de9d81e365a0f164b1fe2143d61468116.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/c1776efc25fe88e06ba87f8c0649422de9d81e365a0f164b1fe2143d61468116.jpg)

![d6c47eede23cab343b81a20b663b4d0a8e9b4a553e911de48a1c8ed7c1ab01c6.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/d6c47eede23cab343b81a20b663b4d0a8e9b4a553e911de48a1c8ed7c1ab01c6.jpg)

![f3373cb0929bcca812d56c7575b105f37a3209569e3eee7c91ee499bfd61c556.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/f3373cb0929bcca812d56c7575b105f37a3209569e3eee7c91ee499bfd61c556.jpg)

![ff48c79d8af9f72824eb65611e03fc392af4cddfd27e9897425f688fc704ba5c.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/images/ff48c79d8af9f72824eb65611e03fc392af4cddfd27e9897425f688fc704ba5c.jpg)

### Tables

![0932396f4f06d099ed7ea76bd05d367a6d731ed41887fc7a89a3345f5d5337fb.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/0932396f4f06d099ed7ea76bd05d367a6d731ed41887fc7a89a3345f5d5337fb.jpg)

![1713596ff70a3b588498d36165c5b42a8d83723a1a7151ad8032d4c069d3154f.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/1713596ff70a3b588498d36165c5b42a8d83723a1a7151ad8032d4c069d3154f.jpg)

![1a51daa4fc8e257a7d312cd74a97a4555088e021b63bf1230025986a18f288cf.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/1a51daa4fc8e257a7d312cd74a97a4555088e021b63bf1230025986a18f288cf.jpg)

![217b28f644da7bcddad0cefaecdb35138da8924fe6eff7bd4f60866258282596.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/217b28f644da7bcddad0cefaecdb35138da8924fe6eff7bd4f60866258282596.jpg)

![231528310fb6c5c43023619a65fa3f43601e66fc2363d3dcf043e05d8cf20659.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/231528310fb6c5c43023619a65fa3f43601e66fc2363d3dcf043e05d8cf20659.jpg)

![248c3029f152f9ec9e27803b48f5422b48455ece4907a7c46fb6ca1c5c307fb1.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/248c3029f152f9ec9e27803b48f5422b48455ece4907a7c46fb6ca1c5c307fb1.jpg)

![5239cd4f28bb1ffe2517f297563ee6451097865235525a1a525f8b23109f2a59.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/5239cd4f28bb1ffe2517f297563ee6451097865235525a1a525f8b23109f2a59.jpg)

![59bd49ea2e8d823c19f0ce9aeaf058b3582cd6825e619fc6957e291a6535e6ed.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/59bd49ea2e8d823c19f0ce9aeaf058b3582cd6825e619fc6957e291a6535e6ed.jpg)

![698f12d7b26baee9016bc3b466590420d0e3215a6835ec38ee59b12cfd6b045b.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/698f12d7b26baee9016bc3b466590420d0e3215a6835ec38ee59b12cfd6b045b.jpg)

![738dd73eb87a504ced2634e621cb14065a2d229a41a51fde0bc90e6aa3649f14.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/738dd73eb87a504ced2634e621cb14065a2d229a41a51fde0bc90e6aa3649f14.jpg)

![7cd6258ef31ae585c02b398a50b87f66b1006f0540eda04c657aae0cba1e6c4e.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/7cd6258ef31ae585c02b398a50b87f66b1006f0540eda04c657aae0cba1e6c4e.jpg)

![8cccb41f7f82ba1c758636047711565087386416a802025d411c6e80a838874b.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/8cccb41f7f82ba1c758636047711565087386416a802025d411c6e80a838874b.jpg)

![9cdb0138b23bef29b3dfe7e8bc1bba1728f743352097255d5efb1c12758e2ff9.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/9cdb0138b23bef29b3dfe7e8bc1bba1728f743352097255d5efb1c12758e2ff9.jpg)

![a62a1873b820e5086e4eeeb91e8d1697252e4fa9fe910e3f3ae2cbdeae701b5d.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/a62a1873b820e5086e4eeeb91e8d1697252e4fa9fe910e3f3ae2cbdeae701b5d.jpg)

![c88a9137ee978bf217c2f9ceeee32d89bd7b625153aeb0fb15f1f6ec7d1f7e79.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/c88a9137ee978bf217c2f9ceeee32d89bd7b625153aeb0fb15f1f6ec7d1f7e79.jpg)

![caba247257ea4040272e9ccb49f8ec8d076eb3f8321ca11dd6f2c947ce9cc499.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/caba247257ea4040272e9ccb49f8ec8d076eb3f8321ca11dd6f2c947ce9cc499.jpg)

![cf455539a862d0f11916e07353bb0316b373fca8b5be19b85c5fb8a1de29f8b7.jpg](../iclr_results/347_Time-MoE_ Billion-Scale Time Series Foundation Models with Mixture of Experts/tables/cf455539a862d0f11916e07353bb0316b373fca8b5be19b85c5fb8a1de29f8b7.jpg)

## Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics


### Images

![0a5a2873c745cf604c65f8048fa6a4fb08dcac594aba8a8770df3a50bb065d1d.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/images/0a5a2873c745cf604c65f8048fa6a4fb08dcac594aba8a8770df3a50bb065d1d.jpg)

![153f16d3efa531f21700fefe66cf07daeed390bc5c39585199960412cb8ceba5.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/images/153f16d3efa531f21700fefe66cf07daeed390bc5c39585199960412cb8ceba5.jpg)

![606045c55e2059f522354a0b70329deb6cfa9abc59df9a3cb28367484f74eec7.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/images/606045c55e2059f522354a0b70329deb6cfa9abc59df9a3cb28367484f74eec7.jpg)

![b3f3bf201ec977f86100dd37adb4aa5dc9fe972b51c59f1140163859f0e015a8.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/images/b3f3bf201ec977f86100dd37adb4aa5dc9fe972b51c59f1140163859f0e015a8.jpg)

![c89bfa3763739bbfcc70df5b844256e4e3981a18844521fbcbd8911af00c33b9.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/images/c89bfa3763739bbfcc70df5b844256e4e3981a18844521fbcbd8911af00c33b9.jpg)

![d328f00c25372d7fa56253d8f1d6b547ba17004720e78e1f26fc6f4b45915ca4.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/images/d328f00c25372d7fa56253d8f1d6b547ba17004720e78e1f26fc6f4b45915ca4.jpg)

![e1a625a69a3a303308b389a5302066fd4587d6882d5181f0e21e8fbe51b8af42.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/images/e1a625a69a3a303308b389a5302066fd4587d6882d5181f0e21e8fbe51b8af42.jpg)

### Tables

![12609eb581dc643de540f6f96b61003553745ab88d760f154ae70b258093885a.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/tables/12609eb581dc643de540f6f96b61003553745ab88d760f154ae70b258093885a.jpg)

![178d16242bff1192c89be8e534fa88b982c0f50b3ac8b22236bb08f16da8519e.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/tables/178d16242bff1192c89be8e534fa88b982c0f50b3ac8b22236bb08f16da8519e.jpg)

![7901f315b7cf108f5b1ca19a1b711a1eb2ee5167303f1d4c8bfa8fd414774a1a.jpg](../iclr_results/348_Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics/tables/7901f315b7cf108f5b1ca19a1b711a1eb2ee5167303f1d4c8bfa8fd414774a1a.jpg)

## BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics


### Images

![08d16c04593ee6ab5f8a831920de50571de7b083b4646f3f08229ca00855ae21.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/08d16c04593ee6ab5f8a831920de50571de7b083b4646f3f08229ca00855ae21.jpg)

![12172fa96b6fb0bc9fd1c059703a53e09f18179420e88ca5f856aaea89bb2803.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/12172fa96b6fb0bc9fd1c059703a53e09f18179420e88ca5f856aaea89bb2803.jpg)

![214a7e103a8a5765b0a810db2c75000732d278d2fd19d025b751a30726c301b4.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/214a7e103a8a5765b0a810db2c75000732d278d2fd19d025b751a30726c301b4.jpg)

![215f05b7681abb6d0820a9a65ca590e6bbf34ef2c805f9bb70b031095d4fde82.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/215f05b7681abb6d0820a9a65ca590e6bbf34ef2c805f9bb70b031095d4fde82.jpg)

![29164c13c7abe1f4145562f4012bf336ee5f5b65ccf0e047a31063df303e274d.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/29164c13c7abe1f4145562f4012bf336ee5f5b65ccf0e047a31063df303e274d.jpg)

![4ad7a60a65cf05a4a62fad4ef082ef16aabf86f6c78b9a5210ee995e017dfda0.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/4ad7a60a65cf05a4a62fad4ef082ef16aabf86f6c78b9a5210ee995e017dfda0.jpg)

![502fc64736e079a1b44cf6827a8d80604b5f4e2d73dc994e64096736c5c40141.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/502fc64736e079a1b44cf6827a8d80604b5f4e2d73dc994e64096736c5c40141.jpg)

![57d214ef5902018bbd01023e3d26abff507190873d4ac239185439a89cb9904a.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/57d214ef5902018bbd01023e3d26abff507190873d4ac239185439a89cb9904a.jpg)

![6b4008a309630d0c58ede4211da6abe5d4c740824bd166b83bf139c0ee45ba6e.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/6b4008a309630d0c58ede4211da6abe5d4c740824bd166b83bf139c0ee45ba6e.jpg)

![6f9ed29b27cd826282f4d6d0f7f37ea5f664acedfbc9e0364a4099502028c805.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/6f9ed29b27cd826282f4d6d0f7f37ea5f664acedfbc9e0364a4099502028c805.jpg)

![83a4b2ac8a5a06ba7c5e0de15ba8717e717817b0551adc2052598248e5cc0e82.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/83a4b2ac8a5a06ba7c5e0de15ba8717e717817b0551adc2052598248e5cc0e82.jpg)

![873e2898e0b66eca37880d0f88dccbdb66bc0d21d793a6025a061a69a584b1cf.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/873e2898e0b66eca37880d0f88dccbdb66bc0d21d793a6025a061a69a584b1cf.jpg)

![a2774dcccff7fdf8d4620abb556fd41c35ff1de180a92ba5d3ec166d0d6c54d6.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/a2774dcccff7fdf8d4620abb556fd41c35ff1de180a92ba5d3ec166d0d6c54d6.jpg)

![d58a0f67f3ed261bd276d3e45db771712b06d6692d1ab1ec03282f3fe9562e05.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/d58a0f67f3ed261bd276d3e45db771712b06d6692d1ab1ec03282f3fe9562e05.jpg)

![fc0d7cdcbf68db1160ed9fc7a0383e87b826548ad05f863b82527916b65243a0.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/images/fc0d7cdcbf68db1160ed9fc7a0383e87b826548ad05f863b82527916b65243a0.jpg)

### Tables

![3099f93b5def71eb89c86ab166fd80ceb4cd165e3cdfa1327afdca0ec41e8969.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/3099f93b5def71eb89c86ab166fd80ceb4cd165e3cdfa1327afdca0ec41e8969.jpg)

![31c8818d220b5fa06131a274f1d92b337e1d0763980536dca77e80b9df706282.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/31c8818d220b5fa06131a274f1d92b337e1d0763980536dca77e80b9df706282.jpg)

![7ac977c865509ea6db2481cf69b6fb31c7c1f83c30ae81cf40ef6e7b8f33064e.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/7ac977c865509ea6db2481cf69b6fb31c7c1f83c30ae81cf40ef6e7b8f33064e.jpg)

![83ea26f5ed09f0637add397874d70fba9b4bb14f2778671e9379a92a16a46c52.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/83ea26f5ed09f0637add397874d70fba9b4bb14f2778671e9379a92a16a46c52.jpg)

![be0e2ebe883e1d49abe284d30ccf7f81ce6ceb3878847b8dd3505923667ffa51.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/be0e2ebe883e1d49abe284d30ccf7f81ce6ceb3878847b8dd3505923667ffa51.jpg)

![daf3894a2650eb89b37b881273bb024b295171efb2e6b9250cfb35339b5f24f4.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/daf3894a2650eb89b37b881273bb024b295171efb2e6b9250cfb35339b5f24f4.jpg)

![dd2f72b892d6c6c548ef195f28bec106d4bdd8d12a8ebf348ce74d36e461aef8.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/dd2f72b892d6c6c548ef195f28bec106d4bdd8d12a8ebf348ce74d36e461aef8.jpg)

![dd4db37b3f850e379c737d501428fb56abe87c873f6ae47a7316c6743f040dca.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/dd4db37b3f850e379c737d501428fb56abe87c873f6ae47a7316c6743f040dca.jpg)

![e6cd269ed5e0732fa1b34532f544b68af3cb2612093131cb14ba668bc29a16a0.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/e6cd269ed5e0732fa1b34532f544b68af3cb2612093131cb14ba668bc29a16a0.jpg)

![eba0219b7db8d97408f95a8bb9e15d2a1fcb841431a8c31873efc112eb899438.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/eba0219b7db8d97408f95a8bb9e15d2a1fcb841431a8c31873efc112eb899438.jpg)

![f65e341eb68c9502974fb14cb4b7129e5ce34d0b20bf8552085ba4c75bf04bb1.jpg](../iclr_results/349_BirdSet_ A Large-Scale Dataset for Audio Classification in Avian Bioacoustics/tables/f65e341eb68c9502974fb14cb4b7129e5ce34d0b20bf8552085ba4c75bf04bb1.jpg)

## Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling


### Images

![02f4c12c225b490131a89a2a2f65d43f787cd7771e97acaaacb4a94109cf5ea6.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/02f4c12c225b490131a89a2a2f65d43f787cd7771e97acaaacb4a94109cf5ea6.jpg)

![0b8f9af27cae28e73b63db4513c444420f58a21618f92efc1573ab8d77a2d182.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/0b8f9af27cae28e73b63db4513c444420f58a21618f92efc1573ab8d77a2d182.jpg)

![0e1d7c1766a9b47b0d7a88f3ede31aa4d862b730def2cd96fc5a57996a50e2f6.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/0e1d7c1766a9b47b0d7a88f3ede31aa4d862b730def2cd96fc5a57996a50e2f6.jpg)

![19c3810fa8d437523d93c4b7d29039b6de511514ea36627fe3c8a430b6b20f47.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/19c3810fa8d437523d93c4b7d29039b6de511514ea36627fe3c8a430b6b20f47.jpg)

![251a7a6c954af98db0ec64bd0ed8feeb739c7ea6f8f494674a9b716d9599c004.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/251a7a6c954af98db0ec64bd0ed8feeb739c7ea6f8f494674a9b716d9599c004.jpg)

![2d3f1c7f340338a75e78fb3a3f5e0f6c52676471d291afafb6503a091374aabb.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/2d3f1c7f340338a75e78fb3a3f5e0f6c52676471d291afafb6503a091374aabb.jpg)

![33659dbef4b4a58ca08927365e76a04accec895dd02c9faea8451cc8d25f5b9f.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/33659dbef4b4a58ca08927365e76a04accec895dd02c9faea8451cc8d25f5b9f.jpg)

![36c4d40ee79de6fab5260c1314626e8c431c40b4cf15f9f0c99913cd6875ae0e.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/36c4d40ee79de6fab5260c1314626e8c431c40b4cf15f9f0c99913cd6875ae0e.jpg)

![453bc8b205de95d6c4ce6c7db1ebcf9a64c7d6260f770330ed85912bbbf6786c.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/453bc8b205de95d6c4ce6c7db1ebcf9a64c7d6260f770330ed85912bbbf6786c.jpg)

![4bcc5dc64094d233623b768e32fec74291252c0e2667f2278f930c26ecb0ac30.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/4bcc5dc64094d233623b768e32fec74291252c0e2667f2278f930c26ecb0ac30.jpg)

![5f50dfcd8c22ae046cd76c89a7406fa6f47c77a84e021e707ba1089d97108f02.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/5f50dfcd8c22ae046cd76c89a7406fa6f47c77a84e021e707ba1089d97108f02.jpg)

![954a1786b64cb7785dcff845541c510f6ee67790ab45694351970363fc110674.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/954a1786b64cb7785dcff845541c510f6ee67790ab45694351970363fc110674.jpg)

![973fd75df76e0c9ec78989575df81395260d586a591a910c63c9c7efbe2abcee.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/973fd75df76e0c9ec78989575df81395260d586a591a910c63c9c7efbe2abcee.jpg)

![9c1d0b711d5d39cd9519427857c5d14735ff0cac0c5e88dd5bfcdf87cbfe1fb7.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/9c1d0b711d5d39cd9519427857c5d14735ff0cac0c5e88dd5bfcdf87cbfe1fb7.jpg)

![9d0867901571f873230d802b3e5c49557f167c1ea8e978aaa20a3650447fb556.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/9d0867901571f873230d802b3e5c49557f167c1ea8e978aaa20a3650447fb556.jpg)

![a3d16e01c01a07422b1bbcadc1defe99a7ab6909728184f59c09b4ea12971913.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/a3d16e01c01a07422b1bbcadc1defe99a7ab6909728184f59c09b4ea12971913.jpg)

![b2906e143feb2d817a2f05376d6506d1b0ae4f97540611aa9261b7849d094b4b.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/b2906e143feb2d817a2f05376d6506d1b0ae4f97540611aa9261b7849d094b4b.jpg)

![cac9f6a3cd73a0216c9839ea67ff8ec823449bd0a2dac60f8c9ed872b2d56aee.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/cac9f6a3cd73a0216c9839ea67ff8ec823449bd0a2dac60f8c9ed872b2d56aee.jpg)

![d476f65b7b58fa6dd23aee63979d2213f87e21f854cd3fa61de82b3fed18b2de.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/d476f65b7b58fa6dd23aee63979d2213f87e21f854cd3fa61de82b3fed18b2de.jpg)

![d9b684116d965f51face9baf33ef3aa71a4e8f3e934052e716836999f2ad1661.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/d9b684116d965f51face9baf33ef3aa71a4e8f3e934052e716836999f2ad1661.jpg)

![db25263056ce5e4a34efccd1b9e9f999d0ef1cf4f7bdb79927dcf7bd94a4e2d5.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/images/db25263056ce5e4a34efccd1b9e9f999d0ef1cf4f7bdb79927dcf7bd94a4e2d5.jpg)

### Tables

![09f00289763f3969c0892abdfc6aa493a6901609a502108e81081eddb8521c50.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/09f00289763f3969c0892abdfc6aa493a6901609a502108e81081eddb8521c50.jpg)

![0a84200638e09fbaa0e6f82a789cbcababd62e2c4e1edece94f4ecf5763ede29.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/0a84200638e09fbaa0e6f82a789cbcababd62e2c4e1edece94f4ecf5763ede29.jpg)

![12d6afa93ace75cf2790bb6ea968f2a70387cf6e90475ec2a85f8b152fcef552.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/12d6afa93ace75cf2790bb6ea968f2a70387cf6e90475ec2a85f8b152fcef552.jpg)

![1dafab8a6e19d3c491b7a26d4508915708622322bae70ccdda8d0eb8d48682d4.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/1dafab8a6e19d3c491b7a26d4508915708622322bae70ccdda8d0eb8d48682d4.jpg)

![29cf484d683987d13f2b3457b51f64165557fed111ccb9ccb3276d12f821cf9e.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/29cf484d683987d13f2b3457b51f64165557fed111ccb9ccb3276d12f821cf9e.jpg)

![2b8c355b09755304dc79c8f81266b0e4bfbc4c441f597e1a294b69fdae6bb5ae.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/2b8c355b09755304dc79c8f81266b0e4bfbc4c441f597e1a294b69fdae6bb5ae.jpg)

![37957a4a6f569a823d0be2e7d0fd52340491db0efbeb41269ee9d29a4533e1a7.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/37957a4a6f569a823d0be2e7d0fd52340491db0efbeb41269ee9d29a4533e1a7.jpg)

![3d704544701b0b3c4ac9f95b583e18abe0a17cd2bb9f66700775614e20c0cc2e.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/3d704544701b0b3c4ac9f95b583e18abe0a17cd2bb9f66700775614e20c0cc2e.jpg)

![4a653002cce61c4527c8bb827a69180310f4850e5a0d388440d4f8eb933458bc.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/4a653002cce61c4527c8bb827a69180310f4850e5a0d388440d4f8eb933458bc.jpg)

![5d1fc60e240e597eea5f7ec431622a6618efa97f25988a5cb495b43975617c1a.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/5d1fc60e240e597eea5f7ec431622a6618efa97f25988a5cb495b43975617c1a.jpg)

![6009c947dea2800349387f1b3a2f5ab42531110e9d7c44234998821e8e582d85.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/6009c947dea2800349387f1b3a2f5ab42531110e9d7c44234998821e8e582d85.jpg)

![63ff7cac73865bb8ef60eee5b1274e0ef521c46c1e56577636df36d8a00e261b.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/63ff7cac73865bb8ef60eee5b1274e0ef521c46c1e56577636df36d8a00e261b.jpg)

![65d1cca980ab8b620a224e526adf3057253948ff32dabf0ed700aede3e4bbefa.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/65d1cca980ab8b620a224e526adf3057253948ff32dabf0ed700aede3e4bbefa.jpg)

![692e79f8a1de258c212e0f4f191570c382c87f36332b4fbc4ba07e334feb865d.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/692e79f8a1de258c212e0f4f191570c382c87f36332b4fbc4ba07e334feb865d.jpg)

![786f0d635aa943c1001c3717f79a6735ee01aae5e94d425c6347d0d02177517b.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/786f0d635aa943c1001c3717f79a6735ee01aae5e94d425c6347d0d02177517b.jpg)

![7e24849c1958d1ea8ebb67ed02c91866cd728ffab54d062230eee3b260390b1d.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/7e24849c1958d1ea8ebb67ed02c91866cd728ffab54d062230eee3b260390b1d.jpg)

![b1f4c8a183f80c1448090ffc154b6bb1b2603a1addab4e9ad7b1a5c9e305fce2.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/b1f4c8a183f80c1448090ffc154b6bb1b2603a1addab4e9ad7b1a5c9e305fce2.jpg)

![b37709a321335613cc4f774befb23ad6e3e80549ff76f0172cb5fc0faef63e14.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/b37709a321335613cc4f774befb23ad6e3e80549ff76f0172cb5fc0faef63e14.jpg)

![cf60b7ba4b6bdec9194acef5da13ddfdb69a14856f7a53ca53967150762c7f51.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/cf60b7ba4b6bdec9194acef5da13ddfdb69a14856f7a53ca53967150762c7f51.jpg)

![e3d5fa27f6040d764771aa1bff702fd889d794bb1de463c6fa73570305f644d6.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/e3d5fa27f6040d764771aa1bff702fd889d794bb1de463c6fa73570305f644d6.jpg)

![f4f628496c01a83ceca179af854287adc0ae8a277d3cb8ae112547992ef2fa6b.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/f4f628496c01a83ceca179af854287adc0ae8a277d3cb8ae112547992ef2fa6b.jpg)

![f5e0474c2b358d2693f2fe9d7cdf3dd0c5af491f34dce7976b3a30bd6a5d48ad.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/f5e0474c2b358d2693f2fe9d7cdf3dd0c5af491f34dce7976b3a30bd6a5d48ad.jpg)

![fadd575e5705867b43181748e3eb70565922b31b48933316bc56276d90874b3c.jpg](../iclr_results/350_Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling/tables/fadd575e5705867b43181748e3eb70565922b31b48933316bc56276d90874b3c.jpg)

## Training-Free Activation Sparsity in Large Language Models


### Images

![082a7b50c9f6446e8381c13ba82516971f8e18015c1365b5f3c1012c634f6456.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/082a7b50c9f6446e8381c13ba82516971f8e18015c1365b5f3c1012c634f6456.jpg)

![12c2e01164208b8b80f06e70d165e9a5a18da844802faaa4b5193193aa206c69.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/12c2e01164208b8b80f06e70d165e9a5a18da844802faaa4b5193193aa206c69.jpg)

![1c936cc2f9e527e6e7daa494d9404cfb40ab2c81b7149a595a89b0ad020f25bf.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/1c936cc2f9e527e6e7daa494d9404cfb40ab2c81b7149a595a89b0ad020f25bf.jpg)

![1e286ba41541527675a04d1d63006637ede23faa0704e16d673347a0dc5ab8be.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/1e286ba41541527675a04d1d63006637ede23faa0704e16d673347a0dc5ab8be.jpg)

![62494631bc30f0c8e1b49305440cca79b7233a93724e28e007d217806dd0058b.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/62494631bc30f0c8e1b49305440cca79b7233a93724e28e007d217806dd0058b.jpg)

![bdcbb09ac58059a54ae9fbe943100de7b10f862179097f3b21ca477d232e17d3.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/bdcbb09ac58059a54ae9fbe943100de7b10f862179097f3b21ca477d232e17d3.jpg)

![d0b3be7d0aa4a116fbde8dd764e84c0ee76e1a23c8e8d2cc337e124cba97b886.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/d0b3be7d0aa4a116fbde8dd764e84c0ee76e1a23c8e8d2cc337e124cba97b886.jpg)

![d3e17cc042f891cfd580086cea241388d1ae2eda2bd9d7b171eceb831b70b65d.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/d3e17cc042f891cfd580086cea241388d1ae2eda2bd9d7b171eceb831b70b65d.jpg)

![eb8f480cef3f614306a8bfa23301e9e21a6bfbe33c6050466c18cdbf4650a183.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/eb8f480cef3f614306a8bfa23301e9e21a6bfbe33c6050466c18cdbf4650a183.jpg)

![ef2981b812997136698abc23337cf293aec104e7a00256a98def18c6b9b3df80.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/images/ef2981b812997136698abc23337cf293aec104e7a00256a98def18c6b9b3df80.jpg)

### Tables

![148afac9ae39203ee846849f77c3be9ed8c615231387949926e291edaa995eb2.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/148afac9ae39203ee846849f77c3be9ed8c615231387949926e291edaa995eb2.jpg)

![271f20a9ab29267c3b0c67307e3e5fba8b4b119fa9c9079ee739c5ae98990ed6.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/271f20a9ab29267c3b0c67307e3e5fba8b4b119fa9c9079ee739c5ae98990ed6.jpg)

![46aa4d809762db1f5077db0cb92fb6e339322f9923b01e34c2492dfc2e7d9ebf.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/46aa4d809762db1f5077db0cb92fb6e339322f9923b01e34c2492dfc2e7d9ebf.jpg)

![70c0516615b4188e3097bd41ad77f7c2fdd7da8605d3a9967dd696706cec654d.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/70c0516615b4188e3097bd41ad77f7c2fdd7da8605d3a9967dd696706cec654d.jpg)

![9e56324f2864f7f1aaae74da9882ad049757a0e471c95a0313ccb9c834fb96ae.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/9e56324f2864f7f1aaae74da9882ad049757a0e471c95a0313ccb9c834fb96ae.jpg)

![ab92cc77bdd93acc25f918d4fce7b20e9a27db0b50555be2e2cb7e1a527ced8d.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/ab92cc77bdd93acc25f918d4fce7b20e9a27db0b50555be2e2cb7e1a527ced8d.jpg)

![adba13556851d8b464500eb535555f4e0879f605f8034468982afcb415196c1a.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/adba13556851d8b464500eb535555f4e0879f605f8034468982afcb415196c1a.jpg)

![af5f80e8ca5ddaf38404011630f3066cb39cf814a54cff21c702fd128207be2c.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/af5f80e8ca5ddaf38404011630f3066cb39cf814a54cff21c702fd128207be2c.jpg)

![c41fe96d72293c39bcee120352a7e21c2a4350bcbf00e2964fb41d309f3664b7.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/c41fe96d72293c39bcee120352a7e21c2a4350bcbf00e2964fb41d309f3664b7.jpg)

![c482516d82149ae322ae9a692a9621175885012a119117dbd02872d35307334f.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/c482516d82149ae322ae9a692a9621175885012a119117dbd02872d35307334f.jpg)

![c9a5919a666ca24e9269c06a43c2ec42918462486faa3b5fc24e4540c4ce7e7a.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/c9a5919a666ca24e9269c06a43c2ec42918462486faa3b5fc24e4540c4ce7e7a.jpg)

![eb14576aeeb0a48e4ab55abadef8b3720eb01a4c1904e718880c71b1f4306e5f.jpg](../iclr_results/351_Training-Free Activation Sparsity in Large Language Models/tables/eb14576aeeb0a48e4ab55abadef8b3720eb01a4c1904e718880c71b1f4306e5f.jpg)

## How Feature Learning Can Improve Neural Scaling Laws


### Images

![08c7cd1e5602811b7221c1f76eb78b7ecfdcb3094c201e3f0abff19e58848d5c.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/08c7cd1e5602811b7221c1f76eb78b7ecfdcb3094c201e3f0abff19e58848d5c.jpg)

![1c93e550af0f5a2db3166e5aa661f93a700103e9c039de2af777dc604b7405e9.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/1c93e550af0f5a2db3166e5aa661f93a700103e9c039de2af777dc604b7405e9.jpg)

![34091cc815dc868f68566db22186ac9ab161355b62b429d5fb8c55649be417c5.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/34091cc815dc868f68566db22186ac9ab161355b62b429d5fb8c55649be417c5.jpg)

![66a7c8f5f5aeb03c26e37904814d7db347cca9646bf56826073d635fc480e757.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/66a7c8f5f5aeb03c26e37904814d7db347cca9646bf56826073d635fc480e757.jpg)

![7618983e9d0dd3ba2652ab0e9aad66a8736ea316d018fb14340fecc9cabb0d72.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/7618983e9d0dd3ba2652ab0e9aad66a8736ea316d018fb14340fecc9cabb0d72.jpg)

![785356c8749ca46ebe82e9ce69c9eecaeaedc280500e3d33452a20075cd630a1.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/785356c8749ca46ebe82e9ce69c9eecaeaedc280500e3d33452a20075cd630a1.jpg)

![8488960ff5e68955d6ecbd96574390d9119454899f783e8172123f1af5a45ce1.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/8488960ff5e68955d6ecbd96574390d9119454899f783e8172123f1af5a45ce1.jpg)

![8536c6aff895f0765947cfbead2ba4f817f86fb644a3a949d080845b68664d2c.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/8536c6aff895f0765947cfbead2ba4f817f86fb644a3a949d080845b68664d2c.jpg)

![9ea3e7a0b7feb3aa23bd6c5d124515c399ca6ea033498f02186bd459c2ef2b38.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/9ea3e7a0b7feb3aa23bd6c5d124515c399ca6ea033498f02186bd459c2ef2b38.jpg)

![a96e1060a96595a161cd8fd9f486004efa1d794bfc1e88943cf6166eca1d42cc.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/a96e1060a96595a161cd8fd9f486004efa1d794bfc1e88943cf6166eca1d42cc.jpg)

![bb7baf5cd1fb19598d39e85380a1c6844ceb675d37bd23480651ce4b8a293e80.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/bb7baf5cd1fb19598d39e85380a1c6844ceb675d37bd23480651ce4b8a293e80.jpg)

![bbca0294144ed521445aa816aea16192f980f57b4aa12410526e01f31b5e4c45.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/bbca0294144ed521445aa816aea16192f980f57b4aa12410526e01f31b5e4c45.jpg)

![da34454883c4af3e32eaa62e40043fbb5db2434b71871238bb0843a5cf96f963.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/images/da34454883c4af3e32eaa62e40043fbb5db2434b71871238bb0843a5cf96f963.jpg)

### Tables

![206fce2cab45749e40e9332affa491eb662f92276f3641c97d55695b806fd5da.jpg](../iclr_results/352_How Feature Learning Can Improve Neural Scaling Laws/tables/206fce2cab45749e40e9332affa491eb662f92276f3641c97d55695b806fd5da.jpg)

## Beyond Next Token Prediction: Patch-Level Training for Large Language Models


### Images

![00834a9797a9780dbdfa65e7f0bd50e998a5757897aa596b9319b669cfa5b9a5.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/00834a9797a9780dbdfa65e7f0bd50e998a5757897aa596b9319b669cfa5b9a5.jpg)

![21496cecba87826bb0e6c7cd0612c5df68aeeb40dbb44901492e15267b783aba.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/21496cecba87826bb0e6c7cd0612c5df68aeeb40dbb44901492e15267b783aba.jpg)

![233eb086de73a3394b5177b48bf1ba457255b522cea983541544d786e8e1a54c.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/233eb086de73a3394b5177b48bf1ba457255b522cea983541544d786e8e1a54c.jpg)

![5dd6a6a80923b4d436ae055980d2f32ae975abf90c5acbafd8de5c1f07e20a47.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/5dd6a6a80923b4d436ae055980d2f32ae975abf90c5acbafd8de5c1f07e20a47.jpg)

![64d2bdc64f11c8e47839f8a1da81eae94a702b6440e12a8fa82e37084cf9b43c.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/64d2bdc64f11c8e47839f8a1da81eae94a702b6440e12a8fa82e37084cf9b43c.jpg)

![7ac58149463c661a2c231cb7030414b27cc64fa491b2c006521cff94747767bf.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/7ac58149463c661a2c231cb7030414b27cc64fa491b2c006521cff94747767bf.jpg)

![85241a16a245912a65fb116ebf24699ea572d50e4189ca4d5a50feedd40a2efe.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/85241a16a245912a65fb116ebf24699ea572d50e4189ca4d5a50feedd40a2efe.jpg)

![b717e8993ca7e60bbc64e45c36ee315adeea2412cedadc8d0086fdea0d963631.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/b717e8993ca7e60bbc64e45c36ee315adeea2412cedadc8d0086fdea0d963631.jpg)

![ccf27b88b16e7e74880e296805e662dbe423ebb5a5b8bcb54b74772bfe6906b8.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/ccf27b88b16e7e74880e296805e662dbe423ebb5a5b8bcb54b74772bfe6906b8.jpg)

![f8907d902da93a35f0071328bf8bbf9cd9228009d63ce0bcf332fd48d0a25c8e.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/f8907d902da93a35f0071328bf8bbf9cd9228009d63ce0bcf332fd48d0a25c8e.jpg)

![f98e56379146cc7025810e4f3c343914022ce2b5c695bf8236d487db1192798b.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/images/f98e56379146cc7025810e4f3c343914022ce2b5c695bf8236d487db1192798b.jpg)

### Tables

![16ed604c3c30ef499e95fb0bb7cb72990c5dcfc18ddee6d3636a518a76b70137.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/tables/16ed604c3c30ef499e95fb0bb7cb72990c5dcfc18ddee6d3636a518a76b70137.jpg)

![222c348333b853e47edfec3784efb7c084e6ee4c4733533d0427e84eb9aa3873.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/tables/222c348333b853e47edfec3784efb7c084e6ee4c4733533d0427e84eb9aa3873.jpg)

![777ce8279e2b1be137bd91bfc694d6cb2180151e9fa3bdb7b7241bf6a287dc6b.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/tables/777ce8279e2b1be137bd91bfc694d6cb2180151e9fa3bdb7b7241bf6a287dc6b.jpg)

![8f6693e8b2205a6aa03801ff420b2dd2b8b47551310b7f47527acd175cac61bc.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/tables/8f6693e8b2205a6aa03801ff420b2dd2b8b47551310b7f47527acd175cac61bc.jpg)

![a29443fc60e134c52a7ccfae15f1272c0b3a4f90d4f58fa41a5ebfb38c7ba69d.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/tables/a29443fc60e134c52a7ccfae15f1272c0b3a4f90d4f58fa41a5ebfb38c7ba69d.jpg)

![d26f9c59f6ded83b6ccfad121035073c1419769123651309b1167d6cb527ae86.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/tables/d26f9c59f6ded83b6ccfad121035073c1419769123651309b1167d6cb527ae86.jpg)

![e4def89226ceb2a9fe049c1523b9d266340f41650d84a9168ae3d6d632811c4b.jpg](../iclr_results/353_Beyond Next Token Prediction_ Patch-Level Training for Large Language Models/tables/e4def89226ceb2a9fe049c1523b9d266340f41650d84a9168ae3d6d632811c4b.jpg)

## Exact Certification of (Graph) Neural Networks Against Label Poisoning


### Images

![090587aea3dfa0ca6f146b58293197289b93ba283e2859a7d5f54c4142badd89.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/090587aea3dfa0ca6f146b58293197289b93ba283e2859a7d5f54c4142badd89.jpg)

![16bed740b40cb4713ad6e2e2cecc205116a47d07f068aa3dd03909734e096c37.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/16bed740b40cb4713ad6e2e2cecc205116a47d07f068aa3dd03909734e096c37.jpg)

![1e85efb59840e5e3fb7d82a44102c936248bbc80e6fed1dae56f7e6145c676fe.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/1e85efb59840e5e3fb7d82a44102c936248bbc80e6fed1dae56f7e6145c676fe.jpg)

![29c4463db60f6a6f2006018b58c4c871f58b7a09cbcf676a2a90074951747a7b.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/29c4463db60f6a6f2006018b58c4c871f58b7a09cbcf676a2a90074951747a7b.jpg)

![3dc72aca9093f729aebf28ea9853f406b24356550e0a72396d690637bf151a72.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/3dc72aca9093f729aebf28ea9853f406b24356550e0a72396d690637bf151a72.jpg)

![4756f1e3d5373f776fd7db933fc881854eab3282b35e41a7614748624da41a97.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/4756f1e3d5373f776fd7db933fc881854eab3282b35e41a7614748624da41a97.jpg)

![595261bf54449cbe3548dbeb8d85f230860215f2dc36b6306188c026765ec772.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/595261bf54449cbe3548dbeb8d85f230860215f2dc36b6306188c026765ec772.jpg)

![5ca173d5bed87e80e6c163b3f0086202a2505bef76303a213a5c1fccbd249af4.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/5ca173d5bed87e80e6c163b3f0086202a2505bef76303a213a5c1fccbd249af4.jpg)

![686badaf6c737673959df6f3a8098e33ea05452cbc0dc4702b83fd2600010411.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/686badaf6c737673959df6f3a8098e33ea05452cbc0dc4702b83fd2600010411.jpg)

![6abd5a048b68c0def06c9ab1cec2c63ec3bfb8c5ac9d8c9bba0ba726cd0ddb6f.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/6abd5a048b68c0def06c9ab1cec2c63ec3bfb8c5ac9d8c9bba0ba726cd0ddb6f.jpg)

![8e180595d7dcf8d0f77d560de8f7c082eeb03d57160b063d51a178ca6bd8edaa.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/8e180595d7dcf8d0f77d560de8f7c082eeb03d57160b063d51a178ca6bd8edaa.jpg)

![a47434c68a9b071e5cca767a51d33f9dc88477dd0921864707dd99d9aa472a78.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/a47434c68a9b071e5cca767a51d33f9dc88477dd0921864707dd99d9aa472a78.jpg)

![ae4f823570e1453c24bb8565755d4e27eacf0b169705e70f2208efdf2ee6ed85.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/ae4f823570e1453c24bb8565755d4e27eacf0b169705e70f2208efdf2ee6ed85.jpg)

![c1d1c500b988dea022a8d2f0ee315aa3e9e11e45031f963c888bfc7be6873404.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/c1d1c500b988dea022a8d2f0ee315aa3e9e11e45031f963c888bfc7be6873404.jpg)

![c31fa926f36902172b555234b38f53aa5a12e32f40a9fa0b6d8f2c16e2068619.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/c31fa926f36902172b555234b38f53aa5a12e32f40a9fa0b6d8f2c16e2068619.jpg)

![eef97d9d5823fb4b8de8b05aaea06ee36ec164b606ba9d182b0ede15d301e7e3.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/images/eef97d9d5823fb4b8de8b05aaea06ee36ec164b606ba9d182b0ede15d301e7e3.jpg)

### Tables

![1c09f4e188cfe0d7ef1e4982253f699f958531065fda6e9984a6f3d45e43c954.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/1c09f4e188cfe0d7ef1e4982253f699f958531065fda6e9984a6f3d45e43c954.jpg)

![2e506005c0535aa6a743891d6ac02527f7b289d80b69d0aeb303890fa952dd37.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/2e506005c0535aa6a743891d6ac02527f7b289d80b69d0aeb303890fa952dd37.jpg)

![36fb800ec9b9f203697fcf87cc17ea5d5cf6205d5a79b0a32476f5a99c697dc7.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/36fb800ec9b9f203697fcf87cc17ea5d5cf6205d5a79b0a32476f5a99c697dc7.jpg)

![3a7287ad91d80abe7e1716f8465a7219dbd987638fd5bab1f1f5bb7ec7b73cb6.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/3a7287ad91d80abe7e1716f8465a7219dbd987638fd5bab1f1f5bb7ec7b73cb6.jpg)

![5d7bb19d83747953d3d668264aa2fe9b4d98999f49f3dd5e9e55293a352e8555.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/5d7bb19d83747953d3d668264aa2fe9b4d98999f49f3dd5e9e55293a352e8555.jpg)

![62ab03a6b2e9b2f3c79a56a8ad16f4d55762b1f986bd3d420655558532d79026.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/62ab03a6b2e9b2f3c79a56a8ad16f4d55762b1f986bd3d420655558532d79026.jpg)

![8f0652bafd68ca325a21d03ff3c93c05e7af9338a218d0e3e8f7b71d42c1a817.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/8f0652bafd68ca325a21d03ff3c93c05e7af9338a218d0e3e8f7b71d42c1a817.jpg)

![9c2f4aac2b3eb4f02436c4e820c2fcbd7d9bd805e469e2091f20acbb2e6a34ae.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/9c2f4aac2b3eb4f02436c4e820c2fcbd7d9bd805e469e2091f20acbb2e6a34ae.jpg)

![a76c6d89fb2671169a91b3a878b7a2f476849647d2ee43a2b50102978c8612ca.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/a76c6d89fb2671169a91b3a878b7a2f476849647d2ee43a2b50102978c8612ca.jpg)

![b49e38547c1510ff1f2621baca3a13a0f3cf1475f850b44e1a517a69d8eb1dff.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/b49e38547c1510ff1f2621baca3a13a0f3cf1475f850b44e1a517a69d8eb1dff.jpg)

![c74e939f257b1bbe9c2e0e2c15c236d8800b773c364e1f107df6dccb1d8504e3.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/c74e939f257b1bbe9c2e0e2c15c236d8800b773c364e1f107df6dccb1d8504e3.jpg)

![cfbbc0d4fdec9eb9b37411937958037e24414fbbb218bcda99ad42d0523b21c7.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/cfbbc0d4fdec9eb9b37411937958037e24414fbbb218bcda99ad42d0523b21c7.jpg)

![fd6aa25a82ead349fcfdf6c36554eba549f46b7531a2626f11d8bb0811355ab7.jpg](../iclr_results/354_Exact Certification of (Graph) Neural Networks Against Label Poisoning/tables/fd6aa25a82ead349fcfdf6c36554eba549f46b7531a2626f11d8bb0811355ab7.jpg)

## Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees


### Images

![01205ecab58e2dc606c9473558ca239f9ac81fdb9cd273e0158581b807c8c221.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/01205ecab58e2dc606c9473558ca239f9ac81fdb9cd273e0158581b807c8c221.jpg)

![2aec1f68cd5f95110a690a061faa0f792779d06e7fa34bd3fca649a5c93c9abf.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/2aec1f68cd5f95110a690a061faa0f792779d06e7fa34bd3fca649a5c93c9abf.jpg)

![2b6173870d31f0953564f6021f15363ad510373146d2d339f2df6abc87d84e0b.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/2b6173870d31f0953564f6021f15363ad510373146d2d339f2df6abc87d84e0b.jpg)

![325809c595330dced80bc1d1d8581cc4e1bd97572d8ea49634f00c5c0910f2aa.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/325809c595330dced80bc1d1d8581cc4e1bd97572d8ea49634f00c5c0910f2aa.jpg)

![3f4aa4484de0feafeabfaf0d437ebeef5e1177d55b806466dce45d0d0bbbf868.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/3f4aa4484de0feafeabfaf0d437ebeef5e1177d55b806466dce45d0d0bbbf868.jpg)

![589738bcec569e21ac1c5a96c7b4e4e079f579807e96420131072b09512ea562.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/589738bcec569e21ac1c5a96c7b4e4e079f579807e96420131072b09512ea562.jpg)

![a0af53f2923dfb5f78f161a85f831b910a27d4f8402df346ecd350b1d2e6208c.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/a0af53f2923dfb5f78f161a85f831b910a27d4f8402df346ecd350b1d2e6208c.jpg)

![b8f56600009ce71029c81d0bea48c77d5b4442d081559356caeeacb9cb6ca9d1.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/b8f56600009ce71029c81d0bea48c77d5b4442d081559356caeeacb9cb6ca9d1.jpg)

![ba3ecfb8272ea612a38c5b7d60ab2ccdfefd3db51b3b379012b4e3fb4d28f396.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/ba3ecfb8272ea612a38c5b7d60ab2ccdfefd3db51b3b379012b4e3fb4d28f396.jpg)

![caa7dcbee8f457c208335a43c34ca0b6156b1fbdda94bf59949b3fac86144a67.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/caa7dcbee8f457c208335a43c34ca0b6156b1fbdda94bf59949b3fac86144a67.jpg)

![dcc395caa13390c3aaa8244d2f6a5d851d8f2d9a98b43453cfa22aaa6b4f538d.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/images/dcc395caa13390c3aaa8244d2f6a5d851d8f2d9a98b43453cfa22aaa6b4f538d.jpg)

### Tables

![11767afd4d0ddc0f3ff6cd72431d5d618f76738a9dc95d1e67c41db112319143.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/tables/11767afd4d0ddc0f3ff6cd72431d5d618f76738a9dc95d1e67c41db112319143.jpg)

![20e9ab8d7ab60667c36ef99d1892c50ef49629410fd16df31cbd361397f64829.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/tables/20e9ab8d7ab60667c36ef99d1892c50ef49629410fd16df31cbd361397f64829.jpg)

![409e6522084755754aaf109f45fff9697cd4ae20b2f4f9f8496f4c24b6b07b13.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/tables/409e6522084755754aaf109f45fff9697cd4ae20b2f4f9f8496f4c24b6b07b13.jpg)

![7cea7bd1eaea0696ea17699752cec41419e57570f38acc9ec1028d62a08937f6.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/tables/7cea7bd1eaea0696ea17699752cec41419e57570f38acc9ec1028d62a08937f6.jpg)

![c44abfdd3cbdd5754afefecdb18f38fd00d4793de04ad68d0007537aed10b5bc.jpg](../iclr_results/355_Decentralized Sporadic Federated Learning_ A Unified Algorithmic Framework with Convergence Guarante/tables/c44abfdd3cbdd5754afefecdb18f38fd00d4793de04ad68d0007537aed10b5bc.jpg)

## Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification


### Images

![03e8f81cf13803a4c567f2d255185e4a2135f97c7327ac69796a2f02b6ca6392.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/03e8f81cf13803a4c567f2d255185e4a2135f97c7327ac69796a2f02b6ca6392.jpg)

![04194c420aa598fd13ca8940c6106bc81049d90cceca63dd7d44fc9ef976e660.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/04194c420aa598fd13ca8940c6106bc81049d90cceca63dd7d44fc9ef976e660.jpg)

![1ae3e6ed3d048472c8db0b08e47a3e1c4af520fa58dbb00f1de286874f94b67d.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/1ae3e6ed3d048472c8db0b08e47a3e1c4af520fa58dbb00f1de286874f94b67d.jpg)

![3e89dea10554811b04a04c947d3288ccf454ce1714d2622be7799e2c183cf3bb.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/3e89dea10554811b04a04c947d3288ccf454ce1714d2622be7799e2c183cf3bb.jpg)

![4c78449ba0b175c8deaebec5ecdd60fe014cbdd6d9f520d035da822cdfe6efe3.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/4c78449ba0b175c8deaebec5ecdd60fe014cbdd6d9f520d035da822cdfe6efe3.jpg)

![532a820db2be53c37437cac7c6e5ca88a235b594dafcdf48b3699a3d738fd2fe.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/532a820db2be53c37437cac7c6e5ca88a235b594dafcdf48b3699a3d738fd2fe.jpg)

![54191dd44e8221ccac2725aac461bc3a8f7f7e5b8efbfc77e9a97485fda156d7.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/54191dd44e8221ccac2725aac461bc3a8f7f7e5b8efbfc77e9a97485fda156d7.jpg)

![723857dd90320ad5b676b6dab1365dfe01caf52a1170f0ca5bb8e983e96082f6.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/723857dd90320ad5b676b6dab1365dfe01caf52a1170f0ca5bb8e983e96082f6.jpg)

![8636e2b4ef920996b1b4f256a5c4e8626df298ec7878c72c2161a5846ac81d4d.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/8636e2b4ef920996b1b4f256a5c4e8626df298ec7878c72c2161a5846ac81d4d.jpg)

![c55f033c8f6d6e780d728628f725eea452908e011ad1c33bc9781a3688326f8f.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/c55f033c8f6d6e780d728628f725eea452908e011ad1c33bc9781a3688326f8f.jpg)

![cff9a3f6d8cc231ad070736e0f61b68b708c5c89b8126af4ee909e673fcc88ad.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/cff9a3f6d8cc231ad070736e0f61b68b708c5c89b8126af4ee909e673fcc88ad.jpg)

![d5de7eee8c9a93401601d07f5d9c173efb0849ba4b6b55634070c4e7f6763637.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/d5de7eee8c9a93401601d07f5d9c173efb0849ba4b6b55634070c4e7f6763637.jpg)

![dc98ad637d05ef358e814c019f58e34d8e865897025ac8f22566dd47f79ccd4d.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/dc98ad637d05ef358e814c019f58e34d8e865897025ac8f22566dd47f79ccd4d.jpg)

![e2963138ab9bf1a9bb81a156534282294816eb2f81daf5ceb03b0378c1fa8d6c.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/e2963138ab9bf1a9bb81a156534282294816eb2f81daf5ceb03b0378c1fa8d6c.jpg)

![e9700dabacae237d3138523442aed7e148f10c601b15a82fcc6649eac3c01267.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/e9700dabacae237d3138523442aed7e148f10c601b15a82fcc6649eac3c01267.jpg)

![ea84d91255480b35e28aa5a270e5b203163d785a0570d4275e147a74bc325191.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/ea84d91255480b35e28aa5a270e5b203163d785a0570d4275e147a74bc325191.jpg)

![f70c9440c0e3240b79133b2b56e023185af771fc3ccdc37594cba60d15535b90.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/images/f70c9440c0e3240b79133b2b56e023185af771fc3ccdc37594cba60d15535b90.jpg)

### Tables

![0dd25121e95387dd8796508f2cc8f75d8fc1752fa287135757f1b6f8e0b113a2.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/0dd25121e95387dd8796508f2cc8f75d8fc1752fa287135757f1b6f8e0b113a2.jpg)

![0f32c29587c7f57703454e8db11d11d9af9e3712bdffba164e59fb320d541bea.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/0f32c29587c7f57703454e8db11d11d9af9e3712bdffba164e59fb320d541bea.jpg)

![394bc8eafd9f4f5cf2371afb0c0b87c5b44b0b7e05c3cea775b08949899de998.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/394bc8eafd9f4f5cf2371afb0c0b87c5b44b0b7e05c3cea775b08949899de998.jpg)

![39a5ebc76e6a11a8c4289dd326b848b0600137d95c9319f98f69b1c86d8c6ce0.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/39a5ebc76e6a11a8c4289dd326b848b0600137d95c9319f98f69b1c86d8c6ce0.jpg)

![41102a927cad3bb71297eb4fe303a5a2c08541c4beec5879f700901d78a2ee56.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/41102a927cad3bb71297eb4fe303a5a2c08541c4beec5879f700901d78a2ee56.jpg)

![489d821ca8d57dd3e4465041ab8345c2d7b2ab8499557e7697a607894150c433.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/489d821ca8d57dd3e4465041ab8345c2d7b2ab8499557e7697a607894150c433.jpg)

![683624dfd8334120da1f7fc0481a8a4d7d7b33ba52ffcbb1f800f7741064f255.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/683624dfd8334120da1f7fc0481a8a4d7d7b33ba52ffcbb1f800f7741064f255.jpg)

![71cefc6d9380ac4fc97dc74ad94172f636d0cfd9bff1206a06c738412022b3d8.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/71cefc6d9380ac4fc97dc74ad94172f636d0cfd9bff1206a06c738412022b3d8.jpg)

![7ac3df77a11945eac5d6ad68bb7eeaa821e856619400e6e0661854fa6e4bdd52.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/7ac3df77a11945eac5d6ad68bb7eeaa821e856619400e6e0661854fa6e4bdd52.jpg)

![80509de56a5b0cd9a5091204a4cc62f9dc59e64e19dc490a306ec2a199812311.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/80509de56a5b0cd9a5091204a4cc62f9dc59e64e19dc490a306ec2a199812311.jpg)

![93b26f5389081ed1844894d73535da0c4f2fdc61299c4bef4212b0609f7e9289.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/93b26f5389081ed1844894d73535da0c4f2fdc61299c4bef4212b0609f7e9289.jpg)

![a739e4fefcd7b75f7b0384ea5eb3cbb4f9e9cb5e1288c8c51826ad3c4cd36f22.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/a739e4fefcd7b75f7b0384ea5eb3cbb4f9e9cb5e1288c8c51826ad3c4cd36f22.jpg)

![ac2bc8022f84bb54630ca3fbee61e0779b03ca986a2ba143caca38a1c8014004.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/ac2bc8022f84bb54630ca3fbee61e0779b03ca986a2ba143caca38a1c8014004.jpg)

![b6cb3633ba13e8a889186b32cc5210987bd03cee0769031c495237cf155cb9f2.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/b6cb3633ba13e8a889186b32cc5210987bd03cee0769031c495237cf155cb9f2.jpg)

![d5db2bfb80c2ad7f93f56467c24372d684da4fa2f453988b088661a4b5a98dac.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/d5db2bfb80c2ad7f93f56467c24372d684da4fa2f453988b088661a4b5a98dac.jpg)

![dfe309b4396a4a6ffd4db6297cd59c2f135480339a599c63023cb16fda7fceb5.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/dfe309b4396a4a6ffd4db6297cd59c2f135480339a599c63023cb16fda7fceb5.jpg)

![e39fdcb0485f9eb6764ae18b2caeb6103407c7252ae349b2bcb53fb3195f841b.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/e39fdcb0485f9eb6764ae18b2caeb6103407c7252ae349b2bcb53fb3195f841b.jpg)

![e746aa6e1346a7e91fdbae8c48a73b546c2291b05a4202d5abf6ce9e5c638f1a.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/e746aa6e1346a7e91fdbae8c48a73b546c2291b05a4202d5abf6ce9e5c638f1a.jpg)

![f43604a65ae9a63e05d0df2f155075ea85c4035c1b6153bcf8784e17649f33d1.jpg](../iclr_results/356_Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification/tables/f43604a65ae9a63e05d0df2f155075ea85c4035c1b6153bcf8784e17649f33d1.jpg)

## X-ALMA: Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale


### Images

![41ecc35a2f76864a9d67e3330e8a9d7c92d81f302be7d21fc1e5328bc9ce74fa.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/images/41ecc35a2f76864a9d67e3330e8a9d7c92d81f302be7d21fc1e5328bc9ce74fa.jpg)

![77ec2f490ecaf17c88d92b0e6a2efa020e2fc29b4a770ee8ec7f4d0efc8778c2.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/images/77ec2f490ecaf17c88d92b0e6a2efa020e2fc29b4a770ee8ec7f4d0efc8778c2.jpg)

![8697ce4eb099d581577806ec9bb74387921d02c1dd3b511d276d1b09e2ffb92e.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/images/8697ce4eb099d581577806ec9bb74387921d02c1dd3b511d276d1b09e2ffb92e.jpg)

![8e797423be63b43aaefce1983b7c5c76e8c0731a419b645dfe6ef06a0d2e3053.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/images/8e797423be63b43aaefce1983b7c5c76e8c0731a419b645dfe6ef06a0d2e3053.jpg)

![c31b8791f26e406c4b111e926419d3728da1b4cb6646152030e3013c156d56a4.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/images/c31b8791f26e406c4b111e926419d3728da1b4cb6646152030e3013c156d56a4.jpg)

### Tables

![0452c709125ef204b02daa823630276f23353a1362d67e2159860e865a2ddf14.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/0452c709125ef204b02daa823630276f23353a1362d67e2159860e865a2ddf14.jpg)

![0bea0624a492004cc2aa71cbf82b75a1b1cf0d575ba2976ce1560aec8eb33d7c.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/0bea0624a492004cc2aa71cbf82b75a1b1cf0d575ba2976ce1560aec8eb33d7c.jpg)

![24c60fce47f386c865600755f53c4981d96c8072ccee657ee067b680569c7edf.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/24c60fce47f386c865600755f53c4981d96c8072ccee657ee067b680569c7edf.jpg)

![3f4086061f34a0aeece59d460d96661b6d3510bd7ea3050f118b4869d4abe329.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/3f4086061f34a0aeece59d460d96661b6d3510bd7ea3050f118b4869d4abe329.jpg)

![48f78b76acc274a7bcd8436f3cf3eebb980b55e09dff86d448909b45e700d870.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/48f78b76acc274a7bcd8436f3cf3eebb980b55e09dff86d448909b45e700d870.jpg)

![4cd96ef6e1b4495a7fe6021cc4acc5b0e145efdc0ce1edfbbda92c600844f3e3.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/4cd96ef6e1b4495a7fe6021cc4acc5b0e145efdc0ce1edfbbda92c600844f3e3.jpg)

![5505eec0d6e93fc459bb8504d948d12196c9113c2e44de616d3160e0506ea7c7.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/5505eec0d6e93fc459bb8504d948d12196c9113c2e44de616d3160e0506ea7c7.jpg)

![77f5c568578c8c7c708b21a47362a6ea2d29f7d6b3cce96c33fe7182b7bc30c2.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/77f5c568578c8c7c708b21a47362a6ea2d29f7d6b3cce96c33fe7182b7bc30c2.jpg)

![87529bf3adeb273e7d3361f57a8870d3336ef2e30d17b86dd91d1fa285562e36.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/87529bf3adeb273e7d3361f57a8870d3336ef2e30d17b86dd91d1fa285562e36.jpg)

![947f281971cb9018e52c00ed9294008813ba9f4abb95ceee2118edacc128b312.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/947f281971cb9018e52c00ed9294008813ba9f4abb95ceee2118edacc128b312.jpg)

![97eb4b133569988d628dd3f15d1d016eb78b8b386146518cbaa9477aff1cca64.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/97eb4b133569988d628dd3f15d1d016eb78b8b386146518cbaa9477aff1cca64.jpg)

![a8896fd6a7a5a8aac4beaa30e43d7e7e3563699872b8442f35575e589e95abba.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/a8896fd6a7a5a8aac4beaa30e43d7e7e3563699872b8442f35575e589e95abba.jpg)

![d2cf9c0935459a6b5d45ee10f70ffd358b78c0e5e7954ea5d80f538873ece328.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/d2cf9c0935459a6b5d45ee10f70ffd358b78c0e5e7954ea5d80f538873ece328.jpg)

![d542361ef637a6f6738e5df822a62e8e89a3c4f04a63bb1a250c49de2109dea1.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/d542361ef637a6f6738e5df822a62e8e89a3c4f04a63bb1a250c49de2109dea1.jpg)

![d892087d6a33f9285f3e71eca5d9f2b0bd31fbb1947e1cc6d2b1f8e35ed0c6fc.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/d892087d6a33f9285f3e71eca5d9f2b0bd31fbb1947e1cc6d2b1f8e35ed0c6fc.jpg)

![e108a2ab1b87978644f3badc193f3d8e468912930c0a9a5ed29d2ee0d4cc1319.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/e108a2ab1b87978644f3badc193f3d8e468912930c0a9a5ed29d2ee0d4cc1319.jpg)

![e425e5278f23d575b1343d84ffcb8f9ef218cae1c0874b4e6d051f84cccbf6ca.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/e425e5278f23d575b1343d84ffcb8f9ef218cae1c0874b4e6d051f84cccbf6ca.jpg)

![f728c66975fb062ee0c760315e6565eda59ea02e2caabf1b19a71b4ae3654069.jpg](../iclr_results/357_X-ALMA_ Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale/tables/f728c66975fb062ee0c760315e6565eda59ea02e2caabf1b19a71b4ae3654069.jpg)

## TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models


### Images

![069573f4c41b4ccafffcb8631bfee6c218b679a0c79a8028aa6ab8919b3259d7.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/images/069573f4c41b4ccafffcb8631bfee6c218b679a0c79a8028aa6ab8919b3259d7.jpg)

![43043dc5811fa0291f6714d58f841bba18e10d53d8f83152aab1f6216fbbf9de.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/images/43043dc5811fa0291f6714d58f841bba18e10d53d8f83152aab1f6216fbbf9de.jpg)

![4ebe6ea419b9a9e6539f85b2be06f31c94f4d6852c97a1af91a9c2e76a3d3997.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/images/4ebe6ea419b9a9e6539f85b2be06f31c94f4d6852c97a1af91a9c2e76a3d3997.jpg)

### Tables

![0624fb9cfead17e467a280530ce22f9a30fc95f96669ff6561eb92a82bd17e91.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/0624fb9cfead17e467a280530ce22f9a30fc95f96669ff6561eb92a82bd17e91.jpg)

![1c0cafbb8095aa06a24f796f1d03e301329e5c774c2f0b42482cfc23aa36a55e.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/1c0cafbb8095aa06a24f796f1d03e301329e5c774c2f0b42482cfc23aa36a55e.jpg)

![2254c0467d3d11efc5b12006d375b326d112b242fa63dcadaab07c2e96651ebb.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/2254c0467d3d11efc5b12006d375b326d112b242fa63dcadaab07c2e96651ebb.jpg)

![5fdf30aa65a99aa221075c2b000c1ce02823c6059639276a192ea1daf4fb6754.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/5fdf30aa65a99aa221075c2b000c1ce02823c6059639276a192ea1daf4fb6754.jpg)

![6bd813ea25933aea41b10a6a5b26bb72592d9ff8cd37182fff4edd7269759be4.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/6bd813ea25933aea41b10a6a5b26bb72592d9ff8cd37182fff4edd7269759be4.jpg)

![7e341b91f5061ecfc676193830c233099fc0e750d454c85bf8e1dcf7e2d197fd.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/7e341b91f5061ecfc676193830c233099fc0e750d454c85bf8e1dcf7e2d197fd.jpg)

![99817f6d3b173e89659aabbfcbef38d348670214374aa77000bc1c0a7d0aeddd.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/99817f6d3b173e89659aabbfcbef38d348670214374aa77000bc1c0a7d0aeddd.jpg)

![9ccc2836edcd428a75bc854c616cfb37ef5fbf1466d30d4cf7b590c740ad2c31.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/9ccc2836edcd428a75bc854c616cfb37ef5fbf1466d30d4cf7b590c740ad2c31.jpg)

![b923d25223560fbd5754eb08fbe18dfbf1a577168a15fd2164459e8b82013092.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/b923d25223560fbd5754eb08fbe18dfbf1a577168a15fd2164459e8b82013092.jpg)

![d5d697d2f91a2b23a526af756fd96a51011dd11c3f57dc5a923b818ae1028fa7.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/d5d697d2f91a2b23a526af756fd96a51011dd11c3f57dc5a923b818ae1028fa7.jpg)

![e2ad12048ce7a1d0f4a1b7bbe809f7ee99e8ad4d7f27a68c7495e62164f1c543.jpg](../iclr_results/358_TAID_ Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Mod/tables/e2ad12048ce7a1d0f4a1b7bbe809f7ee99e8ad4d7f27a68c7495e62164f1c543.jpg)

## Wasserstein Distances, Neuronal Entanglement, and Sparsity


### Images

![0554b23f6ba66c8161d633d9cb926ef624df1878d5f8a6e513b1ad4542e8a22c.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/0554b23f6ba66c8161d633d9cb926ef624df1878d5f8a6e513b1ad4542e8a22c.jpg)

![10052e19c4f9c067d8ec1d2f929bdaadaa5ddf7e700ff2fba241d1b44c955664.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/10052e19c4f9c067d8ec1d2f929bdaadaa5ddf7e700ff2fba241d1b44c955664.jpg)

![1698443cdfe0f68e82bca29819bbd2a329cf62b11c9fd24e545f9bd37e675da7.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/1698443cdfe0f68e82bca29819bbd2a329cf62b11c9fd24e545f9bd37e675da7.jpg)

![1b20512534741bcfce3bc768df83d6306ed30821df6c489b77ddf2eb91982070.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/1b20512534741bcfce3bc768df83d6306ed30821df6c489b77ddf2eb91982070.jpg)

![1b707d44d1845e3a48449300d3863945283f19ed30e55aab22615a7faf0a489f.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/1b707d44d1845e3a48449300d3863945283f19ed30e55aab22615a7faf0a489f.jpg)

![20544ea152377eac969395f4006c2482c2cb647a413ff76bd19a77b19fcf9e0c.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/20544ea152377eac969395f4006c2482c2cb647a413ff76bd19a77b19fcf9e0c.jpg)

![2f30d1ad5d036ea30d5da90fe86df1f3220f166a8e4679f83254c9c2ac17acca.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/2f30d1ad5d036ea30d5da90fe86df1f3220f166a8e4679f83254c9c2ac17acca.jpg)

![2ffde62849ec79e2cc3db1b4bfc8cad218108156d827dc09b1831ce09a7186dc.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/2ffde62849ec79e2cc3db1b4bfc8cad218108156d827dc09b1831ce09a7186dc.jpg)

![316e8d7cd75e6ba33effcec942e7cc889b08e66aa6469b330a674494309d5de9.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/316e8d7cd75e6ba33effcec942e7cc889b08e66aa6469b330a674494309d5de9.jpg)

![3597fd7e3b7ad6cb01da45dbda2e061581906d32724a913f974e9886fe7c0bc1.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/3597fd7e3b7ad6cb01da45dbda2e061581906d32724a913f974e9886fe7c0bc1.jpg)

![421a3bf7b20281a1357e4f2c7bd4b71856a14442c34554915165379799b806b5.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/421a3bf7b20281a1357e4f2c7bd4b71856a14442c34554915165379799b806b5.jpg)

![45c537fc48b22555a732155c076c84ce058abfd79fbde01e68d1756e8d4574df.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/45c537fc48b22555a732155c076c84ce058abfd79fbde01e68d1756e8d4574df.jpg)

![47c2b1382311e2e49b060d8cbf4e0730f6a07d9efc47cd644d1b5e2f2c5320dd.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/47c2b1382311e2e49b060d8cbf4e0730f6a07d9efc47cd644d1b5e2f2c5320dd.jpg)

![5966a1f34b4eb0afd333d64568aba54bee01d7a6f70dec9a0510c397b83817cf.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/5966a1f34b4eb0afd333d64568aba54bee01d7a6f70dec9a0510c397b83817cf.jpg)

![60a8fdf4b5d995d2496f7834c1a87e910df981803aa685e6749d092fa4fbae4f.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/60a8fdf4b5d995d2496f7834c1a87e910df981803aa685e6749d092fa4fbae4f.jpg)

![66569abebf8d8d826c196de3de028270edb50959a60b2ce48445c643fafaa7e9.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/66569abebf8d8d826c196de3de028270edb50959a60b2ce48445c643fafaa7e9.jpg)

![933ad2bb7083884f114a2e936cbe29bb385d5220dbf53a8e5edf87270f18b2aa.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/933ad2bb7083884f114a2e936cbe29bb385d5220dbf53a8e5edf87270f18b2aa.jpg)

![a5d53d9b6e4eec29a6b410a1f362474dfa09f62d2c0720ec85f9d52038a2f30a.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/a5d53d9b6e4eec29a6b410a1f362474dfa09f62d2c0720ec85f9d52038a2f30a.jpg)

![a644e71918b8a03a0a28f0124950dbf8d3ce98d3e7818895dabcf157920d7a56.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/a644e71918b8a03a0a28f0124950dbf8d3ce98d3e7818895dabcf157920d7a56.jpg)

![b054b1c23e2cc59184452b0910c274f984bbc053ae9884671166f1abe7fb2da4.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/b054b1c23e2cc59184452b0910c274f984bbc053ae9884671166f1abe7fb2da4.jpg)

![b280f40ddaab05c3b76bcda7dbcf0b7ab1d4903e0044c7773bd7c8d84caa63aa.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/b280f40ddaab05c3b76bcda7dbcf0b7ab1d4903e0044c7773bd7c8d84caa63aa.jpg)

![b56a888139f4b63dc6e76acacb8ecca9fec1c82cf0d5694f9dbd937df9f603b2.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/b56a888139f4b63dc6e76acacb8ecca9fec1c82cf0d5694f9dbd937df9f603b2.jpg)

![bf719240222f8b17de4b2a1c0e21d514851a6e371980d3a349631826c6f62783.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/bf719240222f8b17de4b2a1c0e21d514851a6e371980d3a349631826c6f62783.jpg)

![ef474b5a34b64ce7a61372db9e18ad743aa6b86694543a12c2e958ffb2bf0d1c.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/images/ef474b5a34b64ce7a61372db9e18ad743aa6b86694543a12c2e958ffb2bf0d1c.jpg)

### Tables

![7d6f252742ac6a2780cb31ede3db3d50438fe7cca38d251012723a1799c47628.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/tables/7d6f252742ac6a2780cb31ede3db3d50438fe7cca38d251012723a1799c47628.jpg)

![7fa64290f52f8bd541e6089978ce5a312deb8854f7da48414af3ad0d2fb4a984.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/tables/7fa64290f52f8bd541e6089978ce5a312deb8854f7da48414af3ad0d2fb4a984.jpg)

![a8444bd98b60283325845c62013e255a55bfec2d4cd8a2006ece0e5a57713bb3.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/tables/a8444bd98b60283325845c62013e255a55bfec2d4cd8a2006ece0e5a57713bb3.jpg)

![abe818a94f4f42332b2ab7d1d179e1783b7dd773d8b9233dd73b1b24b865a93b.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/tables/abe818a94f4f42332b2ab7d1d179e1783b7dd773d8b9233dd73b1b24b865a93b.jpg)

![bf0c9b002a5a97e886d8ecdb86ae5a1a9cc895a835bf7d264a3c1552f4f5f76d.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/tables/bf0c9b002a5a97e886d8ecdb86ae5a1a9cc895a835bf7d264a3c1552f4f5f76d.jpg)

![dce7aba69bcb1d6d3fb5058d71db13f187e58bbe990eb7a13985967fb6e24313.jpg](../iclr_results/359_Wasserstein Distances, Neuronal Entanglement, and Sparsity/tables/dce7aba69bcb1d6d3fb5058d71db13f187e58bbe990eb7a13985967fb6e24313.jpg)

## Geometric Inductive Biases of Deep Networks: The Role of Data and Architecture


### Images

![25f8368366ad3dbbaf1debbd0d2b76c0d3f5e0d4911c7b3e1fecd4a561489fee.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/25f8368366ad3dbbaf1debbd0d2b76c0d3f5e0d4911c7b3e1fecd4a561489fee.jpg)

![31cc55948debde5868877370985a1eeafd74d39508189cac0400945d1a907cc6.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/31cc55948debde5868877370985a1eeafd74d39508189cac0400945d1a907cc6.jpg)

![3bfde201f2b64e86cd9b84ae5a047c9aaf58a3212364e2e6fd055581746d8c90.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/3bfde201f2b64e86cd9b84ae5a047c9aaf58a3212364e2e6fd055581746d8c90.jpg)

![44898bd6d44cc0991b2a18541956a20e3085753b3e2609d9824fa905f424af58.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/44898bd6d44cc0991b2a18541956a20e3085753b3e2609d9824fa905f424af58.jpg)

![4b938860632f41cbdc3c956b83f67c9450daa9ebaf847776d8c5a6cef460d425.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/4b938860632f41cbdc3c956b83f67c9450daa9ebaf847776d8c5a6cef460d425.jpg)

![4f1c10bf614b076b2069b3ede062fc957ec1ae667d3b71a5fd8b7341be8dca8a.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/4f1c10bf614b076b2069b3ede062fc957ec1ae667d3b71a5fd8b7341be8dca8a.jpg)

![539d7d67117f53f870fd12a36ff4227a6efd5c64be400b5b2b6e661df33cc196.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/539d7d67117f53f870fd12a36ff4227a6efd5c64be400b5b2b6e661df33cc196.jpg)

![63ba57a655f3fa2b60d663a71cd7ba6ba61c25a5c1e88d802ac06a79007c7a8c.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/63ba57a655f3fa2b60d663a71cd7ba6ba61c25a5c1e88d802ac06a79007c7a8c.jpg)

![6db78b4c4c5b9c8e25627282539df5345c760394885b8acb018fda37ad7df3b4.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/6db78b4c4c5b9c8e25627282539df5345c760394885b8acb018fda37ad7df3b4.jpg)

![80b8448ec50adeda3abc000ca58ceed725791617a5f17dc934d88d6de96fac1d.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/80b8448ec50adeda3abc000ca58ceed725791617a5f17dc934d88d6de96fac1d.jpg)

![8c20a29d8026e1532115ff5e3adbbc3f0b93f871c767683c1cad089d258d578b.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/8c20a29d8026e1532115ff5e3adbbc3f0b93f871c767683c1cad089d258d578b.jpg)

![af34325aa38682c599fd80e510a0e0cb443a08c3587757c433516f048c3b43a6.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/af34325aa38682c599fd80e510a0e0cb443a08c3587757c433516f048c3b43a6.jpg)

![b8b6b32660fafb93be94465e5e40272dcf1fa1a4b9c67a3443b7abee88d556d6.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/b8b6b32660fafb93be94465e5e40272dcf1fa1a4b9c67a3443b7abee88d556d6.jpg)

![bc5d474bc6d73796c9763da663774e9d7ee1a5f5fbcb0ae9e1ff3ed844cae0b4.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/bc5d474bc6d73796c9763da663774e9d7ee1a5f5fbcb0ae9e1ff3ed844cae0b4.jpg)

![cb6a7e4c434af6f7b013598d9b58eb076915ce3555dd9afb8a77aaeba077676a.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/images/cb6a7e4c434af6f7b013598d9b58eb076915ce3555dd9afb8a77aaeba077676a.jpg)

### Tables

![69af4de8e07e45f900c82eec9108f64cf511ab6d0fa14ebe3b4237744a6f6597.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/tables/69af4de8e07e45f900c82eec9108f64cf511ab6d0fa14ebe3b4237744a6f6597.jpg)

![de18b02996eddf8744458d83cb5b1284d41c86476538a1027bffb56e6798e4b2.jpg](../iclr_results/360_Geometric Inductive Biases of Deep Networks_ The Role of Data and Architecture/tables/de18b02996eddf8744458d83cb5b1284d41c86476538a1027bffb56e6798e4b2.jpg)

## Online Preference Alignment for Language Models via Count-based Exploration


### Images

![6c4ee006991964346ff75243f1e4abb5e2e58fa93dd800973942731934187de8.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/images/6c4ee006991964346ff75243f1e4abb5e2e58fa93dd800973942731934187de8.jpg)

### Tables

![15cb5a7ff1277b2a3b57000f14dfb36673e807620584468d63030f58827b02ae.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/15cb5a7ff1277b2a3b57000f14dfb36673e807620584468d63030f58827b02ae.jpg)

![28c317e41df3bad41a57a651ae29b0c648741243d3a628d2fe764d7582f750bd.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/28c317e41df3bad41a57a651ae29b0c648741243d3a628d2fe764d7582f750bd.jpg)

![8859cac15a7f0721f901be68d9ab7a154053fdcacc8d90a3c40a59c719036749.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/8859cac15a7f0721f901be68d9ab7a154053fdcacc8d90a3c40a59c719036749.jpg)

![8d6b696f894858c5ff76567fdd3d79f964961884c8e439183ac3d6f3aa71b654.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/8d6b696f894858c5ff76567fdd3d79f964961884c8e439183ac3d6f3aa71b654.jpg)

![962e078daa1a2ba29607e0bbf22ff489eb8a0d967a5dd89e9e3fd375ad3937d8.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/962e078daa1a2ba29607e0bbf22ff489eb8a0d967a5dd89e9e3fd375ad3937d8.jpg)

![b430fedd3238ad8dc25053e6da569fec5cfc9d4e6b39a967ebbb1e168aa70065.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/b430fedd3238ad8dc25053e6da569fec5cfc9d4e6b39a967ebbb1e168aa70065.jpg)

![b4fc1f4e489da141b41cee77db2d554fd50615802c7872c1a523755764becb44.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/b4fc1f4e489da141b41cee77db2d554fd50615802c7872c1a523755764becb44.jpg)

![bfe861f75110fc9c8bbd406f8fc4b20699810923925782418a52d8e5a94b2533.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/bfe861f75110fc9c8bbd406f8fc4b20699810923925782418a52d8e5a94b2533.jpg)

![d12246abfc2a05d6a88bb091a3a4294510888f5e6f44cebd82a99525d1519d7a.jpg](../iclr_results/361_Online Preference Alignment for Language Models via Count-based Exploration/tables/d12246abfc2a05d6a88bb091a3a4294510888f5e6f44cebd82a99525d1519d7a.jpg)

## Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution


### Images

![27c26da71330966f3c62169cab2af39b7adb911c520bbebf556cefc3e5dc070d.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/images/27c26da71330966f3c62169cab2af39b7adb911c520bbebf556cefc3e5dc070d.jpg)

![4bb5631a317785f7c5d86acc63fa895e45e2b42a6b21fc0f4a560ea41d9c4dbc.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/images/4bb5631a317785f7c5d86acc63fa895e45e2b42a6b21fc0f4a560ea41d9c4dbc.jpg)

![638db487ed41f53e6d4ee84a17337f72b374897362337f7387d01761033ee21a.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/images/638db487ed41f53e6d4ee84a17337f72b374897362337f7387d01761033ee21a.jpg)

![9bb5f430e57ba4e8d86cdf3ec375a20474cdfc50cc1fafe251fe63d00fe21f79.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/images/9bb5f430e57ba4e8d86cdf3ec375a20474cdfc50cc1fafe251fe63d00fe21f79.jpg)

![9e31ee8c128dd1eb366cf3476d34434179486245b6acf1bac205dcede7ae7cbe.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/images/9e31ee8c128dd1eb366cf3476d34434179486245b6acf1bac205dcede7ae7cbe.jpg)

![fa662aa8ff06d70e5d1ca115c7aca125c7225afaee3f02f4b87025f64a167d00.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/images/fa662aa8ff06d70e5d1ca115c7aca125c7225afaee3f02f4b87025f64a167d00.jpg)

### Tables

![187d114859b011b121432c85a2ed21ff0ac268d393d6a5153a388ac003a00808.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/187d114859b011b121432c85a2ed21ff0ac268d393d6a5153a388ac003a00808.jpg)

![1e133977379ec7281f048dffa76d90977d5aeffdf2d35db9a8c12c41a7911e73.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/1e133977379ec7281f048dffa76d90977d5aeffdf2d35db9a8c12c41a7911e73.jpg)

![33f65dacbd88f2ace26b81d01a5696d906bae7fc5a4e5dc812161e6b3a419a55.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/33f65dacbd88f2ace26b81d01a5696d906bae7fc5a4e5dc812161e6b3a419a55.jpg)

![4e3fc6b7fefbdb5022945f27566f974caa4840a70cd4d5fd6c845d8734a51e88.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/4e3fc6b7fefbdb5022945f27566f974caa4840a70cd4d5fd6c845d8734a51e88.jpg)

![4e8afc3ca46fecfdf2864ca88f3bfe0534462a57639225ec5f16bfa7ec67608e.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/4e8afc3ca46fecfdf2864ca88f3bfe0534462a57639225ec5f16bfa7ec67608e.jpg)

![51fc0deff17c1b4c08454ab666fa2c9725c33847a5f625d81ea566da8ca7eb85.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/51fc0deff17c1b4c08454ab666fa2c9725c33847a5f625d81ea566da8ca7eb85.jpg)

![694d8f796463a192d6e89ccca9de77924cb0e3dee6437c8af64403624a48e1b6.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/694d8f796463a192d6e89ccca9de77924cb0e3dee6437c8af64403624a48e1b6.jpg)

![a425c275cb2680dcb40bcceec59b8e752728de1fd357358636a2407b54834521.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/a425c275cb2680dcb40bcceec59b8e752728de1fd357358636a2407b54834521.jpg)

![bf2aa35473991221899c467d569905766a31af09670e50ab85388b282775cafb.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/bf2aa35473991221899c467d569905766a31af09670e50ab85388b282775cafb.jpg)

![ed18555e4daf72089dee713b47c4aeb7337d543e04f9b83bbabece81361866ea.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/ed18555e4daf72089dee713b47c4aeb7337d543e04f9b83bbabece81361866ea.jpg)

![ef48e9e7d290555158a1371ee91c4067a0a8192d8a54a39da12719f8e40b032c.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/ef48e9e7d290555158a1371ee91c4067a0a8192d8a54a39da12719f8e40b032c.jpg)

![f2968d3bc9800dea944d91a3baf1471ba5ba3b33bda58ecf75c3403db8fa7605.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/f2968d3bc9800dea944d91a3baf1471ba5ba3b33bda58ecf75c3403db8fa7605.jpg)

![fe8865eb02708945c63771bbf56f4636a59ee61cdc720f761de38bcface6609a.jpg](../iclr_results/362_Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution/tables/fe8865eb02708945c63771bbf56f4636a59ee61cdc720f761de38bcface6609a.jpg)

## BodyGen: Advancing Towards Efficient Embodiment Co-Design


### Images

![2b4e3a4abb48ef3ed6a6c7ec5d9f73e5ac5d04e3d6ead4a0a9ede06c65c4008b.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/2b4e3a4abb48ef3ed6a6c7ec5d9f73e5ac5d04e3d6ead4a0a9ede06c65c4008b.jpg)

![584c6cf7ff26ae24706a3d20949da184e4026d56f2f9cf4b3be5567b81de8b57.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/584c6cf7ff26ae24706a3d20949da184e4026d56f2f9cf4b3be5567b81de8b57.jpg)

![68b075e99a9891010d396833eed599750aec6d29450596ff8e96112114123817.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/68b075e99a9891010d396833eed599750aec6d29450596ff8e96112114123817.jpg)

![6dcac61dad224dfca94819422be20d816cf03a9a202c9547fb0d321cc9176a43.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/6dcac61dad224dfca94819422be20d816cf03a9a202c9547fb0d321cc9176a43.jpg)

![731c760656eef022ef2f0bc8f31d65d5ec10db4e0f9532904e7d58e807a0847f.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/731c760656eef022ef2f0bc8f31d65d5ec10db4e0f9532904e7d58e807a0847f.jpg)

![739967f255518ca34bb83dff7696b69a4a26e7625eb260ccfb8593ee8d051d4c.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/739967f255518ca34bb83dff7696b69a4a26e7625eb260ccfb8593ee8d051d4c.jpg)

![7a3038a08b5c8fed598b028a387f1b4f34f9cef1d9a6b62dea4a032f55122d82.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/7a3038a08b5c8fed598b028a387f1b4f34f9cef1d9a6b62dea4a032f55122d82.jpg)

![87e4528fef42c42236bb8117a714b5e8464b690664748a995cff762abfa6ede6.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/87e4528fef42c42236bb8117a714b5e8464b690664748a995cff762abfa6ede6.jpg)

![9a11a4db26108ad968813673151e02341bd719a1556dff5749ee912236e13fee.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/9a11a4db26108ad968813673151e02341bd719a1556dff5749ee912236e13fee.jpg)

![9bee79e01c67122fe64ed7baaf14d58b5d0713b9df6eae4217a7d2b1e8e4c231.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/9bee79e01c67122fe64ed7baaf14d58b5d0713b9df6eae4217a7d2b1e8e4c231.jpg)

![c26a80714aea63dec305c3584584001cbe110824046bfa01eae9e47f12610f71.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/c26a80714aea63dec305c3584584001cbe110824046bfa01eae9e47f12610f71.jpg)

![c9fe3144dc4720a315353f6b6809b067e65ce054dd63adaa02d2edaa0709f3df.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/c9fe3144dc4720a315353f6b6809b067e65ce054dd63adaa02d2edaa0709f3df.jpg)

![d7f10c3bbf5072f47257b68e7024a212c0b37e92e00e306ef88e172d2e9ffd28.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/d7f10c3bbf5072f47257b68e7024a212c0b37e92e00e306ef88e172d2e9ffd28.jpg)

![f5b9e552a501386cf2ed0be4ac4894761e1cbcb5e8f53d8b6b389a0cba7565d4.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/images/f5b9e552a501386cf2ed0be4ac4894761e1cbcb5e8f53d8b6b389a0cba7565d4.jpg)

### Tables

![10545436ce7bec554f680c2e0ac1446f5547d87c53b0830caf57c6c03a251dae.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/tables/10545436ce7bec554f680c2e0ac1446f5547d87c53b0830caf57c6c03a251dae.jpg)

![258fb0c7c1c5cbede9975cff70abca14dc31cccc9b417a0bb815b7dcbda7907a.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/tables/258fb0c7c1c5cbede9975cff70abca14dc31cccc9b417a0bb815b7dcbda7907a.jpg)

![33a3d7a751bc7581604f4657c56eeab7d2248f274a1fb3d115c56f2c0a32dd6a.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/tables/33a3d7a751bc7581604f4657c56eeab7d2248f274a1fb3d115c56f2c0a32dd6a.jpg)

![5350340a00eb64e1e55cd9fe8b0f7e247c725e0d374254bca1760617103efcdc.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/tables/5350340a00eb64e1e55cd9fe8b0f7e247c725e0d374254bca1760617103efcdc.jpg)

![ca87d8a09088f5dfd138d749884ba05395d615a495a095648f39428053e5b1d4.jpg](../iclr_results/363_BodyGen_ Advancing Towards Efficient Embodiment Co-Design/tables/ca87d8a09088f5dfd138d749884ba05395d615a495a095648f39428053e5b1d4.jpg)

## Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models


### Images

![37a460426fc9c6608fc616e3bb8ac668cfdc8d2d14e690e86751c6340b12c6b2.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/37a460426fc9c6608fc616e3bb8ac668cfdc8d2d14e690e86751c6340b12c6b2.jpg)

![4848288bcf1e80319a9f27385b6579a00a72ec2a3a9e9d114f13bbfc7105b40b.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/4848288bcf1e80319a9f27385b6579a00a72ec2a3a9e9d114f13bbfc7105b40b.jpg)

![9eb22d05c175fa12bf6fc481e6a5cf03f1f09d7e6fb00be105dd3fe720ad7a97.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/9eb22d05c175fa12bf6fc481e6a5cf03f1f09d7e6fb00be105dd3fe720ad7a97.jpg)

![c154fd06b6c73b37566ee472ccd755e1204432dd5ce379d02d330c26aa2abeb5.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/c154fd06b6c73b37566ee472ccd755e1204432dd5ce379d02d330c26aa2abeb5.jpg)

![c3f885e2f06a0ccb2fc481b31add89d2e3b6d0f45909589d09ddd235d75ef85f.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/c3f885e2f06a0ccb2fc481b31add89d2e3b6d0f45909589d09ddd235d75ef85f.jpg)

![c4c04244cc2ac358b148a34f791eaf5e2492821d33f1baf49d82a11bb16770e1.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/c4c04244cc2ac358b148a34f791eaf5e2492821d33f1baf49d82a11bb16770e1.jpg)

![c738f22c50207bdd6fd59be0e8d4e65c74ccaacef662c93ee420937195aeb32d.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/c738f22c50207bdd6fd59be0e8d4e65c74ccaacef662c93ee420937195aeb32d.jpg)

![e065cad3ab9aa84dfc471873a2849e232cf1bfe2b2cd217eef5eb3c85581c86a.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/images/e065cad3ab9aa84dfc471873a2849e232cf1bfe2b2cd217eef5eb3c85581c86a.jpg)

### Tables

![23ba1fe92799004d19d767ebe378d90892ed3e2b4f25b2e62c5af11c5c331046.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/23ba1fe92799004d19d767ebe378d90892ed3e2b4f25b2e62c5af11c5c331046.jpg)

![3fd71862ce3ecafea662113c15592894b58acb1d48a8301e00cba6912283e2c5.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/3fd71862ce3ecafea662113c15592894b58acb1d48a8301e00cba6912283e2c5.jpg)

![47e3e41be697244e88a1b6fd8d195b1f49e9be95c90f6add79506cb645a28d7f.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/47e3e41be697244e88a1b6fd8d195b1f49e9be95c90f6add79506cb645a28d7f.jpg)

![538c86911e08c5ff6602d45572192ea4b11c23b2a83fe7e779d4a454bb005192.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/538c86911e08c5ff6602d45572192ea4b11c23b2a83fe7e779d4a454bb005192.jpg)

![5954f9c98d43a25aff62866a16be815c96a8e17e9ddbe53569cb759f86a2fc97.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/5954f9c98d43a25aff62866a16be815c96a8e17e9ddbe53569cb759f86a2fc97.jpg)

![5a0e89982a6ef82ded3fc75bcca5ad0a9fbf995e2f75bfffb13dd74e7bdeb672.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/5a0e89982a6ef82ded3fc75bcca5ad0a9fbf995e2f75bfffb13dd74e7bdeb672.jpg)

![5bc1897791600911cdbc16bdc27f2da4e453979207013df3aba3739a45336708.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/5bc1897791600911cdbc16bdc27f2da4e453979207013df3aba3739a45336708.jpg)

![69674ba205b97e9f64bcdfa891f4729ddbaf838c37c6acd46cd03245c88cafac.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/69674ba205b97e9f64bcdfa891f4729ddbaf838c37c6acd46cd03245c88cafac.jpg)

![728da127e5e2dd9fd361b8643011057035bc3d60fc1e5a9da4b93d65066935bd.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/728da127e5e2dd9fd361b8643011057035bc3d60fc1e5a9da4b93d65066935bd.jpg)

![9cc28c36e7bac369f3cf997e1add160784e32ab59a26878556fd23abc3a72cb2.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/9cc28c36e7bac369f3cf997e1add160784e32ab59a26878556fd23abc3a72cb2.jpg)

![9d0adc8b4c802defaf0b5813b4d906bc4b90f8767ccac98460b59adab6b12f60.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/9d0adc8b4c802defaf0b5813b4d906bc4b90f8767ccac98460b59adab6b12f60.jpg)

![fa470548ff62d59b1a0a072502f26d9b1de335f4b660237b985bbd8df80e2052.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/fa470548ff62d59b1a0a072502f26d9b1de335f4b660237b985bbd8df80e2052.jpg)

![fc475c685d3c8675a3c50d9b9ac3b05cb24166d2076b7b6ad4e0b6371271c852.jpg](../iclr_results/364_Self-play with Execution Feedback_ Improving Instruction-following Capabilities of Large Language Mo/tables/fc475c685d3c8675a3c50d9b9ac3b05cb24166d2076b7b6ad4e0b6371271c852.jpg)

## One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt


### Images

![01ae5650143e65fad8abb3c9c329152e3b4ee6db2eb7d7f1b080489679824042.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/01ae5650143e65fad8abb3c9c329152e3b4ee6db2eb7d7f1b080489679824042.jpg)

![0d03c44685948151513709ca63a5ce3d0f7cba216ba1a75aebe2627b862be649.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/0d03c44685948151513709ca63a5ce3d0f7cba216ba1a75aebe2627b862be649.jpg)

![127cb832a3a0077945ec8668a5f19765fba5f85782497b271fed11eb25b63b67.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/127cb832a3a0077945ec8668a5f19765fba5f85782497b271fed11eb25b63b67.jpg)

![15476771a54f6af88f61af3bae814cfe2abb8fba64c84b1c9b48ecdc714edee8.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/15476771a54f6af88f61af3bae814cfe2abb8fba64c84b1c9b48ecdc714edee8.jpg)

![16a345e1aaad7878993e321e9f7554ca6affb40a59045c9f01fcc7bf28289c44.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/16a345e1aaad7878993e321e9f7554ca6affb40a59045c9f01fcc7bf28289c44.jpg)

![2fd73f99ce6148608fbd5b53e02c62f49ac51433d5d0fdfae1f2b2c734c8d6b9.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/2fd73f99ce6148608fbd5b53e02c62f49ac51433d5d0fdfae1f2b2c734c8d6b9.jpg)

![347999978e089b017d4bdbf0d29b78272e7ef4e7b51081416dcbce54951fc7fd.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/347999978e089b017d4bdbf0d29b78272e7ef4e7b51081416dcbce54951fc7fd.jpg)

![38d08301fa185368a9deaca9b73d6806c0a63665aa15f0c853d532b73116e645.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/38d08301fa185368a9deaca9b73d6806c0a63665aa15f0c853d532b73116e645.jpg)

![3c42754db79650e5095bd5b5e8c91a4261ea27675dad96e377699119e7eb9f08.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/3c42754db79650e5095bd5b5e8c91a4261ea27675dad96e377699119e7eb9f08.jpg)

![46a98dca7a02d8eedd088457a62ec7f118a352bad4eaf65c1d7821022ef378c4.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/46a98dca7a02d8eedd088457a62ec7f118a352bad4eaf65c1d7821022ef378c4.jpg)

![4c04b0b98a5487478d34d6f147af6c1fc0cdebda992f7a027ea0dafcfca6b448.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/4c04b0b98a5487478d34d6f147af6c1fc0cdebda992f7a027ea0dafcfca6b448.jpg)

![6356e9d1d45df8487899ee4992a6dbc2f1a4fea0af420710ed21bffc7cf6e98f.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/6356e9d1d45df8487899ee4992a6dbc2f1a4fea0af420710ed21bffc7cf6e98f.jpg)

![6a11844776fdbee0a09829b273d9b427f521dfabfa7a08f10c758a3e9d258113.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/6a11844776fdbee0a09829b273d9b427f521dfabfa7a08f10c758a3e9d258113.jpg)

![781bd516832017400fe89e1378a4321623143bfc5ff937424932b1774e8d54a2.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/781bd516832017400fe89e1378a4321623143bfc5ff937424932b1774e8d54a2.jpg)

![782a546619d14d4552a1f86158597cab1125292167243fd0e3656854deb7cddb.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/782a546619d14d4552a1f86158597cab1125292167243fd0e3656854deb7cddb.jpg)

![7a5a229c72538c2b4e412578e92071b4cdf5bdfaa2b4954a8d2f169c814c617e.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/7a5a229c72538c2b4e412578e92071b4cdf5bdfaa2b4954a8d2f169c814c617e.jpg)

![7b6d7ca0026a7d303241393a955ba9acf090ecea00b0b6d3d427cfd32620b462.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/7b6d7ca0026a7d303241393a955ba9acf090ecea00b0b6d3d427cfd32620b462.jpg)

![8c3a8ecb418d5fb194d067d113764e2bdae4edc910ff1cab3cffc8af9ec632e0.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/8c3a8ecb418d5fb194d067d113764e2bdae4edc910ff1cab3cffc8af9ec632e0.jpg)

![940dbbd9f0117e554429224c25d9cc5301f7a6c67dbc7227d9b6fad9027241d9.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/940dbbd9f0117e554429224c25d9cc5301f7a6c67dbc7227d9b6fad9027241d9.jpg)

![b4216cd3581084b279ce98b64ec22c72917392e39d1df7e7a4d5af13a7768467.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/b4216cd3581084b279ce98b64ec22c72917392e39d1df7e7a4d5af13a7768467.jpg)

![c96c83ac9a1a76dae8186495bf018e39715fa3f4cbc75dfa12846dfb3bc98176.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/c96c83ac9a1a76dae8186495bf018e39715fa3f4cbc75dfa12846dfb3bc98176.jpg)

![caa5d7220dd2a0e7925f85aa54197a37de5dd51287b70c8fce9573cbe63ad1e9.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/caa5d7220dd2a0e7925f85aa54197a37de5dd51287b70c8fce9573cbe63ad1e9.jpg)

![ee44e4265e6107f82d661ecdd324a624b2f017918540e567ee3699eabf818ed6.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/ee44e4265e6107f82d661ecdd324a624b2f017918540e567ee3699eabf818ed6.jpg)

![f0dccd5f5d780757fde78168f0cd324db9129dd3af1bda011ecce655e2f55dcb.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/f0dccd5f5d780757fde78168f0cd324db9129dd3af1bda011ecce655e2f55dcb.jpg)

![f206ee160d7b6c61fcc635f3627387a47c7544d68613c9b7dc68bcae8fd7f41b.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/f206ee160d7b6c61fcc635f3627387a47c7544d68613c9b7dc68bcae8fd7f41b.jpg)

![fb93c2154115a8cc5e65669550c18e7f1f67af0c6e3225d362f5887ce8e4f834.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/images/fb93c2154115a8cc5e65669550c18e7f1f67af0c6e3225d362f5887ce8e4f834.jpg)

### Tables

![14e2418a7f6005c25f22ec6bfa04608a761129c03e01d98924ae74bfd02beca7.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/tables/14e2418a7f6005c25f22ec6bfa04608a761129c03e01d98924ae74bfd02beca7.jpg)

![2bc74ca09f906dd55897910058cb086f291d464d984331c9fd262544be369952.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/tables/2bc74ca09f906dd55897910058cb086f291d464d984331c9fd262544be369952.jpg)

![a1ff0fec9d1ca0186e050eda45ed9e6f6504e9b74823055c4946137c4d2d31b7.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/tables/a1ff0fec9d1ca0186e050eda45ed9e6f6504e9b74823055c4946137c4d2d31b7.jpg)

![e1f55e73682ef2be58e084e1a27b347578114691d750a432c82aadd1a35bd38d.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/tables/e1f55e73682ef2be58e084e1a27b347578114691d750a432c82aadd1a35bd38d.jpg)

![f7a64e766ce172ba2a623af5461cb9a408fb9a4a2fec775b08ab18f3f5c0f489.jpg](../iclr_results/365_One-Prompt-One-Story_ Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt/tables/f7a64e766ce172ba2a623af5461cb9a408fb9a4a2fec775b08ab18f3f5c0f489.jpg)

## Linear SCM Identification in the Presence of Confounders and Gaussian Noise


### Tables

![b018adb43fe3448d4b74307efc8c59b64b5993bd2fd86783f0e7ec97c43f4375.jpg](../iclr_results/366_Linear SCM Identification in the Presence of Confounders and Gaussian Noise/tables/b018adb43fe3448d4b74307efc8c59b64b5993bd2fd86783f0e7ec97c43f4375.jpg)

## AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs


### Images

![0a72f0533ba1cee836a3807d0bc52de1d4d2446be5ab3907d0f44d23f159c6b0.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/0a72f0533ba1cee836a3807d0bc52de1d4d2446be5ab3907d0f44d23f159c6b0.jpg)

![159f27b3db5ed03bc79dbefb4eecff08972037f0294b93993dbaad19e3bce958.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/159f27b3db5ed03bc79dbefb4eecff08972037f0294b93993dbaad19e3bce958.jpg)

![33d8a9d95ca6d0f989e39d3697874c677d636870f4439e929081e3d6053b21ca.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/33d8a9d95ca6d0f989e39d3697874c677d636870f4439e929081e3d6053b21ca.jpg)

![5a2825b244464e16ecaf2f89fd0bd9f8c4b94f4922a326f64d8a0f1819192281.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/5a2825b244464e16ecaf2f89fd0bd9f8c4b94f4922a326f64d8a0f1819192281.jpg)

![80c24a9836b74d2db5d3741f4296065481b59cd6425cfef77fd81f24a7afa3dd.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/80c24a9836b74d2db5d3741f4296065481b59cd6425cfef77fd81f24a7afa3dd.jpg)

![9b6a602ebed2d6177c76285cbb57620069a1140c8a1ddd2f986b3fe4aabda8da.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/9b6a602ebed2d6177c76285cbb57620069a1140c8a1ddd2f986b3fe4aabda8da.jpg)

![c6fa9cb44938c63a0d682cb455c199f60ebc5125a3c50ca2a5984a3da368b4e6.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/c6fa9cb44938c63a0d682cb455c199f60ebc5125a3c50ca2a5984a3da368b4e6.jpg)

![d0bdf771d243a4f080a48bec7445f3dc3019749d7592d6645ab275917ffc1035.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/images/d0bdf771d243a4f080a48bec7445f3dc3019749d7592d6645ab275917ffc1035.jpg)

### Tables

![11c7f03fa00cceaffb54ae667e8e174fa560537a655e1d394eb897d96e374745.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/11c7f03fa00cceaffb54ae667e8e174fa560537a655e1d394eb897d96e374745.jpg)

![37364d25a2bdff3ee1f7e5f8c6474b1607f61957dfdca50718373068c4ba2d12.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/37364d25a2bdff3ee1f7e5f8c6474b1607f61957dfdca50718373068c4ba2d12.jpg)

![384949d2824640b2356f2c0881f99a80f791925d7a6950af4d208fee476bd067.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/384949d2824640b2356f2c0881f99a80f791925d7a6950af4d208fee476bd067.jpg)

![41b19a8a9ce196038e58a4930836736446224a361a954731271421294d6a2d0a.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/41b19a8a9ce196038e58a4930836736446224a361a954731271421294d6a2d0a.jpg)

![603f3851dff51ef8aad6193f7d289bbbf0acab49c4aa7028dec43621e7104583.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/603f3851dff51ef8aad6193f7d289bbbf0acab49c4aa7028dec43621e7104583.jpg)

![c3d4ce545b0ae619c6bc159edad6a9a3f01c9506e634418de32aad6487c4cefa.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/c3d4ce545b0ae619c6bc159edad6a9a3f01c9506e634418de32aad6487c4cefa.jpg)

![cb839f08c3166963960b6a067480559e8d8ac3954add0cebf5921688d65a0e01.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/cb839f08c3166963960b6a067480559e8d8ac3954add0cebf5921688d65a0e01.jpg)

![d12b43b54df22714488d738ec044b1111faa01813998d3437ba5c5ae1bba880f.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/d12b43b54df22714488d738ec044b1111faa01813998d3437ba5c5ae1bba880f.jpg)

![d327fb04064eedab8263811ebdc92b7b508e606ed081a2db9502fd4e0014c295.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/d327fb04064eedab8263811ebdc92b7b508e606ed081a2db9502fd4e0014c295.jpg)

![d44ffbb1d16cc1068f6b9fd482b8bfc8111a6d051c2a8a915686d2eb5e4f27c2.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/d44ffbb1d16cc1068f6b9fd482b8bfc8111a6d051c2a8a915686d2eb5e4f27c2.jpg)

![e82eca5b49256db15883d2fbba1d3f9d8b1d27e737cbfc5f9c5ea7eee4954263.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/e82eca5b49256db15883d2fbba1d3f9d8b1d27e737cbfc5f9c5ea7eee4954263.jpg)

![e9b31e3501325a67ea35b7fed1d289020a183d2f315d83f32f76cb95d5545137.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/e9b31e3501325a67ea35b7fed1d289020a183d2f315d83f32f76cb95d5545137.jpg)

![ebc9cb34b4f0f8ab38caad02aeba69da8907f0988e9e6dcbecd68ce8d91ef03c.jpg](../iclr_results/367_AutoDAN-Turbo_ A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs/tables/ebc9cb34b4f0f8ab38caad02aeba69da8907f0988e9e6dcbecd68ce8d91ef03c.jpg)

## NetFormer: An interpretable model for recovering dynamical connectivity in neuronal population dynamics


### Images

![23789f290b1eeb9eeee8c2da996ac56fc44b0410ad1bd425383f3272b2bbbd23.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/23789f290b1eeb9eeee8c2da996ac56fc44b0410ad1bd425383f3272b2bbbd23.jpg)

![3ed3fbaf5d2ad3f99ef2988af3160e23216daac346f491320a9bd0194ef198d2.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/3ed3fbaf5d2ad3f99ef2988af3160e23216daac346f491320a9bd0194ef198d2.jpg)

![5c0c0d83803e4f262efe8ed9f098dc7a0c02df8daede94dadb32a31adb5317ba.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/5c0c0d83803e4f262efe8ed9f098dc7a0c02df8daede94dadb32a31adb5317ba.jpg)

![6166785da2ee2997c5d5f92ac616ef2a72a64d1c38b480eb33da0f0f48af5f61.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/6166785da2ee2997c5d5f92ac616ef2a72a64d1c38b480eb33da0f0f48af5f61.jpg)

![6307fc2b17396a1f6bb0a7269ca598bd39d0ef552701c1e613b82b32090791ad.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/6307fc2b17396a1f6bb0a7269ca598bd39d0ef552701c1e613b82b32090791ad.jpg)

![738a524c60d06de8ca6f8e71fe6334af3f6b5286c733431abb13a2cde8686665.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/738a524c60d06de8ca6f8e71fe6334af3f6b5286c733431abb13a2cde8686665.jpg)

![93506a9522f12e09f74add9668b7805904c0bf91c717f61188bd394a43e5392b.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/93506a9522f12e09f74add9668b7805904c0bf91c717f61188bd394a43e5392b.jpg)

![9ebcb9b1004e2c9f5e55f1c081c36b6d6e7a1b940ae775fc583cd34d21775a5d.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/9ebcb9b1004e2c9f5e55f1c081c36b6d6e7a1b940ae775fc583cd34d21775a5d.jpg)

![a679a1dc9faf4c887fcaba3ee1f65e544af89312a4ec3bb90898176fa89eebf6.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/a679a1dc9faf4c887fcaba3ee1f65e544af89312a4ec3bb90898176fa89eebf6.jpg)

![b4c7a3bf2a7cc1533c94f6d54c7faed9e7394f62ed4c7b232dacb3a864fe24ed.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/b4c7a3bf2a7cc1533c94f6d54c7faed9e7394f62ed4c7b232dacb3a864fe24ed.jpg)

![b527ad09f9d81652f802150c4796a1726a334a25b5d581f6beac3890fb936417.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/b527ad09f9d81652f802150c4796a1726a334a25b5d581f6beac3890fb936417.jpg)

![b56e6668b7b9f96d29cfa84b0d308486ed40b3e22f9958b3f312de4222893768.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/b56e6668b7b9f96d29cfa84b0d308486ed40b3e22f9958b3f312de4222893768.jpg)

![ceae6e8d555a22a1807e47ed1719e7440d48e1c120fe1d8f434634a1f0fdac33.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/ceae6e8d555a22a1807e47ed1719e7440d48e1c120fe1d8f434634a1f0fdac33.jpg)

![ed42b773320d8623edd3fa1a2f6cdf0d247d22360afd7de7fd1f5e3a82d4d5b2.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/ed42b773320d8623edd3fa1a2f6cdf0d247d22360afd7de7fd1f5e3a82d4d5b2.jpg)

![eec82713cb0a61497b5369653e2bd210978ff43e525e168aac535eb49868386a.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/images/eec82713cb0a61497b5369653e2bd210978ff43e525e168aac535eb49868386a.jpg)

### Tables

![0125e40a1f7da083edee9827ccb331891145f12156fb8ede6799014c828fee13.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/tables/0125e40a1f7da083edee9827ccb331891145f12156fb8ede6799014c828fee13.jpg)

![2086e352297e157453aee445f73291c80b123d769b2aa99b2c2c4f992289489a.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/tables/2086e352297e157453aee445f73291c80b123d769b2aa99b2c2c4f992289489a.jpg)

![3b04f1240093cd79ba18e4da374aa05cf65592a53ff7326624545949078c9f79.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/tables/3b04f1240093cd79ba18e4da374aa05cf65592a53ff7326624545949078c9f79.jpg)

![4ada70b821271b96116389989098671c6dd0566d78d4f548df685c936fb40718.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/tables/4ada70b821271b96116389989098671c6dd0566d78d4f548df685c936fb40718.jpg)

![6c5e9c92b854ab0c1f83e09da9dc98ceb45fc59101decf948e39dff7fcd8889c.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/tables/6c5e9c92b854ab0c1f83e09da9dc98ceb45fc59101decf948e39dff7fcd8889c.jpg)

![e48fe7ece619833d9104459508b3669c3335ae54964d3c47beefbe38aa0c2005.jpg](../iclr_results/368_NetFormer_ An interpretable model for recovering dynamical connectivity in neuronal population dynam/tables/e48fe7ece619833d9104459508b3669c3335ae54964d3c47beefbe38aa0c2005.jpg)

## MotionAura: Generating High-Quality and Motion Consistent Videos using Discrete Diffusion


### Images

![0933725de447f92ca5d0535bf86e2e694b21b82243ced9269a3f334730c25c70.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/0933725de447f92ca5d0535bf86e2e694b21b82243ced9269a3f334730c25c70.jpg)

![0f9cdc40176b3ded1b182f620fdf907233ef2565084cf9a115d860a36694dfb4.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/0f9cdc40176b3ded1b182f620fdf907233ef2565084cf9a115d860a36694dfb4.jpg)

![173422ec2f74d7501b6de5c816b1479cb94ee81c239810056276b8457be3f22c.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/173422ec2f74d7501b6de5c816b1479cb94ee81c239810056276b8457be3f22c.jpg)

![21eb913c4d427ce628f09dbfdf04aa0f3445a3539766000081721a190b456eda.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/21eb913c4d427ce628f09dbfdf04aa0f3445a3539766000081721a190b456eda.jpg)

![22ab23976dac09f38a9d4749f868272665dc34f3f9a567576a3c1018d0b0596a.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/22ab23976dac09f38a9d4749f868272665dc34f3f9a567576a3c1018d0b0596a.jpg)

![26874279ecfdb13989beb110841c3ecd8b8234a25e3b6bd6254f2e9bde8a227a.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/26874279ecfdb13989beb110841c3ecd8b8234a25e3b6bd6254f2e9bde8a227a.jpg)

![29f5d46942c85785982f27f3fe632968edaae74f4b6125015a8d851997edb012.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/29f5d46942c85785982f27f3fe632968edaae74f4b6125015a8d851997edb012.jpg)

![33a3027641569b840144bf8123a272b5e71124c1f45c21a15f4fc123d6b447dd.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/33a3027641569b840144bf8123a272b5e71124c1f45c21a15f4fc123d6b447dd.jpg)

![343ca886f1344cba29cab6c6183dfde6a0f6e9d420f62cc739f750272030279f.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/343ca886f1344cba29cab6c6183dfde6a0f6e9d420f62cc739f750272030279f.jpg)

![358fa061cfa5a9381ae44c0e43c3e98a7acd55a19011df80744f79a90254a5ba.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/358fa061cfa5a9381ae44c0e43c3e98a7acd55a19011df80744f79a90254a5ba.jpg)

![37dc00c34c850ce079d36fdbac1a75a13641212e95197b5b153ff7034320d517.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/37dc00c34c850ce079d36fdbac1a75a13641212e95197b5b153ff7034320d517.jpg)

![38e5fdd42b50922c220f07a19886ec57e6ea2199ec8244414e42253b26f4f2b0.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/38e5fdd42b50922c220f07a19886ec57e6ea2199ec8244414e42253b26f4f2b0.jpg)

![4255d44ffa52338cd90854a08d4500ab1a335b383aa2a3df6f0f636e88c4b613.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/4255d44ffa52338cd90854a08d4500ab1a335b383aa2a3df6f0f636e88c4b613.jpg)

![4f9d43f0ec1c15f4e206b1aca6c888ad596703c942dd80dc9c3189a55ec29c2a.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/4f9d43f0ec1c15f4e206b1aca6c888ad596703c942dd80dc9c3189a55ec29c2a.jpg)

![527dc995ca5b1ae3f1a1c68fefab1fa51dac2551e24de6f8305a13d211c6532d.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/527dc995ca5b1ae3f1a1c68fefab1fa51dac2551e24de6f8305a13d211c6532d.jpg)

![53569659b5e7ab775676f0e107e8c63aa45598f67ce4abc3d851b9ee28a44385.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/53569659b5e7ab775676f0e107e8c63aa45598f67ce4abc3d851b9ee28a44385.jpg)

![5662ec8bc34523d22998dcdf50b64659b0901d2cc7a53b34848e02bc21513fb3.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/5662ec8bc34523d22998dcdf50b64659b0901d2cc7a53b34848e02bc21513fb3.jpg)

![589998e7a72e098f25a56301cc2d65cbbe42f03a69690aa7d5a7b44e88d07503.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/589998e7a72e098f25a56301cc2d65cbbe42f03a69690aa7d5a7b44e88d07503.jpg)

![5b6b0339d0f32e44b71b850b086401ea4f47e6d93d3f4287b0711a7172f22b14.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/5b6b0339d0f32e44b71b850b086401ea4f47e6d93d3f4287b0711a7172f22b14.jpg)

![645484fa0864e21a6cac577dabb0f4e4c25df69097afb60ad530f100e3e6e5e0.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/645484fa0864e21a6cac577dabb0f4e4c25df69097afb60ad530f100e3e6e5e0.jpg)

![781a67b7cf46e26d87fd6f74665b67ab8b8aecf511997456d4c7be0940bce242.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/781a67b7cf46e26d87fd6f74665b67ab8b8aecf511997456d4c7be0940bce242.jpg)

![79ddee9fd92eaefb7a5c77ad26e869adc94118662ec25d7ab2be2fb2518002ab.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/79ddee9fd92eaefb7a5c77ad26e869adc94118662ec25d7ab2be2fb2518002ab.jpg)

![7a64ef08b8a6c32f6dc67e5706b441cf159ae98207e1536f0f9688da95f2ca90.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/7a64ef08b8a6c32f6dc67e5706b441cf159ae98207e1536f0f9688da95f2ca90.jpg)

![840b3cbb02cb4aaa5f7e77af87a6c5da11d5e86b715daa81335ec9acac78df18.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/840b3cbb02cb4aaa5f7e77af87a6c5da11d5e86b715daa81335ec9acac78df18.jpg)

![88091e9d5cd431c89c5a3fce93ae8da963e08e53e51d40d8cdf0d1a3a4fe762e.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/88091e9d5cd431c89c5a3fce93ae8da963e08e53e51d40d8cdf0d1a3a4fe762e.jpg)

![a17279bd5b8cb64da6cc8a1b4bc79e0620bed8f2ff86bd307318a1a94847c58c.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/a17279bd5b8cb64da6cc8a1b4bc79e0620bed8f2ff86bd307318a1a94847c58c.jpg)

![a9b5f4cf856ec91779a93a816405ad368b67e5b7195d393e8bdd95c4de5026e1.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/a9b5f4cf856ec91779a93a816405ad368b67e5b7195d393e8bdd95c4de5026e1.jpg)

![ada544d802ed0fc5873b33d239ece61639556d718547ab1138f3f5375cbb2b97.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/ada544d802ed0fc5873b33d239ece61639556d718547ab1138f3f5375cbb2b97.jpg)

![b38d662feccbff55bd6580b2ae4471b12f101d9ad6349104a0ca89eb35b01be9.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/b38d662feccbff55bd6580b2ae4471b12f101d9ad6349104a0ca89eb35b01be9.jpg)

![b7d6eb7598f830d54236c6c240d08f1758cfc1c9e163b63c89ad690ec22be94a.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/b7d6eb7598f830d54236c6c240d08f1758cfc1c9e163b63c89ad690ec22be94a.jpg)

![bc8d2b617767482d00095143bcb8fff0f923c0b5f89dcc5c0a69113d5377328a.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/bc8d2b617767482d00095143bcb8fff0f923c0b5f89dcc5c0a69113d5377328a.jpg)

![bebe8c888ad4e6c43aeea8d446fafef56521093dcddccb8651e0b402682ed164.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/bebe8c888ad4e6c43aeea8d446fafef56521093dcddccb8651e0b402682ed164.jpg)

![cd8974153f5778497600b5d555b2c270926bf992b8bf800331eb48173218ce0b.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/cd8974153f5778497600b5d555b2c270926bf992b8bf800331eb48173218ce0b.jpg)

![cec65e7c8ce0acdfba4d8ee1f877694bd8b9260e3b0383acaad39de48db343b8.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/cec65e7c8ce0acdfba4d8ee1f877694bd8b9260e3b0383acaad39de48db343b8.jpg)

![d06a5f9eb83f01208e60d3b5db0d3d0e3df57508c591d6c7b2300fe1bb6175ca.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/d06a5f9eb83f01208e60d3b5db0d3d0e3df57508c591d6c7b2300fe1bb6175ca.jpg)

![d4caf88a369be6f3cb1f6d8c759182d026ee795b9f3e7d6aac8d8fef75fca8a0.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/d4caf88a369be6f3cb1f6d8c759182d026ee795b9f3e7d6aac8d8fef75fca8a0.jpg)

![d62c8df45b99cc8b070fee98664b03ccbfd6f7e73d3faaf30b3df4d329a21beb.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/d62c8df45b99cc8b070fee98664b03ccbfd6f7e73d3faaf30b3df4d329a21beb.jpg)

![db186af62cf8189e4522348db6dc7b62abc6f9b132df6ce07895d427cd406405.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/db186af62cf8189e4522348db6dc7b62abc6f9b132df6ce07895d427cd406405.jpg)

![e58960cea558539e54bc1e3398bb31fde55a6be0543a7246f7799e13c543ca98.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/e58960cea558539e54bc1e3398bb31fde55a6be0543a7246f7799e13c543ca98.jpg)

![e806bab45f35f54dfd9239bfa11bbd3d1730229eb9ab2a4f77a4550694afd0b8.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/e806bab45f35f54dfd9239bfa11bbd3d1730229eb9ab2a4f77a4550694afd0b8.jpg)

![ed5552c15587314680f6e518513374f06d80827700067083bd495fb8302594e6.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/ed5552c15587314680f6e518513374f06d80827700067083bd495fb8302594e6.jpg)

![f4500d839347ddf5f71ba6d9f33fcf3245aa95f881019ef3247b256251eb9271.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/f4500d839347ddf5f71ba6d9f33fcf3245aa95f881019ef3247b256251eb9271.jpg)

![f461766a92e71445b8ade14a221571bdfec694620ef7b3826a05eb29f0fd96a5.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/f461766a92e71445b8ade14a221571bdfec694620ef7b3826a05eb29f0fd96a5.jpg)

![f6903323ca9e8b07ae8f777ad59ba14fd79d3ea53d92f5b5c1b4019482efc4bb.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/f6903323ca9e8b07ae8f777ad59ba14fd79d3ea53d92f5b5c1b4019482efc4bb.jpg)

![f7c4d35c7601f54d136f9b3ccd500e93d8cdfa729cdde7a8e322e2a143f5841c.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/images/f7c4d35c7601f54d136f9b3ccd500e93d8cdfa729cdde7a8e322e2a143f5841c.jpg)

### Tables

![26a80b1549cbf7b91bc7959a953eb1e14452a88c5080b968623e144b50f2d5d2.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/26a80b1549cbf7b91bc7959a953eb1e14452a88c5080b968623e144b50f2d5d2.jpg)

![4afabf1ca18ca309be5560c33245996eccc67c49bf2e540faf3795a629ffa19c.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/4afabf1ca18ca309be5560c33245996eccc67c49bf2e540faf3795a629ffa19c.jpg)

![556407624874e55bb1e845b2ec1976570950e03f95a1eae76b8ca8e10bcb4fe9.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/556407624874e55bb1e845b2ec1976570950e03f95a1eae76b8ca8e10bcb4fe9.jpg)

![60c08cf077eeda8a33a23608c75556d61a74218c85e79732fdf07d4b276844e8.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/60c08cf077eeda8a33a23608c75556d61a74218c85e79732fdf07d4b276844e8.jpg)

![673b1eee61236c62c542d033b8c3744fb7e12867893989b242d49ff87cde9640.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/673b1eee61236c62c542d033b8c3744fb7e12867893989b242d49ff87cde9640.jpg)

![828737b43e9c20cd6cd57672efafea801461903eded011a7b67d828d3a559a4f.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/828737b43e9c20cd6cd57672efafea801461903eded011a7b67d828d3a559a4f.jpg)

![889595e49f60f5e7e3cb3e3bd1b0241fe91930b62eb15e507aea763bbc6343b8.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/889595e49f60f5e7e3cb3e3bd1b0241fe91930b62eb15e507aea763bbc6343b8.jpg)

![a067bd00e5a590c2668d8efc5528d98c912dc774db449c270f93a69992d0593b.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/a067bd00e5a590c2668d8efc5528d98c912dc774db449c270f93a69992d0593b.jpg)

![a9c75f79dfb49d7ab4eb46b041420214126351208b66de44599f030be5ba9d2a.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/a9c75f79dfb49d7ab4eb46b041420214126351208b66de44599f030be5ba9d2a.jpg)

![d679b558ecd3c0993cd9dbe40c29ecc5cbbc68d4973c32a1e255c96bbad69ee3.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/d679b558ecd3c0993cd9dbe40c29ecc5cbbc68d4973c32a1e255c96bbad69ee3.jpg)

![e4e6eaebe0aaf5125f0ee1a3ffdf0bcbe927044b8d9132599050ead125f821ac.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/e4e6eaebe0aaf5125f0ee1a3ffdf0bcbe927044b8d9132599050ead125f821ac.jpg)

![f0d12257acd533847b3374d0c9c07cb6de5e0899e34ec761e98c88acb9e1bbad.jpg](../iclr_results/369_MotionAura_ Generating High-Quality and Motion Consistent Videos using Discrete Diffusion/tables/f0d12257acd533847b3374d0c9c07cb6de5e0899e34ec761e98c88acb9e1bbad.jpg)

## Harnessing Diversity for Important Data Selection in Pretraining Large Language Models


### Images

![66b9a6adb02f327e1a82ef10bcfeabde4ee96c5ab1e28daf58912a1b45af0530.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/images/66b9a6adb02f327e1a82ef10bcfeabde4ee96c5ab1e28daf58912a1b45af0530.jpg)

![99e6b7e1f2d9b0c0e1cd1cdf2c153b85b161caa8b7eac973955b5d930a95392e.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/images/99e6b7e1f2d9b0c0e1cd1cdf2c153b85b161caa8b7eac973955b5d930a95392e.jpg)

![9bd3026565c0e0df157202a3f68581f3919f229e79a70c5b30b1ecd9acaf657e.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/images/9bd3026565c0e0df157202a3f68581f3919f229e79a70c5b30b1ecd9acaf657e.jpg)

![a23bf2613989ec517e5863cdc1ca7f7f222c877883a00e8471a4958dffe6635f.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/images/a23bf2613989ec517e5863cdc1ca7f7f222c877883a00e8471a4958dffe6635f.jpg)

![b154e61627082389b5b21c337a5d3775d6f1cc253cf6b8a4d388a353ca76bbff.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/images/b154e61627082389b5b21c337a5d3775d6f1cc253cf6b8a4d388a353ca76bbff.jpg)

![d185204699354a0036fee092133ba7bb83d9bf9f700b17254d2f7992247228cf.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/images/d185204699354a0036fee092133ba7bb83d9bf9f700b17254d2f7992247228cf.jpg)

![f7a783a9921437c9b8679e4af550f4736f683e53e67ba20cc0061134940d2ad6.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/images/f7a783a9921437c9b8679e4af550f4736f683e53e67ba20cc0061134940d2ad6.jpg)

### Tables

![115c9e3bbfeea5d79315ef21b109d7c964d28b8140a8a11dc06881a7d50434e4.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/115c9e3bbfeea5d79315ef21b109d7c964d28b8140a8a11dc06881a7d50434e4.jpg)

![1c1699aa19d17054ed198402d21283ca9285365e69eeb60638aa112879319342.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/1c1699aa19d17054ed198402d21283ca9285365e69eeb60638aa112879319342.jpg)

![1c43c44d1921faab49cc9a03e07c89a5abb63bd3a7690c9021837d05b446fa58.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/1c43c44d1921faab49cc9a03e07c89a5abb63bd3a7690c9021837d05b446fa58.jpg)

![1ed295e5f27a5f5289e14d9a2cb68683c794c2322414fdd59aad26e06d735344.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/1ed295e5f27a5f5289e14d9a2cb68683c794c2322414fdd59aad26e06d735344.jpg)

![30dde450f7960aff7020cb762823f3a02272776f18fd8fecb55b35cb22b85780.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/30dde450f7960aff7020cb762823f3a02272776f18fd8fecb55b35cb22b85780.jpg)

![4c74d8a9c5540cadad06b0341b432e10d27926dd6149ed66d8bf2ebaf3051089.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/4c74d8a9c5540cadad06b0341b432e10d27926dd6149ed66d8bf2ebaf3051089.jpg)

![71457a0de3ce3cf93b8b06ef32aafabad245d55196c88696f964e4f624e1818e.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/71457a0de3ce3cf93b8b06ef32aafabad245d55196c88696f964e4f624e1818e.jpg)

![74fbe2662281b62803d7cf2cc31b9c07f7bd7d0519571def228acac02f29f160.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/74fbe2662281b62803d7cf2cc31b9c07f7bd7d0519571def228acac02f29f160.jpg)

![7b9d4abe9489f14d2df782a8b3aff20df776dcbe336f8d311f48a3ee51473cb3.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/7b9d4abe9489f14d2df782a8b3aff20df776dcbe336f8d311f48a3ee51473cb3.jpg)

![7fb070ac9d59b387d4202747f71b329fd0c93900ef8fc2b8f7f8f22bf6092898.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/7fb070ac9d59b387d4202747f71b329fd0c93900ef8fc2b8f7f8f22bf6092898.jpg)

![8001732b8cd17e7a40af0b48bc77e684c2f5a63e172223e833d7b3d292d3d42e.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/8001732b8cd17e7a40af0b48bc77e684c2f5a63e172223e833d7b3d292d3d42e.jpg)

![8cb79c487035544e23aba223000d9432ab8a211b6429d0112f932e2984c244e0.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/8cb79c487035544e23aba223000d9432ab8a211b6429d0112f932e2984c244e0.jpg)

![8f2f02bd93f64cd35060adffe19fd929f424104b787401cc8887815ff4cd8885.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/8f2f02bd93f64cd35060adffe19fd929f424104b787401cc8887815ff4cd8885.jpg)

![9ee452f56aabd9f3bd186e14e335bd499b58d520a8e3c9c4f608bc0cc438cf2d.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/9ee452f56aabd9f3bd186e14e335bd499b58d520a8e3c9c4f608bc0cc438cf2d.jpg)

![a5fe73a63cb323d69f09065c9a89c763387a923b2fe0edc84386483a2adfe4f6.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/a5fe73a63cb323d69f09065c9a89c763387a923b2fe0edc84386483a2adfe4f6.jpg)

![ba543cd2ec26ddc300588dc3f7dfcc6a890c6f057b0a0f1ae89f2406c0e9fb95.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/ba543cd2ec26ddc300588dc3f7dfcc6a890c6f057b0a0f1ae89f2406c0e9fb95.jpg)

![e4f8320d072df44e1ed0ec5215ae9743792b72d191ffb5050c86ab5c869fe872.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/e4f8320d072df44e1ed0ec5215ae9743792b72d191ffb5050c86ab5c869fe872.jpg)

![edd21081e259a2ba79554f62db08e801c92acb52ad1371eff2e8797ee96d4d06.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/edd21081e259a2ba79554f62db08e801c92acb52ad1371eff2e8797ee96d4d06.jpg)

![f4376f626ab4cb1ffd710f444ed31175bab714a93da6f3690b2eb6589f4cc95c.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/f4376f626ab4cb1ffd710f444ed31175bab714a93da6f3690b2eb6589f4cc95c.jpg)

![f51ff110ced454c2e354d43f636ad0abdf32bab00dd2072c5abd4874ae72d9c0.jpg](../iclr_results/370_Harnessing Diversity for Important Data Selection in Pretraining Large Language Models/tables/f51ff110ced454c2e354d43f636ad0abdf32bab00dd2072c5abd4874ae72d9c0.jpg)

## Generating Freeform Endoskeletal Robots


### Images

![0a5b78a92f00d5f186bd9a861dcf68034585b19c448ebc7d18c644f881975d63.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/0a5b78a92f00d5f186bd9a861dcf68034585b19c448ebc7d18c644f881975d63.jpg)

![1be0f34b5d11cacd586098d438afdc38807dbec5c948c5bc315ebb62354f8cbe.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/1be0f34b5d11cacd586098d438afdc38807dbec5c948c5bc315ebb62354f8cbe.jpg)

![26bb1c7677cb0fd3f342358f529e63af3e55cb4062d96d2c340c424f7ef07422.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/26bb1c7677cb0fd3f342358f529e63af3e55cb4062d96d2c340c424f7ef07422.jpg)

![28c660ab036ed3807372248b32bd0be0936c7e4f0ffb608d94e5181c1d64604e.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/28c660ab036ed3807372248b32bd0be0936c7e4f0ffb608d94e5181c1d64604e.jpg)

![30b6e595995d61c3a8324dd1f3da140210fdc8b35d7fd4fcc140bea7068883ea.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/30b6e595995d61c3a8324dd1f3da140210fdc8b35d7fd4fcc140bea7068883ea.jpg)

![50577b366e13ee5dc1e8be605c4ee0dc581fb53365001dd265fb01c71f6af70c.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/50577b366e13ee5dc1e8be605c4ee0dc581fb53365001dd265fb01c71f6af70c.jpg)

![56fd510bc6f96768908099c9ec44c3e33fba37cdf7a0d74222d28d52303409d5.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/56fd510bc6f96768908099c9ec44c3e33fba37cdf7a0d74222d28d52303409d5.jpg)

![5f61746f6e53ca2b785eef213ac33c5ea3326d7eeefbc7ece42179368d3d7d42.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/5f61746f6e53ca2b785eef213ac33c5ea3326d7eeefbc7ece42179368d3d7d42.jpg)

![65703eedde9000eaddbeb61ee7e500d681153ca8a0f88c549deb85c514646c39.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/65703eedde9000eaddbeb61ee7e500d681153ca8a0f88c549deb85c514646c39.jpg)

![6c13428feaf5b9833753dd405d810e6c1a091cec38ec4480c5d97b4787df079f.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/6c13428feaf5b9833753dd405d810e6c1a091cec38ec4480c5d97b4787df079f.jpg)

![72a7d37a27ce0e2e57789c8c107933ea1a3e9c979f54bd9ea2bd967c353368d7.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/72a7d37a27ce0e2e57789c8c107933ea1a3e9c979f54bd9ea2bd967c353368d7.jpg)

![7b4cd4ffae6dba647dc9a8e50d25b294e293b630010882774e609f1e86c49005.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/7b4cd4ffae6dba647dc9a8e50d25b294e293b630010882774e609f1e86c49005.jpg)

![90d95b3454575431224d00375bf064bb2cecba0f177d4ef9c78adf304f303464.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/90d95b3454575431224d00375bf064bb2cecba0f177d4ef9c78adf304f303464.jpg)

![992d3bc80873be747afd5b7a51f51f192e493b533f3e2758c587876c5bae598b.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/992d3bc80873be747afd5b7a51f51f192e493b533f3e2758c587876c5bae598b.jpg)

![9a23706aaf10f97e07d863a7d247a19db349188d8ed0c32b033df21542994b1e.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/9a23706aaf10f97e07d863a7d247a19db349188d8ed0c32b033df21542994b1e.jpg)

![a48e64881f91b0ca98f7602e67d859c9dd57f15dbfe9f733ba094596f654f1cd.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/a48e64881f91b0ca98f7602e67d859c9dd57f15dbfe9f733ba094596f654f1cd.jpg)

![be126f42166d4a7d02a4d2bedd2e70d917c66ab57a95aa4479f6ca7d6abbfda0.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/be126f42166d4a7d02a4d2bedd2e70d917c66ab57a95aa4479f6ca7d6abbfda0.jpg)

![c128ae276902e56e1d819b8b8e46da8ed00dd1d644b344935fc62deabfe2eead.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/images/c128ae276902e56e1d819b8b8e46da8ed00dd1d644b344935fc62deabfe2eead.jpg)

### Tables

![30c714e59c06d79359c3ecd81e8138165debcc711bd7598407867d5819e5f9a8.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/30c714e59c06d79359c3ecd81e8138165debcc711bd7598407867d5819e5f9a8.jpg)

![356ed9b15429d4d4a781e8030d5f1fabef97d341c0e42246b8ad5649fcc0419f.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/356ed9b15429d4d4a781e8030d5f1fabef97d341c0e42246b8ad5649fcc0419f.jpg)

![50290ea00df0b1bcaad97ff3e90ea464651f91e852214fa837f4a29a9fd09ac7.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/50290ea00df0b1bcaad97ff3e90ea464651f91e852214fa837f4a29a9fd09ac7.jpg)

![62a1056a57225a46a093d21ffa721dc1632fab359f33d37d64194a220b05b5e5.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/62a1056a57225a46a093d21ffa721dc1632fab359f33d37d64194a220b05b5e5.jpg)

![735b9210458fc854a5f045169999d327d7cd238317f7b7bd8aa1789d96702517.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/735b9210458fc854a5f045169999d327d7cd238317f7b7bd8aa1789d96702517.jpg)

![aa73c263033240b6a306b5d7896d54aad8c8339f27379fa762af1ec0883101d6.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/aa73c263033240b6a306b5d7896d54aad8c8339f27379fa762af1ec0883101d6.jpg)

![f6ceab7203b2c4266e3010f1c13b614d7b35934d4fe2fd2fa61e1c2def9a5f7a.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/f6ceab7203b2c4266e3010f1c13b614d7b35934d4fe2fd2fa61e1c2def9a5f7a.jpg)

![f7b497d4eab24a4867bdfc1615a1bd81bb1b392cd6c55d342b893454708bffb8.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/f7b497d4eab24a4867bdfc1615a1bd81bb1b392cd6c55d342b893454708bffb8.jpg)

![fb602bc58571255e22bd05c346be37ce87d590e62cc4eeec4361327c2fe7545d.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/fb602bc58571255e22bd05c346be37ce87d590e62cc4eeec4361327c2fe7545d.jpg)

![ff0642f4ab81ec9682bfc81c58cba8bf1afd469cc58fce6320379f2ca9d84b99.jpg](../iclr_results/371_Generating Freeform Endoskeletal Robots/tables/ff0642f4ab81ec9682bfc81c58cba8bf1afd469cc58fce6320379f2ca9d84b99.jpg)

## DARE the Extreme: Revisiting Delta-Parameter Pruning For Fine-Tuned Models


### Images

![11ca789cafbeef07bfc22bf8f8835115e8157fdc2a258fdb38d5738c10cd9cd2.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/11ca789cafbeef07bfc22bf8f8835115e8157fdc2a258fdb38d5738c10cd9cd2.jpg)

![14bfb154b5ba8a3f8523e29a2f9672a1f3bfbef6fef3f5b27d0667d26ed0b968.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/14bfb154b5ba8a3f8523e29a2f9672a1f3bfbef6fef3f5b27d0667d26ed0b968.jpg)

![26132383eef18dea92bd9801c18015ca0682f7eb86988d4a034bce0e9bfbee7c.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/26132383eef18dea92bd9801c18015ca0682f7eb86988d4a034bce0e9bfbee7c.jpg)

![3f8e2ae1a73ebb84fd2ebd182c4a8dcca6d376357baea4865af592f50c175cc9.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/3f8e2ae1a73ebb84fd2ebd182c4a8dcca6d376357baea4865af592f50c175cc9.jpg)

![55e27ca7c55839510e61373e5b58240483d052127e5ed3a66979175768587758.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/55e27ca7c55839510e61373e5b58240483d052127e5ed3a66979175768587758.jpg)

![7f1938346a79d5ce11d79acbbdc81244709d4ddd3b11b7d125009cbff9b735dc.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/7f1938346a79d5ce11d79acbbdc81244709d4ddd3b11b7d125009cbff9b735dc.jpg)

![b15cab18b54a6e5129b2625925f26fde5976880091ff87b700770ac3108fa733.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/b15cab18b54a6e5129b2625925f26fde5976880091ff87b700770ac3108fa733.jpg)

![beb88a2e8bd83192785fdd0df2376be38872cec814b7049b1689827cbd6e0b09.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/beb88a2e8bd83192785fdd0df2376be38872cec814b7049b1689827cbd6e0b09.jpg)

![e2ec7087d7c8e3813556abffb49649770317e116b95f9f6ae67d977309ebe82e.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/e2ec7087d7c8e3813556abffb49649770317e116b95f9f6ae67d977309ebe82e.jpg)

![f722cba0c662fe557ae79317a7b0d3b409af350ab15149c1e2c467bd1323ad49.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/f722cba0c662fe557ae79317a7b0d3b409af350ab15149c1e2c467bd1323ad49.jpg)

![fc77cbee69dbe33c68070e5ee715ef0538e93d0caadd5da2df2cb4c043a1704b.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/fc77cbee69dbe33c68070e5ee715ef0538e93d0caadd5da2df2cb4c043a1704b.jpg)

![ff5913dc0e3fbfa726e85de4bafafcc55573864ef04084b99c9d8ddbb2bfd9ae.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/images/ff5913dc0e3fbfa726e85de4bafafcc55573864ef04084b99c9d8ddbb2bfd9ae.jpg)

### Tables

![00cc9ea2263a76fdbae9c01e61360744e7210e3093e6e614287cf657436627fa.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/00cc9ea2263a76fdbae9c01e61360744e7210e3093e6e614287cf657436627fa.jpg)

![075aa81251e5036ae8ab775ca40b134d76a5e2201c42f9b1e0ac7a6d7e0b0385.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/075aa81251e5036ae8ab775ca40b134d76a5e2201c42f9b1e0ac7a6d7e0b0385.jpg)

![21f21679f944c7e1a02247c6633831ef92bfe2e97ec36c7fdd1305eb45d26154.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/21f21679f944c7e1a02247c6633831ef92bfe2e97ec36c7fdd1305eb45d26154.jpg)

![450088a8428ecbfa14385e1dce0234c08aae41db9514dceaee7e497774fc0ca5.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/450088a8428ecbfa14385e1dce0234c08aae41db9514dceaee7e497774fc0ca5.jpg)

![5faccecb812b87eab6a2dfe5c2af6891552004d3d1ff8e586c650237e2146aab.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/5faccecb812b87eab6a2dfe5c2af6891552004d3d1ff8e586c650237e2146aab.jpg)

![691e1536c1028b4d65c16c4f637c790f0b38c3cd63d6167e727c0a11786d70a5.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/691e1536c1028b4d65c16c4f637c790f0b38c3cd63d6167e727c0a11786d70a5.jpg)

![6e93c873401e059494a03482b3ca8abac3c294fbbf75a3ee8b160b096dacd5a0.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/6e93c873401e059494a03482b3ca8abac3c294fbbf75a3ee8b160b096dacd5a0.jpg)

![8bb46250e932e86409b3635154e327e959a25bc024a3867f34179c55bff07c07.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/8bb46250e932e86409b3635154e327e959a25bc024a3867f34179c55bff07c07.jpg)

![9039a6295bd80fded4034edb3ce5c587b85114be359be3b65a19e964325ba4f7.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/9039a6295bd80fded4034edb3ce5c587b85114be359be3b65a19e964325ba4f7.jpg)

![956f4833c79345ddaea8e53e59b676fb79e7e48949efb24f9ed779ff677dfa60.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/956f4833c79345ddaea8e53e59b676fb79e7e48949efb24f9ed779ff677dfa60.jpg)

![bd8c297845c38d4192b77555ce21cc04354eb13475b136b9fb069b55e4935bc4.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/bd8c297845c38d4192b77555ce21cc04354eb13475b136b9fb069b55e4935bc4.jpg)

![d1421cd1e9f227d104a19ea573206cf19d323351dde1feda81060870e5d039ca.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/d1421cd1e9f227d104a19ea573206cf19d323351dde1feda81060870e5d039ca.jpg)

![e9815261107fb2199f4d01ca49670625378452deb132d52922e5dbf01477360d.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/e9815261107fb2199f4d01ca49670625378452deb132d52922e5dbf01477360d.jpg)

![f82e6cdcec026b8ce7949eaaacbf7fe8be34f2f43003f7913baade8de8fb0b30.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/f82e6cdcec026b8ce7949eaaacbf7fe8be34f2f43003f7913baade8de8fb0b30.jpg)

![f9100b5e19030022162b3edb9b08e74ed5befc80b7b90900fce55e252891d40f.jpg](../iclr_results/372_DARE the Extreme_ Revisiting Delta-Parameter Pruning For Fine-Tuned Models/tables/f9100b5e19030022162b3edb9b08e74ed5befc80b7b90900fce55e252891d40f.jpg)

## Dense Video Object Captioning from Disjoint Supervision


### Images

![446489be469671ec1b226b936ebec01bd79038d96c0f65c092f55140124decc3.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/images/446489be469671ec1b226b936ebec01bd79038d96c0f65c092f55140124decc3.jpg)

![4d271abfcef4ebb982e0b2a5cced622d37e760ab6ad65647d4bce1624e05b221.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/images/4d271abfcef4ebb982e0b2a5cced622d37e760ab6ad65647d4bce1624e05b221.jpg)

![51b6b00bd48e79126c7aa05d4316f8d9fb791d943dc434d005374f7ced55fc6c.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/images/51b6b00bd48e79126c7aa05d4316f8d9fb791d943dc434d005374f7ced55fc6c.jpg)

![a92c8b7b054bf81f5b7f655a17bcc57a632b5bd3af0eeafbe2eb4fab6939c8da.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/images/a92c8b7b054bf81f5b7f655a17bcc57a632b5bd3af0eeafbe2eb4fab6939c8da.jpg)

### Tables

![1dd0b85e8773a7864a00c4a36d002d55e90d1ac9aec321f29524175054ae96e8.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/1dd0b85e8773a7864a00c4a36d002d55e90d1ac9aec321f29524175054ae96e8.jpg)

![2ddf37e1f6b6b24f70a150c157c2fd6d4233673590b92433bea61c324f241610.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/2ddf37e1f6b6b24f70a150c157c2fd6d4233673590b92433bea61c324f241610.jpg)

![329f3d5e3e8e66d5fb669bd802e79a72a846a53627edbfa3e92ba94a87c3d784.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/329f3d5e3e8e66d5fb669bd802e79a72a846a53627edbfa3e92ba94a87c3d784.jpg)

![3f00da407323ba791611641975177c94c030eaf4231bdec32354c3392cd86fe7.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/3f00da407323ba791611641975177c94c030eaf4231bdec32354c3392cd86fe7.jpg)

![4c0725cdabb19edc14c41b1a58c55a1c054f333399da298f2c5d89051e203545.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/4c0725cdabb19edc14c41b1a58c55a1c054f333399da298f2c5d89051e203545.jpg)

![548bb1a179aea7aee3b15a94125c5ce65880f47b392990bdc46b2ae08bf82064.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/548bb1a179aea7aee3b15a94125c5ce65880f47b392990bdc46b2ae08bf82064.jpg)

![58518476c41b4e9501fc1a1ee59e3bc1ca4e2d854bdb8e1b3ed2305de5644f0c.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/58518476c41b4e9501fc1a1ee59e3bc1ca4e2d854bdb8e1b3ed2305de5644f0c.jpg)

![5cd4c03e20c42c2c8d50dea292a28b35c5a9d63f992a54311775ac31eeba9037.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/5cd4c03e20c42c2c8d50dea292a28b35c5a9d63f992a54311775ac31eeba9037.jpg)

![736b0425f66489f835b214b33223bcdc7453c558d8782eab8da3fd8570748733.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/736b0425f66489f835b214b33223bcdc7453c558d8782eab8da3fd8570748733.jpg)

![85e9a2a6ab671c373c7fd1b51a36e0b01cee5a2d17aa6c0d0dd5ccc1f646cfd0.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/85e9a2a6ab671c373c7fd1b51a36e0b01cee5a2d17aa6c0d0dd5ccc1f646cfd0.jpg)

![9eef9c9cbc73b21f1c4320d1d696c72c823b52793caf2f310ad865b55dceac7d.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/9eef9c9cbc73b21f1c4320d1d696c72c823b52793caf2f310ad865b55dceac7d.jpg)

![defd9005f807eb7fe25efcc7ca0222053d4a76d1f805175be093e5a84abc2db9.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/defd9005f807eb7fe25efcc7ca0222053d4a76d1f805175be093e5a84abc2db9.jpg)

![f334f1ec73eaf5889b5fb59d4b0780f0cc4a6dc21fc988e5c92f285077652fdb.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/f334f1ec73eaf5889b5fb59d4b0780f0cc4a6dc21fc988e5c92f285077652fdb.jpg)

![fb043250c7e466a717e624b498c97fe2a3a6070a29b0483f44d3ca51492de63e.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/fb043250c7e466a717e624b498c97fe2a3a6070a29b0483f44d3ca51492de63e.jpg)

![fc2530073af9ac5cad267edc4d18d5c366f4361f7bd8d1725b5a5782215f6921.jpg](../iclr_results/373_Dense Video Object Captioning from Disjoint Supervision/tables/fc2530073af9ac5cad267edc4d18d5c366f4361f7bd8d1725b5a5782215f6921.jpg)

## Targeted Attack Improves Protection against Unauthorized Diffusion Customization

### Images

![35b56681825799f110662d0f963dedc9663f4247d39c76429ca2a1b5921b923c.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/35b56681825799f110662d0f963dedc9663f4247d39c76429ca2a1b5921b923c.jpg)

![363af44915c8d500355784a13dcc8722a4763f2c536a57719647a32a689381d7.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/363af44915c8d500355784a13dcc8722a4763f2c536a57719647a32a689381d7.jpg)

![484c00f7facf5d7c338dfd064d5361fcc4ae05d2e7baeb04cc1924a50fc52fb5.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/484c00f7facf5d7c338dfd064d5361fcc4ae05d2e7baeb04cc1924a50fc52fb5.jpg)

![4e457857a8c3197546c28108315e01b69f202d9e2bfa7e9241618ae4f1f731c0.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/4e457857a8c3197546c28108315e01b69f202d9e2bfa7e9241618ae4f1f731c0.jpg)

![5a2bc2ec93b9a4b2b7e6ef52a3b3639960abd3843edd0c02b1ba4752e7c2ea94.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/5a2bc2ec93b9a4b2b7e6ef52a3b3639960abd3843edd0c02b1ba4752e7c2ea94.jpg)

![6c3ff043e132d0f75397fea252b186dc384b1f17deea31c1781eca7aabecff4f.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/6c3ff043e132d0f75397fea252b186dc384b1f17deea31c1781eca7aabecff4f.jpg)

![6ca84d3f474d38673566723b6069947c1a74821c17811dac0d8a31ec2a411c5e.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/6ca84d3f474d38673566723b6069947c1a74821c17811dac0d8a31ec2a411c5e.jpg)

![8290773d6545c8d3c3d72bed9abe3de290f3341c8712bb32a17aed6c0277b3c3.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/8290773d6545c8d3c3d72bed9abe3de290f3341c8712bb32a17aed6c0277b3c3.jpg)

![8631c480e607da53688f776223428f4bd9e65d940d2dc4c40cf1c18fbf93291d.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/8631c480e607da53688f776223428f4bd9e65d940d2dc4c40cf1c18fbf93291d.jpg)

![aa63eb7edcf7c4f7273db09ba1360acfbc111af65b2e04f05f6d98e3dec05053.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/aa63eb7edcf7c4f7273db09ba1360acfbc111af65b2e04f05f6d98e3dec05053.jpg)

![b42e6789fe9f10e18ec39bc94b0cbc6b77b7d43a76f93ff0bc3a578193a8db98.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/b42e6789fe9f10e18ec39bc94b0cbc6b77b7d43a76f93ff0bc3a578193a8db98.jpg)

![b6d65410a3e87919c757f718aee85a59b01dcbf4d818d331b5cc1119e1f1b22e.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/b6d65410a3e87919c757f718aee85a59b01dcbf4d818d331b5cc1119e1f1b22e.jpg)

![bc194070cf5919ac1051e23620504fa383d6a6af03dff1c3a65d59432788f6c6.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/bc194070cf5919ac1051e23620504fa383d6a6af03dff1c3a65d59432788f6c6.jpg)

![c192c1a2e43b5e31e0396a55b0f0a4050afb4cad70ac3e9b9c648894dcef8e17.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/c192c1a2e43b5e31e0396a55b0f0a4050afb4cad70ac3e9b9c648894dcef8e17.jpg)

![cfad365775d7dbb1b4455e3b31671303880cfb3f13297748434d28f8cba44f7c.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/cfad365775d7dbb1b4455e3b31671303880cfb3f13297748434d28f8cba44f7c.jpg)

![d592b333a073a157732c4ebbf6862fb6f3f94a0c78c9550812fb7e4271b96a09.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/d592b333a073a157732c4ebbf6862fb6f3f94a0c78c9550812fb7e4271b96a09.jpg)

![d915eaa48e271eae9b0bddd66aa3df1e39792ef371f5ae255d80e83f5205fe87.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/d915eaa48e271eae9b0bddd66aa3df1e39792ef371f5ae255d80e83f5205fe87.jpg)

![f9f56ea25d449c02f90a35417e1257b044bc0a7364aeb94beeae618ee751e374.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/f9f56ea25d449c02f90a35417e1257b044bc0a7364aeb94beeae618ee751e374.jpg)

![fa2913f7ad3d9d056e9196aa840d95bff099baf91994ed9f29d78d1164e34dfa.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/images/fa2913f7ad3d9d056e9196aa840d95bff099baf91994ed9f29d78d1164e34dfa.jpg)

### Tables

![080053211312c05618289eeb40b9502b0f9230562bb64423a0bb67e7dc4714e2.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/tables/080053211312c05618289eeb40b9502b0f9230562bb64423a0bb67e7dc4714e2.jpg)

![09afb51ec37e14b997c807801b6b41e4e974b87488bb71b23632ce4edc117f53.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/tables/09afb51ec37e14b997c807801b6b41e4e974b87488bb71b23632ce4edc117f53.jpg)

![1f6ed3cf9cf79e971a2484c8eef0aadc452d07a4ef4017016a2153f48335ffd9.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/tables/1f6ed3cf9cf79e971a2484c8eef0aadc452d07a4ef4017016a2153f48335ffd9.jpg)

![54b7f8bb522ea9026cc96cf8629c8f46ba9aee45563bb5555bc3adeaacc9a55d.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/tables/54b7f8bb522ea9026cc96cf8629c8f46ba9aee45563bb5555bc3adeaacc9a55d.jpg)

![60b797808a833e5db939e3bc0b9aac5e0f59e8b89dc28140c598d8f3c4eb6de7.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/tables/60b797808a833e5db939e3bc0b9aac5e0f59e8b89dc28140c598d8f3c4eb6de7.jpg)

![b0774e93b0026e3083784dc6ce478dcd4637725dbaecbd13a4b4b98f2d555c4d.jpg](../iclr_results/374_Targeted Attack Improves Protection against Unauthorized Diffusion Customization/tables/b0774e93b0026e3083784dc6ce478dcd4637725dbaecbd13a4b4b98f2d555c4d.jpg)
