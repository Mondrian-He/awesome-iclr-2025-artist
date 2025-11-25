# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 29 of 100

## 目录 (Table of Contents)

1. [DICE: End-to-end Deformation Capture of Hand-Face Interactions from a Single Image](#DICE-End-to-end-Deformation-Capture-of-Hand-Face-Interactions-from-a-Single-Image)
2. [High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers.](#High-quality-Text-to-3D-Character-Generation-with-SparseCubes-and-Sparse-Transformers)
3. [Broadening Target Distributions for Accelerated Diffusion Models via a Novel Analysis Approach](#Broadening-Target-Distributions-for-Accelerated-Diffusion-Models-via-a-Novel-Analysis-Approach)
4. [Warm Diffusion: Recipe for Blur-Noise Mixture Diffusion Models](#Warm-Diffusion-Recipe-for-Blur-Noise-Mixture-Diffusion-Models)
5. [CircuitFusion: Multimodal Circuit Representation Learning for Agile Chip Design](#CircuitFusion-Multimodal-Circuit-Representation-Learning-for-Agile-Chip-Design)
6. [Lossy Compression with Pretrained Diffusion Models](#Lossy-Compression-with-Pretrained-Diffusion-Models)
7. [MoLEx: Mixture of Layer Experts for Fine-tuning with Sparse Upcycling](#MoLEx-Mixture-of-Layer-Experts-for-Fine-tuning-with-Sparse-Upcycling)
8. [PaLD: Detection of Text Partially Written by Large Language Models](#PaLD-Detection-of-Text-Partially-Written-by-Large-Language-Models)
9. [Graph Transformers Dream of Electric Flow](#Graph-Transformers-Dream-of-Electric-Flow)
10. [$InterLCM$: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration](#InterLCM-Low-Quality-Images-as-Intermediate-States-of-Latent-Consistency-Models-for-Effective-Blind-Face-Restoration)
11. [Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology](#Towards-Realistic-UAV-Vision-Language-Navigation-Platform-Benchmark-and-Methodology)
12. [Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers](#Mechanism-and-Emergence-of-Stacked-Attention-Heads-in-Multi-Layer-Transformers)
13. [PhyloLM: Inferring the Phylogeny of Large Language Models and Predicting their Performances in Benchmarks](#PhyloLM-Inferring-the-Phylogeny-of-Large-Language-Models-and-Predicting-their-Performances-in-Benchmarks)
14. [Reasoning of Large Language Models over Knowledge Graphs with Super-Relations](#Reasoning-of-Large-Language-Models-over-Knowledge-Graphs-with-Super-Relations)
15. [Biologically Plausible Brain Graph Transformer](#Biologically-Plausible-Brain-Graph-Transformer)
16. [Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning](#Multimodal-Large-Language-Models-for-Inverse-Molecular-Design-with-Retrosynthetic-Planning)
17. [A Causal Lens for Learning Long-term Fair Policies](#A-Causal-Lens-for-Learning-Long-term-Fair-Policies)
18. [Smoothing the Shift: Towards Stable Test-Time Adaptation under Complex Multimodal Noises](#Smoothing-the-Shift-Towards-Stable-Test-Time-Adaptation-under-Complex-Multimodal-Noises)
19. [Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Perception](#Aligning-Generative-Denoising-with-Discriminative-Objectives-Unleashes-Diffusion-for-Visual-Perception)
20. [N-ForGOT: Towards Not-forgetting and Generalization of Open Temporal Graph Learning](#N-ForGOT-Towards-Not-forgetting-and-Generalization-of-Open-Temporal-Graph-Learning)
21. [Systematic Outliers in Large Language Models](#Systematic-Outliers-in-Large-Language-Models)
22. [Protecting against simultaneous data poisoning attacks](#Protecting-against-simultaneous-data-poisoning-attacks)
23. [SEMDICE: Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation](#SEMDICE-Off-policy-State-Entropy-Maximization-via-Stationary-Distribution-Correction-Estimation)
24. [Generating Likely Counterfactuals Using Sum-Product Networks](#Generating-Likely-Counterfactuals-Using-Sum-Product-Networks)
25. [Metric-Driven Attributions for Vision Transformers](#Metric-Driven-Attributions-for-Vision-Transformers)
26. [Greener GRASS: Enhancing GNNs with Encoding, Rewiring, and Attention](#Greener-GRASS-Enhancing-GNNs-with-Encoding-Rewiring-and-Attention)
27. [TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting](#TS-LIF-A-Temporal-Segment-Spiking-Neuron-Network-for-Time-Series-Forecasting)
28. [Exposure Bracketing Is All You Need For A High-Quality Image](#Exposure-Bracketing-Is-All-You-Need-For-A-High-Quality-Image)
29. [Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving](#Semi-Supervised-Vision-Centric-3D-Occupancy-World-Model-for-Autonomous-Driving)
30. [Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions](#Enhanced-Diffusion-Sampling-via-Extrapolation-with-Multiple-ODE-Solutions)
31. [A Statistical Framework for Ranking LLM-based Chatbots](#A-Statistical-Framework-for-Ranking-LLM-based-Chatbots)
32. [OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting](#OSTQuant-Refining-Large-Language-Model-Quantization-with-Orthogonal-and-Scaling-Transformations-for-Better-Distribution-Fitting)
33. [TULIP: Token-length Upgraded CLIP](#TULIP-Token-length-Upgraded-CLIP)
34. [Scaling Stick-Breaking Attention: An Efficient Implementation and In-depth Study](#Scaling-Stick-Breaking-Attention-An-Efficient-Implementation-and-In-depth-Study)
35. [Gated Delta Networks: Improving Mamba2 with Delta Rule](#Gated-Delta-Networks-Improving-Mamba2-with-Delta-Rule)
36. [Global Well-posedness and Convergence Analysis of Score-based Generative Models via Sharp Lipschitz Estimates](#Global-Well-posedness-and-Convergence-Analysis-of-Score-based-Generative-Models-via-Sharp-Lipschitz-Estimates)

---


## DICE: End-to-end Deformation Capture of Hand-Face Interactions from a Single Image

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

## DICE: End-to-end Deformation Capture of Hand-Face Interactions from a Single Image


### Images

![1e3cf9993955ebec232de699de88c1ad7d33124d4dc79a9f511d9f99045c7217.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/1e3cf9993955ebec232de699de88c1ad7d33124d4dc79a9f511d9f99045c7217.jpg)

![211d4400672e47821d01ffb55a6b88c99f9989415aedceb445ab6af002048eca.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/211d4400672e47821d01ffb55a6b88c99f9989415aedceb445ab6af002048eca.jpg)

![2781e042871f64bf594fefe712812d340845bc0b9d5afcdaee733c9c6ed417db.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/2781e042871f64bf594fefe712812d340845bc0b9d5afcdaee733c9c6ed417db.jpg)

![69f977c0839f1fd6d841b6226077f3d0772f1f032684a9bc36a273215a9f13d6.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/69f977c0839f1fd6d841b6226077f3d0772f1f032684a9bc36a273215a9f13d6.jpg)

![82e4113f8f1fea567ca96f31c8ad702d3cdc9c30872f0afdf487c09c4351746b.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/82e4113f8f1fea567ca96f31c8ad702d3cdc9c30872f0afdf487c09c4351746b.jpg)

![864f9f5d53e5924504b028042f4675f922253f6caa780b14d9e5675083e3e703.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/864f9f5d53e5924504b028042f4675f922253f6caa780b14d9e5675083e3e703.jpg)

![b09ec5a801abbed6615ca3f3981e5c446f2871a1cb69bbbbeef6dc008c922bbf.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/b09ec5a801abbed6615ca3f3981e5c446f2871a1cb69bbbbeef6dc008c922bbf.jpg)

![e7d132c0a917721d7d0a8e68aaf66088d4af21530639b5524e51544f5dab9b2a.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/e7d132c0a917721d7d0a8e68aaf66088d4af21530639b5524e51544f5dab9b2a.jpg)

![f286fd7d572d153aa1b5e7d1459bf267ef6c164dc3e0420bddde70e9212e0c90.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/images/f286fd7d572d153aa1b5e7d1459bf267ef6c164dc3e0420bddde70e9212e0c90.jpg)

### Tables

![01e1d10976da0aa308f88448989dc8837a945337e276921997c6d07e8572f7cf.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/tables/01e1d10976da0aa308f88448989dc8837a945337e276921997c6d07e8572f7cf.jpg)

![2b1cd683de74acfc9cba194529c88f405f85127cac6b172ac2a591a161deeb26.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/tables/2b1cd683de74acfc9cba194529c88f405f85127cac6b172ac2a591a161deeb26.jpg)

![3e6f001aaf149d0e32cbdcaca20d5086f8655889d6fb290f157c2ce13cc1d19b.jpg](../iclr_results/1057_DICE_ End-to-end Deformation Capture of Hand-Face Interactions from a Single Image/tables/3e6f001aaf149d0e32cbdcaca20d5086f8655889d6fb290f157c2ce13cc1d19b.jpg)

## High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers.


### Images

![09518b8427c90500f5b7339243c6e28984ceb51ddc9c44d0b52f179bea5befd8.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/09518b8427c90500f5b7339243c6e28984ceb51ddc9c44d0b52f179bea5befd8.jpg)

![178aa9600e841ade39a6c7ccda8468d42ef4c55b2ecef6c9cd15dd16dbd264e8.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/178aa9600e841ade39a6c7ccda8468d42ef4c55b2ecef6c9cd15dd16dbd264e8.jpg)

![1ec2f11e42ebd57432a3e99e3f1a1cbc82deb67927d0bb5c77c144c38a234feb.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/1ec2f11e42ebd57432a3e99e3f1a1cbc82deb67927d0bb5c77c144c38a234feb.jpg)

![2418276771b8a5114428401895726aa690645475aa2545497ba7bab36409eeeb.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/2418276771b8a5114428401895726aa690645475aa2545497ba7bab36409eeeb.jpg)

![4899f059160f6b17011cc306ce348b4fd1d09ab07b9cb64fa2a6d17439074728.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/4899f059160f6b17011cc306ce348b4fd1d09ab07b9cb64fa2a6d17439074728.jpg)

![507ec9918830b812b58b17856483b681b86baf516de470641a04186c2e51d0fe.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/507ec9918830b812b58b17856483b681b86baf516de470641a04186c2e51d0fe.jpg)

![521e655c9f830426863bc24a2ed7723236318bd073072b6e9d650030cefd79d8.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/521e655c9f830426863bc24a2ed7723236318bd073072b6e9d650030cefd79d8.jpg)

![5a4481205199cd7ee7d62ec91faf5b868002891959e5c66d0c3d2f5f94d133ae.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/5a4481205199cd7ee7d62ec91faf5b868002891959e5c66d0c3d2f5f94d133ae.jpg)

![74cca7a21dfdb77ecfc7b5d4181f028f3a0a08371714a12bf6d78dada81005b0.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/74cca7a21dfdb77ecfc7b5d4181f028f3a0a08371714a12bf6d78dada81005b0.jpg)

![99acf5d656cde5f662696c2f973e791d74bb8c368aef2cf9782287b7b57b5806.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/99acf5d656cde5f662696c2f973e791d74bb8c368aef2cf9782287b7b57b5806.jpg)

![c228313b77f4c304390079902ea9b7abdd7f7b76075f1c014acde66748809c41.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/c228313b77f4c304390079902ea9b7abdd7f7b76075f1c014acde66748809c41.jpg)

![c73fb1dc28ab023267bf6aff2b75d96370c93e31d03fee0e0f38c8fe2ab0c08b.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/c73fb1dc28ab023267bf6aff2b75d96370c93e31d03fee0e0f38c8fe2ab0c08b.jpg)

![e3d0056da3adc95f2d1aeb8aa34a5c59542bc67ae00a35561263f11a6966d372.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/e3d0056da3adc95f2d1aeb8aa34a5c59542bc67ae00a35561263f11a6966d372.jpg)

![f2db79178530276bf71f59129c48bb971beb57d0fed45d4a4b19a247c37414b4.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./images/f2db79178530276bf71f59129c48bb971beb57d0fed45d4a4b19a247c37414b4.jpg)

### Tables

![17d7053e002145bbd9b29dd3b3840aca7a0f272dec22f8069f0cb1edf4c5538f.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./tables/17d7053e002145bbd9b29dd3b3840aca7a0f272dec22f8069f0cb1edf4c5538f.jpg)

![61788b3f71bef8f28a5dec33ae248a98430f45d1c836639be4aca606fa63f8ce.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./tables/61788b3f71bef8f28a5dec33ae248a98430f45d1c836639be4aca606fa63f8ce.jpg)

![b46e8f39e4b0f4d78534e530e30e4b1235e1dc5e7a27cddf5326e611d4f685d1.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./tables/b46e8f39e4b0f4d78534e530e30e4b1235e1dc5e7a27cddf5326e611d4f685d1.jpg)

![c4306acaf3e8f2d9f112aed9ebe853c3edb5bc566986021504bac79cc01677d9.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./tables/c4306acaf3e8f2d9f112aed9ebe853c3edb5bc566986021504bac79cc01677d9.jpg)

![cacb662f5524b19fe0d59dd0700ad0ca200375ac14c2efa8008a479f700fa668.jpg](../iclr_results/1058_High-quality Text-to-3D Character Generation with SparseCubes and Sparse Transformers./tables/cacb662f5524b19fe0d59dd0700ad0ca200375ac14c2efa8008a479f700fa668.jpg)

## Broadening Target Distributions for Accelerated Diffusion Models via a Novel Analysis Approach


### Images

![9246dd7f6f7a488a2bc195c7279d9f36d8f424f1c2ca9c2e2eea9eb373a91566.jpg](../iclr_results/1059_Broadening Target Distributions for Accelerated Diffusion Models via a Novel Analysis Approach/images/9246dd7f6f7a488a2bc195c7279d9f36d8f424f1c2ca9c2e2eea9eb373a91566.jpg)

### Tables

![6aaeeea95221631bcca55596c2793841c00a3957263ee3f52f2365158260e601.jpg](../iclr_results/1059_Broadening Target Distributions for Accelerated Diffusion Models via a Novel Analysis Approach/tables/6aaeeea95221631bcca55596c2793841c00a3957263ee3f52f2365158260e601.jpg)

## Warm Diffusion: Recipe for Blur-Noise Mixture Diffusion Models


### Images

![0de7bf9723afda752498a2dfa3bfd589935fdcfd7b5552fa11d0bf0f78554fd6.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/0de7bf9723afda752498a2dfa3bfd589935fdcfd7b5552fa11d0bf0f78554fd6.jpg)

![387d319adb4a76015b5ce6af91488fb6328c1299a279fb3a0d1f63444415699d.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/387d319adb4a76015b5ce6af91488fb6328c1299a279fb3a0d1f63444415699d.jpg)

![38a4704353fb3f5e5151858a1011f2e3e73733a6e100772393e719f5c0d9891f.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/38a4704353fb3f5e5151858a1011f2e3e73733a6e100772393e719f5c0d9891f.jpg)

![42d153d49b2250472c0c01511276a321cc3a6bad6de21c432bb7f74e7b187c03.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/42d153d49b2250472c0c01511276a321cc3a6bad6de21c432bb7f74e7b187c03.jpg)

![5a1fb0bce19e4bf38929e5b8ccc17025b6e0d7f7ec49d12549402aebf7c1f2cb.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/5a1fb0bce19e4bf38929e5b8ccc17025b6e0d7f7ec49d12549402aebf7c1f2cb.jpg)

![8293b27c4056c4789166dadc4264611d9ea41e8bf921dc6a991f30ed1bf41070.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/8293b27c4056c4789166dadc4264611d9ea41e8bf921dc6a991f30ed1bf41070.jpg)

![901fe6de279ad4ababd66996fc31e25df0ba57a6b3896e8ba7ead6047ce8744a.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/901fe6de279ad4ababd66996fc31e25df0ba57a6b3896e8ba7ead6047ce8744a.jpg)

![a16b717ab0dc20a776bdb5ccc560243d1f338a2dc9925f86707d00f719957d66.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/a16b717ab0dc20a776bdb5ccc560243d1f338a2dc9925f86707d00f719957d66.jpg)

![a1c86f3688fc783c6858f820584a5047103f4529fe269db817bf6c5996994928.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/a1c86f3688fc783c6858f820584a5047103f4529fe269db817bf6c5996994928.jpg)

![b208981106a5c198facf2b58b1d93991577e2c074da2aaca0b8161331b188206.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/b208981106a5c198facf2b58b1d93991577e2c074da2aaca0b8161331b188206.jpg)

![d517ea140d3e462a1fab9792757561847df68425a70f1c7465089883ab690b29.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/d517ea140d3e462a1fab9792757561847df68425a70f1c7465089883ab690b29.jpg)

![f0e75b57d177cf02f9c6f65869a7f859afea0cd3f59a805059ca49f8cd75d59f.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/images/f0e75b57d177cf02f9c6f65869a7f859afea0cd3f59a805059ca49f8cd75d59f.jpg)

### Tables

![4cf0207ae5c2118d2e65c014986d6a8622e0c77f7aefd5ef738d6e5a8d8fd5e6.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/tables/4cf0207ae5c2118d2e65c014986d6a8622e0c77f7aefd5ef738d6e5a8d8fd5e6.jpg)

![9bf837873c2967a848db13e5471c807c1cad943e90409474bf5c0af2c88f6ea3.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/tables/9bf837873c2967a848db13e5471c807c1cad943e90409474bf5c0af2c88f6ea3.jpg)

![b213eed90214d935e5c4b708bfe057e669e0fc205232c605afe2b174e5672235.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/tables/b213eed90214d935e5c4b708bfe057e669e0fc205232c605afe2b174e5672235.jpg)

![b378259c763378831b2cb8ae6c8b0f09fbb84c64c861356bf96053d1ba5ec017.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/tables/b378259c763378831b2cb8ae6c8b0f09fbb84c64c861356bf96053d1ba5ec017.jpg)

![b8adb89b9f322dacc7bd5e292cced6d95f5ce87a34325789278c05c01e049954.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/tables/b8adb89b9f322dacc7bd5e292cced6d95f5ce87a34325789278c05c01e049954.jpg)

![ba908e3e0a0ee589b2979d92a9bd2ab34363f79c6e5168bf4e9046bc888055c2.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/tables/ba908e3e0a0ee589b2979d92a9bd2ab34363f79c6e5168bf4e9046bc888055c2.jpg)

![d5345fa6a20fdc4c6d41ce8a162e02817bbb94d0242ed11401c37783dc81a952.jpg](../iclr_results/1060_Warm Diffusion_ Recipe for Blur-Noise Mixture Diffusion Models/tables/d5345fa6a20fdc4c6d41ce8a162e02817bbb94d0242ed11401c37783dc81a952.jpg)

## CircuitFusion: Multimodal Circuit Representation Learning for Agile Chip Design


### Images

![20bdce3ad4267b02aefdd76be4e1d4105df804b978210fcc08ddcff2d3203248.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/20bdce3ad4267b02aefdd76be4e1d4105df804b978210fcc08ddcff2d3203248.jpg)

![29f33ba08df54e097c45e79e5f4a57817454f251beae6df63c0d453fed00a6b3.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/29f33ba08df54e097c45e79e5f4a57817454f251beae6df63c0d453fed00a6b3.jpg)

![2b4f5f3722d19c18dc49518a82caff854aef824022048b97729230c2109eb1b3.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/2b4f5f3722d19c18dc49518a82caff854aef824022048b97729230c2109eb1b3.jpg)

![31f5d851241afa4b504b950736212b0d213a0f7efa4ccf147b45349bce29c451.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/31f5d851241afa4b504b950736212b0d213a0f7efa4ccf147b45349bce29c451.jpg)

![84dbe3b35d5f674288b1aaca24167134d54117a4f64f5ae142c690bc4e855eca.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/84dbe3b35d5f674288b1aaca24167134d54117a4f64f5ae142c690bc4e855eca.jpg)

![c7031963d9ae924bdcc64e9374c4805d3fb401d5b4dc64e13782406718837995.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/c7031963d9ae924bdcc64e9374c4805d3fb401d5b4dc64e13782406718837995.jpg)

![da6deb3ed850a82c8b4f446b901363259e18228323e3bd87855e4e8a64d2d8e9.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/da6deb3ed850a82c8b4f446b901363259e18228323e3bd87855e4e8a64d2d8e9.jpg)

![e5f179da1f4b251630df4b9a19fc0e55fc37776b09110c942db0c1f86c93801b.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/e5f179da1f4b251630df4b9a19fc0e55fc37776b09110c942db0c1f86c93801b.jpg)

![ec9638a78df298c1f6caf2baeffbe8b13faa1e255064b6c3a44c372cc3ec5769.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/ec9638a78df298c1f6caf2baeffbe8b13faa1e255064b6c3a44c372cc3ec5769.jpg)

![feab4eccb2c75e0358239581fcacb664a74e9e6d5da5c33257d039c5d67a9369.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/images/feab4eccb2c75e0358239581fcacb664a74e9e6d5da5c33257d039c5d67a9369.jpg)

### Tables

![2d91ec68131f1ad92d48218a7916058b150892abba39b07ffb31188336a1a65d.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/2d91ec68131f1ad92d48218a7916058b150892abba39b07ffb31188336a1a65d.jpg)

![469f7da11926572ce377456a61bbd284ba1f9f86bd403d63c4c5d101978173ac.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/469f7da11926572ce377456a61bbd284ba1f9f86bd403d63c4c5d101978173ac.jpg)

![50650a60af3db17ba0545a13d6640eb624eac4efec67dddad34a8960a1a748d7.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/50650a60af3db17ba0545a13d6640eb624eac4efec67dddad34a8960a1a748d7.jpg)

![5fd9eb309b5621f165b1eb1d81108dbf8deb970a7837425ee639d6d45876a8ef.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/5fd9eb309b5621f165b1eb1d81108dbf8deb970a7837425ee639d6d45876a8ef.jpg)

![6f73fc2eb2b27a454946da27b90988b612d8bd11ea2d9203065a45e038d15b00.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/6f73fc2eb2b27a454946da27b90988b612d8bd11ea2d9203065a45e038d15b00.jpg)

![709ac070289b6eea3ff0d4f3c7711b83ccb5e08fe51388723dc847e900b18444.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/709ac070289b6eea3ff0d4f3c7711b83ccb5e08fe51388723dc847e900b18444.jpg)

![9b6c0c351703d37feb5cb8142c6408ee3b930fad9b117282e0bc65f8e40570dc.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/9b6c0c351703d37feb5cb8142c6408ee3b930fad9b117282e0bc65f8e40570dc.jpg)

![b319c06146cd5c638e72d77f8c00f70999536f2b08f383133784d01488932fee.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/b319c06146cd5c638e72d77f8c00f70999536f2b08f383133784d01488932fee.jpg)

![b676cf3eb99a3ef43babfaa171bc22e5002cd4e95dbc63771ad5410fc33cb942.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/b676cf3eb99a3ef43babfaa171bc22e5002cd4e95dbc63771ad5410fc33cb942.jpg)

![bc658fce2254a84352f8010fb9359f8de93bde20c9bc14876932e0f17a0220f9.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/bc658fce2254a84352f8010fb9359f8de93bde20c9bc14876932e0f17a0220f9.jpg)

![c1009590f26b1efccfb5f18d9becd65f7fcc272af5e88d49682182ae5d513db0.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/c1009590f26b1efccfb5f18d9becd65f7fcc272af5e88d49682182ae5d513db0.jpg)

![dd466e14983983e0e0f1a00726b454bba76d9da0e6281f2aff79bd6195fe7deb.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/dd466e14983983e0e0f1a00726b454bba76d9da0e6281f2aff79bd6195fe7deb.jpg)

![efc21b5b55b89711208e58da08cb7e56551ecfa6e2e9fa6fbdfdd9bd95d2503b.jpg](../iclr_results/1061_CircuitFusion_ Multimodal Circuit Representation Learning for Agile Chip Design/tables/efc21b5b55b89711208e58da08cb7e56551ecfa6e2e9fa6fbdfdd9bd95d2503b.jpg)

## Lossy Compression with Pretrained Diffusion Models


### Images

![09ea8cfb18e1533ad623225bde49c34da4d9ff697c87377bba9212a1f1e347fe.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/images/09ea8cfb18e1533ad623225bde49c34da4d9ff697c87377bba9212a1f1e347fe.jpg)

![18e02a58703bbc247a6cfa1e9cb719958f0ed55c9a365bcf0ec6653fa67c0f2d.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/images/18e02a58703bbc247a6cfa1e9cb719958f0ed55c9a365bcf0ec6653fa67c0f2d.jpg)

![1c87cec51db5736cb0ca71e8f21e7bbf90314f9644b35abfc666818ea2ea6c87.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/images/1c87cec51db5736cb0ca71e8f21e7bbf90314f9644b35abfc666818ea2ea6c87.jpg)

![a52ad33e1d8f00787df2cc1461ba6dfd73ba37ac9f91ceaa1a95f3771878dba6.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/images/a52ad33e1d8f00787df2cc1461ba6dfd73ba37ac9f91ceaa1a95f3771878dba6.jpg)

![c5e29f6da1dbc36c7182890225802aa32d7d19162057cd1c2bbb1da23fd97085.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/images/c5e29f6da1dbc36c7182890225802aa32d7d19162057cd1c2bbb1da23fd97085.jpg)

![f5b5e2fa0ce26784dcc339f0e042f9d98b8a11c9a271d43e10168d2e76429781.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/images/f5b5e2fa0ce26784dcc339f0e042f9d98b8a11c9a271d43e10168d2e76429781.jpg)

![f88d574ab8d519365067fa19330154ca9579751bede8fdd21826d928619d2203.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/images/f88d574ab8d519365067fa19330154ca9579751bede8fdd21826d928619d2203.jpg)

### Tables

![28bf0d7d3162301f292667ff1008dfcc1580dad6721d88b584dd707b44b0890d.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/tables/28bf0d7d3162301f292667ff1008dfcc1580dad6721d88b584dd707b44b0890d.jpg)

![ff459ab5f1915060c3a4c4d6ed90071f38fb2034e0d079fa0cfd0c91e5fd504b.jpg](../iclr_results/1063_Lossy Compression with Pretrained Diffusion Models/tables/ff459ab5f1915060c3a4c4d6ed90071f38fb2034e0d079fa0cfd0c91e5fd504b.jpg)

## MoLEx: Mixture of Layer Experts for Fine-tuning with Sparse Upcycling


### Images

![303ba88b2d86e29c61ea4f668cf38c6f9c7d16d39a440819929eda133345d972.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/images/303ba88b2d86e29c61ea4f668cf38c6f9c7d16d39a440819929eda133345d972.jpg)

![58738910843f620ac3131a3945b9f641c2a91fb7dffb33778f74b1a65ac1e04f.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/images/58738910843f620ac3131a3945b9f641c2a91fb7dffb33778f74b1a65ac1e04f.jpg)

### Tables

![0599d988781af6f0a2f3bfe4a15216a8f122343abcbd641996b8990256212d4a.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/0599d988781af6f0a2f3bfe4a15216a8f122343abcbd641996b8990256212d4a.jpg)

![069f937ee758c6dde1730fa6b8a2f73dd16c7f2534f236710ee133f249e30798.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/069f937ee758c6dde1730fa6b8a2f73dd16c7f2534f236710ee133f249e30798.jpg)

![08ee9e20142be38cffb9376665e8bf131867a843939161cb96473686099206bf.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/08ee9e20142be38cffb9376665e8bf131867a843939161cb96473686099206bf.jpg)

![122cb9b7678744af5fab9e3ebdaf1cde7a66848eaec7526cb5240aec572d8c95.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/122cb9b7678744af5fab9e3ebdaf1cde7a66848eaec7526cb5240aec572d8c95.jpg)

![1582845dbca80fa1a12ad3b1ee61058e410a093f27366c2df9ce26394300b3c8.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/1582845dbca80fa1a12ad3b1ee61058e410a093f27366c2df9ce26394300b3c8.jpg)

![455cdeeede5f3731f2fa3d76663fcba80fb22de074d437303b535ca5632d3342.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/455cdeeede5f3731f2fa3d76663fcba80fb22de074d437303b535ca5632d3342.jpg)

![61a0b6e97022e00c6766b0a8c21bce3984380a48ad8925e3bb13b7593e509947.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/61a0b6e97022e00c6766b0a8c21bce3984380a48ad8925e3bb13b7593e509947.jpg)

![80b5e97bc6065c1d7f38c60927e887e4225477b623a6df8cebd236ae59f03187.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/80b5e97bc6065c1d7f38c60927e887e4225477b623a6df8cebd236ae59f03187.jpg)

![8bb9d266a8c58666fd222aa94a4acf7f5e3e11c8e319a131defaea232f9134a1.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/8bb9d266a8c58666fd222aa94a4acf7f5e3e11c8e319a131defaea232f9134a1.jpg)

![b0c93a9ae1f2c067081edb8d24d41b4d44b49d5232c21dba2de025befa75c9b5.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/b0c93a9ae1f2c067081edb8d24d41b4d44b49d5232c21dba2de025befa75c9b5.jpg)

![b2db11662a96c090b49ba96a41f9b2c1b3936d7f297f720bd032fe8aa4cb77ed.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/b2db11662a96c090b49ba96a41f9b2c1b3936d7f297f720bd032fe8aa4cb77ed.jpg)

![bfee3e2e54fbe98a80a4ad3493fbfcb16f37076e1907074b25abb8080ccea54f.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/bfee3e2e54fbe98a80a4ad3493fbfcb16f37076e1907074b25abb8080ccea54f.jpg)

![ee2fb887a645c18250910ab3db52a8425e0cbf3a62f6594a0ae988b46cd2b05a.jpg](../iclr_results/1064_MoLEx_ Mixture of Layer Experts for Fine-tuning with Sparse Upcycling/tables/ee2fb887a645c18250910ab3db52a8425e0cbf3a62f6594a0ae988b46cd2b05a.jpg)

## PaLD: Detection of Text Partially Written by Large Language Models


### Images

![142654bc16d8d2242aaf44c8fc24fa932625e383274e1ffcd573d1822ae0d81f.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/images/142654bc16d8d2242aaf44c8fc24fa932625e383274e1ffcd573d1822ae0d81f.jpg)

![44fd5c524db4d07ad5bd17c14886c18bb74379e9502abb66f28792f111040c02.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/images/44fd5c524db4d07ad5bd17c14886c18bb74379e9502abb66f28792f111040c02.jpg)

![73d4077b2a08d65c6a6249eb8817c765ac0f2687ee0365fcb18e9c79efc843c7.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/images/73d4077b2a08d65c6a6249eb8817c765ac0f2687ee0365fcb18e9c79efc843c7.jpg)

![80bc5b9c76c432344bcf7a906f7da9787dc3211b036f906a836035dc9b4966f4.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/images/80bc5b9c76c432344bcf7a906f7da9787dc3211b036f906a836035dc9b4966f4.jpg)

![a14912518d38b10c7048280a4abdf4564bbff8782672aad8dcb371c669aa4a0c.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/images/a14912518d38b10c7048280a4abdf4564bbff8782672aad8dcb371c669aa4a0c.jpg)

![ca1bc5b26abea75ddf8d09e838f3e514580831c03eb6cd639a73c6422272a62e.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/images/ca1bc5b26abea75ddf8d09e838f3e514580831c03eb6cd639a73c6422272a62e.jpg)

![e0e25f6fc13847d6c972a64fc40bb058586e6afd872e0138bc2ff16d371245a1.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/images/e0e25f6fc13847d6c972a64fc40bb058586e6afd872e0138bc2ff16d371245a1.jpg)

### Tables

![2c9278f1cb90dbe177e6067e6431e04584f7652dcdc1167855875ee1745d7194.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/2c9278f1cb90dbe177e6067e6431e04584f7652dcdc1167855875ee1745d7194.jpg)

![302da561de8454c58a60badf0c006adbb626cb99553e9562be8e94231b409bf6.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/302da561de8454c58a60badf0c006adbb626cb99553e9562be8e94231b409bf6.jpg)

![449f9c347ae60d1f6810c906e8e5f1e9dd3338b53dfc3a17951bd3617438017a.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/449f9c347ae60d1f6810c906e8e5f1e9dd3338b53dfc3a17951bd3617438017a.jpg)

![4eddecb771b55c38c15cb03b66c277b08c18020344bf9c424adbd5f45a076e1e.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/4eddecb771b55c38c15cb03b66c277b08c18020344bf9c424adbd5f45a076e1e.jpg)

![5702c498d182d547f8c6966c2265570de9ad542e029c3267b055f94d96650f17.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/5702c498d182d547f8c6966c2265570de9ad542e029c3267b055f94d96650f17.jpg)

![5b8a7eca1baf74d80aafb6c8413972ccc675ebe07746ffb6fff4e9e14de34b47.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/5b8a7eca1baf74d80aafb6c8413972ccc675ebe07746ffb6fff4e9e14de34b47.jpg)

![7710ed9e08a54b9e60afe0b28965d24d61ed60787588fc148695811d6abe9028.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/7710ed9e08a54b9e60afe0b28965d24d61ed60787588fc148695811d6abe9028.jpg)

![811c34105be18b7d38e98e822407b6dd930ac951c026970871b218cf9643c504.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/811c34105be18b7d38e98e822407b6dd930ac951c026970871b218cf9643c504.jpg)

![83c6ab7fc2bb5c734e826b7e793daf04b6f3157930f2385534f578eed3acfa9a.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/83c6ab7fc2bb5c734e826b7e793daf04b6f3157930f2385534f578eed3acfa9a.jpg)

![85297d8b72dd024046d0205859030536b3351bcff99c3c0be11faf07fa88cd12.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/85297d8b72dd024046d0205859030536b3351bcff99c3c0be11faf07fa88cd12.jpg)

![9c19a4a844f3b6a2e3b75ad4d9128bfac921ad27992e09b000f96b49b0a10ff4.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/9c19a4a844f3b6a2e3b75ad4d9128bfac921ad27992e09b000f96b49b0a10ff4.jpg)

![ab423a0c9e82d84d46dd5a1607d3621acd5f11330a61d09c9c8ba40e545e7ec7.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/ab423a0c9e82d84d46dd5a1607d3621acd5f11330a61d09c9c8ba40e545e7ec7.jpg)

![ab78e4d4244abf971ff5a513d789b2730b0653f79b6e5453807e16231f9c4266.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/ab78e4d4244abf971ff5a513d789b2730b0653f79b6e5453807e16231f9c4266.jpg)

![ac2bcadc03bcbb327092ced638faa378e91e9005e5436aa4536fb8a77c115110.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/ac2bcadc03bcbb327092ced638faa378e91e9005e5436aa4536fb8a77c115110.jpg)

![ae706531ccb02996c4608f2e7fc193539a7b0047767465e2cfa5f48dd08cec95.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/ae706531ccb02996c4608f2e7fc193539a7b0047767465e2cfa5f48dd08cec95.jpg)

![b0ab08ef471dcb7ec030e9aad72fd3cbaf0efd9237f499e59bc50575eb46dc26.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/b0ab08ef471dcb7ec030e9aad72fd3cbaf0efd9237f499e59bc50575eb46dc26.jpg)

![d6dccba18927ccfc7454f005e59d1f055419bbcbb30dee237dbf493ee988fac6.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/d6dccba18927ccfc7454f005e59d1f055419bbcbb30dee237dbf493ee988fac6.jpg)

![d9ddb3586c78d19b0d85e2b26180433f07fd297e1bd3268d479da21fd05c68d5.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/d9ddb3586c78d19b0d85e2b26180433f07fd297e1bd3268d479da21fd05c68d5.jpg)

![f12418cd2653b7938ab602a086fb77224fda035f9e1c0be2130666a950d68498.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/f12418cd2653b7938ab602a086fb77224fda035f9e1c0be2130666a950d68498.jpg)

![fdeccbbb6ffdae889a85f2275fa5e7efff1ae8432a617ad39a3ab7f1ecbeeb24.jpg](../iclr_results/1065_PaLD_ Detection of Text Partially Written by Large Language Models/tables/fdeccbbb6ffdae889a85f2275fa5e7efff1ae8432a617ad39a3ab7f1ecbeeb24.jpg)

## Graph Transformers Dream of Electric Flow


### Images

![0fac494a84683cbc6328b90b8c0493f12875d10f8084ab3da98eb25ed756747a.jpg](../iclr_results/1066_Graph Transformers Dream of Electric Flow/images/0fac494a84683cbc6328b90b8c0493f12875d10f8084ab3da98eb25ed756747a.jpg)

![382f3a3d4b7af81325902175f5806538d69bfbe152a08da192f338c9f949da96.jpg](../iclr_results/1066_Graph Transformers Dream of Electric Flow/images/382f3a3d4b7af81325902175f5806538d69bfbe152a08da192f338c9f949da96.jpg)

![6aea7770de52175ca7d6d386c9186f15aacb1eb6438116972a16cdd0db6eddb8.jpg](../iclr_results/1066_Graph Transformers Dream of Electric Flow/images/6aea7770de52175ca7d6d386c9186f15aacb1eb6438116972a16cdd0db6eddb8.jpg)

### Tables

![17d640c09a0777798e48f6ffc130b17c079300e0425ab5a2d8a1b21c28fde916.jpg](../iclr_results/1066_Graph Transformers Dream of Electric Flow/tables/17d640c09a0777798e48f6ffc130b17c079300e0425ab5a2d8a1b21c28fde916.jpg)

![c44235ffb007eb4579a5b59c6b53230beda3af70337d4d193d4c0f3c69034d0a.jpg](../iclr_results/1066_Graph Transformers Dream of Electric Flow/tables/c44235ffb007eb4579a5b59c6b53230beda3af70337d4d193d4c0f3c69034d0a.jpg)

## $InterLCM$: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration


### Images

![06ee69134ff2b7d871047942b0297940d7b7b64f71e0a4c5569a8ebfbe7ac471.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/06ee69134ff2b7d871047942b0297940d7b7b64f71e0a4c5569a8ebfbe7ac471.jpg)

![1312e18fde7f3a1a6d903fd0e693e5316cbb9f642d10ac502ed7b7499768f37a.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/1312e18fde7f3a1a6d903fd0e693e5316cbb9f642d10ac502ed7b7499768f37a.jpg)

![1c697ae186d0126c6cd8177818bb1ad878de31862715ebf62ffa7e8a8e21f7aa.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/1c697ae186d0126c6cd8177818bb1ad878de31862715ebf62ffa7e8a8e21f7aa.jpg)

![3f6d9f60ca1960be10a6e79211f88e37419890f9a411b67ea4f4ab5d26450bbd.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/3f6d9f60ca1960be10a6e79211f88e37419890f9a411b67ea4f4ab5d26450bbd.jpg)

![409df7e3fdc379eaee9005ad94076393c3fedf32d7568d6a618f4b1ec1cd1a0d.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/409df7e3fdc379eaee9005ad94076393c3fedf32d7568d6a618f4b1ec1cd1a0d.jpg)

![4a342404694cab3b4413953e19e9e2cf6bb294df2e704ee73edba1148465e7d5.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/4a342404694cab3b4413953e19e9e2cf6bb294df2e704ee73edba1148465e7d5.jpg)

![50e8518a84010bf9e09b08f9dc0f2274092ef230c90e079203c008913633ae95.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/50e8518a84010bf9e09b08f9dc0f2274092ef230c90e079203c008913633ae95.jpg)

![5c110adbeb6ab6f590b2df62e157d854fe47a04cc8fa82932a75c1337422a527.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/5c110adbeb6ab6f590b2df62e157d854fe47a04cc8fa82932a75c1337422a527.jpg)

![6274f578050608ee91d9e19cd3914485b2b1a4958492c33f1bbc3eb24a5d6cca.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/6274f578050608ee91d9e19cd3914485b2b1a4958492c33f1bbc3eb24a5d6cca.jpg)

![662df7f92e0e0a98fd920f9be8ee4ac1c9dcc0cf9f124eb9efbcc6fcfde746ab.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/662df7f92e0e0a98fd920f9be8ee4ac1c9dcc0cf9f124eb9efbcc6fcfde746ab.jpg)

![6bf900da677fc0e4bc0626d950da6f7844a529bdf9e9c5bf02648fb7621dc676.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/6bf900da677fc0e4bc0626d950da6f7844a529bdf9e9c5bf02648fb7621dc676.jpg)

![6fb2b9e5addbf352057a11b6471fa842e099847d7ccd989ea1231d98510e1c86.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/6fb2b9e5addbf352057a11b6471fa842e099847d7ccd989ea1231d98510e1c86.jpg)

![830e90b12222e1eaf3b87db765fbbc2d72ffc306d7c73010266835d507d155cd.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/830e90b12222e1eaf3b87db765fbbc2d72ffc306d7c73010266835d507d155cd.jpg)

![87d7c8f45b91438f9ef692893736a8013ca9e945d571b246f47fe8b28d83fee6.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/87d7c8f45b91438f9ef692893736a8013ca9e945d571b246f47fe8b28d83fee6.jpg)

![8c78b85bbdc62b287c8ec1cd0711cf8ed7dc1fd8e5b1debd8272acf4a0decfcd.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/8c78b85bbdc62b287c8ec1cd0711cf8ed7dc1fd8e5b1debd8272acf4a0decfcd.jpg)

![8d33b44431c44593cbe273e7b4c1bf92daa92483458ea6d9398eb28474216373.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/8d33b44431c44593cbe273e7b4c1bf92daa92483458ea6d9398eb28474216373.jpg)

![95f35102705257045da438b7a71db17a000935e052471ca89dbb4050cdba7a01.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/95f35102705257045da438b7a71db17a000935e052471ca89dbb4050cdba7a01.jpg)

![9ab829ccfe08bfd86306788e7f981e461510886c59b467d8702b97b6e72b82ca.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/9ab829ccfe08bfd86306788e7f981e461510886c59b467d8702b97b6e72b82ca.jpg)

![a19da94b4fadef48d97b9a582c4e4c179d56378da091bc5cee4267e87081fee9.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/a19da94b4fadef48d97b9a582c4e4c179d56378da091bc5cee4267e87081fee9.jpg)

![ab621c874421e534a74220610a347db1214117e152ff59dcb668eb91478a8bc8.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/ab621c874421e534a74220610a347db1214117e152ff59dcb668eb91478a8bc8.jpg)

![abe855737abc81608f7b13281b2b830ca15ddeae76a5aec3973e9354b4ad9c6e.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/abe855737abc81608f7b13281b2b830ca15ddeae76a5aec3973e9354b4ad9c6e.jpg)

![cac18c68c1cd16fbdc8414276614d4d638b1dfcf367d95a4165894f10787c2cc.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/cac18c68c1cd16fbdc8414276614d4d638b1dfcf367d95a4165894f10787c2cc.jpg)

![d2107db1b03a9600a2ca4d61cab70ff79ffee21bcc3a5c0e3f93e913e1b29990.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/d2107db1b03a9600a2ca4d61cab70ff79ffee21bcc3a5c0e3f93e913e1b29990.jpg)

![d4b479f7989a8944c93c1898c84fa618519e24fddbcbc54065218b4112569200.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/d4b479f7989a8944c93c1898c84fa618519e24fddbcbc54065218b4112569200.jpg)

![db64f55fd870aebf3bdb7c03e4c6b3a14490e34405720e41b9ae9eac45ccc620.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/db64f55fd870aebf3bdb7c03e4c6b3a14490e34405720e41b9ae9eac45ccc620.jpg)

![dee9d5a84507e7d3c564ac557d8008e0f45c78df45551d681ecff35ad8a9c8cf.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/dee9d5a84507e7d3c564ac557d8008e0f45c78df45551d681ecff35ad8a9c8cf.jpg)

![e513f6fc8d0eab1bd41012b8c49c582b0436950e7ac985cebdc02dd923a19724.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/e513f6fc8d0eab1bd41012b8c49c582b0436950e7ac985cebdc02dd923a19724.jpg)

![e51fe0ec724a7ed1d321ee2158cdf1a66ea1b4defaa1da878eafb9039f84c8a1.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/e51fe0ec724a7ed1d321ee2158cdf1a66ea1b4defaa1da878eafb9039f84c8a1.jpg)

![f00a19733d3e84fd1dfb4822f86627149e8389c8dc3b4cda8885f0b53044c003.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/images/f00a19733d3e84fd1dfb4822f86627149e8389c8dc3b4cda8885f0b53044c003.jpg)

### Tables

![0a4726d658f321a89c8665d09d736184352c78a532badc025330468d8fdf6dc9.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/0a4726d658f321a89c8665d09d736184352c78a532badc025330468d8fdf6dc9.jpg)

![14b92e6cfe9790c05538c94788522edd98e1717bbc1ea43bf71304bcd05922b1.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/14b92e6cfe9790c05538c94788522edd98e1717bbc1ea43bf71304bcd05922b1.jpg)

![1885f44c31c80c0a2f0792b88f9b4b3e1c173f8751340c2c5e628f9eb646c916.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/1885f44c31c80c0a2f0792b88f9b4b3e1c173f8751340c2c5e628f9eb646c916.jpg)

![19f20c445444cbddb2679acd9c48a45d12d24a53a921abfd4a6f040931a291cd.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/19f20c445444cbddb2679acd9c48a45d12d24a53a921abfd4a6f040931a291cd.jpg)

![4e934e5a40e8d406ecc577de3ffc287cab767e8f684f139161f7834cf43a799f.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/4e934e5a40e8d406ecc577de3ffc287cab767e8f684f139161f7834cf43a799f.jpg)

![98785efac7acfe4c30485475a97c1b9f97e507effac9a953a1762c3c994f8173.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/98785efac7acfe4c30485475a97c1b9f97e507effac9a953a1762c3c994f8173.jpg)

![a068ada894fbf83639117e9e3a2694461bcf9a49e12087547c6298b302378204.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/a068ada894fbf83639117e9e3a2694461bcf9a49e12087547c6298b302378204.jpg)

![afdb487121b93a8e4e22899effffa280c5e1ad2707cc6808122eb41a544f6ab3.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/afdb487121b93a8e4e22899effffa280c5e1ad2707cc6808122eb41a544f6ab3.jpg)

![c8c14168edeaf45de430f1bfaf6288abe0121c8e5ce3728383db190fdc4ee42a.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/c8c14168edeaf45de430f1bfaf6288abe0121c8e5ce3728383db190fdc4ee42a.jpg)

![fca5ea9b2256560bd5e2fa9cf3659c50b102b0e18ec9a9ac3d22767b0d8b7caa.jpg](../iclr_results/1067_$InterLCM$_ Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Bli/tables/fca5ea9b2256560bd5e2fa9cf3659c50b102b0e18ec9a9ac3d22767b0d8b7caa.jpg)

## Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology


### Images

![059b62c05aba0468ee44fb0b83da49cf3cffcf07b3bed642a261fb7a9c5c4478.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/059b62c05aba0468ee44fb0b83da49cf3cffcf07b3bed642a261fb7a9c5c4478.jpg)

![13208cb9c23c6fd462bb4750f29114b58fbb677449a60f31319771f1eb7c0ee7.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/13208cb9c23c6fd462bb4750f29114b58fbb677449a60f31319771f1eb7c0ee7.jpg)

![1ce89a7d1a3764842b38a679fff09fd9a536b8acdac11ac047c9c4b4189f291c.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/1ce89a7d1a3764842b38a679fff09fd9a536b8acdac11ac047c9c4b4189f291c.jpg)

![22f40cfb53cf5d86478a728fbbc56d9690a691635b6bccaf6de70658760baeb8.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/22f40cfb53cf5d86478a728fbbc56d9690a691635b6bccaf6de70658760baeb8.jpg)

![2caf4657486dbb24f097ab8392cdb316f865e5477a3278cea6c03440ba1ed760.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/2caf4657486dbb24f097ab8392cdb316f865e5477a3278cea6c03440ba1ed760.jpg)

![49c3c3895ca8f35bcbb0ab9d891a38317b21c0d5c2f78cd2685812ded1f39413.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/49c3c3895ca8f35bcbb0ab9d891a38317b21c0d5c2f78cd2685812ded1f39413.jpg)

![7350e3056a2b378acd1a369f8ca3732b60fe693799765186ef944dd4b500c1d8.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/7350e3056a2b378acd1a369f8ca3732b60fe693799765186ef944dd4b500c1d8.jpg)

![8cd45f53b1d4439c5ce6d6214766acb6beb88a2ad7199941852973d32fdc7002.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/8cd45f53b1d4439c5ce6d6214766acb6beb88a2ad7199941852973d32fdc7002.jpg)

![94053271ada890e2bb67bbc01a7e64b879a85f00c43a9b93e76b1c2fe59c244b.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/94053271ada890e2bb67bbc01a7e64b879a85f00c43a9b93e76b1c2fe59c244b.jpg)

![d6d3ffacec9fca0ae064d38effda8262a3fcdeb60058de012857a3f6533f1998.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/d6d3ffacec9fca0ae064d38effda8262a3fcdeb60058de012857a3f6533f1998.jpg)

![f6928764e4e139b3cac9c13a990b46c178d0b7d439cd222d6d7cfeb6c2336053.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/images/f6928764e4e139b3cac9c13a990b46c178d0b7d439cd222d6d7cfeb6c2336053.jpg)

### Tables

![2b22aa9251044595966990a9e251d125223fe869195f8781e642e5d5b33753e9.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/2b22aa9251044595966990a9e251d125223fe869195f8781e642e5d5b33753e9.jpg)

![489e096e42fb7249b0f6501f6910f605872bae7960c76765a529e2b8aa580779.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/489e096e42fb7249b0f6501f6910f605872bae7960c76765a529e2b8aa580779.jpg)

![6c01a029727b67a5dfa52a2dcd2ef1ded2792416c94e54e51c32b98be28e9062.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/6c01a029727b67a5dfa52a2dcd2ef1ded2792416c94e54e51c32b98be28e9062.jpg)

![86beea919a150d7bd25426572d5ced829375704567b84363229655747eb854fc.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/86beea919a150d7bd25426572d5ced829375704567b84363229655747eb854fc.jpg)

![8fa1b4f89577d1178cac76c644a39c66d93c4a17dfc72572d807dfb776317158.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/8fa1b4f89577d1178cac76c644a39c66d93c4a17dfc72572d807dfb776317158.jpg)

![abd7bbd82777d9c198df2a61bfed7bb5cf34eaaec196a72bfa74d6ff2719f268.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/abd7bbd82777d9c198df2a61bfed7bb5cf34eaaec196a72bfa74d6ff2719f268.jpg)

![b51ec290824cd186163e0da954a314a49705aa24f8118d2a0d6bd886d7e141f1.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/b51ec290824cd186163e0da954a314a49705aa24f8118d2a0d6bd886d7e141f1.jpg)

![b7d78a3ebb73553d057d29e717a56541813ed24688dd67d288e052c1341455a1.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/b7d78a3ebb73553d057d29e717a56541813ed24688dd67d288e052c1341455a1.jpg)

![ff5dbd9a8c784df864a271f04e260c5680087860368b3e062bacba318a51f6f6.jpg](../iclr_results/1068_Towards Realistic UAV Vision-Language Navigation_ Platform, Benchmark, and Methodology/tables/ff5dbd9a8c784df864a271f04e260c5680087860368b3e062bacba318a51f6f6.jpg)

## Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers


### Images

![24aeb003efee22acabb203006da87db3b1c6bb1138e8bb1fa3836acff8a6c73f.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/24aeb003efee22acabb203006da87db3b1c6bb1138e8bb1fa3836acff8a6c73f.jpg)

![26f9c2346dd03c2e860b202f7eb015731eb59625eb70780c2be6a6bc884554e9.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/26f9c2346dd03c2e860b202f7eb015731eb59625eb70780c2be6a6bc884554e9.jpg)

![66ea779f589cf27464bdd1316602a71e6f2630de22704844c75c3e6f8dde473b.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/66ea779f589cf27464bdd1316602a71e6f2630de22704844c75c3e6f8dde473b.jpg)

![6c4d853a0e46a838ab15c26e4ecd423aafc77c6109e760b3fe1058774a05b5fd.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/6c4d853a0e46a838ab15c26e4ecd423aafc77c6109e760b3fe1058774a05b5fd.jpg)

![7f2898e7a900dc32f201e1da98fef25f7ab5b9c7290f34e668d10fb5f8efe12e.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/7f2898e7a900dc32f201e1da98fef25f7ab5b9c7290f34e668d10fb5f8efe12e.jpg)

![9b592402dc20291b72376c40a4fc30a27f1e6be77b6ffec5b2d8309c26853d16.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/9b592402dc20291b72376c40a4fc30a27f1e6be77b6ffec5b2d8309c26853d16.jpg)

![9c175a237b009bbd2af57e9b63d15232bdeeb6dcff6013cc1279371fda999ead.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/9c175a237b009bbd2af57e9b63d15232bdeeb6dcff6013cc1279371fda999ead.jpg)

![ab7f930dee14fc94b7a16ca386600a32308d525bfb95100cb4ea9c2c03bfff6b.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/ab7f930dee14fc94b7a16ca386600a32308d525bfb95100cb4ea9c2c03bfff6b.jpg)

![ad6a3dad5e0868ae312130c103ec61a86842fceb94ac02fbf00f8d4d1b4b12ce.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/ad6a3dad5e0868ae312130c103ec61a86842fceb94ac02fbf00f8d4d1b4b12ce.jpg)

![bd8702a77098f18f1542a5cdfa30d924f83b658728db39a423c82c3cc4aeae37.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/bd8702a77098f18f1542a5cdfa30d924f83b658728db39a423c82c3cc4aeae37.jpg)

![c2d5e2d0f4f4ed1f12d3a8ec2e008a1e0299db63c86bf12e3f80c72b7b7fa0a2.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/c2d5e2d0f4f4ed1f12d3a8ec2e008a1e0299db63c86bf12e3f80c72b7b7fa0a2.jpg)

![e97e13ed3036ba66e5ac72dd03ba61856d57b0011490a10dfc417a445f7aaf93.jpg](../iclr_results/1069_Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers/images/e97e13ed3036ba66e5ac72dd03ba61856d57b0011490a10dfc417a445f7aaf93.jpg)

## PhyloLM: Inferring the Phylogeny of Large Language Models and Predicting their Performances in Benchmarks


### Images

![04e8dc458223acbb7304bcc31fbf19a85b3476052afa7e166d0b1513c10b6cfa.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/04e8dc458223acbb7304bcc31fbf19a85b3476052afa7e166d0b1513c10b6cfa.jpg)

![132d4696a8c568a8d5d58e35bd89e6e38deca76c54b380d480e972d98785bfe2.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/132d4696a8c568a8d5d58e35bd89e6e38deca76c54b380d480e972d98785bfe2.jpg)

![18d383396475d44ec4de791bbd62ab90ad0ac8ef5c730c2e7901928da759d250.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/18d383396475d44ec4de791bbd62ab90ad0ac8ef5c730c2e7901928da759d250.jpg)

![2a6ea5d9343411f6e057d8337a012473c22ec655016a08f0e262626b5ecc1212.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/2a6ea5d9343411f6e057d8337a012473c22ec655016a08f0e262626b5ecc1212.jpg)

![462ed2f589f5a2c25f53954827c968e5a9200aeb5b6aef738a2297c8bbd5c398.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/462ed2f589f5a2c25f53954827c968e5a9200aeb5b6aef738a2297c8bbd5c398.jpg)

![465d333f14a9bc1eaa008016c1bc47b974227bbe3137282015d88903d4d3cbb5.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/465d333f14a9bc1eaa008016c1bc47b974227bbe3137282015d88903d4d3cbb5.jpg)

![48b45315f0e05291e8b56cab8cf21785f94f8d74ddd6bd4429043fdc4d6ec57b.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/48b45315f0e05291e8b56cab8cf21785f94f8d74ddd6bd4429043fdc4d6ec57b.jpg)

![4d66e69558962296b04fed013102914a868ddaeacd058f3fda0d3f609bc85f2d.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/4d66e69558962296b04fed013102914a868ddaeacd058f3fda0d3f609bc85f2d.jpg)

![4fc54a1cc690a556e5ed4157eee55c4e027f97f3e80af6027143a951dcfc67de.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/4fc54a1cc690a556e5ed4157eee55c4e027f97f3e80af6027143a951dcfc67de.jpg)

![64d963b5394fc66856f4eb6aaa2422f9253236acc6edbb3bcf81b4e75ad83507.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/64d963b5394fc66856f4eb6aaa2422f9253236acc6edbb3bcf81b4e75ad83507.jpg)

![664100162fc0b763f83e762e8d93dac232dc91397320db1dbb3e33a1f9a1781f.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/664100162fc0b763f83e762e8d93dac232dc91397320db1dbb3e33a1f9a1781f.jpg)

![699258dcd40e564966831378251ba6712f3efdbbf34240b73b39c73cb78f2a4d.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/699258dcd40e564966831378251ba6712f3efdbbf34240b73b39c73cb78f2a4d.jpg)

![6f290ffd4981075c39520facf0e9a8ebb457e942a577e129c01cef6291548f17.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/6f290ffd4981075c39520facf0e9a8ebb457e942a577e129c01cef6291548f17.jpg)

![72fe079be1a6d8bec277c62ffcc38a019f2966709a30ad213bb4abd7a2afed6b.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/72fe079be1a6d8bec277c62ffcc38a019f2966709a30ad213bb4abd7a2afed6b.jpg)

![7484fedaee3f9517fc256bace5b1e372c7f34cfca4617ec849152d1c261c946b.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/7484fedaee3f9517fc256bace5b1e372c7f34cfca4617ec849152d1c261c946b.jpg)

![77f541feaf4e5e62733197cfc2b7a527151141996e8f8c16c59204f5a1ff6d85.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/77f541feaf4e5e62733197cfc2b7a527151141996e8f8c16c59204f5a1ff6d85.jpg)

![862411c7e571c86cbef10c81d270454acaf66a6470ef90bfae34360459b77ef5.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/862411c7e571c86cbef10c81d270454acaf66a6470ef90bfae34360459b77ef5.jpg)

![96caad6cd74b181a3dcf18e002892bb1a07199546b9ff62a49bd70ea6132dd88.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/96caad6cd74b181a3dcf18e002892bb1a07199546b9ff62a49bd70ea6132dd88.jpg)

![9d56adb51843a50533480af2e443405b529f91820564541ccb8a6e65eec8ab3c.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/9d56adb51843a50533480af2e443405b529f91820564541ccb8a6e65eec8ab3c.jpg)

![a30499c79c6c248a52b213de37f3279787017a9249fdd6785dc55aebd2953e05.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/a30499c79c6c248a52b213de37f3279787017a9249fdd6785dc55aebd2953e05.jpg)

![ab76d8893b67a30a65b215686fc84d9885d94f4ff168a1117f1b417e91acb9ae.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/ab76d8893b67a30a65b215686fc84d9885d94f4ff168a1117f1b417e91acb9ae.jpg)

![ac20d5fc5d8ccfe948ed1b71a8699bb3a5f347ab7e9d3873d5e36e55c811e6a6.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/ac20d5fc5d8ccfe948ed1b71a8699bb3a5f347ab7e9d3873d5e36e55c811e6a6.jpg)

![ad4e2c1280dc3552204b3cb8245746a322fab4b00e557a0fe8d2f300cd65e9f3.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/ad4e2c1280dc3552204b3cb8245746a322fab4b00e557a0fe8d2f300cd65e9f3.jpg)

![af0c3af2df79d08c855493b719c5cb40e298365d36e6433ed61d809fa4ab9df6.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/af0c3af2df79d08c855493b719c5cb40e298365d36e6433ed61d809fa4ab9df6.jpg)

![bc3e99636c067fb45cab3b5f34d93073adaf723bab37aa199a3be6c0bc6d65a5.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/bc3e99636c067fb45cab3b5f34d93073adaf723bab37aa199a3be6c0bc6d65a5.jpg)

![c76e30c61e2c8bc4a74393a2ff244394f6edb6017fcedadbd24e5ee13dcbf187.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/c76e30c61e2c8bc4a74393a2ff244394f6edb6017fcedadbd24e5ee13dcbf187.jpg)

![cba2653d3b68e3d723a0e49fca3135c9034c9a87b1351f67a9f1da722f5a877c.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/cba2653d3b68e3d723a0e49fca3135c9034c9a87b1351f67a9f1da722f5a877c.jpg)

![e1205a47063e70868f2e330c1f4cd008e4601ddb44a657e23f95dff5ff786cc7.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/e1205a47063e70868f2e330c1f4cd008e4601ddb44a657e23f95dff5ff786cc7.jpg)

![e89b2194c68f5c2485ae48b664bcf8d01b7bc05888a0be4177d599581d023941.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/e89b2194c68f5c2485ae48b664bcf8d01b7bc05888a0be4177d599581d023941.jpg)

![e99fb7a5d6cef764c98b3adabed9dd103251d19606fd6b63c5d8c78b02ea7b54.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/e99fb7a5d6cef764c98b3adabed9dd103251d19606fd6b63c5d8c78b02ea7b54.jpg)

![f265ee4e21ebccb61315196a197aeb7636efc8120ed6ea7ba806845bc4ff25f6.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/f265ee4e21ebccb61315196a197aeb7636efc8120ed6ea7ba806845bc4ff25f6.jpg)

![f5070e9dcca29f2eb3ba0fe806485c716321676a169a796ac8828e52664e6b70.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/f5070e9dcca29f2eb3ba0fe806485c716321676a169a796ac8828e52664e6b70.jpg)

![fb1ed816bcee3ca54226a80a8355e7ee3d9eff2ddc8b0387fcca5e3642f0cc5e.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/fb1ed816bcee3ca54226a80a8355e7ee3d9eff2ddc8b0387fcca5e3642f0cc5e.jpg)

![fcb0c2d3f1b39d6b8792d6b973afb80a4195612b048702a375cf4c5818267973.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/images/fcb0c2d3f1b39d6b8792d6b973afb80a4195612b048702a375cf4c5818267973.jpg)

### Tables

![02fe4f8a4b3f1ab109731a2eb0325440013192c52a0f523181f8ce81998e443c.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/tables/02fe4f8a4b3f1ab109731a2eb0325440013192c52a0f523181f8ce81998e443c.jpg)

![109727020c3018350182a5a9e89f62161e0b6570c50aac3e825ecb01955d898d.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/tables/109727020c3018350182a5a9e89f62161e0b6570c50aac3e825ecb01955d898d.jpg)

![49715e0ef742056b1b1ff0e8d882c899b1f508bf9a9b80520be8b4396279071d.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/tables/49715e0ef742056b1b1ff0e8d882c899b1f508bf9a9b80520be8b4396279071d.jpg)

![619c493d3056cfdbbde2fae6e809967c7ba452bca451ecd5ad9dbe92815d885b.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/tables/619c493d3056cfdbbde2fae6e809967c7ba452bca451ecd5ad9dbe92815d885b.jpg)

![9c4255ccc0512fec62cfadee8d56e9b5a6c4f380eb51ca0b201908b70032acfe.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/tables/9c4255ccc0512fec62cfadee8d56e9b5a6c4f380eb51ca0b201908b70032acfe.jpg)

![e0afc1a46fa59d1e5b90ea9c321c652d1bd71cf42f033dd05d876dc66c19a2ba.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/tables/e0afc1a46fa59d1e5b90ea9c321c652d1bd71cf42f033dd05d876dc66c19a2ba.jpg)

![e37bfa9087d45aa82a1e5566550e0ef6996ee8e990ff48fd8b82759d23e86e0e.jpg](../iclr_results/1070_PhyloLM_ Inferring the Phylogeny of Large Language Models and Predicting their Performances in Bench/tables/e37bfa9087d45aa82a1e5566550e0ef6996ee8e990ff48fd8b82759d23e86e0e.jpg)

## Reasoning of Large Language Models over Knowledge Graphs with Super-Relations


### Images

![15c6fb5144ae9124d1ce8bcd5654205c0265cd26703cf8b05a0dc3f98b051837.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/images/15c6fb5144ae9124d1ce8bcd5654205c0265cd26703cf8b05a0dc3f98b051837.jpg)

![8a1f8f65df3012d5ffffeed76fe8dc0d485b3d570c57172d6ebcda115fd4bea0.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/images/8a1f8f65df3012d5ffffeed76fe8dc0d485b3d570c57172d6ebcda115fd4bea0.jpg)

![b474d0e20a941e8fd2b4123864fcba7b327a3935d3ac81fd40d8d42b4e9e8118.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/images/b474d0e20a941e8fd2b4123864fcba7b327a3935d3ac81fd40d8d42b4e9e8118.jpg)

![bf0ae22f7b8b4a604c921692f67981a06c0fc9b90dbd11777d117256873ff5c6.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/images/bf0ae22f7b8b4a604c921692f67981a06c0fc9b90dbd11777d117256873ff5c6.jpg)

![ed06acc9f39b2dc532b7b1e86166fbce485b7ec239b224ef3b63e0eaf5db26a2.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/images/ed06acc9f39b2dc532b7b1e86166fbce485b7ec239b224ef3b63e0eaf5db26a2.jpg)

![ffddea63dde889431ade66942206caf1f6ba81fa865cdbc87b91c3283ff21cab.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/images/ffddea63dde889431ade66942206caf1f6ba81fa865cdbc87b91c3283ff21cab.jpg)

### Tables

![12de397c54a03e3e2c5c2b6dd6496c413bb0032d2b694ea0606c6d490dca4e82.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/12de397c54a03e3e2c5c2b6dd6496c413bb0032d2b694ea0606c6d490dca4e82.jpg)

![49bcaa496b9a5fb14e14ef629e99fbf364ec4c062c224b1df23109b62f70eb18.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/49bcaa496b9a5fb14e14ef629e99fbf364ec4c062c224b1df23109b62f70eb18.jpg)

![65e097b70822bb27477c87d94136009c50b8ea09d84491c508470be401986e8c.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/65e097b70822bb27477c87d94136009c50b8ea09d84491c508470be401986e8c.jpg)

![6a58fdeca8329261d3c342c799149f5614628538b110d0de5e739104599c878a.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/6a58fdeca8329261d3c342c799149f5614628538b110d0de5e739104599c878a.jpg)

![8fc9f252c398cf7e4e623f941c005df43f810abde47f148957e1d919cd210212.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/8fc9f252c398cf7e4e623f941c005df43f810abde47f148957e1d919cd210212.jpg)

![9b55c100a009665dc1e310faa51430f669f0fc12dd18c2885defe1bb4c445cbe.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/9b55c100a009665dc1e310faa51430f669f0fc12dd18c2885defe1bb4c445cbe.jpg)

![9cded5f089b585b15493aca527b0d962eabb482dc291608cb14ec2d7a9d5cb7b.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/9cded5f089b585b15493aca527b0d962eabb482dc291608cb14ec2d7a9d5cb7b.jpg)

![9ee357bcc5b9e48969eb8d047ed48a8db1c36c03909f6b4275dc3fa0c8aebe8f.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/9ee357bcc5b9e48969eb8d047ed48a8db1c36c03909f6b4275dc3fa0c8aebe8f.jpg)

![b1d3170cb80c1cedbeba21553aeb5a8410e432fe54377e40c37a6cba497c93cc.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/b1d3170cb80c1cedbeba21553aeb5a8410e432fe54377e40c37a6cba497c93cc.jpg)

![b3ae97a85e1a03421c3d8632fbade1db3c6b925902ee8940a249f2f80f9c2c77.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/b3ae97a85e1a03421c3d8632fbade1db3c6b925902ee8940a249f2f80f9c2c77.jpg)

![bd1ed9e4828678f698578d2ac8cedcc1c9372a54912c42f96e625a83a49a7241.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/bd1ed9e4828678f698578d2ac8cedcc1c9372a54912c42f96e625a83a49a7241.jpg)

![d37f9cf15f8ac9880b6f1c4d784f9ec2aec97c43fdece5908140264d0a44bdca.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/d37f9cf15f8ac9880b6f1c4d784f9ec2aec97c43fdece5908140264d0a44bdca.jpg)

![ee83a20df1effe6eae39cec2f05473b52b9ef6b3081573983643f314ac5d5779.jpg](../iclr_results/1071_Reasoning of Large Language Models over Knowledge Graphs with Super-Relations/tables/ee83a20df1effe6eae39cec2f05473b52b9ef6b3081573983643f314ac5d5779.jpg)

## Biologically Plausible Brain Graph Transformer


### Images

![0a090f82cb5e272643cb50615ef0871b6ab61e6093cf9ec7e6ce63d1452568ac.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/0a090f82cb5e272643cb50615ef0871b6ab61e6093cf9ec7e6ce63d1452568ac.jpg)

![39c5b63cb57ac5d80096d057372148631bbdd4241ae3e91cb0aef4d718ec0a97.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/39c5b63cb57ac5d80096d057372148631bbdd4241ae3e91cb0aef4d718ec0a97.jpg)

![3c88b66ee119bda6f052d03d3c314c687ba749e971141c661df09509418bc73b.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/3c88b66ee119bda6f052d03d3c314c687ba749e971141c661df09509418bc73b.jpg)

![46b4bf86022570f227961979dcdd6685956cdc524d386b405f338fc4a03e42ea.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/46b4bf86022570f227961979dcdd6685956cdc524d386b405f338fc4a03e42ea.jpg)

![4778f4a1304da5a1bd1fdd1c69fbd0dae7ca5cc66cd19a0e66788ff78c5953d4.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/4778f4a1304da5a1bd1fdd1c69fbd0dae7ca5cc66cd19a0e66788ff78c5953d4.jpg)

![4b68fc93c3a3c92fa0f64e6835a751d085e7a8175c807377dcc918b79c946505.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/4b68fc93c3a3c92fa0f64e6835a751d085e7a8175c807377dcc918b79c946505.jpg)

![591a3d7a3ddd475d28c6244255ad3d0e9dc461f69076b19f91980a985ee36c5f.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/591a3d7a3ddd475d28c6244255ad3d0e9dc461f69076b19f91980a985ee36c5f.jpg)

![93356f1f4ffa7abf298a590d7db7a49c9691753c9b1bd64e994394134626a679.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/93356f1f4ffa7abf298a590d7db7a49c9691753c9b1bd64e994394134626a679.jpg)

![a0ed5be88a011d6a9fd2015f2bdd9bf5366b62364ec232082e919dc0aeef4890.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/a0ed5be88a011d6a9fd2015f2bdd9bf5366b62364ec232082e919dc0aeef4890.jpg)

![aaabff3ccbe45e35b5a2f891d039331842e51c30dc1fb1d7359a22318a360d04.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/aaabff3ccbe45e35b5a2f891d039331842e51c30dc1fb1d7359a22318a360d04.jpg)

![c1a7140a5eade122523409eff9857e7820544d663361787dfd3a7c2b570b84f1.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/c1a7140a5eade122523409eff9857e7820544d663361787dfd3a7c2b570b84f1.jpg)

![d7f9ad924b108f62fa27bf9d05dbe61b0dab1f558978210118bd0148311d6ef3.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/d7f9ad924b108f62fa27bf9d05dbe61b0dab1f558978210118bd0148311d6ef3.jpg)

![ddc87843620183a2ea782c6795f7ddd1d3f1c25a1e6e64b0af08eb9229914182.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/ddc87843620183a2ea782c6795f7ddd1d3f1c25a1e6e64b0af08eb9229914182.jpg)

![e271c714842a5a2b36a5c886f9056fd99983a8ba21d58cf2e3380fc3030c4ad2.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/e271c714842a5a2b36a5c886f9056fd99983a8ba21d58cf2e3380fc3030c4ad2.jpg)

![e9f5c547353a71470536eefb9b9abb5618f44936addd98646c096cc28f2c727a.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/e9f5c547353a71470536eefb9b9abb5618f44936addd98646c096cc28f2c727a.jpg)

![fb615ea050b84e37d7c13fa968b1842fbc462aea2c088da356cb08670f4a54eb.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/images/fb615ea050b84e37d7c13fa968b1842fbc462aea2c088da356cb08670f4a54eb.jpg)

### Tables

![03f64752a5df42339ad3c8d43fba797f8297b263093384faeb55793c45d309db.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/03f64752a5df42339ad3c8d43fba797f8297b263093384faeb55793c45d309db.jpg)

![07aa0d8ce9d402d2abac0bf6f58e438fe37afac75f5a2418fa938c322764f1a4.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/07aa0d8ce9d402d2abac0bf6f58e438fe37afac75f5a2418fa938c322764f1a4.jpg)

![26632aff05d123187ff1993bf9f5321c211d0a12b4fdd378df23ba9b452c6d56.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/26632aff05d123187ff1993bf9f5321c211d0a12b4fdd378df23ba9b452c6d56.jpg)

![848df1a44357411581a0c0c895fc7de0e8af124f14258058fad672882ca8538d.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/848df1a44357411581a0c0c895fc7de0e8af124f14258058fad672882ca8538d.jpg)

![90c12dd2c2a737a167c78a3c8d687019f14d2e9d3f61d195ee52de4dbe475f35.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/90c12dd2c2a737a167c78a3c8d687019f14d2e9d3f61d195ee52de4dbe475f35.jpg)

![ba511f9ff54a0f1b20c9dba8f5d71e9d9fe54ebc7d167aba942bd18c2de560c6.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/ba511f9ff54a0f1b20c9dba8f5d71e9d9fe54ebc7d167aba942bd18c2de560c6.jpg)

![de28a7c686db680d8440a63fc69a26047cc87bc863b452f4885cf870dbf97a6a.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/de28a7c686db680d8440a63fc69a26047cc87bc863b452f4885cf870dbf97a6a.jpg)

![e9428484cc32633c9002c18b58521dd0d0002c787ad7b5a689aad2c08ad01189.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/e9428484cc32633c9002c18b58521dd0d0002c787ad7b5a689aad2c08ad01189.jpg)

![edeca3bcbde1d54e34bb648cd1e3ff2ca67243476411a864e7c17933edaf2142.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/edeca3bcbde1d54e34bb648cd1e3ff2ca67243476411a864e7c17933edaf2142.jpg)

![f39c79a199a94f1075384cac7f5f27922dc33bb899f14d45083eef2cb973a845.jpg](../iclr_results/1072_Biologically Plausible Brain Graph Transformer/tables/f39c79a199a94f1075384cac7f5f27922dc33bb899f14d45083eef2cb973a845.jpg)

## Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning


### Images

![078d8452f67b2303ca540b61289438d415f32bf26d7b0033526e9dbc46dcf3cc.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/078d8452f67b2303ca540b61289438d415f32bf26d7b0033526e9dbc46dcf3cc.jpg)

![112c47807e0b002b5afa6dcfd0b6b7f1c9e94da407da8b7d32907bcb431672b5.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/112c47807e0b002b5afa6dcfd0b6b7f1c9e94da407da8b7d32907bcb431672b5.jpg)

![14abb1bfc5dbc72cb0e05d0e8fbfd0e535a7a1496f3f687bec6b607210a88c36.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/14abb1bfc5dbc72cb0e05d0e8fbfd0e535a7a1496f3f687bec6b607210a88c36.jpg)

![1e3f72e053fe2f402cabe291cf30855e9fb6dececd32c48a6e690b282ddcaaae.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/1e3f72e053fe2f402cabe291cf30855e9fb6dececd32c48a6e690b282ddcaaae.jpg)

![2b95e907c49bbc060fe3843e67f9b6569f17771aaaa8ac9aea1b94b5cfc562b4.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/2b95e907c49bbc060fe3843e67f9b6569f17771aaaa8ac9aea1b94b5cfc562b4.jpg)

![3ccefeeddeb2727b88e508c48d9c17a45603f58dfede4ce80bd04edf7f48c705.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/3ccefeeddeb2727b88e508c48d9c17a45603f58dfede4ce80bd04edf7f48c705.jpg)

![3e8269e82559360bbfd545ac69831d2c6b5e80cceee1a5fe9c279831ab6cb17c.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/3e8269e82559360bbfd545ac69831d2c6b5e80cceee1a5fe9c279831ab6cb17c.jpg)

![5189555f8d8e04e1af262220314f5b5ac5c835af883c5ae6f8750bbc9528f3e6.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/5189555f8d8e04e1af262220314f5b5ac5c835af883c5ae6f8750bbc9528f3e6.jpg)

![664097803d72cb0be96db62f4d4dd47495f3b9d8ce866834fc86adcffa678f83.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/664097803d72cb0be96db62f4d4dd47495f3b9d8ce866834fc86adcffa678f83.jpg)

![70cab8005d391d3a9c3dacee26ef6fc905a02076887ae6305fe34683713d263d.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/70cab8005d391d3a9c3dacee26ef6fc905a02076887ae6305fe34683713d263d.jpg)

![78d8657701d721cf58365ba4678258e69637f086a3d6f8d68d755534658eb218.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/78d8657701d721cf58365ba4678258e69637f086a3d6f8d68d755534658eb218.jpg)

![7f8acd0c8cf9fe32ac159f450ed2b4326fb8ff7c3293f5479e9483c0484a11ca.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/7f8acd0c8cf9fe32ac159f450ed2b4326fb8ff7c3293f5479e9483c0484a11ca.jpg)

![8300b95742a4f9e4f148795dd6f3dc941aa41681d3a0082c84ba383e862a23e2.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/8300b95742a4f9e4f148795dd6f3dc941aa41681d3a0082c84ba383e862a23e2.jpg)

![938c853cc30d2d995d5375156b497ec5b0aec39f5f7859307d7a2bfc644b8870.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/938c853cc30d2d995d5375156b497ec5b0aec39f5f7859307d7a2bfc644b8870.jpg)

![97bc0b78c86ea73be0af57e7e4001f7fc0c9986a0e7013517e105af110c63bd4.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/97bc0b78c86ea73be0af57e7e4001f7fc0c9986a0e7013517e105af110c63bd4.jpg)

![a2ab9275ad7bea890d431549d2a35ff55ea2386c0eac0cea9a63e31fff51fe55.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/a2ab9275ad7bea890d431549d2a35ff55ea2386c0eac0cea9a63e31fff51fe55.jpg)

![bc8149fe302024a1347b683223b85b0cc2f5cbc6d2cd385c914dfde3ec9910d4.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/bc8149fe302024a1347b683223b85b0cc2f5cbc6d2cd385c914dfde3ec9910d4.jpg)

![c2b165b51709c97b8c58435fa1df1b3c60882738e2375886c3e67f855f10f93f.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/c2b165b51709c97b8c58435fa1df1b3c60882738e2375886c3e67f855f10f93f.jpg)

![cfd9f9f4cd45d9c90f3834d9dbb231e312f158038e8f3cd1dc9a1c6fd7e7ba68.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/images/cfd9f9f4cd45d9c90f3834d9dbb231e312f158038e8f3cd1dc9a1c6fd7e7ba68.jpg)

### Tables

![0e4c38bba8048ed1bd2b7b0924478b60787546251f8f118841cf09d4371898a7.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/tables/0e4c38bba8048ed1bd2b7b0924478b60787546251f8f118841cf09d4371898a7.jpg)

![330c311b853cc915259f24f6e47f36a8a2fe286bcfa5ba36e40a1545f6a9ce40.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/tables/330c311b853cc915259f24f6e47f36a8a2fe286bcfa5ba36e40a1545f6a9ce40.jpg)

![57cbe1f67074d0550039ca9d698905a7449ff287b2939c580f50e2d2eb6bf79a.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/tables/57cbe1f67074d0550039ca9d698905a7449ff287b2939c580f50e2d2eb6bf79a.jpg)

![7e229ebc7474034adbf625f7d1d510bd88285d1a774a7d15f4a283d4ed9f35ec.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/tables/7e229ebc7474034adbf625f7d1d510bd88285d1a774a7d15f4a283d4ed9f35ec.jpg)

![9796befbf6ac2b52ea77765e7351f977ed80e710ac66559cfcbed56da3277269.jpg](../iclr_results/1073_Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning/tables/9796befbf6ac2b52ea77765e7351f977ed80e710ac66559cfcbed56da3277269.jpg)

## A Causal Lens for Learning Long-term Fair Policies


### Images

![157ce9b0394171bccf2a3087b912579167ba7279eccef4235b9aa863ff65fa62.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/157ce9b0394171bccf2a3087b912579167ba7279eccef4235b9aa863ff65fa62.jpg)

![6c6a2ba7d94a8f06f27bf87e98af19016a213df2320ecb059beccaf4c24b62d6.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/6c6a2ba7d94a8f06f27bf87e98af19016a213df2320ecb059beccaf4c24b62d6.jpg)

![81974f43e761838775dc31bfc0534d8e5b342a85fc4788e1ad3d90fd6c09ceba.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/81974f43e761838775dc31bfc0534d8e5b342a85fc4788e1ad3d90fd6c09ceba.jpg)

![90c206f5144a76cb063b524d8551fdfb9984fad11c1664bd8a4c42f5b212b0eb.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/90c206f5144a76cb063b524d8551fdfb9984fad11c1664bd8a4c42f5b212b0eb.jpg)

![9f46f54bbccf76f7b12ca2246273329c33805c5537539b2a278e51fc8010d25f.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/9f46f54bbccf76f7b12ca2246273329c33805c5537539b2a278e51fc8010d25f.jpg)

![cd233632f94ab84fa5e46f5abbcae191cc78c928eae17098386d52864996b70c.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/cd233632f94ab84fa5e46f5abbcae191cc78c928eae17098386d52864996b70c.jpg)

![d71fbd0977c16b9cd79b9c86f4c31769659e1935a606e8a7db61f2cfeff86b02.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/d71fbd0977c16b9cd79b9c86f4c31769659e1935a606e8a7db61f2cfeff86b02.jpg)

![e7307341b093397e56ccf3dbc7dd16689d858bc6eb8e6eacab7b5fa5de2a7fa3.jpg](../iclr_results/1074_A Causal Lens for Learning Long-term Fair Policies/images/e7307341b093397e56ccf3dbc7dd16689d858bc6eb8e6eacab7b5fa5de2a7fa3.jpg)

## Smoothing the Shift: Towards Stable Test-Time Adaptation under Complex Multimodal Noises


### Images

![12afc27bbcca764885d494285edb5a1c2daa26c785b3bd7e5373215f70c83f86.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/12afc27bbcca764885d494285edb5a1c2daa26c785b3bd7e5373215f70c83f86.jpg)

![3fcd7b0587d7b35029ebd837f4c41eed098c6cb1e6cd29c96314bbbb064cc45d.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/3fcd7b0587d7b35029ebd837f4c41eed098c6cb1e6cd29c96314bbbb064cc45d.jpg)

![4d2dda7b1c07f820fc057b04d64fb9079d8b31ec89776928b57d6fc8b893c2f3.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/4d2dda7b1c07f820fc057b04d64fb9079d8b31ec89776928b57d6fc8b893c2f3.jpg)

![52ead40da0b30af79ff3f84d3ff01b403594a34f8b162ddfafa82f9398505d41.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/52ead40da0b30af79ff3f84d3ff01b403594a34f8b162ddfafa82f9398505d41.jpg)

![5fd818ff96bad953bc48635c4b0c90a7863de7a4dab3f29898558fdfbe985fde.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/5fd818ff96bad953bc48635c4b0c90a7863de7a4dab3f29898558fdfbe985fde.jpg)

![6094873b75cfbc4219f1f1d288c995a0851b9faf4dd9e4f55db5d237d6c53cbb.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/6094873b75cfbc4219f1f1d288c995a0851b9faf4dd9e4f55db5d237d6c53cbb.jpg)

![62a192a47c19cae2d1bbe474b4f7a2b44531b7525ec9bcaf624b46f6d6204f75.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/62a192a47c19cae2d1bbe474b4f7a2b44531b7525ec9bcaf624b46f6d6204f75.jpg)

![7ba5562542048c4c93383f43af6cc650105e564bfc347cdf5004bcd16dbfd786.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/7ba5562542048c4c93383f43af6cc650105e564bfc347cdf5004bcd16dbfd786.jpg)

![b3c4f06342f8a506dfde3143c7b699a1809a38c471c0f64492ac94ea2f7b27bc.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/b3c4f06342f8a506dfde3143c7b699a1809a38c471c0f64492ac94ea2f7b27bc.jpg)

![bc872e244ad2c75a0151e7346a39735efa3a139f0cba4b259bf4be3e6a9485c1.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/bc872e244ad2c75a0151e7346a39735efa3a139f0cba4b259bf4be3e6a9485c1.jpg)

![c9583cbe964de9c50713ce011373cf69cb193ac3860baacd73dfadf9bd73433e.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/c9583cbe964de9c50713ce011373cf69cb193ac3860baacd73dfadf9bd73433e.jpg)

![ccc786b1891b9f7262b949d8c9bba0a88b8ad6b2b6368a68539a0459c586dd31.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/ccc786b1891b9f7262b949d8c9bba0a88b8ad6b2b6368a68539a0459c586dd31.jpg)

![d496aac7d0c48eb712c3e1a39c872ef8f9ac666c5b3ea53c4fa480151f79b036.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/images/d496aac7d0c48eb712c3e1a39c872ef8f9ac666c5b3ea53c4fa480151f79b036.jpg)

### Tables

![1959b98d8b5cbf8006e71fe426a27c446b9420d73f034eba0868a9deaaac5e9d.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/1959b98d8b5cbf8006e71fe426a27c446b9420d73f034eba0868a9deaaac5e9d.jpg)

![35bf71633c6ea98d6ab5412744191bb50310a35016cabb4bd1e03277164ce20e.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/35bf71633c6ea98d6ab5412744191bb50310a35016cabb4bd1e03277164ce20e.jpg)

![4221e699d5f9874acbd6c87b19a9c528d6ba3cbf509392f0da244f40c3877138.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/4221e699d5f9874acbd6c87b19a9c528d6ba3cbf509392f0da244f40c3877138.jpg)

![50c31e4a41bddcd1813b26b7f91e40cf0a83938b3c73075e91037d47fafdd28a.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/50c31e4a41bddcd1813b26b7f91e40cf0a83938b3c73075e91037d47fafdd28a.jpg)

![7a3e06630246bea5e1fef1aaf82f1cff399ea745a4322ea3b5747bbd7ad1a3a2.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/7a3e06630246bea5e1fef1aaf82f1cff399ea745a4322ea3b5747bbd7ad1a3a2.jpg)

![de76025e6ba47a4480552c1ec1102f3e3e6f28d1c9b7fc2a382db82877c25b78.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/de76025e6ba47a4480552c1ec1102f3e3e6f28d1c9b7fc2a382db82877c25b78.jpg)

![e8d79e4fb42abaa70e5a2ab07eddbcb05ef415dbbdfdcc15dec0b73898194942.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/e8d79e4fb42abaa70e5a2ab07eddbcb05ef415dbbdfdcc15dec0b73898194942.jpg)

![f110fe4254b505aed765605118879c2f9e5cc918372447764361295892fcd007.jpg](../iclr_results/1075_Smoothing the Shift_ Towards Stable Test-Time Adaptation under Complex Multimodal Noises/tables/f110fe4254b505aed765605118879c2f9e5cc918372447764361295892fcd007.jpg)

## Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Perception


### Images

![0a52599b3cdde990436e85c75ba82294220054db32bef7fc72c17b34e68b0313.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/0a52599b3cdde990436e85c75ba82294220054db32bef7fc72c17b34e68b0313.jpg)

![10e0904f339bfcdc07f81fa47dea44fc7de2879b77bfe92a424abe946911ba4d.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/10e0904f339bfcdc07f81fa47dea44fc7de2879b77bfe92a424abe946911ba4d.jpg)

![4db9f782059144858c8d300d76fb65adce1eb95dcdecc325e4a36edcade268a8.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/4db9f782059144858c8d300d76fb65adce1eb95dcdecc325e4a36edcade268a8.jpg)

![55a12c68b093154c73749376a40596cd9c5c3f4cf77d233046066bf6bd43fa46.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/55a12c68b093154c73749376a40596cd9c5c3f4cf77d233046066bf6bd43fa46.jpg)

![66d3873dbf0486ac8445c6458a7d3df3bed47737da227982a7502e55c44d8021.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/66d3873dbf0486ac8445c6458a7d3df3bed47737da227982a7502e55c44d8021.jpg)

![682d72cf19d2389a40aeaf89bbcb3ca36c1aa626cc60f8ef251158aa7690c589.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/682d72cf19d2389a40aeaf89bbcb3ca36c1aa626cc60f8ef251158aa7690c589.jpg)

![82399e18401bf744e6440035f20f0f786358fa8895c5ae4bce68b6f0a61d42fa.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/82399e18401bf744e6440035f20f0f786358fa8895c5ae4bce68b6f0a61d42fa.jpg)

![88b5055c75c0b0ce3e4f9552652f3ad1a353023c96f8546331499e0482ca696b.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/88b5055c75c0b0ce3e4f9552652f3ad1a353023c96f8546331499e0482ca696b.jpg)

![9069ecb9e3303ed46df57e873e7c71bde1344915b64e7af1fb97faa5f2c2586c.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/9069ecb9e3303ed46df57e873e7c71bde1344915b64e7af1fb97faa5f2c2586c.jpg)

![c09691f0e51095e6f3dfb2b5eba91b770068f2a64254a2e062f7974b861fbaa3.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/c09691f0e51095e6f3dfb2b5eba91b770068f2a64254a2e062f7974b861fbaa3.jpg)

![fc55fe02f6a6290eac79c3f403250264416cd80042df7286ed6e13aa4876346a.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/fc55fe02f6a6290eac79c3f403250264416cd80042df7286ed6e13aa4876346a.jpg)

![ff01fc59ab67f50f1ef2b5147ea1531a0dbc8bbf86824d5585de24989fb3ead8.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/images/ff01fc59ab67f50f1ef2b5147ea1531a0dbc8bbf86824d5585de24989fb3ead8.jpg)

### Tables

![27cef65c4ae747116ffb81c7c99d757e83a3eab48119235d75e454246ec55031.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/27cef65c4ae747116ffb81c7c99d757e83a3eab48119235d75e454246ec55031.jpg)

![3d17a9d48bf309a076192417ee4b47e324eee0433fcabb3386509490996b4a14.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/3d17a9d48bf309a076192417ee4b47e324eee0433fcabb3386509490996b4a14.jpg)

![5dc3c23454cd0f836d0dd56eadabd6d81f7a84cc2322b7da4b5ffa8c51ae7def.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/5dc3c23454cd0f836d0dd56eadabd6d81f7a84cc2322b7da4b5ffa8c51ae7def.jpg)

![6f66f18d4dac4bed7d7a329b9c00663c725eb0026c070d430e3dca116fb64f6d.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/6f66f18d4dac4bed7d7a329b9c00663c725eb0026c070d430e3dca116fb64f6d.jpg)

![70a164a66926237302401327d08a927394a0155f0697a835a090898731a54a7d.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/70a164a66926237302401327d08a927394a0155f0697a835a090898731a54a7d.jpg)

![7c755afd0a0295b90f7290e552dd850012bc5ae6c2292f63b410269ad3f5838d.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/7c755afd0a0295b90f7290e552dd850012bc5ae6c2292f63b410269ad3f5838d.jpg)

![be357cd61254281c204a3760dca50b34b08d295e2da9ab7b4df767120f9a2db5.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/be357cd61254281c204a3760dca50b34b08d295e2da9ab7b4df767120f9a2db5.jpg)

![c1e70f7458d7cdc3a63cb45a4b0d815deeedae2ed812de4d9c69775ce121d0c2.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/c1e70f7458d7cdc3a63cb45a4b0d815deeedae2ed812de4d9c69775ce121d0c2.jpg)

![c2d50a51a8f3d3e00c2e416ad699b60eb30fabbbbc1d677c10241bc87790902d.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/c2d50a51a8f3d3e00c2e416ad699b60eb30fabbbbc1d677c10241bc87790902d.jpg)

![c8e3e9dae8fa9d045075cb30cd366d67436bc5aaa514692553df5b0795665baf.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/c8e3e9dae8fa9d045075cb30cd366d67436bc5aaa514692553df5b0795665baf.jpg)

![d4f02f583023a432aa2047605fe4cf9c8c4bef20625033cb3e42edffdb245ca9.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/d4f02f583023a432aa2047605fe4cf9c8c4bef20625033cb3e42edffdb245ca9.jpg)

![d5c5ed2a76fd2b4658ec0e25ec11281ce181a2ca3acdd5d5ef478ba1e95feb74.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/d5c5ed2a76fd2b4658ec0e25ec11281ce181a2ca3acdd5d5ef478ba1e95feb74.jpg)

![dcb1a2a823e5b9ad07b7ea50461a5dc4606cfd7fac8e0f04c0b8a7529941e2d3.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/dcb1a2a823e5b9ad07b7ea50461a5dc4606cfd7fac8e0f04c0b8a7529941e2d3.jpg)

![e3219c380b0349f0bd914f634893a0f90d49ea9e9996872e8f01c7939d3c01b9.jpg](../iclr_results/1076_Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Percepti/tables/e3219c380b0349f0bd914f634893a0f90d49ea9e9996872e8f01c7939d3c01b9.jpg)

## N-ForGOT: Towards Not-forgetting and Generalization of Open Temporal Graph Learning


### Images

![6402f91a3327f6aab3e3de612bd7361a9deb2c94d53565d9d921e6d0ae419920.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/images/6402f91a3327f6aab3e3de612bd7361a9deb2c94d53565d9d921e6d0ae419920.jpg)

![77dc8e8a02ba0c038794bd051faae43bab85feba88a2f503f1600519fbc9d8ec.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/images/77dc8e8a02ba0c038794bd051faae43bab85feba88a2f503f1600519fbc9d8ec.jpg)

![8978f5e708df86a166155c3f52f0cb95c5ce1d519ab20842f704e555a05eeac4.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/images/8978f5e708df86a166155c3f52f0cb95c5ce1d519ab20842f704e555a05eeac4.jpg)

![a4cb57ffb1369d46e180a5ecc5933d984dbfa9287abd1869d3bd83117e8c7a5d.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/images/a4cb57ffb1369d46e180a5ecc5933d984dbfa9287abd1869d3bd83117e8c7a5d.jpg)

![bddab43c977b920485bf2d1269598ffa59fa35e446ddd00a6e9b6d96e28f346e.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/images/bddab43c977b920485bf2d1269598ffa59fa35e446ddd00a6e9b6d96e28f346e.jpg)

![f0a5a83b01735d05006a6a4e7317768f2bc5d2898e0bff068094d69e17629549.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/images/f0a5a83b01735d05006a6a4e7317768f2bc5d2898e0bff068094d69e17629549.jpg)

![ff7273d5bf994c02d9220d55a1fd3b88f13ae36d731fb8370e7bfef34d518a69.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/images/ff7273d5bf994c02d9220d55a1fd3b88f13ae36d731fb8370e7bfef34d518a69.jpg)

### Tables

![30964c15fe6840a03b0bb8da8dfdd91a1fed5b2e9ea7c517c633c3cc94f19222.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/30964c15fe6840a03b0bb8da8dfdd91a1fed5b2e9ea7c517c633c3cc94f19222.jpg)

![3e4b613183bd4b969c45d0740328ba6e1d307b4f67dbfd63e991fb9ef902a0ea.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/3e4b613183bd4b969c45d0740328ba6e1d307b4f67dbfd63e991fb9ef902a0ea.jpg)

![41bc4d7fcf55e29638d710edaa2d1866bb50a244e4f7ff722f1f0c345bc5ea8e.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/41bc4d7fcf55e29638d710edaa2d1866bb50a244e4f7ff722f1f0c345bc5ea8e.jpg)

![4c59361ce4b825b2a289dec9af7e68723c410b77f68862b8891d503c36924782.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/4c59361ce4b825b2a289dec9af7e68723c410b77f68862b8891d503c36924782.jpg)

![67501f8add5e77052bef3bfe01c99e56a8edb2a04f158fb21110b437d8a23915.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/67501f8add5e77052bef3bfe01c99e56a8edb2a04f158fb21110b437d8a23915.jpg)

![852bb1b29bc194f107fbc0dadb04c8c0f07e2cc390adb1e48a0909eb3ba406c5.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/852bb1b29bc194f107fbc0dadb04c8c0f07e2cc390adb1e48a0909eb3ba406c5.jpg)

![d7c4eddb3765e802585c323b9734a6608ab77f1841e242614bbe62a51f824d29.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/d7c4eddb3765e802585c323b9734a6608ab77f1841e242614bbe62a51f824d29.jpg)

![f7b2a1744733303bdc83faa66edb271ec0c195720f99090c6a952f3420509af4.jpg](../iclr_results/1077_N-ForGOT_ Towards Not-forgetting and Generalization of Open Temporal Graph Learning/tables/f7b2a1744733303bdc83faa66edb271ec0c195720f99090c6a952f3420509af4.jpg)

## Systematic Outliers in Large Language Models


### Images

![05bacc9a61c1c15f5785b14ae358428cb73224011e77c02eec61739a488d8c7f.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/05bacc9a61c1c15f5785b14ae358428cb73224011e77c02eec61739a488d8c7f.jpg)

![142dbabd3be31395e6ad5298b0ade7e3f39cd3fc8ebe02ec4ff5039f1ad64cfb.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/142dbabd3be31395e6ad5298b0ade7e3f39cd3fc8ebe02ec4ff5039f1ad64cfb.jpg)

![28fa3a08a432858d6b8d8d1e6a4523099dee08d7617fc0c9a9875e30957cd1dd.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/28fa3a08a432858d6b8d8d1e6a4523099dee08d7617fc0c9a9875e30957cd1dd.jpg)

![2fb70e71f2bee18c4c12cb01cf1e4a73ac5c02bdb1c9ea9e95f102ac907ac6e5.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/2fb70e71f2bee18c4c12cb01cf1e4a73ac5c02bdb1c9ea9e95f102ac907ac6e5.jpg)

![3c7d7734b55e94bf33a9e98977422fac03b8c73bea7c6381808c57a686277435.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/3c7d7734b55e94bf33a9e98977422fac03b8c73bea7c6381808c57a686277435.jpg)

![3de2082bf24dd93dd837c112e87bd562281c8151f861037fddd2ef3b8e8c9724.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/3de2082bf24dd93dd837c112e87bd562281c8151f861037fddd2ef3b8e8c9724.jpg)

![41ff9770644c87e30e37a810d9f89dce19a9f0999118e0c8d2d5b1fee68f12a0.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/41ff9770644c87e30e37a810d9f89dce19a9f0999118e0c8d2d5b1fee68f12a0.jpg)

![527f00e804260783f5861a4ba3a1001b854802273ae7f43d4b629b2f728688d6.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/527f00e804260783f5861a4ba3a1001b854802273ae7f43d4b629b2f728688d6.jpg)

![528ca87eac0bbd18afd34ae9ca0968ab993625e951ec95fa9f2d7090a83584c1.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/528ca87eac0bbd18afd34ae9ca0968ab993625e951ec95fa9f2d7090a83584c1.jpg)

![79b692c3a4ebcc58c9710df9238284992351c7c0a4678b4ff277e3a58d452c02.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/79b692c3a4ebcc58c9710df9238284992351c7c0a4678b4ff277e3a58d452c02.jpg)

![9049a08db032aa5789691af9ef091517815e0cdae3e6b81ebe2e0c92962c8849.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/9049a08db032aa5789691af9ef091517815e0cdae3e6b81ebe2e0c92962c8849.jpg)

![9ab56715623c293aaad7b27581bbbe3d2a35add75607bad6c09655df41638aa3.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/9ab56715623c293aaad7b27581bbbe3d2a35add75607bad6c09655df41638aa3.jpg)

![a0214d1efce4fcb3e9dabc87609b8b6fdf45c1081a1229702d6a2e41ac386b73.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/a0214d1efce4fcb3e9dabc87609b8b6fdf45c1081a1229702d6a2e41ac386b73.jpg)

![b212525b75c817172cbea703b26f21c60ca2ecaf911f5c278cf77df2a0843975.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/b212525b75c817172cbea703b26f21c60ca2ecaf911f5c278cf77df2a0843975.jpg)

![b6cf9d2fd937ceaea7105c7efe6a98913e1215e2c0b211d217322b01b85de930.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/b6cf9d2fd937ceaea7105c7efe6a98913e1215e2c0b211d217322b01b85de930.jpg)

![c6262abd92cb69eae85566d419005e1533c8eefad918f264a65b9f065595f602.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/c6262abd92cb69eae85566d419005e1533c8eefad918f264a65b9f065595f602.jpg)

![cf8a49a08ba28f45df199d31faa0580797975ad381a88ab0e70466f7faf13854.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/cf8a49a08ba28f45df199d31faa0580797975ad381a88ab0e70466f7faf13854.jpg)

![db8f3a7892807a3bc3189673684a2f0b43b50fcff42f93189868c3757a7c32ff.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/db8f3a7892807a3bc3189673684a2f0b43b50fcff42f93189868c3757a7c32ff.jpg)

![e785a07fb38379c9456170d3dbc53336f464454b37f5545778c774068a668e8b.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/e785a07fb38379c9456170d3dbc53336f464454b37f5545778c774068a668e8b.jpg)

![e99170fa48212d13a32cd3a5bee4ec5557c9601a9b4a5ddb44828dbded5530c2.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/e99170fa48212d13a32cd3a5bee4ec5557c9601a9b4a5ddb44828dbded5530c2.jpg)

![eb9cb329877e215be429fab931814cacd996e6c4d6e93d5991693137015584af.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/eb9cb329877e215be429fab931814cacd996e6c4d6e93d5991693137015584af.jpg)

![ec6da11fea4510fdb2818ba95123a7d4e881ce20d448ffc8d45f5c88a9940ec7.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/ec6da11fea4510fdb2818ba95123a7d4e881ce20d448ffc8d45f5c88a9940ec7.jpg)

![edf4b0e0c221ce3ab794882ac3cd4862162f0309fb5a3117260c4f374f8d3189.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/edf4b0e0c221ce3ab794882ac3cd4862162f0309fb5a3117260c4f374f8d3189.jpg)

![fe2713240b1871958e74ad5aedbd5c353664a5d53219f0addbf3e11699c2b140.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/images/fe2713240b1871958e74ad5aedbd5c353664a5d53219f0addbf3e11699c2b140.jpg)

### Tables

![87184fb808b2a71cd6b39eaaf1a8bd0194b3342749b7f9034693c2f7a0f3ed04.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/tables/87184fb808b2a71cd6b39eaaf1a8bd0194b3342749b7f9034693c2f7a0f3ed04.jpg)

![dfdc1c7f143e90a2804160f5f79e0adc315fb76d35530cdbc877fcfcad00181d.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/tables/dfdc1c7f143e90a2804160f5f79e0adc315fb76d35530cdbc877fcfcad00181d.jpg)

![f8d06dff4db859b66368748848887d25c31fea1d859f79d629de0ec856c96589.jpg](../iclr_results/1078_Systematic Outliers in Large Language Models/tables/f8d06dff4db859b66368748848887d25c31fea1d859f79d629de0ec856c96589.jpg)

## Protecting against simultaneous data poisoning attacks


### Images

![080719fe8ff95b80197729cdebd95cb84fe4ee511e11de53d10ea8dc57fd482b.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/080719fe8ff95b80197729cdebd95cb84fe4ee511e11de53d10ea8dc57fd482b.jpg)

![1118d9e582a2bcc2920018abf882256320b629e737f67803cb2d9bbea5c1fb1c.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/1118d9e582a2bcc2920018abf882256320b629e737f67803cb2d9bbea5c1fb1c.jpg)

![1e2d5bb5308998aebbf5396bf3ff3117bea49827d6c6b4e4f084d3fe5ce3b266.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/1e2d5bb5308998aebbf5396bf3ff3117bea49827d6c6b4e4f084d3fe5ce3b266.jpg)

![21c1d51ddb29706c6ec0d386233b8c101156a2063bc60609830c85ec993f04c7.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/21c1d51ddb29706c6ec0d386233b8c101156a2063bc60609830c85ec993f04c7.jpg)

![369bcc18b38e2ac767831b22a3934c6f4b984c8e97552fb9b80adf160c310597.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/369bcc18b38e2ac767831b22a3934c6f4b984c8e97552fb9b80adf160c310597.jpg)

![655118605f096054cc01304138a8427a29936faf931fa4aed119fa8497d4e07a.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/655118605f096054cc01304138a8427a29936faf931fa4aed119fa8497d4e07a.jpg)

![7e991c90f2ef640d4ff09983448bc1d195795a7e0d1c360e701914b79c2692e9.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/7e991c90f2ef640d4ff09983448bc1d195795a7e0d1c360e701914b79c2692e9.jpg)

![8da5ad725977c71c3b865eaf23c16d6b738025b2471ca3e884dc072c68aff782.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/8da5ad725977c71c3b865eaf23c16d6b738025b2471ca3e884dc072c68aff782.jpg)

![a7ab681fc90b06290494bcf2f263bc92d1cca2c73f86d6b7827c2d3bbcf1e4e6.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/a7ab681fc90b06290494bcf2f263bc92d1cca2c73f86d6b7827c2d3bbcf1e4e6.jpg)

![db215345f09752c8a48bde73f47d94c3abb3aa0df21c1b63699f1c6929b7ef9f.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/db215345f09752c8a48bde73f47d94c3abb3aa0df21c1b63699f1c6929b7ef9f.jpg)

![db9cc79c0ffb5cff26028da05f322d7ef5b7e540249bbb3f585007d59627625f.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/images/db9cc79c0ffb5cff26028da05f322d7ef5b7e540249bbb3f585007d59627625f.jpg)

### Tables

![101ac968f03bb447b690ec099de5d73daa61626c89e25311b75da5d1b38e1a5e.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/tables/101ac968f03bb447b690ec099de5d73daa61626c89e25311b75da5d1b38e1a5e.jpg)

![370939c3ad3d0749a0f2c0f39e80d828fd335d855a267133bd780c1a2632149b.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/tables/370939c3ad3d0749a0f2c0f39e80d828fd335d855a267133bd780c1a2632149b.jpg)

![83fae2c076500f45345143c860d18de8f4da46aa73a3ca2e93d00aa8e0a1c263.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/tables/83fae2c076500f45345143c860d18de8f4da46aa73a3ca2e93d00aa8e0a1c263.jpg)

![ae6f422e811fe3237d857ec26f928b04d47ff09985da8405e7f0194c6f899877.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/tables/ae6f422e811fe3237d857ec26f928b04d47ff09985da8405e7f0194c6f899877.jpg)

![b7a0bd28647a1117e75a6cbd7af3fe06795143a86fcf6cc179dd57d3992269e8.jpg](../iclr_results/1079_Protecting against simultaneous data poisoning attacks/tables/b7a0bd28647a1117e75a6cbd7af3fe06795143a86fcf6cc179dd57d3992269e8.jpg)

## SEMDICE: Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation


### Images

![0ada183b7aaab0eb72dd995a5b976618d160aa0b728e8c2659590ee26572b2e7.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/0ada183b7aaab0eb72dd995a5b976618d160aa0b728e8c2659590ee26572b2e7.jpg)

![239e4db7dc8a7fe941533176c096327c7d9a25a792c942d35f7f99585daae67c.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/239e4db7dc8a7fe941533176c096327c7d9a25a792c942d35f7f99585daae67c.jpg)

![24f53ee2dd4acabc1a6a2c743eeb2e06e6c5277d0075925c4ebbb25f6430cc93.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/24f53ee2dd4acabc1a6a2c743eeb2e06e6c5277d0075925c4ebbb25f6430cc93.jpg)

![34f0233f200c26cace061ba650f1c4d5c15311b50ec0e9138650054992b63ec0.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/34f0233f200c26cace061ba650f1c4d5c15311b50ec0e9138650054992b63ec0.jpg)

![41d403b1f17cf97e91fb19ba6f9da3d6bb04fefaf5220f4e8a3b192a68c25910.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/41d403b1f17cf97e91fb19ba6f9da3d6bb04fefaf5220f4e8a3b192a68c25910.jpg)

![4c71d21cc2ec7935c25c3ca258360345bdb07cafaa05e3b6b3cf35922503b609.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/4c71d21cc2ec7935c25c3ca258360345bdb07cafaa05e3b6b3cf35922503b609.jpg)

![5ac2cf266249d7c3bee1d8757d1b94fb74b97f5b44167098881efb0e90485c80.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/5ac2cf266249d7c3bee1d8757d1b94fb74b97f5b44167098881efb0e90485c80.jpg)

![7b3f0a6e09baf6ffdeafc577d2b736fd545d63de31d17ac929ae7263badfc92d.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/7b3f0a6e09baf6ffdeafc577d2b736fd545d63de31d17ac929ae7263badfc92d.jpg)

![89ce32d177313a6d6506c0bc7375900758a2f093037ddcd1ecf3dabd52062cba.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/89ce32d177313a6d6506c0bc7375900758a2f093037ddcd1ecf3dabd52062cba.jpg)

![939bf09978b26acbccb7ef1abd6cf81d7a6f589ae5e1c68851a1d77ebee594b5.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/939bf09978b26acbccb7ef1abd6cf81d7a6f589ae5e1c68851a1d77ebee594b5.jpg)

![b0adaf508393535c3157902e8dc3b31a636016935bef5939177fe246e689c3f2.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/b0adaf508393535c3157902e8dc3b31a636016935bef5939177fe246e689c3f2.jpg)

![ba1d6cdbbd1151f4b0a9bf5bfed11fc008d58e0c7bbf8e01c2fb34b825b03193.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/ba1d6cdbbd1151f4b0a9bf5bfed11fc008d58e0c7bbf8e01c2fb34b825b03193.jpg)

![c2f0da100c26a475772abcdc3f35c3792c7ae1342d97e21cbf26f724e92f130c.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/images/c2f0da100c26a475772abcdc3f35c3792c7ae1342d97e21cbf26f724e92f130c.jpg)

### Tables

![3ff02ccd5d62bf1b81f83c4364d1dfb8803c16058830698c125d4cbf006d026b.jpg](../iclr_results/1080_SEMDICE_ Off-policy State Entropy Maximization via Stationary Distribution Correction Estimation/tables/3ff02ccd5d62bf1b81f83c4364d1dfb8803c16058830698c125d4cbf006d026b.jpg)

## Generating Likely Counterfactuals Using Sum-Product Networks


### Images

![13e915367fc167053a2a39a9dea46f263479964224dd1f24658bc0517165245d.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/images/13e915367fc167053a2a39a9dea46f263479964224dd1f24658bc0517165245d.jpg)

![2f30373351501c78d9055f281fbe40094459a8cd0e087b91ce308d7dc8e8bd04.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/images/2f30373351501c78d9055f281fbe40094459a8cd0e087b91ce308d7dc8e8bd04.jpg)

![815841d73f86f364038bdddbd5bac17e9cae9cd147a3b6b92ad4fa1b90f6a498.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/images/815841d73f86f364038bdddbd5bac17e9cae9cd147a3b6b92ad4fa1b90f6a498.jpg)

![a1af0dabcc6f1e535075b64a50878770c6ad7946446a0622e06014d541f8c4ef.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/images/a1af0dabcc6f1e535075b64a50878770c6ad7946446a0622e06014d541f8c4ef.jpg)

### Tables

![065ba08bed7cf45385050d25642b296118079383ce2af8f93bdea251f3f9764d.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/065ba08bed7cf45385050d25642b296118079383ce2af8f93bdea251f3f9764d.jpg)

![36f2d7ed00030bede15f042ce7a3747f18898b56557c4d45aa1d147357ceb0bf.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/36f2d7ed00030bede15f042ce7a3747f18898b56557c4d45aa1d147357ceb0bf.jpg)

![44c1396309fb8bb92be9815281b903da5a9596cf149504817f55ddf1fe7f9a11.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/44c1396309fb8bb92be9815281b903da5a9596cf149504817f55ddf1fe7f9a11.jpg)

![4afa98891871e81482dbc5d8ba2dfecfd7cdec32f339c57cbc09b9b368313643.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/4afa98891871e81482dbc5d8ba2dfecfd7cdec32f339c57cbc09b9b368313643.jpg)

![6517cfb62920418631da7d3e4e60b289b13759c183efd0cacbf34831730b6ab2.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/6517cfb62920418631da7d3e4e60b289b13759c183efd0cacbf34831730b6ab2.jpg)

![7364ee8942736a23b60dafa0a483b65fa6cb6e706cc28029a2c8f2729ec5fcdf.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/7364ee8942736a23b60dafa0a483b65fa6cb6e706cc28029a2c8f2729ec5fcdf.jpg)

![7c0a4ddea6fa7ebb83eaa323362f6bc40e469efb3cddd7bb40092c1b6e5e65d7.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/7c0a4ddea6fa7ebb83eaa323362f6bc40e469efb3cddd7bb40092c1b6e5e65d7.jpg)

![7eda23417902baf6cb010aab2e0478816f1a714037fac3b34b6c1ae1b6e35268.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/7eda23417902baf6cb010aab2e0478816f1a714037fac3b34b6c1ae1b6e35268.jpg)

![81ec9a88ba59e1f065ef4d38900279dfe37d8811dbea273c872fab481e7974e2.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/81ec9a88ba59e1f065ef4d38900279dfe37d8811dbea273c872fab481e7974e2.jpg)

![8668ac0a29ca28106127ac32ce9156987e9799ef293c8aa6649209ef2b4da797.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/8668ac0a29ca28106127ac32ce9156987e9799ef293c8aa6649209ef2b4da797.jpg)

![94ceadab4dc538654cf5e5ffb20e4c125263a2a9bcf14f427a63c898bbb7f401.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/94ceadab4dc538654cf5e5ffb20e4c125263a2a9bcf14f427a63c898bbb7f401.jpg)

![96bb2b606e7a294fd422ca7db4bb2cd8e14ebc5f0b0b6656ab5a245d5e08e8fb.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/96bb2b606e7a294fd422ca7db4bb2cd8e14ebc5f0b0b6656ab5a245d5e08e8fb.jpg)

![a68ecc36de6a29b16c8d92f1be3ba05461ea49621e78dc12b645acc0c79cc1ff.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/a68ecc36de6a29b16c8d92f1be3ba05461ea49621e78dc12b645acc0c79cc1ff.jpg)

![b1f8373ccdc897297318ef37b0deea5a829611d140a5c6bd95431f90a6a0d04f.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/b1f8373ccdc897297318ef37b0deea5a829611d140a5c6bd95431f90a6a0d04f.jpg)

![b591079a0f8c4e5ee85690be48f978546cbce0849bb05acb89984c4bdb749533.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/b591079a0f8c4e5ee85690be48f978546cbce0849bb05acb89984c4bdb749533.jpg)

![b87e6fa08825253a00498afcaedd2e600fadb242e7aba9572206e4e0526cca81.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/b87e6fa08825253a00498afcaedd2e600fadb242e7aba9572206e4e0526cca81.jpg)

![db9720983e66e75785d467c9176d94b702040f5625aab300b6bc9376f46baeab.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/db9720983e66e75785d467c9176d94b702040f5625aab300b6bc9376f46baeab.jpg)

![e00bd0e3e4d4e8664bb3e57fd00d47879a6719883b308719d10db0186e0268b1.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/e00bd0e3e4d4e8664bb3e57fd00d47879a6719883b308719d10db0186e0268b1.jpg)

![ec411dda1e33ec2af4c5eca566d913e9d1b84fa8b51db9a49b3380ba3ec043bc.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/ec411dda1e33ec2af4c5eca566d913e9d1b84fa8b51db9a49b3380ba3ec043bc.jpg)

![f188a5438ba371fbf7d3133dae3fbec5c0ac4d67947bdd27153e3596d25fba01.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/f188a5438ba371fbf7d3133dae3fbec5c0ac4d67947bdd27153e3596d25fba01.jpg)

![fda2748f49fffd3aac726847a3bca5c5717500ed8cde19422519bbfe6aa7d843.jpg](../iclr_results/1081_Generating Likely Counterfactuals Using Sum-Product Networks/tables/fda2748f49fffd3aac726847a3bca5c5717500ed8cde19422519bbfe6aa7d843.jpg)

## Metric-Driven Attributions for Vision Transformers


### Images

![08a589476710c58e835c11fe80cdd0f14a5360fe12d1cda5d6b91173c9044306.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/08a589476710c58e835c11fe80cdd0f14a5360fe12d1cda5d6b91173c9044306.jpg)

![33d202b0934d39bba0a049c7ad09a119e1a768ed1821765d4d745ada7124447b.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/33d202b0934d39bba0a049c7ad09a119e1a768ed1821765d4d745ada7124447b.jpg)

![364db3c627e9119227b4f11614afa3b61c709f1a84b4d7c8490d5ae01a88bf31.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/364db3c627e9119227b4f11614afa3b61c709f1a84b4d7c8490d5ae01a88bf31.jpg)

![3ae38c37b508c6316dfcdbd66ac27978256d9346395ffd6d0a7fcc01115893e7.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/3ae38c37b508c6316dfcdbd66ac27978256d9346395ffd6d0a7fcc01115893e7.jpg)

![438eb964c12374bdf4c053468e388ee5d08e5f78e0d7e430695ecaaf50b669cf.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/438eb964c12374bdf4c053468e388ee5d08e5f78e0d7e430695ecaaf50b669cf.jpg)

![4aadd87c161dcafe2495fbbe9f32c5bdc88bdf460d08b5d092c4e4293f60d36b.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/4aadd87c161dcafe2495fbbe9f32c5bdc88bdf460d08b5d092c4e4293f60d36b.jpg)

![4f317cb1222d0b93d41bf5123fe5f5901c407732bd0ba9f268911274252e4932.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/4f317cb1222d0b93d41bf5123fe5f5901c407732bd0ba9f268911274252e4932.jpg)

![578eed3b707df03a817cff94c9082122ba5a616b4ba758a9f85ca4ce5c50fbec.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/578eed3b707df03a817cff94c9082122ba5a616b4ba758a9f85ca4ce5c50fbec.jpg)

![6183b6bc23dad19fa1e614757bfa42f68c19da05c2cd6c3535ac643eb8eae488.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/6183b6bc23dad19fa1e614757bfa42f68c19da05c2cd6c3535ac643eb8eae488.jpg)

![6845f01c90e8708208cd50414ef928e779ae65538bc61fcb15e5e328428a6100.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/6845f01c90e8708208cd50414ef928e779ae65538bc61fcb15e5e328428a6100.jpg)

![72de90225086d4940720000a64f96dc7b29064f5101042db9192dced55e3c7ea.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/72de90225086d4940720000a64f96dc7b29064f5101042db9192dced55e3c7ea.jpg)

![74f1b8646c5d95055b367b94d31bd9500fa0427341e0feacedb81c54539ec923.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/74f1b8646c5d95055b367b94d31bd9500fa0427341e0feacedb81c54539ec923.jpg)

![753c5da89a8e53be19fdc5c9b6d166080111e3d5a244a976e68c94e37e89e0f7.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/753c5da89a8e53be19fdc5c9b6d166080111e3d5a244a976e68c94e37e89e0f7.jpg)

![7e56289705d6d8ee6a13b46eb2c9c016ec793565dc3b431fc09c9b5c651ceb57.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/7e56289705d6d8ee6a13b46eb2c9c016ec793565dc3b431fc09c9b5c651ceb57.jpg)

![89bb0920fa534fa497f5d7254f40e3d8f3d8125bf489d9e8ba93d3814179b39a.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/89bb0920fa534fa497f5d7254f40e3d8f3d8125bf489d9e8ba93d3814179b39a.jpg)

![92b3a297bd3a3d89c0be23cbd9e676345a8d01a2aa6386cdeb3fde3bfcc03519.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/92b3a297bd3a3d89c0be23cbd9e676345a8d01a2aa6386cdeb3fde3bfcc03519.jpg)

![a3430e4ed953ca703a00d1bd83020b6273de6c2b9390d21dfd67f889cd8e1234.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/a3430e4ed953ca703a00d1bd83020b6273de6c2b9390d21dfd67f889cd8e1234.jpg)

![b1a26ca216539138874d6327e21af01964d4af70d9eb39ee08f1010207cd948f.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/b1a26ca216539138874d6327e21af01964d4af70d9eb39ee08f1010207cd948f.jpg)

![b9f8b1f78234cd48410b9155f27a5a68b2df192560c8653d472f5811681e5a84.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/b9f8b1f78234cd48410b9155f27a5a68b2df192560c8653d472f5811681e5a84.jpg)

![c0da4b947dec7e3432c7a87ddb9da1515f0091397a5956b7d8fa5461df2d63ae.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/c0da4b947dec7e3432c7a87ddb9da1515f0091397a5956b7d8fa5461df2d63ae.jpg)

![cc684442652a27b30056d00e9df6c80274d137a01c95c9208b6350a762aa5bdc.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/cc684442652a27b30056d00e9df6c80274d137a01c95c9208b6350a762aa5bdc.jpg)

![d4a6bb11986e6a5555ba40a9f11861d4289e8b4e7f09628101cc8727174caf2f.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/d4a6bb11986e6a5555ba40a9f11861d4289e8b4e7f09628101cc8727174caf2f.jpg)

![d605a56ea0bc21abe0a2e1a802c9a12dfca11d7a323e024248b96a0072f047ab.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/d605a56ea0bc21abe0a2e1a802c9a12dfca11d7a323e024248b96a0072f047ab.jpg)

![e10403061a30752b313646bc16baa5e36dd62fa624e19d3cc25bc75adfe9449a.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/e10403061a30752b313646bc16baa5e36dd62fa624e19d3cc25bc75adfe9449a.jpg)

![e80ae7df7c3f5cbff54d474ec908cfc475a75b02802f834de22186939db37336.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/e80ae7df7c3f5cbff54d474ec908cfc475a75b02802f834de22186939db37336.jpg)

![f3f8238eecfcc9dfdbbd14e53ce95094b2782ce4387d7937ffb570d58d4c323d.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/f3f8238eecfcc9dfdbbd14e53ce95094b2782ce4387d7937ffb570d58d4c323d.jpg)

![fecac3d72bb42ab47fe243f2c05686a03bd1550e428ad6c4bc5d1ed573d32567.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/fecac3d72bb42ab47fe243f2c05686a03bd1550e428ad6c4bc5d1ed573d32567.jpg)

![fff204eebc311dff472b0bf50578322ae153e91c6d8b3b00a1d4706ef2e36e4b.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/images/fff204eebc311dff472b0bf50578322ae153e91c6d8b3b00a1d4706ef2e36e4b.jpg)

### Tables

![03a6dd53e898dca1c754b4e3e58028af69739caf4e4ede5c46a5dac8b8ce0e0c.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/03a6dd53e898dca1c754b4e3e58028af69739caf4e4ede5c46a5dac8b8ce0e0c.jpg)

![065fcda41de54a1cb58ebae6e8177438b06a81bb71814994def1150aebf8629f.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/065fcda41de54a1cb58ebae6e8177438b06a81bb71814994def1150aebf8629f.jpg)

![199d44be0a372fcf2daeaaa558b5a6d053005c62263e15134498f8503cfe4665.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/199d44be0a372fcf2daeaaa558b5a6d053005c62263e15134498f8503cfe4665.jpg)

![1f9b5a08b837d3a89ed789cab46bb65ad737b2a68783278ac9207891852d54b2.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/1f9b5a08b837d3a89ed789cab46bb65ad737b2a68783278ac9207891852d54b2.jpg)

![36b85a1c85c12c006d116453cdfddd595ea08cbb76a6cfb83992709753fe4499.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/36b85a1c85c12c006d116453cdfddd595ea08cbb76a6cfb83992709753fe4499.jpg)

![504fe2115feff83a8a4b31b4ad6facecfbeea87116c2c122a8cef1970462368f.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/504fe2115feff83a8a4b31b4ad6facecfbeea87116c2c122a8cef1970462368f.jpg)

![5aa9c4e81a951f083979716ccefd2d23a0f2f232b340a31d635eab2724dbdbcf.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/5aa9c4e81a951f083979716ccefd2d23a0f2f232b340a31d635eab2724dbdbcf.jpg)

![7432cb0c08b877ad1b9859e6fe6a83e1e80d49571386fc161d178bf2c91d4507.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/7432cb0c08b877ad1b9859e6fe6a83e1e80d49571386fc161d178bf2c91d4507.jpg)

![86c64ce5039e7e813c7bfcf76fc67aa41813f01efc44ca1cc373aff0a3218547.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/86c64ce5039e7e813c7bfcf76fc67aa41813f01efc44ca1cc373aff0a3218547.jpg)

![e565588afa17a179476d0f4c00e1b78758a0fe73a789c36b461855bb6b9ae622.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/e565588afa17a179476d0f4c00e1b78758a0fe73a789c36b461855bb6b9ae622.jpg)

![f2df8ed00ee8e81e137dc69d100245cc33460848d9135755d411047ec3bc3edd.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/f2df8ed00ee8e81e137dc69d100245cc33460848d9135755d411047ec3bc3edd.jpg)

![f330d9dd3ca500ad68d9b07a0fccefc2b01d90ed216dc96c850f301f69175271.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/f330d9dd3ca500ad68d9b07a0fccefc2b01d90ed216dc96c850f301f69175271.jpg)

![f4b080322a99135fc1ac7567eacb42849c11adf2631c5dc41e0b1b3027fbe089.jpg](../iclr_results/1082_Metric-Driven Attributions for Vision Transformers/tables/f4b080322a99135fc1ac7567eacb42849c11adf2631c5dc41e0b1b3027fbe089.jpg)

## Greener GRASS: Enhancing GNNs with Encoding, Rewiring, and Attention


### Images

![52fc8639709f3329c6585645775bc0ce5b1c0532ff9fe1eea765953e0b5d7fcf.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/images/52fc8639709f3329c6585645775bc0ce5b1c0532ff9fe1eea765953e0b5d7fcf.jpg)

![5ea5d53fcfd3ebc2ea02bd7e89ca8d53aff5d82cb42892bc3cfb563452ff77cf.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/images/5ea5d53fcfd3ebc2ea02bd7e89ca8d53aff5d82cb42892bc3cfb563452ff77cf.jpg)

![959432810e20ec142cc5d8e9d0a1746990e6c1279d949bb07b9e674c95b896a6.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/images/959432810e20ec142cc5d8e9d0a1746990e6c1279d949bb07b9e674c95b896a6.jpg)

![c426cb3054cd02e8d629ddc67b5d6de76293d5c4d151a09aa415a4f155942bc8.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/images/c426cb3054cd02e8d629ddc67b5d6de76293d5c4d151a09aa415a4f155942bc8.jpg)

### Tables

![01e034acdbfa88fac739ee555661f58573583ffe1ee8edfe9f731a1f4f0fe039.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/01e034acdbfa88fac739ee555661f58573583ffe1ee8edfe9f731a1f4f0fe039.jpg)

![1c4dbdee5a1f27002bb53b779371b882fe11dd26c99367d1f05cc1379a14e76c.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/1c4dbdee5a1f27002bb53b779371b882fe11dd26c99367d1f05cc1379a14e76c.jpg)

![20556cb835a0177fec58385507def2303f429ca51e2645bd221f0e6562d0db3e.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/20556cb835a0177fec58385507def2303f429ca51e2645bd221f0e6562d0db3e.jpg)

![362c6e530eac5be9a83260afbb7aa6c14ff5cfebb6c9d59ba20b4570c5e34f9b.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/362c6e530eac5be9a83260afbb7aa6c14ff5cfebb6c9d59ba20b4570c5e34f9b.jpg)

![3eb7731b0df5c2fa1e6545714bbc0a860c6e36f49cf3d94d938252448bf165c0.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/3eb7731b0df5c2fa1e6545714bbc0a860c6e36f49cf3d94d938252448bf165c0.jpg)

![53b56040f7427da1f8670d26f9384d95bfaac81f1df9c0d26e28bea9ca164fc4.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/53b56040f7427da1f8670d26f9384d95bfaac81f1df9c0d26e28bea9ca164fc4.jpg)

![734dde077a66044c088fe3e89e90e5c34add8eb572bdd585b9d741ad7ee258b4.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/734dde077a66044c088fe3e89e90e5c34add8eb572bdd585b9d741ad7ee258b4.jpg)

![9994209209d03ff67776a3467d326d9f14a985a2c3aa92dce4dbc60cb5090c34.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/9994209209d03ff67776a3467d326d9f14a985a2c3aa92dce4dbc60cb5090c34.jpg)

![b9576a4d6f525f2a197bdde8e883b61b4352fe71dda73e61d08c42b6be2d4d32.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/b9576a4d6f525f2a197bdde8e883b61b4352fe71dda73e61d08c42b6be2d4d32.jpg)

![bc2f7ec14e28d2c43019316ef6913d9f44d592479d1f11d7b40c082fb109c27e.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/bc2f7ec14e28d2c43019316ef6913d9f44d592479d1f11d7b40c082fb109c27e.jpg)

![c4d57da46a76f91790189a9aff3b88a6f033322bd0fc392b408ec307038bee9b.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/c4d57da46a76f91790189a9aff3b88a6f033322bd0fc392b408ec307038bee9b.jpg)

![e27ca9cf70386d02eca87426b8e45e4e7dec5f2edc84848c33a1d9775092aeeb.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/e27ca9cf70386d02eca87426b8e45e4e7dec5f2edc84848c33a1d9775092aeeb.jpg)

![fd51fa9c4dd9857cfb5c3f76d2960f2a1c31a3d9dbab6eff9297433ec8393d2a.jpg](../iclr_results/1083_Greener GRASS_ Enhancing GNNs with Encoding, Rewiring, and Attention/tables/fd51fa9c4dd9857cfb5c3f76d2960f2a1c31a3d9dbab6eff9297433ec8393d2a.jpg)

## TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting


### Images

![0214f25db273efda6d13cac31860f123b561ba44016e047b66b6db9105a26b9f.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/images/0214f25db273efda6d13cac31860f123b561ba44016e047b66b6db9105a26b9f.jpg)

![513ad60fe0a14d5f290bf24fd56c94404e49d2b359747a1610cfad83fcb71f66.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/images/513ad60fe0a14d5f290bf24fd56c94404e49d2b359747a1610cfad83fcb71f66.jpg)

![80d87af493b164394237ca0ff51bc1230ffdc2a8d069ad8dfff9725421442648.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/images/80d87af493b164394237ca0ff51bc1230ffdc2a8d069ad8dfff9725421442648.jpg)

![8c13f39be529fbeecce1e36b13d88cc6563ab6b23d8da3a8dc8e1a2dbd42d1ff.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/images/8c13f39be529fbeecce1e36b13d88cc6563ab6b23d8da3a8dc8e1a2dbd42d1ff.jpg)

![aa1668f3cc30ef51412afcdb276acb35e6f2e3bd64747d7bbf511c53b9767d51.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/images/aa1668f3cc30ef51412afcdb276acb35e6f2e3bd64747d7bbf511c53b9767d51.jpg)

![b78f84aadc47f3939a22db76d68dad6500f45fc62e115e689f30922e14d0936a.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/images/b78f84aadc47f3939a22db76d68dad6500f45fc62e115e689f30922e14d0936a.jpg)

![cf4710411d036e68c28e9530d41a214860c686cba642021b164366301c69e8f5.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/images/cf4710411d036e68c28e9530d41a214860c686cba642021b164366301c69e8f5.jpg)

### Tables

![23075d9995a359b706f0fae2d74f4e5124cdc5ae470b4d2ea837f7be7bb0a1f6.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/23075d9995a359b706f0fae2d74f4e5124cdc5ae470b4d2ea837f7be7bb0a1f6.jpg)

![2d590f424846fccd81515944c9305165ff2d5a5f8a88370a21ed234e66f208a8.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/2d590f424846fccd81515944c9305165ff2d5a5f8a88370a21ed234e66f208a8.jpg)

![4f98ef36be2c3ec47937f14428874457cc5197e0835099ef201e2c5b3c1736e0.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/4f98ef36be2c3ec47937f14428874457cc5197e0835099ef201e2c5b3c1736e0.jpg)

![6896361ebcecf7c3e71f7bc5cbd1de67c876464a465a372c8d1ef764c7846d45.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/6896361ebcecf7c3e71f7bc5cbd1de67c876464a465a372c8d1ef764c7846d45.jpg)

![7faf0d36aed65becb66dbf92e1bf5059f387d6199666c654dae527c933b8c628.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/7faf0d36aed65becb66dbf92e1bf5059f387d6199666c654dae527c933b8c628.jpg)

![90c4eb16825666f12716a46cca07e8a333b999827d3b1290245c342e256bbafc.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/90c4eb16825666f12716a46cca07e8a333b999827d3b1290245c342e256bbafc.jpg)

![bf89528a3719533b85faab771a1d16e5ebcb349fe7c3302e1c2f1c4b67a97e75.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/bf89528a3719533b85faab771a1d16e5ebcb349fe7c3302e1c2f1c4b67a97e75.jpg)

![c43ab03f8a6afa33034e5e604aa9426285db241ce3bef0ca8e5d44e2ba967850.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/c43ab03f8a6afa33034e5e604aa9426285db241ce3bef0ca8e5d44e2ba967850.jpg)

![cb415cb9c500073f280df1ce8b07e6f06408418b1f6b22c47aa3dc3b30682e2b.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/cb415cb9c500073f280df1ce8b07e6f06408418b1f6b22c47aa3dc3b30682e2b.jpg)

![ce3c760958e8e6d8b9c793b4c27691f6ba2c361f2b8059025689c5e6f938f457.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/ce3c760958e8e6d8b9c793b4c27691f6ba2c361f2b8059025689c5e6f938f457.jpg)

![ec67147322247b432cb6badfb83ec50561dd449970659bbbf415adcf6cf404d0.jpg](../iclr_results/1084_TS-LIF_ A Temporal Segment Spiking Neuron Network for Time Series Forecasting/tables/ec67147322247b432cb6badfb83ec50561dd449970659bbbf415adcf6cf404d0.jpg)

## Exposure Bracketing Is All You Need For A High-Quality Image


### Images

![16c7e50ebc51a00ad100d6685035ef536b39d07dea687ffc74cdf1873705736f.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/16c7e50ebc51a00ad100d6685035ef536b39d07dea687ffc74cdf1873705736f.jpg)

![1c5c1150979e0fc6e828e862b624509badb3281cd401288fa1bba8d6dc66ea68.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/1c5c1150979e0fc6e828e862b624509badb3281cd401288fa1bba8d6dc66ea68.jpg)

![25bfbdad8f7febca987129424ef4f83011cadfdca8ba653f3261fc295adec490.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/25bfbdad8f7febca987129424ef4f83011cadfdca8ba653f3261fc295adec490.jpg)

![2aaf2d48d13ddf2c3486920b87dc3ad2aa75363b3189a84eea36c5362ef02af7.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/2aaf2d48d13ddf2c3486920b87dc3ad2aa75363b3189a84eea36c5362ef02af7.jpg)

![8a0cc4389fda1ea8c57c0a0061701038fc60c29de9fb5f835125dc55ed841dfd.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/8a0cc4389fda1ea8c57c0a0061701038fc60c29de9fb5f835125dc55ed841dfd.jpg)

![9f2dcac08ef7085dc51fb45f5966798d8718cbc96be893dc2025f2be2ebbe800.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/9f2dcac08ef7085dc51fb45f5966798d8718cbc96be893dc2025f2be2ebbe800.jpg)

![ab2c8b36cea3b0ad1d3bffe350f57d452f033a0aaee215b6aead3ac23bf1371a.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/ab2c8b36cea3b0ad1d3bffe350f57d452f033a0aaee215b6aead3ac23bf1371a.jpg)

![c2ca546bf8df9d38c49fce09254398f0c65d7512c7cc97aadf1a7005d5e466bf.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/c2ca546bf8df9d38c49fce09254398f0c65d7512c7cc97aadf1a7005d5e466bf.jpg)

![cd24221d3e41015fc68ba66237fcb9c91b3b87721f621c18acc261e72dd9cb5e.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/cd24221d3e41015fc68ba66237fcb9c91b3b87721f621c18acc261e72dd9cb5e.jpg)

![cdd234c4948c0315b949478237de062ed6d9db2f9852fb7203aeeece0a9d449d.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/cdd234c4948c0315b949478237de062ed6d9db2f9852fb7203aeeece0a9d449d.jpg)

![ced3f76ec23c49d7afca3d1e6d4252ce70ac30910d448416fc878f9cbfde49c6.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/ced3f76ec23c49d7afca3d1e6d4252ce70ac30910d448416fc878f9cbfde49c6.jpg)

![eb4eee09c077641a4ba5dd373b2d1197ee6314fc8e24a9a59ef25fdfc23ef11c.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/eb4eee09c077641a4ba5dd373b2d1197ee6314fc8e24a9a59ef25fdfc23ef11c.jpg)

![faefe42e2c6a5121ea006be042b98060f85ed17e7c14a2db6aa2e1948ff12ca1.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/images/faefe42e2c6a5121ea006be042b98060f85ed17e7c14a2db6aa2e1948ff12ca1.jpg)

### Tables

![384a8a5cc3ec9d9f99f56c93c2e7cc60a7c76e203ae5a30d445619f479b4b96a.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/384a8a5cc3ec9d9f99f56c93c2e7cc60a7c76e203ae5a30d445619f479b4b96a.jpg)

![393fb8bbb498a0d6c164720c22ce2c37b9aee2e7a5fe5d96c778086a4e7ec132.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/393fb8bbb498a0d6c164720c22ce2c37b9aee2e7a5fe5d96c778086a4e7ec132.jpg)

![399783dbd7aabfc59bfb5a3f1974c3209817bffd001daa196430eaee7c8245ed.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/399783dbd7aabfc59bfb5a3f1974c3209817bffd001daa196430eaee7c8245ed.jpg)

![3f26332d60ba329b93a2a570e2f15c983563831b355c008ab50173a8ca2a8d72.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/3f26332d60ba329b93a2a570e2f15c983563831b355c008ab50173a8ca2a8d72.jpg)

![47ece47d3663398a0f96d9d94f6c8f526ad4479fdd2564d7f4092a6a2c47ef73.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/47ece47d3663398a0f96d9d94f6c8f526ad4479fdd2564d7f4092a6a2c47ef73.jpg)

![4b8740a7608b105cbd544d09f2a41bab8a5eed649eb57132fbbd892fdf8abd96.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/4b8740a7608b105cbd544d09f2a41bab8a5eed649eb57132fbbd892fdf8abd96.jpg)

![54a60210230ceb8109a4ed4bca160c3e36892b7d0fc57725e20942f1cfa462e0.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/54a60210230ceb8109a4ed4bca160c3e36892b7d0fc57725e20942f1cfa462e0.jpg)

![804fe38d2dae049a1a92cedcd5a2b167daca1b5006bb328f67f5cc005b611c00.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/804fe38d2dae049a1a92cedcd5a2b167daca1b5006bb328f67f5cc005b611c00.jpg)

![9ab9dda7815a761ba4c0062bc806a5c018d54a5ccd3992533140cf62a0357da9.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/9ab9dda7815a761ba4c0062bc806a5c018d54a5ccd3992533140cf62a0357da9.jpg)

![a333fff3edb86a24f5fd643a731f19ea40fee7146dfe54b725feb856f375c8f3.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/a333fff3edb86a24f5fd643a731f19ea40fee7146dfe54b725feb856f375c8f3.jpg)

![d2eb1997655176fce0f43e86b3ebd92a5b5c121163be977f013517aa721db193.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/d2eb1997655176fce0f43e86b3ebd92a5b5c121163be977f013517aa721db193.jpg)

![e4c6671216014a60ff93056655035c88d75ea88309134459feafefe35ae89534.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/e4c6671216014a60ff93056655035c88d75ea88309134459feafefe35ae89534.jpg)

![e7286c4618e62a96376f4cd852cb77a441f1f084920ea356f6ddd78649986b21.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/e7286c4618e62a96376f4cd852cb77a441f1f084920ea356f6ddd78649986b21.jpg)

![fcc4c5259a9320bb36a02ba1ea99a7648d6d6dc3eefa177db290723b937a15d6.jpg](../iclr_results/1085_Exposure Bracketing Is All You Need For A High-Quality Image/tables/fcc4c5259a9320bb36a02ba1ea99a7648d6d6dc3eefa177db290723b937a15d6.jpg)

## Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving


### Images

![2ee8f2ab7be5a6ee497797e9ab84ff84ecb3b1cf872a6d3dcf06877444949bee.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/images/2ee8f2ab7be5a6ee497797e9ab84ff84ecb3b1cf872a6d3dcf06877444949bee.jpg)

![556fe8a2634c5584aec428fa63f7235926461849a15798e76f3186526481ae0e.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/images/556fe8a2634c5584aec428fa63f7235926461849a15798e76f3186526481ae0e.jpg)

![609fc193a965f10152366a9ddf86074aae31f24e901c96069ddba3cf54e9344e.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/images/609fc193a965f10152366a9ddf86074aae31f24e901c96069ddba3cf54e9344e.jpg)

![6858f23e9a45bde43aa93becfd6b9aeded80a8b2f2c790dc54e9a6e3d746ac17.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/images/6858f23e9a45bde43aa93becfd6b9aeded80a8b2f2c790dc54e9a6e3d746ac17.jpg)

![6ecbcca58516c7b74727d11dd29b114efd2221d3793425f0b03a49d312d9b5ea.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/images/6ecbcca58516c7b74727d11dd29b114efd2221d3793425f0b03a49d312d9b5ea.jpg)

![77ac0b0db71db4ad12cb3c85951605b11d79b6b7d74333877f87a0e3c75d9290.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/images/77ac0b0db71db4ad12cb3c85951605b11d79b6b7d74333877f87a0e3c75d9290.jpg)

### Tables

![09686a168e2caadd855852f17829551a27d61f3668b5b56b5d8fe45047914c14.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/09686a168e2caadd855852f17829551a27d61f3668b5b56b5d8fe45047914c14.jpg)

![2cf89ae4979617f557e9e8dea837769a26baeb80fa13c76decc32ae98f5a2efc.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/2cf89ae4979617f557e9e8dea837769a26baeb80fa13c76decc32ae98f5a2efc.jpg)

![35ed5cdeb3f8346af38e0bc5f4e3c51589be272eefff2df43cc9efbb234a8664.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/35ed5cdeb3f8346af38e0bc5f4e3c51589be272eefff2df43cc9efbb234a8664.jpg)

![3b9bc637f55800c4a0a6be68ab0349dbf5f04fc40cf46f9dbb592c6ddb60dded.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/3b9bc637f55800c4a0a6be68ab0349dbf5f04fc40cf46f9dbb592c6ddb60dded.jpg)

![41b577e4b9ad55b4cff75faf848e063eab94181618a8ca33a009eab023283acc.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/41b577e4b9ad55b4cff75faf848e063eab94181618a8ca33a009eab023283acc.jpg)

![4a248afe94fc3bcb117869255c77917cda98ea85d17b023472ea4181dffa0a52.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/4a248afe94fc3bcb117869255c77917cda98ea85d17b023472ea4181dffa0a52.jpg)

![6bfd9db2462f175aff593cec9eee91c691fa34e40db17811b8d91ce3c441d0a0.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/6bfd9db2462f175aff593cec9eee91c691fa34e40db17811b8d91ce3c441d0a0.jpg)

![7516778ba546f6f75ae90e57009616738d3dc9bf284eddbdc4ef760cc80db01a.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/7516778ba546f6f75ae90e57009616738d3dc9bf284eddbdc4ef760cc80db01a.jpg)

![8b12274eca7b4cfe957ccda80bdfc9b949ed771a414f1ec7845611fba407d9f4.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/8b12274eca7b4cfe957ccda80bdfc9b949ed771a414f1ec7845611fba407d9f4.jpg)

![9a3eef8dc4cec70e993903aa3f7ba3cc3dca071d56492943fe33e2de926f0684.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/9a3eef8dc4cec70e993903aa3f7ba3cc3dca071d56492943fe33e2de926f0684.jpg)

![a59a4f60abc682f5d91c9a34bfceb435d1189b6dfd61e177166df28bd0db22f2.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/a59a4f60abc682f5d91c9a34bfceb435d1189b6dfd61e177166df28bd0db22f2.jpg)

![bfcdfa7df7311186592b55b24ed6e49eb7caf3b38c92922df117eb6b956cb79c.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/bfcdfa7df7311186592b55b24ed6e49eb7caf3b38c92922df117eb6b956cb79c.jpg)

![d9d025870282e6f36fa6467c34538131a711eb415821d1b0f77b63fd8aabd94f.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/d9d025870282e6f36fa6467c34538131a711eb415821d1b0f77b63fd8aabd94f.jpg)

![e8d771de56809d79102b261bdbbdeb3d1cf5200e2c54cb5d2d5c1a0123632064.jpg](../iclr_results/1086_Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving/tables/e8d771de56809d79102b261bdbbdeb3d1cf5200e2c54cb5d2d5c1a0123632064.jpg)

## Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions


### Images

![052793bff5da6181631a4b0bde66e2dff5b52280b34d0d321f28c95fe3c8e29d.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/052793bff5da6181631a4b0bde66e2dff5b52280b34d0d321f28c95fe3c8e29d.jpg)

![0b817c5afab62656bfe2274591f6bb42ec432ddb2f642bc4264b92afce41aa0d.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/0b817c5afab62656bfe2274591f6bb42ec432ddb2f642bc4264b92afce41aa0d.jpg)

![1cd4436d066845f4972aaa8f4e062183ce9781cbdf62cf2b52b6f5c619b84f40.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/1cd4436d066845f4972aaa8f4e062183ce9781cbdf62cf2b52b6f5c619b84f40.jpg)

![2d84770170dce27bc1e72c950b1aa5747106e903bd7939b9cecebc9c9f8065cd.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/2d84770170dce27bc1e72c950b1aa5747106e903bd7939b9cecebc9c9f8065cd.jpg)

![40390b68c0ef221b57e2b05f16f786d8c491e1639e57119a14896fe068ee077b.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/40390b68c0ef221b57e2b05f16f786d8c491e1639e57119a14896fe068ee077b.jpg)

![42a6eae10716fd82ac2325bb475b0b8f2bf2e9c28bbbc5aca238d67271ecfd52.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/42a6eae10716fd82ac2325bb475b0b8f2bf2e9c28bbbc5aca238d67271ecfd52.jpg)

![58835590f2f465f125a541882430bb0ff40361a3f35abd036c7e423cb539a26b.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/58835590f2f465f125a541882430bb0ff40361a3f35abd036c7e423cb539a26b.jpg)

![79ddbc7422d761e83a6b57e02b155ac49536635ac62ded0b1e6ea0fc30df5481.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/79ddbc7422d761e83a6b57e02b155ac49536635ac62ded0b1e6ea0fc30df5481.jpg)

![812c041829b5c217bcc11beb2a4fbc8be12d19186fddb04f71b6c975ecbad850.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/812c041829b5c217bcc11beb2a4fbc8be12d19186fddb04f71b6c975ecbad850.jpg)

![86eb8babb5a25d76e85081d6c14569f8a3b4cb1caefb10f4400fac626adef639.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/86eb8babb5a25d76e85081d6c14569f8a3b4cb1caefb10f4400fac626adef639.jpg)

![903e5b69aa95cefb651bc9eea7262bed57f5b1c78515b2e01810e69f3c944e39.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/903e5b69aa95cefb651bc9eea7262bed57f5b1c78515b2e01810e69f3c944e39.jpg)

![9e98fb69b197f80bf0a222cac80e11a997b2c8189f8eb43cc9f547ae9d129873.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/9e98fb69b197f80bf0a222cac80e11a997b2c8189f8eb43cc9f547ae9d129873.jpg)

![a472245cf4998af8cc1de2f9fdc126622ed171a4f17a730750bf0b3679265f72.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/a472245cf4998af8cc1de2f9fdc126622ed171a4f17a730750bf0b3679265f72.jpg)

![b9b386c2f4a731fc9df4fe6175ce9b89aadd51c9930467ecb5e1b0dee69bc4f2.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/b9b386c2f4a731fc9df4fe6175ce9b89aadd51c9930467ecb5e1b0dee69bc4f2.jpg)

![bec5a9e61d894292afec340a1fd7686537d23f21857c1d706405ce616873ca5b.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/bec5a9e61d894292afec340a1fd7686537d23f21857c1d706405ce616873ca5b.jpg)

![bff818f8d66ffffaef3b59ac45d02134d6a606c5141f050f6cda56c3112c9cb3.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/bff818f8d66ffffaef3b59ac45d02134d6a606c5141f050f6cda56c3112c9cb3.jpg)

![d0f6fdc002d40f6c7f807e1b6ce114f9d84be806a5d630f0568a063b8d6f3f66.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/d0f6fdc002d40f6c7f807e1b6ce114f9d84be806a5d630f0568a063b8d6f3f66.jpg)

![e9ee9523bdd43c9f39b71ca85dd4d02fca7e3a1c1095986e26023d9e15bf31e2.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/e9ee9523bdd43c9f39b71ca85dd4d02fca7e3a1c1095986e26023d9e15bf31e2.jpg)

![f83dcd6c9e56451a11dde2c55d6bdaba72b6db63ddb1e8608b7d1e9dc9e218a8.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/f83dcd6c9e56451a11dde2c55d6bdaba72b6db63ddb1e8608b7d1e9dc9e218a8.jpg)

![fa58200084d01772420c7c11712abcc111af068c3200cf46e74a78ae204552bb.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/images/fa58200084d01772420c7c11712abcc111af068c3200cf46e74a78ae204552bb.jpg)

### Tables

![0147d27d12ba56e4d83ea51f60d55c5049a72492691075d1f00478a6afa932ff.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/0147d27d12ba56e4d83ea51f60d55c5049a72492691075d1f00478a6afa932ff.jpg)

![1886d2d71a9a75a200d722e486272cc9db9d6806d4b55993c79600024e6667a4.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/1886d2d71a9a75a200d722e486272cc9db9d6806d4b55993c79600024e6667a4.jpg)

![4c782afaaf295f192f82cc354986090ab7b290fe2e78b9a69b302dde94542fc7.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/4c782afaaf295f192f82cc354986090ab7b290fe2e78b9a69b302dde94542fc7.jpg)

![61f23af4431ae79269d94ab54efbd461dde390b3ae8dd1627aa8b5780e7ef32e.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/61f23af4431ae79269d94ab54efbd461dde390b3ae8dd1627aa8b5780e7ef32e.jpg)

![70fdf25c79222b23da93619a6ab16ce6e3a965e53fab5faf7a4b3565e97fe4a1.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/70fdf25c79222b23da93619a6ab16ce6e3a965e53fab5faf7a4b3565e97fe4a1.jpg)

![98750dc345aa4102eb07a2002296e79fe23cb8cd0ca8cf773a6baf223006177f.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/98750dc345aa4102eb07a2002296e79fe23cb8cd0ca8cf773a6baf223006177f.jpg)

![c4a9eb084899a18bcb78787d784047c3945cd9504ed9876bb60602a3a3eaf0bb.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/c4a9eb084899a18bcb78787d784047c3945cd9504ed9876bb60602a3a3eaf0bb.jpg)

![e40969513346d2bd53de6104f520b95de8f5ee0a19df18a2b8eb5c213cf7de31.jpg](../iclr_results/1087_Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions/tables/e40969513346d2bd53de6104f520b95de8f5ee0a19df18a2b8eb5c213cf7de31.jpg)

## A Statistical Framework for Ranking LLM-based Chatbots


### Images

![136446299f3a67a1975d46ebc3a86b3ff62057c0daaff237dd4a554654fc668b.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/136446299f3a67a1975d46ebc3a86b3ff62057c0daaff237dd4a554654fc668b.jpg)

![238df98e79d195cab769797f858c285cb8d4a70182f8663d36e8ea10396e4294.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/238df98e79d195cab769797f858c285cb8d4a70182f8663d36e8ea10396e4294.jpg)

![6c34dbc3a9f0901f0c3ac0db595430eb7edb869c8d495bff127de7c297b5a9f6.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/6c34dbc3a9f0901f0c3ac0db595430eb7edb869c8d495bff127de7c297b5a9f6.jpg)

![6e3ee2ae5ebeb1073938800f16e5b11997c912ebdeb65505d0485be82aedecf9.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/6e3ee2ae5ebeb1073938800f16e5b11997c912ebdeb65505d0485be82aedecf9.jpg)

![87be80c9d730f25d507a54398bb9beb598fd0455644f4aefcf4a4a1c35131a82.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/87be80c9d730f25d507a54398bb9beb598fd0455644f4aefcf4a4a1c35131a82.jpg)

![8cea774508a0cdf2e8e91f711402f5cb11bc2abb3e2e2111b502f998675ca47c.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/8cea774508a0cdf2e8e91f711402f5cb11bc2abb3e2e2111b502f998675ca47c.jpg)

![8ec77b3f7114d098a1af2b46a597cbd91e38b8ae23e33462cec8e06b6805a147.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/8ec77b3f7114d098a1af2b46a597cbd91e38b8ae23e33462cec8e06b6805a147.jpg)

![b6c2772bca15fd894dd12adcdb0758a524ef4fec17040d27cb32a2a2d9e60262.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/b6c2772bca15fd894dd12adcdb0758a524ef4fec17040d27cb32a2a2d9e60262.jpg)

![c92b4c6ebc76598ed748eec771887a8f98211627edff27bab26744e738ff206e.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/c92b4c6ebc76598ed748eec771887a8f98211627edff27bab26744e738ff206e.jpg)

![cf91db95894a7a079e789e6433bf08791b11e124573b91855cf628831cb13db6.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/images/cf91db95894a7a079e789e6433bf08791b11e124573b91855cf628831cb13db6.jpg)

### Tables

![031dbddd3cac1210fe760b9114732c8737f2b90734454ddad0cefcab6fc0d1ae.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/tables/031dbddd3cac1210fe760b9114732c8737f2b90734454ddad0cefcab6fc0d1ae.jpg)

![417aef7514912f96616adc26b056bc97640c528833edf628615d22cde3368b39.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/tables/417aef7514912f96616adc26b056bc97640c528833edf628615d22cde3368b39.jpg)

![db3cbfea5a862e4a4bac190ce8ba96805e218b9d6b732d99d444044d8881524c.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/tables/db3cbfea5a862e4a4bac190ce8ba96805e218b9d6b732d99d444044d8881524c.jpg)

![f45cac4e9c3aca6f87b5216cff7b5f31e11ee0949c69731d0902093fee5ad094.jpg](../iclr_results/1088_A Statistical Framework for Ranking LLM-based Chatbots/tables/f45cac4e9c3aca6f87b5216cff7b5f31e11ee0949c69731d0902093fee5ad094.jpg)

## OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting


### Images

![012f8e0958aa71231a07fde6d842f358819356a812af9623a1f9e102c9af5dbc.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/012f8e0958aa71231a07fde6d842f358819356a812af9623a1f9e102c9af5dbc.jpg)

![15c1c9965afe22a2db6eb724f09e5eb8b0f81efdaf6f88c3f7e4d48b97254642.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/15c1c9965afe22a2db6eb724f09e5eb8b0f81efdaf6f88c3f7e4d48b97254642.jpg)

![42ec36754ab875c4a872ca3b46ef15f2817c9a280a738a297f904e5cfcf317b9.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/42ec36754ab875c4a872ca3b46ef15f2817c9a280a738a297f904e5cfcf317b9.jpg)

![7c56422bdb1bf49efa1d49384b40a521a057a59c2191b9024c18c6182acb39d8.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/7c56422bdb1bf49efa1d49384b40a521a057a59c2191b9024c18c6182acb39d8.jpg)

![87aa679046b2ab6d1f6fc3ab13b87901e48ca425b1269a86c82473a7b3533381.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/87aa679046b2ab6d1f6fc3ab13b87901e48ca425b1269a86c82473a7b3533381.jpg)

![8c90f4d5fc314cdddc9d6b097a30dd6976a9790bf01b9754601f62b5c80150e5.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/8c90f4d5fc314cdddc9d6b097a30dd6976a9790bf01b9754601f62b5c80150e5.jpg)

![8ddec2708ea07d7de2a1befda1e41aeceb49039415da023349ed139bc58ce5b1.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/8ddec2708ea07d7de2a1befda1e41aeceb49039415da023349ed139bc58ce5b1.jpg)

![970cecfa142e1fde1758895b7cfb9129f097210513d1776036678ce972b4ae17.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/970cecfa142e1fde1758895b7cfb9129f097210513d1776036678ce972b4ae17.jpg)

![b9e22b6337abb5e6ba1b87c6dd10b16c56c3917837eb16470ba02d5a993b8000.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/b9e22b6337abb5e6ba1b87c6dd10b16c56c3917837eb16470ba02d5a993b8000.jpg)

![bfe605c976d5bfe25b3cdddb4b4ce72c7fba8e5b658080350ca3f9f5b7dfcb46.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/bfe605c976d5bfe25b3cdddb4b4ce72c7fba8e5b658080350ca3f9f5b7dfcb46.jpg)

![ce6b57f0f65c3c7814964a365d219afed3dd3465f12ff25200463d4f51520d8b.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/ce6b57f0f65c3c7814964a365d219afed3dd3465f12ff25200463d4f51520d8b.jpg)

![d5153e65ad32e7ad1fc974ef04a9469fa39c415f336ec4787f8f70c71ccc0a92.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/d5153e65ad32e7ad1fc974ef04a9469fa39c415f336ec4787f8f70c71ccc0a92.jpg)

![ed48663b385d82e55f09b3501bc2f1e00b0dfa1ddcdf8dd2d423e8dcaad26de3.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/ed48663b385d82e55f09b3501bc2f1e00b0dfa1ddcdf8dd2d423e8dcaad26de3.jpg)

![ed6f0bd35d327746421e03f2f4b490f893b939c79e89d7917bcee9da968f9ba0.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/images/ed6f0bd35d327746421e03f2f4b490f893b939c79e89d7917bcee9da968f9ba0.jpg)

### Tables

![1673c9fbac730db5fa0936091bb10a04039c1af05483d5f59ecad364a016f3bb.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/1673c9fbac730db5fa0936091bb10a04039c1af05483d5f59ecad364a016f3bb.jpg)

![356d1517aab464aadc9f4f7f16521f538c631a10e1d3041edecc1bf4a1e0aa3d.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/356d1517aab464aadc9f4f7f16521f538c631a10e1d3041edecc1bf4a1e0aa3d.jpg)

![3a98e35b277596c4e559d1e12e83aebfb61e39a8a4ee13f93fff2890a439940d.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/3a98e35b277596c4e559d1e12e83aebfb61e39a8a4ee13f93fff2890a439940d.jpg)

![3ddfb31140f9bba5dfecb16ffd5a0e2f23726ecc34aafacee6b0c7e58cd6f193.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/3ddfb31140f9bba5dfecb16ffd5a0e2f23726ecc34aafacee6b0c7e58cd6f193.jpg)

![624c9268e646c19d870f67b8eb00b4d519757b814d2f12acab8a9c0575f6889b.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/624c9268e646c19d870f67b8eb00b4d519757b814d2f12acab8a9c0575f6889b.jpg)

![638ec8e6859c5ee5f466499c544a97a05c49186fbbc0c7b7efe75d2c5b97d875.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/638ec8e6859c5ee5f466499c544a97a05c49186fbbc0c7b7efe75d2c5b97d875.jpg)

![6571ba1444b0f657f5fe6c88cfa4d287701335046aff849e28ad4d565f31ee95.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/6571ba1444b0f657f5fe6c88cfa4d287701335046aff849e28ad4d565f31ee95.jpg)

![659b8c0a2df7ffffa06774054f51a62a3fd0eed075164040c53418c9669f7f35.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/659b8c0a2df7ffffa06774054f51a62a3fd0eed075164040c53418c9669f7f35.jpg)

![84d9a8ccaa80e9c6d0636caa6b6a52690f493ea74c16b2cf6a7c578f7f7c2933.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/84d9a8ccaa80e9c6d0636caa6b6a52690f493ea74c16b2cf6a7c578f7f7c2933.jpg)

![9863fee6fc9a7b73d8c1b0e9d8668ddbc937b2616420d6469248be89b05be3e3.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/9863fee6fc9a7b73d8c1b0e9d8668ddbc937b2616420d6469248be89b05be3e3.jpg)

![aecd48175ac759feb59b5a214b32eeecf21056a3a9401aa6d42bb5206e97c5bd.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/aecd48175ac759feb59b5a214b32eeecf21056a3a9401aa6d42bb5206e97c5bd.jpg)

![b1f4f8fdee2cbebb8e0cdcb93821ff78c9a726e205fd55c9e48990b0192dc65e.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/b1f4f8fdee2cbebb8e0cdcb93821ff78c9a726e205fd55c9e48990b0192dc65e.jpg)

![c877f2e7e8dfa3fe6ee9d8acf6e0cc4fa7843643c73a62aad53bf94866e68325.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/c877f2e7e8dfa3fe6ee9d8acf6e0cc4fa7843643c73a62aad53bf94866e68325.jpg)

![e48e2f3a09b16d5c5b138b8b7a30c38c4cb709f6a65bf2d9d20874e441040960.jpg](../iclr_results/1089_OSTQuant_ Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for/tables/e48e2f3a09b16d5c5b138b8b7a30c38c4cb709f6a65bf2d9d20874e441040960.jpg)

## TULIP: Token-length Upgraded CLIP


### Images

![03de6427d8667a4253c82daad5046ec44e6320f3432054d3548dbd46ffe15d51.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/03de6427d8667a4253c82daad5046ec44e6320f3432054d3548dbd46ffe15d51.jpg)

![0c4669a87a2b491c3476ff89ea0a2790134d44c35a48e2a69c8adc112a11b25e.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/0c4669a87a2b491c3476ff89ea0a2790134d44c35a48e2a69c8adc112a11b25e.jpg)

![0f988fcfc5bd5e722392ecde851565e3201fb1028e9a4b19049cdb2ceaedde92.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/0f988fcfc5bd5e722392ecde851565e3201fb1028e9a4b19049cdb2ceaedde92.jpg)

![1675a3e3cb67d11b01d7bfb6d39f214a9f3af5ddef5911f34312b9c5d63d7927.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/1675a3e3cb67d11b01d7bfb6d39f214a9f3af5ddef5911f34312b9c5d63d7927.jpg)

![38ac8b9a7718f94ee927db9137c0af98c3a31062772c38329e35588bba510ac4.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/38ac8b9a7718f94ee927db9137c0af98c3a31062772c38329e35588bba510ac4.jpg)

![3c4478de437160e4624705567ed8ec90ab1db3ff79b640940f1a3b2ca87adb79.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/3c4478de437160e4624705567ed8ec90ab1db3ff79b640940f1a3b2ca87adb79.jpg)

![4282b0cc03e1702b1f7bb510a527eab25c5106a124f5b6b96c9a497992cdf183.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/4282b0cc03e1702b1f7bb510a527eab25c5106a124f5b6b96c9a497992cdf183.jpg)

![4f151a255515e06bcbf5c432e0d70834a80dd7f277ac73361accd074708ee739.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/4f151a255515e06bcbf5c432e0d70834a80dd7f277ac73361accd074708ee739.jpg)

![54ed356b7775d92a64de5690446f604fe36f267250e458d7242067b6a579c03c.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/54ed356b7775d92a64de5690446f604fe36f267250e458d7242067b6a579c03c.jpg)

![57c724906ca3f799dad4a303f3fbfc96cbb8540074f661e5ea56b7f14c729f93.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/57c724906ca3f799dad4a303f3fbfc96cbb8540074f661e5ea56b7f14c729f93.jpg)

![6c06db7d0a969590d43acf9b3bdf5393bc1c8266a6eb665c75a6cfd72186dea2.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/6c06db7d0a969590d43acf9b3bdf5393bc1c8266a6eb665c75a6cfd72186dea2.jpg)

![76651dc94ea1cb275d1387ecae5d2694b2550c0fd0af597e924d9228660ef470.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/76651dc94ea1cb275d1387ecae5d2694b2550c0fd0af597e924d9228660ef470.jpg)

![7ea6ffaf77d8f7c39d3cf6efc790215b5c3e2c9932a5c34ed4463ddd6ee1deed.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/7ea6ffaf77d8f7c39d3cf6efc790215b5c3e2c9932a5c34ed4463ddd6ee1deed.jpg)

![88485f06b2ea7975ad58986f76cf2de001a54636e173e5e7a38842243e1d89e6.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/88485f06b2ea7975ad58986f76cf2de001a54636e173e5e7a38842243e1d89e6.jpg)

![8b4cd87a787dac23ba83134f788f92c54dd3a495c4013d54ceb300f634fa3a81.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/8b4cd87a787dac23ba83134f788f92c54dd3a495c4013d54ceb300f634fa3a81.jpg)

![9683664a45f8996017e0d1a4d3da1334eae62d42a9d8ce80b34f34506adc779f.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/9683664a45f8996017e0d1a4d3da1334eae62d42a9d8ce80b34f34506adc779f.jpg)

![9705787cb9c1bd114b5049d89508a0a04ac2b982d2d837acb1c28fc21fbe8e47.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/9705787cb9c1bd114b5049d89508a0a04ac2b982d2d837acb1c28fc21fbe8e47.jpg)

![9c01492102542563cbe97c3565fb821ffc12cb9d6da2fa1d5744867c17559cfb.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/9c01492102542563cbe97c3565fb821ffc12cb9d6da2fa1d5744867c17559cfb.jpg)

![a59c2c20a6750b11da160b7fe76582bced14b1242b74437b6556c1b97382717e.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/a59c2c20a6750b11da160b7fe76582bced14b1242b74437b6556c1b97382717e.jpg)

![a90d31b2adf50fc3dd96e0d9e9a8f93a7b63926dd779600b861f3484cda99952.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/a90d31b2adf50fc3dd96e0d9e9a8f93a7b63926dd779600b861f3484cda99952.jpg)

![aa4f4f2e83732832036a9550e58eac552180199e5e2ae2e1e35f2a32d0449a8b.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/aa4f4f2e83732832036a9550e58eac552180199e5e2ae2e1e35f2a32d0449a8b.jpg)

![af0fea61de4ad559a714e00ce9829d2424b962d6864c12a0ec59c83ec203e968.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/af0fea61de4ad559a714e00ce9829d2424b962d6864c12a0ec59c83ec203e968.jpg)

![b1d90482c0f8925645386e50db137694eef96cdece88c18556cc7eea0ce98bc2.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/b1d90482c0f8925645386e50db137694eef96cdece88c18556cc7eea0ce98bc2.jpg)

![b2c36e7cd261547672be36aa3d82d3c4c3ad6b9043f9d5b23acb8b712dcf8c50.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/b2c36e7cd261547672be36aa3d82d3c4c3ad6b9043f9d5b23acb8b712dcf8c50.jpg)

![bd0ecd0f33bf95f097c52bf0609ce524a7d143e3071c7f132298d880aab34ffa.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/bd0ecd0f33bf95f097c52bf0609ce524a7d143e3071c7f132298d880aab34ffa.jpg)

![be2be338e8bbac5453652c77618dc151ca25cf84703e762481cba42c97665f88.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/be2be338e8bbac5453652c77618dc151ca25cf84703e762481cba42c97665f88.jpg)

![c874146a055580ae5bf16123a128680507f932e9ebad28f759fd768d34079622.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/c874146a055580ae5bf16123a128680507f932e9ebad28f759fd768d34079622.jpg)

![c8feae9f4cb54c817033b12f76faff023010b9478ec3ab0bec597f2fd5f3ee1b.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/c8feae9f4cb54c817033b12f76faff023010b9478ec3ab0bec597f2fd5f3ee1b.jpg)

![cebeaff7e0d790f77daa97607fa9cc0211c724a15f460919a7dd7c711aa44123.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/cebeaff7e0d790f77daa97607fa9cc0211c724a15f460919a7dd7c711aa44123.jpg)

![d2f2f4f5a3473a974ee4ede11232ea608912e5e3ca06c7869a8807663b69379a.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/d2f2f4f5a3473a974ee4ede11232ea608912e5e3ca06c7869a8807663b69379a.jpg)

![d4ec9c8c18b9f3ff15324867e04f8fb40ad28eb893f05a610e558c74928fc237.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/d4ec9c8c18b9f3ff15324867e04f8fb40ad28eb893f05a610e558c74928fc237.jpg)

![e8135f9d0af263ddd8fbf8165144f136e460eb84ffa9e61f9cd5188e888ef3eb.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/e8135f9d0af263ddd8fbf8165144f136e460eb84ffa9e61f9cd5188e888ef3eb.jpg)

![ecd654ce13256a8f92a08d8ec81cc1ebf8c12a0b818b21270fcf2eb01f5efe73.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/ecd654ce13256a8f92a08d8ec81cc1ebf8c12a0b818b21270fcf2eb01f5efe73.jpg)

![f07ef60ede6aac079cf8019ffc91e382b0f8861badb68256b93668377e2af204.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/f07ef60ede6aac079cf8019ffc91e382b0f8861badb68256b93668377e2af204.jpg)

![f0dd9104a90641cdd7a329a48547e55927f259b96fea2358be02c5a62db43192.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/f0dd9104a90641cdd7a329a48547e55927f259b96fea2358be02c5a62db43192.jpg)

![f20fe03ef3cd7156d4176d622c2dc275676fca38dafc59a0937747f539200752.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/images/f20fe03ef3cd7156d4176d622c2dc275676fca38dafc59a0937747f539200752.jpg)

### Tables

![0190570ea87e6e41205a3449d69519802628c92296cdbab04bc3a4cc49fa99b9.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/0190570ea87e6e41205a3449d69519802628c92296cdbab04bc3a4cc49fa99b9.jpg)

![0de69eb22627aea69062aaf9789f7817fb5df65df80008371069594d60ae373a.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/0de69eb22627aea69062aaf9789f7817fb5df65df80008371069594d60ae373a.jpg)

![14bc5ce2e226e1ee3da343f693f3f80b557ce0f3b2da74aa77e1448bb4e7e3fd.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/14bc5ce2e226e1ee3da343f693f3f80b557ce0f3b2da74aa77e1448bb4e7e3fd.jpg)

![332824561a2d4b244cf9668308479e9b90c005da19fc6a5c4ebad96194f5de5c.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/332824561a2d4b244cf9668308479e9b90c005da19fc6a5c4ebad96194f5de5c.jpg)

![37c9ab6d8355ffb0c8ad6626f99f51825dadaef7e3165fb356eb57eff7b6ae45.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/37c9ab6d8355ffb0c8ad6626f99f51825dadaef7e3165fb356eb57eff7b6ae45.jpg)

![83ca3aa4c5aeaf543793163224c6249073d5d687ab699bd6e74b7fd9629b0846.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/83ca3aa4c5aeaf543793163224c6249073d5d687ab699bd6e74b7fd9629b0846.jpg)

![fbd4a364ee58f564dce70a6a862eb5f357d9ef7cfe155d6fc06897df5232d3c7.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/fbd4a364ee58f564dce70a6a862eb5f357d9ef7cfe155d6fc06897df5232d3c7.jpg)

![fe3e93367dd0fc0c6234e44b2400e84e74bc28a48f9b1b7b230b707f8130587c.jpg](../iclr_results/1090_TULIP_ Token-length Upgraded CLIP/tables/fe3e93367dd0fc0c6234e44b2400e84e74bc28a48f9b1b7b230b707f8130587c.jpg)

## Scaling Stick-Breaking Attention: An Efficient Implementation and In-depth Study


### Images

![020a8b671dfe37ac97086e0a5237d802bc5624e38f064fc430630533a392e29b.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/020a8b671dfe37ac97086e0a5237d802bc5624e38f064fc430630533a392e29b.jpg)

![05ebd9ab8599955c876e952184f9fae78321c8a636dcaad7b68601ec9b56beed.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/05ebd9ab8599955c876e952184f9fae78321c8a636dcaad7b68601ec9b56beed.jpg)

![16a30f5130f06197b61ca09b324fab4702845f2080bb9ddd26104fe5ef3c9744.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/16a30f5130f06197b61ca09b324fab4702845f2080bb9ddd26104fe5ef3c9744.jpg)

![38db4c7586cd25bcf4a8c8794237b048ce0d2be225b4309a525af891e8531032.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/38db4c7586cd25bcf4a8c8794237b048ce0d2be225b4309a525af891e8531032.jpg)

![7b72155f389b81c8b094b63c61458012de5c0808624fd99b9626dc4e5e38723b.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/7b72155f389b81c8b094b63c61458012de5c0808624fd99b9626dc4e5e38723b.jpg)

![8ddd0ca0c86545965570ed8a43d8898932cc31ed430506246dfc2570792781b6.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/8ddd0ca0c86545965570ed8a43d8898932cc31ed430506246dfc2570792781b6.jpg)

![98b0c74a5bdbaeae7929510baea65fbc24e8fcf03a6654b564c81c7e4be27bbc.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/98b0c74a5bdbaeae7929510baea65fbc24e8fcf03a6654b564c81c7e4be27bbc.jpg)

![be8f16f056db4a0132125ce55f4f991243b3b95e5d57626c8f31e708e8a704ea.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/be8f16f056db4a0132125ce55f4f991243b3b95e5d57626c8f31e708e8a704ea.jpg)

![c2536e528d2b6ae3e4dbfe123270ca077345c3958ef81e6f3f590df9eb230aa3.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/images/c2536e528d2b6ae3e4dbfe123270ca077345c3958ef81e6f3f590df9eb230aa3.jpg)

### Tables

![03bc72014abe3bd904caa3698214282b896d382e238f720f60e0b7c8090eac41.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/tables/03bc72014abe3bd904caa3698214282b896d382e238f720f60e0b7c8090eac41.jpg)

![295e5ce05c17b80a6bef7c4a6ca559429d8db2a0007cdb8e918f27471c6b629c.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/tables/295e5ce05c17b80a6bef7c4a6ca559429d8db2a0007cdb8e918f27471c6b629c.jpg)

![5e8b911a2e83ae9b74d4a316cf2634899b5d6cd3b96ce586442fc3b383109175.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/tables/5e8b911a2e83ae9b74d4a316cf2634899b5d6cd3b96ce586442fc3b383109175.jpg)

![7e38b1c658030c735baa61de39c590116a9decb098f99f0515f1a90d159f75ea.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/tables/7e38b1c658030c735baa61de39c590116a9decb098f99f0515f1a90d159f75ea.jpg)

![bc49d631676ed60810f3258f1bb3dfbf1773c261038691d5205988c4f4564dab.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/tables/bc49d631676ed60810f3258f1bb3dfbf1773c261038691d5205988c4f4564dab.jpg)

![dfa29955cf82b41fa70ad115a0fadb68f83d6b3f583cd29d00e1c10240124255.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/tables/dfa29955cf82b41fa70ad115a0fadb68f83d6b3f583cd29d00e1c10240124255.jpg)

![e2e6f712f171b547391a7825226b39979b1671239bb47936f48a7fc25ce98110.jpg](../iclr_results/1091_Scaling Stick-Breaking Attention_ An Efficient Implementation and In-depth Study/tables/e2e6f712f171b547391a7825226b39979b1671239bb47936f48a7fc25ce98110.jpg)

## Gated Delta Networks: Improving Mamba2 with Delta Rule


### Images

![09b7225b25872b3023498d8c4582e62338c39098779d2ea14bd994a2433912cf.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/images/09b7225b25872b3023498d8c4582e62338c39098779d2ea14bd994a2433912cf.jpg)

![b0e6cc449f95ccfb9d4cc414b20db150bba2e40f398fa94163b4b0a7b49a6e24.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/images/b0e6cc449f95ccfb9d4cc414b20db150bba2e40f398fa94163b4b0a7b49a6e24.jpg)

![f838e03886c7b890bd4f12b5a8d1fc63fddca1e3dd62bac007d6ba74ce274205.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/images/f838e03886c7b890bd4f12b5a8d1fc63fddca1e3dd62bac007d6ba74ce274205.jpg)

### Tables

![45c449b8b6af246d1d50e489798c8a38f9a330bd553b20c3ea79b4e26493c57b.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/tables/45c449b8b6af246d1d50e489798c8a38f9a330bd553b20c3ea79b4e26493c57b.jpg)

![75fbfbf14942d8e366159251bf7e2c15de0f0a0612da00982e6be3f5378f14b3.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/tables/75fbfbf14942d8e366159251bf7e2c15de0f0a0612da00982e6be3f5378f14b3.jpg)

![a02415c3750127547e08093172cabe6149a61e6f2b1bd375c0fbd679f3b4600c.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/tables/a02415c3750127547e08093172cabe6149a61e6f2b1bd375c0fbd679f3b4600c.jpg)

![aeec47a579939fa755e3fe66d9f9f0e54fdb78d5f4f4399b4d76fd5b3cb1e923.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/tables/aeec47a579939fa755e3fe66d9f9f0e54fdb78d5f4f4399b4d76fd5b3cb1e923.jpg)

![bf17e7774e820538bb98b0d42667ea4939f048665cf5fcb103beaef32082a7c8.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/tables/bf17e7774e820538bb98b0d42667ea4939f048665cf5fcb103beaef32082a7c8.jpg)

![e93ea3c8916e680dd540698f1cd95bbbff2e258d9e4312fcd79279bf54f5c64e.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/tables/e93ea3c8916e680dd540698f1cd95bbbff2e258d9e4312fcd79279bf54f5c64e.jpg)

![fef0b6533791e56b2b856f30cddfcc6831533ab887e6df89f612c3f8bde57e73.jpg](../iclr_results/1092_Gated Delta Networks_ Improving Mamba2 with Delta Rule/tables/fef0b6533791e56b2b856f30cddfcc6831533ab887e6df89f612c3f8bde57e73.jpg)

## Global Well-posedness and Convergence Analysis of Score-based Generative Models via Sharp Lipschitz Estimates

### Images

![65fb1a89d3f5715a6d10ea3f8318447752f6ed5e923cef5abd522854ce3419c3.jpg](../iclr_results/1093_Global Well-posedness and Convergence Analysis of Score-based Generative Models via Sharp Lipschitz /images/65fb1a89d3f5715a6d10ea3f8318447752f6ed5e923cef5abd522854ce3419c3.jpg)
