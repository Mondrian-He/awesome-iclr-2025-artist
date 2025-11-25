# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 42 of 100

## 目录 (Table of Contents)

1. [Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond](#Rethinking-LLM-Unlearning-Objectives-A-Gradient-Perspective-and-Go-Beyond)
2. [GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack](#GSBAK-top-K-Geometric-Score-based-Black-box-Attack)
3. [Investigating the Pre-Training Dynamics of In-Context Learning: Task Recognition vs. Task Learning](#Investigating-the-Pre-Training-Dynamics-of-In-Context-Learning-Task-Recognition-vs-Task-Learning)
4. [Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs](#Straight-to-Zero-Why-Linearly-Decaying-the-Learning-Rate-to-Zero-Works-Best-for-LLMs)
5. [TDDBench: A Benchmark for Training data detection](#TDDBench-A-Benchmark-for-Training-data-detection)
6. [AutoG: Towards automatic graph construction from tabular data](#AutoG-Towards-automatic-graph-construction-from-tabular-data)
7. [DelTA: An Online Document-Level Translation Agent Based on Multi-Level Memory](#DelTA-An-Online-Document-Level-Translation-Agent-Based-on-Multi-Level-Memory)
8. [Mind Control through Causal Inference: Predicting Clean Images from Poisoned Data](#Mind-Control-through-Causal-Inference-Predicting-Clean-Images-from-Poisoned-Data)
9. [Can Knowledge Editing Really Correct Hallucinations?](#Can-Knowledge-Editing-Really-Correct-Hallucinations)
10. [OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination](#OvercookedV2-Rethinking-Overcooked-for-Zero-Shot-Coordination)
11. [Adversarial Search Engine Optimization for Large Language Models](#Adversarial-Search-Engine-Optimization-for-Large-Language-Models)
12. [Causal Representation Learning from Multimodal Biomedical Observations](#Causal-Representation-Learning-from-Multimodal-Biomedical-Observations)
13. [Towards Robust Multimodal Open-set Test-time Adaptation via Adaptive Entropy-aware Optimization](#Towards-Robust-Multimodal-Open-set-Test-time-Adaptation-via-Adaptive-Entropy-aware-Optimization)
14. [Mixture Compressor for Mixture-of-Experts LLMs Gains More](#Mixture-Compressor-for-Mixture-of-Experts-LLMs-Gains-More)
15. [Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction](#Efficient-Exploration-and-Discriminative-World-Model-Learning-with-an-Object-Centric-Abstraction)
16. [Watch Less, Do More: Implicit Skill Discovery for Video-Conditioned Policy](#Watch-Less-Do-More-Implicit-Skill-Discovery-for-Video-Conditioned-Policy)
17. [SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation](#SAFREE-Training-Free-and-Adaptive-Guard-for-Safe-Text-to-Image-And-Video-Generation)
18. [Causal Graphical Models for Vision-Language Compositional Understanding](#Causal-Graphical-Models-for-Vision-Language-Compositional-Understanding)
19. [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](#Jailbreaking-Leading-Safety-Aligned-LLMs-with-Simple-Adaptive-Attacks)
20. [Immunogenicity Prediction with Dual Attention Enables Vaccine Target Selection](#Immunogenicity-Prediction-with-Dual-Attention-Enables-Vaccine-Target-Selection)
21. [Privately Counting Partially Ordered Data](#Privately-Counting-Partially-Ordered-Data)
22. [Mining your own secrets: Diffusion Classifier Scores for Continual Personalization of Text-to-Image Diffusion Models](#Mining-your-own-secrets-Diffusion-Classifier-Scores-for-Continual-Personalization-of-Text-to-Image-Diffusion-Models)
23. [Large Language Models are Interpretable Learners](#Large-Language-Models-are-Interpretable-Learners)
24. [No Location Left Behind: Measuring and Improving the Fairness of Implicit Representations for Earth Data](#No-Location-Left-Behind-Measuring-and-Improving-the-Fairness-of-Implicit-Representations-for-Earth-Data)
25. [Bridging and Modeling Correlations in Pairwise Data for Direct Preference Optimization](#Bridging-and-Modeling-Correlations-in-Pairwise-Data-for-Direct-Preference-Optimization)
26. [DiTTo-TTS: Diffusion Transformers for Scalable Text-to-Speech without Domain-Specific Factors](#DiTTo-TTS-Diffusion-Transformers-for-Scalable-Text-to-Speech-without-Domain-Specific-Factors)
27. [Uncertainty-Aware Decoding with Minimum Bayes Risk](#Uncertainty-Aware-Decoding-with-Minimum-Bayes-Risk)
28. [GEVRM: Goal-Expressive Video Generation Model For Robust Visual Manipulation](#GEVRM-Goal-Expressive-Video-Generation-Model-For-Robust-Visual-Manipulation)
29. [Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration](#Posterior-Mean-Rectified-Flow-Towards-Minimum-MSE-Photo-Realistic-Image-Restoration)
30. [PWM: Policy Learning with Multi-Task World Models](#PWM-Policy-Learning-with-Multi-Task-World-Models)
31. [SAGEPhos: Sage Bio-Coupled and Augmented Fusion for Phosphorylation Site Detection](#SAGEPhos-Sage-Bio-Coupled-and-Augmented-Fusion-for-Phosphorylation-Site-Detection)
32. [OBI-Bench: Can LMMs Aid in Study of Ancient Script on Oracle Bones?](#OBI-Bench-Can-LMMs-Aid-in-Study-of-Ancient-Script-on-Oracle-Bones)
33. [NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics](#NatureLM-audio-an-Audio-Language-Foundation-Model-for-Bioacoustics)
34. [GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering](#GI-GS-Global-Illumination-Decomposition-on-Gaussian-Splatting-for-Inverse-Rendering)
35. [Real-Time Video Generation with Pyramid Attention Broadcast](#Real-Time-Video-Generation-with-Pyramid-Attention-Broadcast)
36. [HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation](#HAMSTER-Hierarchical-Action-Models-for-Open-World-Robot-Manipulation)
37. [Improving Complex Reasoning with Dynamic Prompt Corruption: A Soft Prompt Optimization Approach](#Improving-Complex-Reasoning-with-Dynamic-Prompt-Corruption-A-Soft-Prompt-Optimization-Approach)

---


## Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond

### Images

![1d268afa9b35c30ce7ebcd083a5326a5c5d60d4303bd32dcea789a40e2bf7d27.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/1d268afa9b35c30ce7ebcd083a5326a5c5d60d4303bd32dcea789a40e2bf7d27.jpg)

![2a158ceb7714ae25e442a0db96197b82407ea0f8ffcf010d4621004ba5efd7a4.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/2a158ceb7714ae25e442a0db96197b82407ea0f8ffcf010d4621004ba5efd7a4.jpg)

![4e79035c0cd65438617b144f524fedeaa5af2ca4aaa18f3760672bf33390af8e.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/4e79035c0cd65438617b144f524fedeaa5af2ca4aaa18f3760672bf33390af8e.jpg)

![5e19ec07c33e124fa46f3e1d5582b91548419fab8faa039a0daf7ec4610766d8.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/5e19ec07c33e124fa46f3e1d5582b91548419fab8faa039a0daf7ec4610766d8.jpg)

![81343e7c1a8c299b12eb50a4ee9fdf166599781ba4a97e4e6655bdafb51b8c6a.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/81343e7c1a8c299b12eb50a4ee9fdf166599781ba4a97e4e6655bdafb51b8c6a.jpg)

![9202e45f55a071b23ec3e5312769386743f66bd92629ab2828d26babba4a0976.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/9202e45f55a071b23ec3e5312769386743f66bd92629ab2828d26babba4a0976.jpg)

![99975b5616b3153762909fb3f2fb406afbde6f8c48056e9ce9fff1b1fb943b60.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/99975b5616b3153762909fb3f2fb406afbde6f8c48056e9ce9fff1b1fb943b60.jpg)

![dbe9ec000edd171ed5c2b56103bf54437c9bfae3a793646ec576c75467232c58.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/dbe9ec000edd171ed5c2b56103bf54437c9bfae3a793646ec576c75467232c58.jpg)

![e6676764b287514c89d0cbb7c980a05b7ef9819b6d41bfa356f15b485a44fe59.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/e6676764b287514c89d0cbb7c980a05b7ef9819b6d41bfa356f15b485a44fe59.jpg)

![f0185f277ddc004ba7037aebbd825154051d449c623fd98a4d6ec9a603d756b7.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f0185f277ddc004ba7037aebbd825154051d449c623fd98a4d6ec9a603d756b7.jpg)

![f0fbebbeab339406e87c08671f60db9303b61eee9d1df92c9cf7ffe0e9d6a9b8.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f0fbebbeab339406e87c08671f60db9303b61eee9d1df92c9cf7ffe0e9d6a9b8.jpg)

![f25e06f0cd4feda481b87bb8fa59c22568def92710a66ccc32abedca84d323ea.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f25e06f0cd4feda481b87bb8fa59c22568def92710a66ccc32abedca84d323ea.jpg)

![f31f95cf85c3e523a2c58bb0a0e57cb76044583fd7e61c6f404fc9a5f1789cb7.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f31f95cf85c3e523a2c58bb0a0e57cb76044583fd7e61c6f404fc9a5f1789cb7.jpg)

![f6dc741281fc84694da8bec513561a6182e994536ee4d01ddea4517e91d71243.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f6dc741281fc84694da8bec513561a6182e994536ee4d01ddea4517e91d71243.jpg)

### Tables

![18ebef067a02af2971b3e20f3a22c5b46cf5911cf2ad5c1994c36bb523e4776b.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/tables/18ebef067a02af2971b3e20f3a22c5b46cf5911cf2ad5c1994c36bb523e4776b.jpg)

![2a7448554e37e096b1de4bbe4f70ccf9f5335f5c7226885211d5edc64d5a0a9b.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/tables/2a7448554e37e096b1de4bbe4f70ccf9f5335f5c7226885211d5edc64d5a0a9b.jpg)

![6d8e6d3badc33041768386d644672ebd05786731b25c569cea601e853be95143.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/tables/6d8e6d3badc33041768386d644672ebd05786731b25c569cea601e853be95143.jpg)

## Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond


### Images

![125907e6c65d1be9326b76f0acae3ae0888d2e285e8d09c9a061fe6881a002b8.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/125907e6c65d1be9326b76f0acae3ae0888d2e285e8d09c9a061fe6881a002b8.jpg)

![1278eee71e171309d7e248c8346b524ff2823bb680b2edcfb6887e36b5cfcbc7.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/1278eee71e171309d7e248c8346b524ff2823bb680b2edcfb6887e36b5cfcbc7.jpg)

![144ddcab6d3e8b2311b1114500da579d4319f549ab6dcedffdc8639cbc8b6af5.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/144ddcab6d3e8b2311b1114500da579d4319f549ab6dcedffdc8639cbc8b6af5.jpg)

![637fbe825c8be47ef4843686d5eba6ed5736de39aff32511598c2f3cf6d383e5.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/637fbe825c8be47ef4843686d5eba6ed5736de39aff32511598c2f3cf6d383e5.jpg)

![67afe4e6dc8ae1c54be2e45644e664207e76c878aae426c21a20e1c8cdbf72f9.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/67afe4e6dc8ae1c54be2e45644e664207e76c878aae426c21a20e1c8cdbf72f9.jpg)

![6c3f7cd511572d5900fe0c6d6b37fc76f33b06ea6357a18ed9338917fc0e3037.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/6c3f7cd511572d5900fe0c6d6b37fc76f33b06ea6357a18ed9338917fc0e3037.jpg)

![85c3efda0bd1eed3eaf72035313943032d43fac99fa234d7b71a7789b5c4f8a6.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/85c3efda0bd1eed3eaf72035313943032d43fac99fa234d7b71a7789b5c4f8a6.jpg)

![8fcbe6cc380359c7a80c47e7ff5baae36ed23025064c255bbb8cd39f7e59e99a.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/8fcbe6cc380359c7a80c47e7ff5baae36ed23025064c255bbb8cd39f7e59e99a.jpg)

![a0db097295d5d0a3f33b0bf0706925c1f383197ca97372625f09c264e23dc54a.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/a0db097295d5d0a3f33b0bf0706925c1f383197ca97372625f09c264e23dc54a.jpg)

![a44f6b7fefcb1dc995f23bf9156d5985b2cfd7ccd069617e3313bfed69e9d27c.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/a44f6b7fefcb1dc995f23bf9156d5985b2cfd7ccd069617e3313bfed69e9d27c.jpg)

![bd28cca191b93b7edab3725615344e0cb80862e1aa46f73bc3eb4a20e6fd1a16.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/bd28cca191b93b7edab3725615344e0cb80862e1aa46f73bc3eb4a20e6fd1a16.jpg)

![c7878814a2ffb9f757de37d5dcbadce8931b2c9d319d3c047fd55cf130bba67e.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/c7878814a2ffb9f757de37d5dcbadce8931b2c9d319d3c047fd55cf130bba67e.jpg)

![c94d6fb0f5e7b926173e68efaaa474f26cb4d44191a6138c9f97a6cf07e85971.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/c94d6fb0f5e7b926173e68efaaa474f26cb4d44191a6138c9f97a6cf07e85971.jpg)

![cadf8c590ee8fdf44a78bbea62fd077b3208f4328e4fd053b2f1f6808aff6d2d.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/cadf8c590ee8fdf44a78bbea62fd077b3208f4328e4fd053b2f1f6808aff6d2d.jpg)

![e4ee26ee0dc77ed8075770c6b6abe76afb6821054f19541c9e9a87657ba4d944.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/e4ee26ee0dc77ed8075770c6b6abe76afb6821054f19541c9e9a87657ba4d944.jpg)

![fd5e35a52e54fa5addc3936d4facfbfbc766ef3b38d9f984c25669205736b653.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/fd5e35a52e54fa5addc3936d4facfbfbc766ef3b38d9f984c25669205736b653.jpg)

![ff355a7a5d2e34e1184a1d7067f7dd789e65055fdb814e5cea384d8bd0a32c28.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/images/ff355a7a5d2e34e1184a1d7067f7dd789e65055fdb814e5cea384d8bd0a32c28.jpg)

### Tables

![0859e693407c1fc24aa90f8c6b3439bd4f78ee5dfe64d6fce0c856cc40d358e3.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/0859e693407c1fc24aa90f8c6b3439bd4f78ee5dfe64d6fce0c856cc40d358e3.jpg)

![2576c71d0abe9dad103eaaa35a1eb5fd9dcd2f4900100e271ebc79bf3741c54a.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/2576c71d0abe9dad103eaaa35a1eb5fd9dcd2f4900100e271ebc79bf3741c54a.jpg)

![31b821309ca4848ab1468fadae22b62550a161eccdcfd841c169d8a2089b6123.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/31b821309ca4848ab1468fadae22b62550a161eccdcfd841c169d8a2089b6123.jpg)

![3c7f9aba7ad64809f70664e8ad08a07b4dea1e6e4bb4057276bb6b5463d50327.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/3c7f9aba7ad64809f70664e8ad08a07b4dea1e6e4bb4057276bb6b5463d50327.jpg)

![3ee2a3e8bcd7af3600ec0b1040fa1f405514b27b3ab60883889d5aae568451a3.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/3ee2a3e8bcd7af3600ec0b1040fa1f405514b27b3ab60883889d5aae568451a3.jpg)

![6523b254677e254ec5b14626ecc2ea4d3539e218147f761e6d103a79eb6c7958.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/6523b254677e254ec5b14626ecc2ea4d3539e218147f761e6d103a79eb6c7958.jpg)

![707fb76fce402da3bcc1a745be8f87a9e654ed8afeb6f0e6131b8293657ae251.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/707fb76fce402da3bcc1a745be8f87a9e654ed8afeb6f0e6131b8293657ae251.jpg)

![76c7feb2029c50271e7dc0e1fe44f1686dea1393eb8de5bf887a7fbee80c42ec.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/76c7feb2029c50271e7dc0e1fe44f1686dea1393eb8de5bf887a7fbee80c42ec.jpg)

![94812d448c6d791b069d89fda74dbfe288ab24e48b874892f502be58c013c847.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/94812d448c6d791b069d89fda74dbfe288ab24e48b874892f502be58c013c847.jpg)

![cca56dcb4dcfa045ca46211c2e0f875d07e691e334fd665f1270b76334849345.jpg](../iclr_results/1528_Rethinking%20LLM%20Unlearning%20Objectives_%20A%20Gradient%20Perspective%20and%20Go%20Beyond/tables/cca56dcb4dcfa045ca46211c2e0f875d07e691e334fd665f1270b76334849345.jpg)

## GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack


### Images

![004076ab7eb2507689f7533fb28f9988abecb57790a069e5677c022cee099531.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/004076ab7eb2507689f7533fb28f9988abecb57790a069e5677c022cee099531.jpg)

![14f65b2cad6b839538144c2c484c9928d32d380bd081fadfcfed5b4361032489.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/14f65b2cad6b839538144c2c484c9928d32d380bd081fadfcfed5b4361032489.jpg)

![19c278ed35066fe1f9d1bb850207c535d1384a91f172af74d7420c0240b2c8e6.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/19c278ed35066fe1f9d1bb850207c535d1384a91f172af74d7420c0240b2c8e6.jpg)

![2ca7036af4ec9bce891541dbe14cf96295d259a722008972344fa35aad8a79cb.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/2ca7036af4ec9bce891541dbe14cf96295d259a722008972344fa35aad8a79cb.jpg)

![31d1b1eee4db39e9cb488157861a48ad850b3494f50d2738fbe05aadb537eccf.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/31d1b1eee4db39e9cb488157861a48ad850b3494f50d2738fbe05aadb537eccf.jpg)

![4b1373271317287f9255396a72664012b7d9c82aebad6bd6c081eeaf4eb0cfd0.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/4b1373271317287f9255396a72664012b7d9c82aebad6bd6c081eeaf4eb0cfd0.jpg)

![4bf2e3cdc2ccba7eb0f9d900c15910f6bebb04b59c497540c8299baf3717a9a6.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/4bf2e3cdc2ccba7eb0f9d900c15910f6bebb04b59c497540c8299baf3717a9a6.jpg)

![52d0aad3e7955ad04cac0c0419201b6e73965a95b37278d244a844a4fd4dd282.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/52d0aad3e7955ad04cac0c0419201b6e73965a95b37278d244a844a4fd4dd282.jpg)

![6468cf8e4f4f6b242392fa5d78f9023dcec152c5fb8953ef3b2caea98f4ac697.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/6468cf8e4f4f6b242392fa5d78f9023dcec152c5fb8953ef3b2caea98f4ac697.jpg)

![6f64144a704a67f9b61125a505da0fb1328265eb3e355bf1791b922a8ee85e9e.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/6f64144a704a67f9b61125a505da0fb1328265eb3e355bf1791b922a8ee85e9e.jpg)

![72077aa489e5a114b221445ea8bb1b3772b8626a2e1af10ce33c2e5061c31a78.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/72077aa489e5a114b221445ea8bb1b3772b8626a2e1af10ce33c2e5061c31a78.jpg)

![7dbcbf30fdb8543f2b78555dada8a9ac4c2c5684942b5f86026dad26533d7aa7.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/7dbcbf30fdb8543f2b78555dada8a9ac4c2c5684942b5f86026dad26533d7aa7.jpg)

![7e62681f8bde6edd7ee722304fec7ed4261a28eb1828a06b39562d390e08db56.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/7e62681f8bde6edd7ee722304fec7ed4261a28eb1828a06b39562d390e08db56.jpg)

![8165adc2d4da306310af8d4d45f4e79e2e6a71cb1cdcfdc8c698d3ce9784b822.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/8165adc2d4da306310af8d4d45f4e79e2e6a71cb1cdcfdc8c698d3ce9784b822.jpg)

![83c6e5821bc0d26834bac2447bf489921c0b19df6337038619fedfb24b6a4676.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/83c6e5821bc0d26834bac2447bf489921c0b19df6337038619fedfb24b6a4676.jpg)

![896dba8d9d12531f40204108ef9fb2f9cc8d817d56534907ec374af44beba873.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/896dba8d9d12531f40204108ef9fb2f9cc8d817d56534907ec374af44beba873.jpg)

![922c54444917d4754f045196cff1f5ce8a8b6e311c5616e6b29717bfc2703e70.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/922c54444917d4754f045196cff1f5ce8a8b6e311c5616e6b29717bfc2703e70.jpg)

![9332cdeff1d34c7112c479f6d0c519e1e856525b04219b3736b3543fba7220be.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/9332cdeff1d34c7112c479f6d0c519e1e856525b04219b3736b3543fba7220be.jpg)

![98d179b5aff1d1df39b8b1453e0c3ad25ea3b48f1828a5e60308bc81d6553db2.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/98d179b5aff1d1df39b8b1453e0c3ad25ea3b48f1828a5e60308bc81d6553db2.jpg)

![a7e3589d921c8bcea0a59247ca103ac834e5f6cca4fbd2f60db50cd9b0a2b7e3.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/a7e3589d921c8bcea0a59247ca103ac834e5f6cca4fbd2f60db50cd9b0a2b7e3.jpg)

![b1aba4da55657f0bc8b6922b3e05a47b718ee86181168cfd0db2bf781b738106.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/b1aba4da55657f0bc8b6922b3e05a47b718ee86181168cfd0db2bf781b738106.jpg)

![b36bae5229253ebf6d4f1f3f69759d63a5d2a6104ad2cee9717f969f3b17e6fd.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/b36bae5229253ebf6d4f1f3f69759d63a5d2a6104ad2cee9717f969f3b17e6fd.jpg)

![b8c30f2589b713304232ed63609b6cc62781cd3610cbaba56e31b3db6d0af721.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/b8c30f2589b713304232ed63609b6cc62781cd3610cbaba56e31b3db6d0af721.jpg)

![c0bc25c7c6dcfd7b50cd050652115d3d92215e8950947bd94e2ac32cc8746188.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/c0bc25c7c6dcfd7b50cd050652115d3d92215e8950947bd94e2ac32cc8746188.jpg)

![c115a9fb84b550a75750cca79029d1bcc021bdb4528c0839d26c15b1d5881fc1.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/c115a9fb84b550a75750cca79029d1bcc021bdb4528c0839d26c15b1d5881fc1.jpg)

![c35628b255745a3c2e85ff41b56414ac188b180cff27c7ec7f6c08e7e26bfab4.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/c35628b255745a3c2e85ff41b56414ac188b180cff27c7ec7f6c08e7e26bfab4.jpg)

![ccdf9ebc11d1c1027cdeed3231d5fd1013f0ce69ff648de772b4a0c88890c144.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/ccdf9ebc11d1c1027cdeed3231d5fd1013f0ce69ff648de772b4a0c88890c144.jpg)

![cd96d06d8480df3854de4553adec12c94239d1429beece1c6739525cd6ce2132.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/cd96d06d8480df3854de4553adec12c94239d1429beece1c6739525cd6ce2132.jpg)

![d458656074e1b28517470936f5183c2e658dc9dd4fac2a02b3d07b93223d3ea8.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/d458656074e1b28517470936f5183c2e658dc9dd4fac2a02b3d07b93223d3ea8.jpg)

![e0dec3c58eb66a55b687bd61155e4e35fe68c11ae01cede0f1816f03d8ba1cb1.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/e0dec3c58eb66a55b687bd61155e4e35fe68c11ae01cede0f1816f03d8ba1cb1.jpg)

![e41e30dca36bad263cddf35d387c15313cf68e1a5abd12a451429854f5a98209.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/e41e30dca36bad263cddf35d387c15313cf68e1a5abd12a451429854f5a98209.jpg)

![e7dd26eb95e248691b1d84de6596b6b803db9dfbe71e3e050f811205c1ff27fd.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/e7dd26eb95e248691b1d84de6596b6b803db9dfbe71e3e050f811205c1ff27fd.jpg)

![f170005993523346b3815fc287e36cde3e2e375323d13f37fbf44ae85e904589.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/f170005993523346b3815fc287e36cde3e2e375323d13f37fbf44ae85e904589.jpg)

![fb51c36ce0af69ca4d8399be8b5f14b77c667e2c18c7e2d12ac3b514bb579ae4.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/images/fb51c36ce0af69ca4d8399be8b5f14b77c667e2c18c7e2d12ac3b514bb579ae4.jpg)

### Tables

![115bbf143918f633d567626153da8217d7c984b7ede9492e1d96fa6230945b20.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/115bbf143918f633d567626153da8217d7c984b7ede9492e1d96fa6230945b20.jpg)

![1c2c715dfd7efaec4904bb67b4dfbc4090a07ffd7bf3667bd43d71b24b00fe18.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/1c2c715dfd7efaec4904bb67b4dfbc4090a07ffd7bf3667bd43d71b24b00fe18.jpg)

![2b0b182c95b3a56671fafa53f996b9dc0d51ef68ff766dd900aa4a5745f32919.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/2b0b182c95b3a56671fafa53f996b9dc0d51ef68ff766dd900aa4a5745f32919.jpg)

![4194c2d0ec48bf67a33d553a802d945be3b5d8831797c3cf1c53baba1e9713b6.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/4194c2d0ec48bf67a33d553a802d945be3b5d8831797c3cf1c53baba1e9713b6.jpg)

![588a3b12685cf1c887173f70f306550b53e9cbe4f069aaf552de47d6923fc43b.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/588a3b12685cf1c887173f70f306550b53e9cbe4f069aaf552de47d6923fc43b.jpg)

![67aca6944057b600287a85b0475ac1c241c12cf1595ce089cd331d39d86813e0.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/67aca6944057b600287a85b0475ac1c241c12cf1595ce089cd331d39d86813e0.jpg)

![9b81e4df8a80bb98e743315141dd23957edf163545656f705ab38a1271819bab.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/9b81e4df8a80bb98e743315141dd23957edf163545656f705ab38a1271819bab.jpg)

![9e2993f0241051526f3a5d3b2aeb91a00c8e422f0037fc96a468f453592b64b6.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/9e2993f0241051526f3a5d3b2aeb91a00c8e422f0037fc96a468f453592b64b6.jpg)

![a866566dcfddb7fa74066722464612cb54b1b8f2c2ea04a5b108122d8568b424.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/a866566dcfddb7fa74066722464612cb54b1b8f2c2ea04a5b108122d8568b424.jpg)

![bbbaf7110aaf0924224d6b6d676ded721065a557e8b1c8bdf01c2e0805d0ec70.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/bbbaf7110aaf0924224d6b6d676ded721065a557e8b1c8bdf01c2e0805d0ec70.jpg)

![d35aa35a3fdf4409fa36d9703c0c2462b9cf20069760df7c8f8c125c729e0d10.jpg](../iclr_results/1529_GSBA%24%5EK%24_%20%24top%24-%24K%24%20Geometric%20Score-based%20Black-box%20Attack/tables/d35aa35a3fdf4409fa36d9703c0c2462b9cf20069760df7c8f8c125c729e0d10.jpg)

## Investigating the Pre-Training Dynamics of In-Context Learning: Task Recognition vs. Task Learning


### Images

![0101097cc880ff0c18b5bd2caf1fae22521989d8dd36c5528d8c72a06ae940fd.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/images/0101097cc880ff0c18b5bd2caf1fae22521989d8dd36c5528d8c72a06ae940fd.jpg)

![022a6cea241a39573f6fa05b827b672a1ad1354b7598dc441a50b6e6f94fbdc9.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/images/022a6cea241a39573f6fa05b827b672a1ad1354b7598dc441a50b6e6f94fbdc9.jpg)

![934627385f513ad7cc4d3040417e582c4e05c92e531a8819a3c0d48d1767e956.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/images/934627385f513ad7cc4d3040417e582c4e05c92e531a8819a3c0d48d1767e956.jpg)

![a87e182a1b7b2ec863fa45df19919d3becc53bfa6c603e794219060ed9a73160.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/images/a87e182a1b7b2ec863fa45df19919d3becc53bfa6c603e794219060ed9a73160.jpg)

![d0527fee4b9b814a28778762a81d81089e80fc9c03c50bf177d5751817fb0901.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/images/d0527fee4b9b814a28778762a81d81089e80fc9c03c50bf177d5751817fb0901.jpg)

![f418b455ca4626d8ecd9ca453418c67ac7de0dc7177edaff711acd6053b5273e.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/images/f418b455ca4626d8ecd9ca453418c67ac7de0dc7177edaff711acd6053b5273e.jpg)

### Tables

![1b08133350bb80b53e094edb547458fbb521625ba9be9cac5383e2408ee68bb2.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/tables/1b08133350bb80b53e094edb547458fbb521625ba9be9cac5383e2408ee68bb2.jpg)

![224ffa9f348c4ed3dc9fa6a9637db7c099517503f341c1c95bc4bd3b2c39d43a.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/tables/224ffa9f348c4ed3dc9fa6a9637db7c099517503f341c1c95bc4bd3b2c39d43a.jpg)

![30e3833d6e744f414a69836a212acc895331004333f4e91a2e470f14b03f8447.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/tables/30e3833d6e744f414a69836a212acc895331004333f4e91a2e470f14b03f8447.jpg)

![34d45d92b3960b3286c59dfe04891e4642091bf3323dcc4feb8d4da2956dd791.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/tables/34d45d92b3960b3286c59dfe04891e4642091bf3323dcc4feb8d4da2956dd791.jpg)

![92ed1130ebd38f0037a2350d951274f132f153030ebb86ba1d2ab32f7d6779da.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/tables/92ed1130ebd38f0037a2350d951274f132f153030ebb86ba1d2ab32f7d6779da.jpg)

![de81be2332e5f01ce4a1cf484aadb23b61bab81c04b17b06f32bc873907db87e.jpg](../iclr_results/1530_Investigating%20the%20Pre-Training%20Dynamics%20of%20In-Context%20Learning_%20Task%20Recognition%20vs.%20Task%20Learning/tables/de81be2332e5f01ce4a1cf484aadb23b61bab81c04b17b06f32bc873907db87e.jpg)

## Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs


### Images

![07d32bc7450469c765f2edb7c2665712d9f87c123875925df64a76ac55f033fa.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/07d32bc7450469c765f2edb7c2665712d9f87c123875925df64a76ac55f033fa.jpg)

![1042ea7e4e125d99e2d992ae6b794d85ecd00b3b7cf29ce27b0c76250b545b9c.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/1042ea7e4e125d99e2d992ae6b794d85ecd00b3b7cf29ce27b0c76250b545b9c.jpg)

![18766b7757ae559eba4c59091de92e067d8ff5eb59b696d52135c652fa8250f8.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/18766b7757ae559eba4c59091de92e067d8ff5eb59b696d52135c652fa8250f8.jpg)

![1ad90f08f1cca99319c679f2b82e427716c8013753644ae1bb44b3668c9a0afd.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/1ad90f08f1cca99319c679f2b82e427716c8013753644ae1bb44b3668c9a0afd.jpg)

![1bbecfb29b48237cb2825855d4985fb5d38fae0d1bd6aa84939106af68b3cd01.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/1bbecfb29b48237cb2825855d4985fb5d38fae0d1bd6aa84939106af68b3cd01.jpg)

![1e1e3931bfad9497e8c4224dd6a76a5eaf89cff39bd54c390b1db06b3b6e569c.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/1e1e3931bfad9497e8c4224dd6a76a5eaf89cff39bd54c390b1db06b3b6e569c.jpg)

![2775352e5388b822934e513672294536c53644b104f2472f4174c68a93421c2c.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/2775352e5388b822934e513672294536c53644b104f2472f4174c68a93421c2c.jpg)

![376be877648fd657ce6f6a3c3f62de46767ec49695592508221eaef431d90fe3.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/376be877648fd657ce6f6a3c3f62de46767ec49695592508221eaef431d90fe3.jpg)

![3f1c2c41c5dcfdfd2ba5447f7b60153857d9517d8d5065a0017606c01480795b.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/3f1c2c41c5dcfdfd2ba5447f7b60153857d9517d8d5065a0017606c01480795b.jpg)

![4fcd9acc49b08ab4968fbe9b77cdb2278d598c646861105a45e981361806a1c4.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/4fcd9acc49b08ab4968fbe9b77cdb2278d598c646861105a45e981361806a1c4.jpg)

![544479d429d4756809f4aaaf37ca618f64ba67af8e5e141832740410be15dc7f.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/544479d429d4756809f4aaaf37ca618f64ba67af8e5e141832740410be15dc7f.jpg)

![671bbd9750b7327d63d63631d10c747945549e45d3e63227a7273c58a2e24438.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/671bbd9750b7327d63d63631d10c747945549e45d3e63227a7273c58a2e24438.jpg)

![68fc33b44892bdfc1d4eaa27287ecc2ae3a1169408f3976018aef8472d7ba9d9.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/68fc33b44892bdfc1d4eaa27287ecc2ae3a1169408f3976018aef8472d7ba9d9.jpg)

![69565702fe0cb6ea58c739dd1d8f85e3adb53054972fadd773dfeaf23397770e.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/69565702fe0cb6ea58c739dd1d8f85e3adb53054972fadd773dfeaf23397770e.jpg)

![72c4888888daab690562f8bb4627226bff5eecd5c6830af2575f5afd55a031c9.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/72c4888888daab690562f8bb4627226bff5eecd5c6830af2575f5afd55a031c9.jpg)

![7bbbe21e1b9f0b22ab08d3cfb8d243088d08ef0b6abbd3077680d67f1230f996.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/7bbbe21e1b9f0b22ab08d3cfb8d243088d08ef0b6abbd3077680d67f1230f996.jpg)

![7bc49c1e2c1a9421d59a56eadb8073a76a4e7d0d14cedbda642f1f236bf2d52b.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/7bc49c1e2c1a9421d59a56eadb8073a76a4e7d0d14cedbda642f1f236bf2d52b.jpg)

![8344ba973e771c65d5d99f21e4940a77f8697ddd82a6e1c3f942a7d2f5fcba69.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/8344ba973e771c65d5d99f21e4940a77f8697ddd82a6e1c3f942a7d2f5fcba69.jpg)

![8b54033f6f47028f3bed14faa82575b8e4de72e5def77bf11777694da58529de.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/8b54033f6f47028f3bed14faa82575b8e4de72e5def77bf11777694da58529de.jpg)

![95fddac09bc87131f6a9595b3984f4074cec3621bef121ad1774641804cd8022.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/95fddac09bc87131f6a9595b3984f4074cec3621bef121ad1774641804cd8022.jpg)

![9f8db8cc0d387c45b4b07da325b1f53cc019ffd202c6d101abe43e73017e5290.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/9f8db8cc0d387c45b4b07da325b1f53cc019ffd202c6d101abe43e73017e5290.jpg)

![ac5304ff1bf68ec69bc3dd1a9d2260204ac96968799166ed17f0fba541720f94.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/ac5304ff1bf68ec69bc3dd1a9d2260204ac96968799166ed17f0fba541720f94.jpg)

![b497a5a2069c8e0dc28aeb9253af131d601d2d0be19cc0858101fbbba6b9efcb.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/b497a5a2069c8e0dc28aeb9253af131d601d2d0be19cc0858101fbbba6b9efcb.jpg)

![b7307ad18be0535623bc386f889267437f36f45ea6a1a9e2e57b11ad70b6dd7e.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/b7307ad18be0535623bc386f889267437f36f45ea6a1a9e2e57b11ad70b6dd7e.jpg)

![c5a92baf54cb5925711bf06abcbb27a19b93ccea716ccf18e31c1ab11ff732f6.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/c5a92baf54cb5925711bf06abcbb27a19b93ccea716ccf18e31c1ab11ff732f6.jpg)

![cc03be01ffad4317b6acd34660f6cbf9831118f4098f3f78805533f73296137e.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/cc03be01ffad4317b6acd34660f6cbf9831118f4098f3f78805533f73296137e.jpg)

![cd8343135b6f36d3acba1aefcf6d553058098941d07588d249a62b6309e656f4.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/cd8343135b6f36d3acba1aefcf6d553058098941d07588d249a62b6309e656f4.jpg)

![db2b85ef0aa769d8bee4d7accc22481575c3e01d11a84fe06de5570619314204.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/db2b85ef0aa769d8bee4d7accc22481575c3e01d11a84fe06de5570619314204.jpg)

![eb71fc3f0487b34ea8e87048d02f230b1b1bf02a8aaf86db1ea38c52fa355961.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/images/eb71fc3f0487b34ea8e87048d02f230b1b1bf02a8aaf86db1ea38c52fa355961.jpg)

### Tables

![00385f97f2c2e3c071517fa0272c85a8a39bd8a7e883a6a16ed0dca1a5871aa6.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/tables/00385f97f2c2e3c071517fa0272c85a8a39bd8a7e883a6a16ed0dca1a5871aa6.jpg)

![030699fd934d9099bf0c0f1f34fed18bcd6514eef58ae4334e7e71ca11edc71c.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/tables/030699fd934d9099bf0c0f1f34fed18bcd6514eef58ae4334e7e71ca11edc71c.jpg)

![a6eb8314ca8dfecdb75b5c5be03ca0fb9fa82317300f6fe69667a8e4a5cd847e.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/tables/a6eb8314ca8dfecdb75b5c5be03ca0fb9fa82317300f6fe69667a8e4a5cd847e.jpg)

![f349179e0f28f87be0c9a145fda00c162a7f0b6b885355614702f461e7ad5ca6.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/tables/f349179e0f28f87be0c9a145fda00c162a7f0b6b885355614702f461e7ad5ca6.jpg)

![fddd127cb23e1de3fa8c58f8bf8ad5420aa80c7cbf4952742e78a987117f1054.jpg](../iclr_results/1531_Straight%20to%20Zero_%20Why%20Linearly%20Decaying%20the%20Learning%20Rate%20to%20Zero%20Works%20Best%20for%20LLMs/tables/fddd127cb23e1de3fa8c58f8bf8ad5420aa80c7cbf4952742e78a987117f1054.jpg)

## TDDBench: A Benchmark for Training data detection


### Images

![1dfe1335dea2a8a74e6e629d50f6005ad1cca74431584e797d2b5943a172f421.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/images/1dfe1335dea2a8a74e6e629d50f6005ad1cca74431584e797d2b5943a172f421.jpg)

![3c869c790137ddac139716895e435341d70a3df4b525e91c4c20c017d33d5ec4.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/images/3c869c790137ddac139716895e435341d70a3df4b525e91c4c20c017d33d5ec4.jpg)

![e3d04157dc6f90cb03e75aaa88d8168f4e44a5ca73078ee57e13cb27c4419072.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/images/e3d04157dc6f90cb03e75aaa88d8168f4e44a5ca73078ee57e13cb27c4419072.jpg)

![eecaf6f8b32cc2fa0a19248ce5d1b0ee3e4de93b75d6ca094e0c9520374578a5.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/images/eecaf6f8b32cc2fa0a19248ce5d1b0ee3e4de93b75d6ca094e0c9520374578a5.jpg)

### Tables

![1490547a86ebae0e0ea1e271583dc7e5f315d178acdf1e14e47fd89339d25967.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/1490547a86ebae0e0ea1e271583dc7e5f315d178acdf1e14e47fd89339d25967.jpg)

![1c806334302b6b20b172cea7074e699b55a9b3d96032098aecdd0a471ece0666.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/1c806334302b6b20b172cea7074e699b55a9b3d96032098aecdd0a471ece0666.jpg)

![275b2ce9deeca5a72cfc16ce21417e9e66b4b6abbd7b4905bd44cd91903e9341.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/275b2ce9deeca5a72cfc16ce21417e9e66b4b6abbd7b4905bd44cd91903e9341.jpg)

![2da2061ae9cbc9e248655dda2b56db84217d5acdb7cd1c6bc3e7218ef8a60063.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/2da2061ae9cbc9e248655dda2b56db84217d5acdb7cd1c6bc3e7218ef8a60063.jpg)

![39a45d8ac0a2ae0d7727e76f3dc63d0df5d64ab00bcb6a8f6912fbac83f9b34d.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/39a45d8ac0a2ae0d7727e76f3dc63d0df5d64ab00bcb6a8f6912fbac83f9b34d.jpg)

![3ae63278c317a5700183dfc077240fc57d900f217d3e3cc731901df15d52369e.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/3ae63278c317a5700183dfc077240fc57d900f217d3e3cc731901df15d52369e.jpg)

![40ba8a8342da0f9fbfdd30edafc853afd21f7d12507d14f68434fde9a58ab50b.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/40ba8a8342da0f9fbfdd30edafc853afd21f7d12507d14f68434fde9a58ab50b.jpg)

![4392244bc5a84b18b3bdd9517fd7e57bbbc03e3af9ad66918ec41312fd300041.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/4392244bc5a84b18b3bdd9517fd7e57bbbc03e3af9ad66918ec41312fd300041.jpg)

![4a47e0e5795acea83d2225e97de8ec95cc2b8f2b128e9e185a163aacf183b776.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/4a47e0e5795acea83d2225e97de8ec95cc2b8f2b128e9e185a163aacf183b776.jpg)

![7a861af8a14671117c7853779511cad72b626d03bf04d967ec87eaa13e4e6421.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/7a861af8a14671117c7853779511cad72b626d03bf04d967ec87eaa13e4e6421.jpg)

![7d69b3d69ce4ec5ad75e22792aec1362c3a37c2d17540587567cdf1d9bd3d790.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/7d69b3d69ce4ec5ad75e22792aec1362c3a37c2d17540587567cdf1d9bd3d790.jpg)

![8c5c4cde2311c5d22e608a10b76a635407e49c2d2c578f040e05827671437a98.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/8c5c4cde2311c5d22e608a10b76a635407e49c2d2c578f040e05827671437a98.jpg)

![8f4aa46fa716555ee3c4dbd5b4941174a85a9e2b96d35b753b4c9f4d2292c8f2.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/8f4aa46fa716555ee3c4dbd5b4941174a85a9e2b96d35b753b4c9f4d2292c8f2.jpg)

![90cae37f008218bd784f629b8005de2416f4a3208208e060277d875de0e3db5b.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/90cae37f008218bd784f629b8005de2416f4a3208208e060277d875de0e3db5b.jpg)

![92b5a5f6d16ac585bd2f161a93f99841c9a082fb31feb354a6f4d5adb3d61bde.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/92b5a5f6d16ac585bd2f161a93f99841c9a082fb31feb354a6f4d5adb3d61bde.jpg)

![9cfb77150b81e0ebcb7584c810858cd8cbd51e328189ee9de3e39c984e5e9b4a.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/9cfb77150b81e0ebcb7584c810858cd8cbd51e328189ee9de3e39c984e5e9b4a.jpg)

![a01bde74b610f7cd91b152f7ff6ec797a3e3544b05cd459877b73ba8df3bc0e9.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/a01bde74b610f7cd91b152f7ff6ec797a3e3544b05cd459877b73ba8df3bc0e9.jpg)

![a9f66de5eb0ebe57e429688b0556ebf35b82b8bde6cdec5b731648c0699003f4.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/a9f66de5eb0ebe57e429688b0556ebf35b82b8bde6cdec5b731648c0699003f4.jpg)

![b4c84d02c4042f88da12c4151bda613074049b4b18e87b140d176ba3f723e4e4.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/b4c84d02c4042f88da12c4151bda613074049b4b18e87b140d176ba3f723e4e4.jpg)

![c0315bd871a0b94b2f99679616aa102f52f4c3da2081b93e60716db403851292.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/c0315bd871a0b94b2f99679616aa102f52f4c3da2081b93e60716db403851292.jpg)

![de58aa559ef3c67bccb4f88358189145872ac9e4679efe8f03d3a8e5e3ab717b.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/de58aa559ef3c67bccb4f88358189145872ac9e4679efe8f03d3a8e5e3ab717b.jpg)

![f87d40e7c8ae27e4469ec2a950086460f5541be78d7070f241ae5b07763bb567.jpg](../iclr_results/1532_TDDBench_%20A%20Benchmark%20for%20Training%20data%20detection/tables/f87d40e7c8ae27e4469ec2a950086460f5541be78d7070f241ae5b07763bb567.jpg)

## AutoG: Towards automatic graph construction from tabular data


### Images

![0143382b0dd7096f4462b5ad7dca5a5a5dc6eca264dca128ba06116f5fc141d6.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/0143382b0dd7096f4462b5ad7dca5a5a5dc6eca264dca128ba06116f5fc141d6.jpg)

![0fcdb9a0da4e0c2989f8172bfbed16ae533bab5ef29a3fe987703fe61ee1db56.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/0fcdb9a0da4e0c2989f8172bfbed16ae533bab5ef29a3fe987703fe61ee1db56.jpg)

![11525ad659f6bb0be6446446227d1cd5f7282e1e28ad9dc8244516135d489c25.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/11525ad659f6bb0be6446446227d1cd5f7282e1e28ad9dc8244516135d489c25.jpg)

![230cf419b22ff15adaa26d18a169e34666307197fc759fad3bc528531dd92c36.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/230cf419b22ff15adaa26d18a169e34666307197fc759fad3bc528531dd92c36.jpg)

![253820d6392cc7b8e8c9185c89fc1029ce88a8ca952fd225a5d1bf5620206f30.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/253820d6392cc7b8e8c9185c89fc1029ce88a8ca952fd225a5d1bf5620206f30.jpg)

![31b9840d5d2239926ea1e5e47da323b7d09a0776a5f2e94bb2877d4d53377c8d.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/31b9840d5d2239926ea1e5e47da323b7d09a0776a5f2e94bb2877d4d53377c8d.jpg)

![347dd3d7cfec37ce2daf284ccb7c59fa4147e408dba5b8290e63c3fdc8671659.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/347dd3d7cfec37ce2daf284ccb7c59fa4147e408dba5b8290e63c3fdc8671659.jpg)

![394e0d85ceba583f4338f29181b7e5ea295dbf95f018f5f399f0e1550967aedb.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/394e0d85ceba583f4338f29181b7e5ea295dbf95f018f5f399f0e1550967aedb.jpg)

![53d73792cc81c9995110ff073f28ae0815dd790ee1e5656429116bcd6dfbb858.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/53d73792cc81c9995110ff073f28ae0815dd790ee1e5656429116bcd6dfbb858.jpg)

![5794d301514c31035bfaeb4adb73d42f003f851464ec82b1b2ba331e49605883.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/5794d301514c31035bfaeb4adb73d42f003f851464ec82b1b2ba331e49605883.jpg)

![5e340810f6164afd1dcf7bf8a780c770754bad2bdd1a38369cf45b4a9bdf053a.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/5e340810f6164afd1dcf7bf8a780c770754bad2bdd1a38369cf45b4a9bdf053a.jpg)

![5f682ffd7d863120673e1c8884f1741f2beb6364253b9c3897d3ef9e358a22d0.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/5f682ffd7d863120673e1c8884f1741f2beb6364253b9c3897d3ef9e358a22d0.jpg)

![6af8359b6326dde57b9b55e049b96b6650392ba19dc6977287910f50c5a7afd4.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/6af8359b6326dde57b9b55e049b96b6650392ba19dc6977287910f50c5a7afd4.jpg)

![7923f1565d001bf0eab03452d264ccb2fb97bc2e2aae32452bdb667ff904e0ef.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/7923f1565d001bf0eab03452d264ccb2fb97bc2e2aae32452bdb667ff904e0ef.jpg)

![7c03402fe519ea83dab34c069c986dfd205a5d34ce2cde65756ba6b0c2bdcf76.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/7c03402fe519ea83dab34c069c986dfd205a5d34ce2cde65756ba6b0c2bdcf76.jpg)

![80f4d6b43dfc22057c741aa10fcdc76f489734fe817d3413ad839938c0287c40.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/80f4d6b43dfc22057c741aa10fcdc76f489734fe817d3413ad839938c0287c40.jpg)

![88b59384c6d8520b246b7e66ae301b1738c886ac5a233fec4d06307ce9684ef4.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/88b59384c6d8520b246b7e66ae301b1738c886ac5a233fec4d06307ce9684ef4.jpg)

![95d0c184c91382f8cd9ca1dad362df9591659b3088a6a3d0a1207dd87c15670b.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/95d0c184c91382f8cd9ca1dad362df9591659b3088a6a3d0a1207dd87c15670b.jpg)

![9923888e9e3591c1adea3a546f1366bb52d18bde3e2b20b3f8ea47d63f4e946e.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/9923888e9e3591c1adea3a546f1366bb52d18bde3e2b20b3f8ea47d63f4e946e.jpg)

![9bbffa2739f2e0f7408a3965a939e8da34402fdda3a3c713790618a9570d2bc1.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/9bbffa2739f2e0f7408a3965a939e8da34402fdda3a3c713790618a9570d2bc1.jpg)

![a26658d2dbfaa432e6c3ab9ed31b092235f8dae81c22d4f2ea6275e50c748e36.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/a26658d2dbfaa432e6c3ab9ed31b092235f8dae81c22d4f2ea6275e50c748e36.jpg)

![b0fef49370615d6317d6e3bb4a306ca3a8da060e5dcadc50124330fb7335c3ba.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/b0fef49370615d6317d6e3bb4a306ca3a8da060e5dcadc50124330fb7335c3ba.jpg)

![c27621fde8b1ff5240a5599d6e1fe32baaa4da883cfcf9a4bac93590bf5a6cd5.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/c27621fde8b1ff5240a5599d6e1fe32baaa4da883cfcf9a4bac93590bf5a6cd5.jpg)

![de1bd892f09b0f9548bdf98122a7ce27a0586b8574dd3498c87557c911573883.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/images/de1bd892f09b0f9548bdf98122a7ce27a0586b8574dd3498c87557c911573883.jpg)

### Tables

![05236918d7bfd9a23b8e693452b5c980a95b4d17683630137229952931109cb7.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/05236918d7bfd9a23b8e693452b5c980a95b4d17683630137229952931109cb7.jpg)

![06d04c88e1a587df3fe9ed8ab945fbc3a5cea785e47b8ac0f7a95d435e1dd94c.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/06d04c88e1a587df3fe9ed8ab945fbc3a5cea785e47b8ac0f7a95d435e1dd94c.jpg)

![1dac5a27607b181aabc6a46bcb1c404da1155eb38740d904830823da1d2e7550.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/1dac5a27607b181aabc6a46bcb1c404da1155eb38740d904830823da1d2e7550.jpg)

![47d093d665c30166cd8a2c2dcc6949a641715960a5953661591cf54380176f59.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/47d093d665c30166cd8a2c2dcc6949a641715960a5953661591cf54380176f59.jpg)

![4aa3217592bc89a2127a67ee24941a23f9988d5663c29a9920bb259d940e89e9.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/4aa3217592bc89a2127a67ee24941a23f9988d5663c29a9920bb259d940e89e9.jpg)

![9837bc70228c663e735060bc67566f229dafc717f119b1adad710e011865847d.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/9837bc70228c663e735060bc67566f229dafc717f119b1adad710e011865847d.jpg)

![d837be2194a4fc2adf7efb5194998103878e900517d25f15233c25528f83f773.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/d837be2194a4fc2adf7efb5194998103878e900517d25f15233c25528f83f773.jpg)

![fd5eb3c905e5760387dbc3d1fb99d2d0157f7814be771efb42be4a9ff1e646b4.jpg](../iclr_results/1533_AutoG_%20Towards%20automatic%20graph%20construction%20from%20tabular%20data/tables/fd5eb3c905e5760387dbc3d1fb99d2d0157f7814be771efb42be4a9ff1e646b4.jpg)

## DelTA: An Online Document-Level Translation Agent Based on Multi-Level Memory


### Images

![0ee76f6d798727e3193e8ba6ab48bac34813d01e07e8e4265d193d7a31842e50.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/images/0ee76f6d798727e3193e8ba6ab48bac34813d01e07e8e4265d193d7a31842e50.jpg)

![1719d34427406e8b7f778a824dedbd542c2fdef2bcc02db6ba63559199819a8c.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/images/1719d34427406e8b7f778a824dedbd542c2fdef2bcc02db6ba63559199819a8c.jpg)

![ab390148872dac10f17f80b6240ee9798a326f078f8b99352e7afc467714a1de.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/images/ab390148872dac10f17f80b6240ee9798a326f078f8b99352e7afc467714a1de.jpg)

![d06cdc8368126d5d18b87f77ecf2b619334a55aca63c5a5ebd146dffcb9728e7.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/images/d06cdc8368126d5d18b87f77ecf2b619334a55aca63c5a5ebd146dffcb9728e7.jpg)

### Tables

![0ee7dddefc51c7b0ce1eb97cc3e81c726dd9ad0ad5abaa8d8825b0bd14198f70.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/0ee7dddefc51c7b0ce1eb97cc3e81c726dd9ad0ad5abaa8d8825b0bd14198f70.jpg)

![184eeed6555d7f54f9fa9bd12d740644cc8ea4f485fbdf92444a509ca040e383.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/184eeed6555d7f54f9fa9bd12d740644cc8ea4f485fbdf92444a509ca040e383.jpg)

![1aee744c126f97d0751b6dabbec8864d4dcd4d0ac3dfdccde8e600331896d2f3.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/1aee744c126f97d0751b6dabbec8864d4dcd4d0ac3dfdccde8e600331896d2f3.jpg)

![2374840eb23cffa4102164227763c400177b9b13fd3969ac8485051a564e7c67.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/2374840eb23cffa4102164227763c400177b9b13fd3969ac8485051a564e7c67.jpg)

![4e353be45cb4fc756bf21823c3acf34ec928143e969dcd8434e1aaa9a42233ae.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/4e353be45cb4fc756bf21823c3acf34ec928143e969dcd8434e1aaa9a42233ae.jpg)

![6720ba36935f2f54bfe29e0ab01275f5f7b1290afb85c26cab90cd86a862fc71.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/6720ba36935f2f54bfe29e0ab01275f5f7b1290afb85c26cab90cd86a862fc71.jpg)

![725a46f7398196825ff31c849de70cbdfe4fe7a5a81779cf40099c6d8208a412.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/725a46f7398196825ff31c849de70cbdfe4fe7a5a81779cf40099c6d8208a412.jpg)

![a811da71c8cc5d1cc8629bf90f10b1b0d64fdba03d0876918267267d5ee53ae6.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/a811da71c8cc5d1cc8629bf90f10b1b0d64fdba03d0876918267267d5ee53ae6.jpg)

![b30067b29aa63ebac813b6c1a30f036aa3a1787ad952bc164494a1afbdb74524.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/b30067b29aa63ebac813b6c1a30f036aa3a1787ad952bc164494a1afbdb74524.jpg)

![c00803bdf880370b0861b11a90f82aee0ff5a6863a37a65b3e4d5e4691562474.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/c00803bdf880370b0861b11a90f82aee0ff5a6863a37a65b3e4d5e4691562474.jpg)

![c14fc65fc318d944e4440b546d2898d5c7158da27ad72c622a38443ab00cf311.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/c14fc65fc318d944e4440b546d2898d5c7158da27ad72c622a38443ab00cf311.jpg)

![cdc36cc57e4fe105f92417fc867dd0658348d9ef70b8eb7613df7a83a0d8cfde.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/cdc36cc57e4fe105f92417fc867dd0658348d9ef70b8eb7613df7a83a0d8cfde.jpg)

![f4027a50d196df7b99d51b29cdb3a5126c6e64a6549c219ff77e202a019d5119.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/f4027a50d196df7b99d51b29cdb3a5126c6e64a6549c219ff77e202a019d5119.jpg)

![fc4cd8fd5bcd04d20b7098c069911054345a8dfc89966998a4c1045a872ace4b.jpg](../iclr_results/1534_DelTA_%20An%20Online%20Document-Level%20Translation%20Agent%20Based%20on%20Multi-Level%20Memory/tables/fc4cd8fd5bcd04d20b7098c069911054345a8dfc89966998a4c1045a872ace4b.jpg)

## Mind Control through Causal Inference: Predicting Clean Images from Poisoned Data


### Images

![1b8894b42144e5f10324c5c49d1337c3915902d1ae7b2f3b81aa0b1bdac23eda.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/1b8894b42144e5f10324c5c49d1337c3915902d1ae7b2f3b81aa0b1bdac23eda.jpg)

![21531a634f229d90a2a4fe611a2035cee106be99dc5183860a7532bf7fc0bf13.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/21531a634f229d90a2a4fe611a2035cee106be99dc5183860a7532bf7fc0bf13.jpg)

![308f93d9eed9e291b8697b372b124d94c5f511314b3bd834fa223cf2487bfc14.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/308f93d9eed9e291b8697b372b124d94c5f511314b3bd834fa223cf2487bfc14.jpg)

![492bc78b8825629563d9be05e247e4ef9ede67d876b7ff8791a892ec5942f333.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/492bc78b8825629563d9be05e247e4ef9ede67d876b7ff8791a892ec5942f333.jpg)

![53f65361a68fabe80969807e256ad83a94e5b28a3d657dce7ece7b4b6e382bd5.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/53f65361a68fabe80969807e256ad83a94e5b28a3d657dce7ece7b4b6e382bd5.jpg)

![545b4097cadb88097eabe258437fe3be2ac6a508962974d86e95944eb47f5901.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/545b4097cadb88097eabe258437fe3be2ac6a508962974d86e95944eb47f5901.jpg)

![68ef3f40e009338931f75665b8f9947b3d8ed6da335c2312e45c92e4b50bd125.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/68ef3f40e009338931f75665b8f9947b3d8ed6da335c2312e45c92e4b50bd125.jpg)

![698961a4774e342dec51108a085a2ea824d4b2555f1a530373dffa742b6c75f1.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/698961a4774e342dec51108a085a2ea824d4b2555f1a530373dffa742b6c75f1.jpg)

![6d172f0b8fda8051684a7a7d1474be8f21b98f00ff39da7a704c13aff2c2a0f9.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/6d172f0b8fda8051684a7a7d1474be8f21b98f00ff39da7a704c13aff2c2a0f9.jpg)

![7f8d50128cbb6beb7f8a1e4a1dc4101e277e062d5d1612dc9152269e1cf394bb.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/7f8d50128cbb6beb7f8a1e4a1dc4101e277e062d5d1612dc9152269e1cf394bb.jpg)

![a9d90ac1875293ebf995c158a3bfb889be6fef6759063ddc1cfe39c9659b80b1.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/a9d90ac1875293ebf995c158a3bfb889be6fef6759063ddc1cfe39c9659b80b1.jpg)

![aa8523a2b350cb18a89cef4f543392f13fb8cf7a212bcad9f0ed2d6afe4634ca.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/aa8523a2b350cb18a89cef4f543392f13fb8cf7a212bcad9f0ed2d6afe4634ca.jpg)

![b278eac8ebecade2aad39e62b71e470d46ce370792f24a0625e054127073f4a0.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/b278eac8ebecade2aad39e62b71e470d46ce370792f24a0625e054127073f4a0.jpg)

![cfb5a3969471d53eaf0107eaf0acf4a682c7d1cb3e7dc1d2351d9ae837cfcd97.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/cfb5a3969471d53eaf0107eaf0acf4a682c7d1cb3e7dc1d2351d9ae837cfcd97.jpg)

![d7765e7ba869fba04bba166f1db8a9e6ebe5ae363736972b783eb2d44bfc8659.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/images/d7765e7ba869fba04bba166f1db8a9e6ebe5ae363736972b783eb2d44bfc8659.jpg)

### Tables

![0e1daca969719df8929fb8fe2405f751a79cbcfa298de96b19449a113caed2b0.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/0e1daca969719df8929fb8fe2405f751a79cbcfa298de96b19449a113caed2b0.jpg)

![13295920cb4f12baf272cbcaf06788c426ed3a8316389ed91e966665fb26fb6f.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/13295920cb4f12baf272cbcaf06788c426ed3a8316389ed91e966665fb26fb6f.jpg)

![3e75323279a2d0cda3cb272ef48cfa178a472871d4c3e8609cf03be2e14f74b3.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/3e75323279a2d0cda3cb272ef48cfa178a472871d4c3e8609cf03be2e14f74b3.jpg)

![40270685672fad1204e2c8e74bae46bfab7b725f1b30ae7f7d9b3b4174027092.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/40270685672fad1204e2c8e74bae46bfab7b725f1b30ae7f7d9b3b4174027092.jpg)

![4a9efa1fc3ea02d02061100713226dfe1531765011c42c72f796332a4e431748.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/4a9efa1fc3ea02d02061100713226dfe1531765011c42c72f796332a4e431748.jpg)

![5729c8457a005768d5642793b5453fa45b70a677ace39b4fa6fe3d8cf54952cf.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/5729c8457a005768d5642793b5453fa45b70a677ace39b4fa6fe3d8cf54952cf.jpg)

![7137c594b68cda79cfea92d4f62b10800a0763468651baf37fc7196017f543dc.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/7137c594b68cda79cfea92d4f62b10800a0763468651baf37fc7196017f543dc.jpg)

![72eb7fba12ecd714666420abec4dcdde614aed02645029f4314231addab796fd.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/72eb7fba12ecd714666420abec4dcdde614aed02645029f4314231addab796fd.jpg)

![7bd3666afc45c4b41863000b16b9236fcafb61ed177bae4d9362eb3d7e816d3f.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/7bd3666afc45c4b41863000b16b9236fcafb61ed177bae4d9362eb3d7e816d3f.jpg)

![855c0eb9c15c0d663f20d465e96985e70d3425f703e1b46c80fc2d09b86c7104.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/855c0eb9c15c0d663f20d465e96985e70d3425f703e1b46c80fc2d09b86c7104.jpg)

![85ca53c5b7c2d8c65baaa5b574085dfddb5df7b2a6cff53e18a4194e13383d6c.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/85ca53c5b7c2d8c65baaa5b574085dfddb5df7b2a6cff53e18a4194e13383d6c.jpg)

![a95bcc09a0b71d56cfec7f436d3333b8830310adb835a524234cfbe81e592b21.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/a95bcc09a0b71d56cfec7f436d3333b8830310adb835a524234cfbe81e592b21.jpg)

![d80664c054d2aee31b045fa742fa40a7dc659296806ae48a11311e9cb4ffdfab.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/d80664c054d2aee31b045fa742fa40a7dc659296806ae48a11311e9cb4ffdfab.jpg)

![fedde2cc2776182560aa84f0622f09d2d78be4929185b07e244b34211cd0544c.jpg](../iclr_results/1535_Mind%20Control%20through%20Causal%20Inference_%20Predicting%20Clean%20Images%20from%20Poisoned%20Data/tables/fedde2cc2776182560aa84f0622f09d2d78be4929185b07e244b34211cd0544c.jpg)

## Can Knowledge Editing Really Correct Hallucinations?


### Images

![0b438ec955522674567cb6fdb7b8f494438c0054c7d46ecda0023aceb0c0fc2f.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/0b438ec955522674567cb6fdb7b8f494438c0054c7d46ecda0023aceb0c0fc2f.jpg)

![161ced33298424ae129a4046cd6a4965e07de7d63d17fd215fd83ddc881f53a0.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/161ced33298424ae129a4046cd6a4965e07de7d63d17fd215fd83ddc881f53a0.jpg)

![183c1dc543f04c7b7f4d7bfbe8e55963f47cae02156f38e6f6d3e6fd3aa9bae2.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/183c1dc543f04c7b7f4d7bfbe8e55963f47cae02156f38e6f6d3e6fd3aa9bae2.jpg)

![2948fc487e56d3645f32a3a89d9f991eef72208cc17e07ae7e411cc136cdf682.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/2948fc487e56d3645f32a3a89d9f991eef72208cc17e07ae7e411cc136cdf682.jpg)

![35fefb9bce274317a6c28af7837faccbfb9e4c4290360ffdaeedc1c1874de31d.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/35fefb9bce274317a6c28af7837faccbfb9e4c4290360ffdaeedc1c1874de31d.jpg)

![3c60d51150fe482428c61f07a40fc531c6f3e0c1acc27c4e70429c310eea659f.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/3c60d51150fe482428c61f07a40fc531c6f3e0c1acc27c4e70429c310eea659f.jpg)

![5c2bef7f7b076d5fd98b50d0cb6d6ee242e8f6ef67087a4b17c727b6963bc765.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/5c2bef7f7b076d5fd98b50d0cb6d6ee242e8f6ef67087a4b17c727b6963bc765.jpg)

![63a2ffc3c9813057ec4e11b8b1c1c3ddb894790abab59a1abedbe999f651de82.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/63a2ffc3c9813057ec4e11b8b1c1c3ddb894790abab59a1abedbe999f651de82.jpg)

![67c3b73eb91eb891f9624bb0615f496f08072bb02e7a0170af2327a636dbf63e.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/67c3b73eb91eb891f9624bb0615f496f08072bb02e7a0170af2327a636dbf63e.jpg)

![77df48abe239136cfac01ec7c28a4bd679cef5d90dad0c731adeda1c47ce60b3.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/77df48abe239136cfac01ec7c28a4bd679cef5d90dad0c731adeda1c47ce60b3.jpg)

![8167d6c3a4c58f5bb23d336aaa3dd2b429ec31f27ee9aac7426f8734567ef6a2.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/8167d6c3a4c58f5bb23d336aaa3dd2b429ec31f27ee9aac7426f8734567ef6a2.jpg)

![917dd6a476b481a83c1ec3e6a54b1f301152fda1ec53fe3183f9e3202af85fd4.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/917dd6a476b481a83c1ec3e6a54b1f301152fda1ec53fe3183f9e3202af85fd4.jpg)

![93e8330e99cdf7799e849d1a414bdeff50503b5ea5746ef1bf096cfdab35e0ca.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/93e8330e99cdf7799e849d1a414bdeff50503b5ea5746ef1bf096cfdab35e0ca.jpg)

![b38cf0bd2cfbff78636e3dc59774a1860b67ab588369e06e6442dcc4d9c794e6.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/b38cf0bd2cfbff78636e3dc59774a1860b67ab588369e06e6442dcc4d9c794e6.jpg)

![c4cffff017f68b809e53100e7940fe703f18e3c92b3362c22d055c4bc0810e4e.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/c4cffff017f68b809e53100e7940fe703f18e3c92b3362c22d055c4bc0810e4e.jpg)

![c8028e7227bf94a4a37cd3497c548a08c52d722a05077269dec1a1382fcdc560.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/c8028e7227bf94a4a37cd3497c548a08c52d722a05077269dec1a1382fcdc560.jpg)

![d98ce114b786109485fef3f03da800b433730fd985eac21545194b2976aaa886.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/d98ce114b786109485fef3f03da800b433730fd985eac21545194b2976aaa886.jpg)

![f9d6264adaeff870e0f81ed8b534f9673e733f98ec0bb0b6e49a4ee2e83d1ebd.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/images/f9d6264adaeff870e0f81ed8b534f9673e733f98ec0bb0b6e49a4ee2e83d1ebd.jpg)

### Tables

![b54e3cb7b3c951222845278e432436260eb4018161a9208736a89cfc79163507.jpg](../iclr_results/1536_Can%20Knowledge%20Editing%20Really%20Correct%20Hallucinations_/tables/b54e3cb7b3c951222845278e432436260eb4018161a9208736a89cfc79163507.jpg)

## OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination


### Images

![05b814cc9b14a6f35489332f9d06879aa75d99d32fa348b6bdde1124bf3c7aa8.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/05b814cc9b14a6f35489332f9d06879aa75d99d32fa348b6bdde1124bf3c7aa8.jpg)

![1a8e80c630cbdfca95bf611040d00507f5b811175fa09985e76728289bf90bb5.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/1a8e80c630cbdfca95bf611040d00507f5b811175fa09985e76728289bf90bb5.jpg)

![31a9d0d806abd46115d992640128e9e396c9aeaeba913b43244220dbba90648b.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/31a9d0d806abd46115d992640128e9e396c9aeaeba913b43244220dbba90648b.jpg)

![4c3485cb257af797338b94befa1f4895974d442ab752cf2f608002574355f447.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/4c3485cb257af797338b94befa1f4895974d442ab752cf2f608002574355f447.jpg)

![545e2ed72d6f7db94ef22b2e70aa2dc8622f3fe7e8d6fcd90fe5005789e93948.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/545e2ed72d6f7db94ef22b2e70aa2dc8622f3fe7e8d6fcd90fe5005789e93948.jpg)

![5fcdece80f084008dbd84fc2de245c8ad1a99b872839fd273508a0d62da4ea60.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/5fcdece80f084008dbd84fc2de245c8ad1a99b872839fd273508a0d62da4ea60.jpg)

![6663b4c10aa7350362d7e33fe1f32ce86e669ada87af37d1b4e035b7a7119827.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/6663b4c10aa7350362d7e33fe1f32ce86e669ada87af37d1b4e035b7a7119827.jpg)

![7e43a98e4791b772aa39e8f9c49e21ca9c07da4ab5551f502aecc5f967ca5953.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/7e43a98e4791b772aa39e8f9c49e21ca9c07da4ab5551f502aecc5f967ca5953.jpg)

![8e48a7b37742706084eb08ab23ba78ccca591d1aaaccaf09313836f0e2bb7d4c.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/8e48a7b37742706084eb08ab23ba78ccca591d1aaaccaf09313836f0e2bb7d4c.jpg)

![9165dcdb6b3bf9ac3776d33a3263c712b36680492593c95cfd162c93993962ca.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/9165dcdb6b3bf9ac3776d33a3263c712b36680492593c95cfd162c93993962ca.jpg)

![91f37487736af0fb8265a456b6fd3201dbe25261c05f339a5bdd9444fe2d1003.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/91f37487736af0fb8265a456b6fd3201dbe25261c05f339a5bdd9444fe2d1003.jpg)

![96b18d656696659b25fd2eb4bdd984e27cc4e038a6ee868f1a2709f83b15fd24.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/96b18d656696659b25fd2eb4bdd984e27cc4e038a6ee868f1a2709f83b15fd24.jpg)

![a994f568da516bfa5a5ff45288c82e75cc0010386b9a0a4e1dfa7635d2c7185a.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/a994f568da516bfa5a5ff45288c82e75cc0010386b9a0a4e1dfa7635d2c7185a.jpg)

![b77021692d85a3a9844c29a3004b3fe98e6a34fbc732b3e481102d72b2191865.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/b77021692d85a3a9844c29a3004b3fe98e6a34fbc732b3e481102d72b2191865.jpg)

![c4fb92764235c93b4cc644e10d4b68f895f2e3da7594c639aa743b099a94d2cf.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/c4fb92764235c93b4cc644e10d4b68f895f2e3da7594c639aa743b099a94d2cf.jpg)

![c509086adc582f0ddfaa4b6e5ea278c598776145448af946095af470cce832f2.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/c509086adc582f0ddfaa4b6e5ea278c598776145448af946095af470cce832f2.jpg)

![d046f604e8ac8032b9b7a5127c28ed5fb2c4bec24c26a3cb52ad29a7a99c8d5f.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/d046f604e8ac8032b9b7a5127c28ed5fb2c4bec24c26a3cb52ad29a7a99c8d5f.jpg)

![d8ad427cdc62bc893f68624c22e2b31f6799218166ca88b6a0fa8daa85069390.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/d8ad427cdc62bc893f68624c22e2b31f6799218166ca88b6a0fa8daa85069390.jpg)

![e68c1def727538cadc2a3ee52f1236ae9a9a361cd69a35eb4616f75dec39337b.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/e68c1def727538cadc2a3ee52f1236ae9a9a361cd69a35eb4616f75dec39337b.jpg)

![f353a5c3e5a31b94aa069446a97d4f0f25b243c486cef82f4b6f636ed7279510.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/images/f353a5c3e5a31b94aa069446a97d4f0f25b243c486cef82f4b6f636ed7279510.jpg)

### Tables

![16358a2593069a32d44caa87aaeaded54ba3495ababf51ebe597613e67a1239f.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/16358a2593069a32d44caa87aaeaded54ba3495ababf51ebe597613e67a1239f.jpg)

![1b4f7bba67dc72aee59eb0429c93ec77f5fc4ff25d60ccfd5a87012d79a93e09.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/1b4f7bba67dc72aee59eb0429c93ec77f5fc4ff25d60ccfd5a87012d79a93e09.jpg)

![29c43793b36e9e97b4c64143eb8d23a6af946ceae9a102436a1c7acf600ea114.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/29c43793b36e9e97b4c64143eb8d23a6af946ceae9a102436a1c7acf600ea114.jpg)

![3467a39e9986642c88e05427a65f8ff6d92101352c4070ead28e6c1cc685ec12.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/3467a39e9986642c88e05427a65f8ff6d92101352c4070ead28e6c1cc685ec12.jpg)

![423f8dc4f9018e0c539d66d8d3278f81373ffc4b40b9c0d34d385f344e6dabcf.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/423f8dc4f9018e0c539d66d8d3278f81373ffc4b40b9c0d34d385f344e6dabcf.jpg)

![51caa84dcfb4dcee9903d1df29b2a509fe14f3f43b189479c47af8493910954f.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/51caa84dcfb4dcee9903d1df29b2a509fe14f3f43b189479c47af8493910954f.jpg)

![560d1e13f7e3ec1d6d786384f9e60fad6511bb6e686132f9743d019fb821acea.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/560d1e13f7e3ec1d6d786384f9e60fad6511bb6e686132f9743d019fb821acea.jpg)

![6cf596da552709d94e35361ac7bf277f3b66670fe1e921b1cf7b6bf82cabfb8f.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/6cf596da552709d94e35361ac7bf277f3b66670fe1e921b1cf7b6bf82cabfb8f.jpg)

![8a176d557242e13938090c7f436845120d20c9d3334985e815166c3b1c71a412.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/8a176d557242e13938090c7f436845120d20c9d3334985e815166c3b1c71a412.jpg)

![9fa7bee3825f2a66119e5e8bd290b148b79f4807f00da472a9a3c7c4a274b204.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/9fa7bee3825f2a66119e5e8bd290b148b79f4807f00da472a9a3c7c4a274b204.jpg)

![bd253c0c797bd2b8749b832af3ccc6ad0fe5b4c1fb0be6fe95c05f0687703ce3.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/bd253c0c797bd2b8749b832af3ccc6ad0fe5b4c1fb0be6fe95c05f0687703ce3.jpg)

![eeeb60a51551ce2ebb874cd8d641aa25ad6e60eed3ba93df29bfeef8fcea9d9a.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/eeeb60a51551ce2ebb874cd8d641aa25ad6e60eed3ba93df29bfeef8fcea9d9a.jpg)

![f0f17813669cf08b71aac494c69de468c3417a2cc6e545478cbd85a733bfc519.jpg](../iclr_results/1537_OvercookedV2_%20Rethinking%20Overcooked%20for%20Zero-Shot%20Coordination/tables/f0f17813669cf08b71aac494c69de468c3417a2cc6e545478cbd85a733bfc519.jpg)

## Adversarial Search Engine Optimization for Large Language Models


### Images

![0ca8682e686b6d8e6ef25f7f9ea2bb12032aedcf3f5a6164600aa998b63521c2.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/0ca8682e686b6d8e6ef25f7f9ea2bb12032aedcf3f5a6164600aa998b63521c2.jpg)

![24471de05746e2dd33066e81258e54464be4faea08cf108be98161ec94655dce.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/24471de05746e2dd33066e81258e54464be4faea08cf108be98161ec94655dce.jpg)

![28b6552dee3f634b765aad1e4c2f167f2d7bd2b88231768c9889decee44b90c6.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/28b6552dee3f634b765aad1e4c2f167f2d7bd2b88231768c9889decee44b90c6.jpg)

![2a0ca645f8f664618c4243e570b131b425a2f25f2e0943827274aa36ee821b54.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/2a0ca645f8f664618c4243e570b131b425a2f25f2e0943827274aa36ee821b54.jpg)

![41aaff364324354be6ff829db57b6f0031e8b4ff3aaf1beef2906b7e04cf639e.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/41aaff364324354be6ff829db57b6f0031e8b4ff3aaf1beef2906b7e04cf639e.jpg)

![51be0b4946885ae403e14dc7d6973c17d9785c32a9fdbcda80459ed06af24fe2.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/51be0b4946885ae403e14dc7d6973c17d9785c32a9fdbcda80459ed06af24fe2.jpg)

![53b8fd597f9ec3ad0a66b6b27e5fe94afb407123f5276f544efa35b3e79ad44b.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/53b8fd597f9ec3ad0a66b6b27e5fe94afb407123f5276f544efa35b3e79ad44b.jpg)

![54b2aa3ba2abc7fda4e2c4413e593c9739df75dc25f0e9719aa6a328ad146f3b.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/54b2aa3ba2abc7fda4e2c4413e593c9739df75dc25f0e9719aa6a328ad146f3b.jpg)

![5bd20f2c6db5c07e25972cff96eb2674f6257a0ef8c705483c1cfd731ed51ea6.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/5bd20f2c6db5c07e25972cff96eb2674f6257a0ef8c705483c1cfd731ed51ea6.jpg)

![675b78ab7bbdefec428c66912ad7f6f9f6fe99d0e2f6cf5b6b7f38c857c13e3a.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/675b78ab7bbdefec428c66912ad7f6f9f6fe99d0e2f6cf5b6b7f38c857c13e3a.jpg)

![8347b33a1c317713ad7b8efa1d80f6b4617c15a97183717b214db5373adb6de8.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/8347b33a1c317713ad7b8efa1d80f6b4617c15a97183717b214db5373adb6de8.jpg)

![85e6aaa19038c7a0f7e77bdd2ab42502af80f6da178ac926fc9cc6ac3b924fbc.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/85e6aaa19038c7a0f7e77bdd2ab42502af80f6da178ac926fc9cc6ac3b924fbc.jpg)

![866a7a12d2e674f198dd70c6426fb2bf82f71562659cd56be64b4eebf175cb57.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/866a7a12d2e674f198dd70c6426fb2bf82f71562659cd56be64b4eebf175cb57.jpg)

![8953dc091801f0f00aa2168c46381cda97c0c946a6e94620bb949c726ed5b7ff.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/8953dc091801f0f00aa2168c46381cda97c0c946a6e94620bb949c726ed5b7ff.jpg)

![9008b9c389c2bf638d2d5e58a4bf5b10a1d5bc9d2384753d963e680b01d31b43.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/9008b9c389c2bf638d2d5e58a4bf5b10a1d5bc9d2384753d963e680b01d31b43.jpg)

![9423ab748272c33e43baed582b91ffe74c212b89c79c673297b11d2ec9b58618.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/9423ab748272c33e43baed582b91ffe74c212b89c79c673297b11d2ec9b58618.jpg)

![a42919c922b1a9055d0b0b34269b48e5fc7555a2e1ff547ae77370838c587177.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/a42919c922b1a9055d0b0b34269b48e5fc7555a2e1ff547ae77370838c587177.jpg)

![aae92f73323fb73e9f7bd1b581ef50acaacdb6ded9d3860cd3e0294bb9523edf.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/aae92f73323fb73e9f7bd1b581ef50acaacdb6ded9d3860cd3e0294bb9523edf.jpg)

![b5b69c8eca20f1c2ea14dd4fdebba49e019daecffecf75c384ad42cc185c2b10.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/b5b69c8eca20f1c2ea14dd4fdebba49e019daecffecf75c384ad42cc185c2b10.jpg)

![b78dae5087788c6082a1c218b97f6c1faf69eb3adb7c506cc1d6a959f7478aa5.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/b78dae5087788c6082a1c218b97f6c1faf69eb3adb7c506cc1d6a959f7478aa5.jpg)

![bae81f5a18996467e75e385fdedd62c5188213de1f4e28e56f34bbfb374a49c6.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/bae81f5a18996467e75e385fdedd62c5188213de1f4e28e56f34bbfb374a49c6.jpg)

![cb2de11f1d2a92915488aefaa2fa71264ee5e7deaf5d05c354bf8c32ebe8c204.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/cb2de11f1d2a92915488aefaa2fa71264ee5e7deaf5d05c354bf8c32ebe8c204.jpg)

![d293910e835b012fc341cb59c910208176c6f7d3573700e3cedf9dc6a341c5fe.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/d293910e835b012fc341cb59c910208176c6f7d3573700e3cedf9dc6a341c5fe.jpg)

![da8de550027add7b57d22b95bfb3101e8608c1ec3613fadf821dce73b3c8fd62.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/da8de550027add7b57d22b95bfb3101e8608c1ec3613fadf821dce73b3c8fd62.jpg)

![dd3a184caf26f9eab5283ab51d0d5f8ca92a2ea2446182444583202e260fbd91.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/dd3a184caf26f9eab5283ab51d0d5f8ca92a2ea2446182444583202e260fbd91.jpg)

![e07f2cac73b9667d2e3cf5047191362d668908c0ddd68d675edce22c050b2c07.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/e07f2cac73b9667d2e3cf5047191362d668908c0ddd68d675edce22c050b2c07.jpg)

![e33ada0e1032f95f79fb5e2c282a7f87ad2125adc5345f9cbd95d62b463950b3.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/e33ada0e1032f95f79fb5e2c282a7f87ad2125adc5345f9cbd95d62b463950b3.jpg)

![f0594ae834f0891ab887b31460d366b342b5bcc0347c924d762de97000a85c74.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/f0594ae834f0891ab887b31460d366b342b5bcc0347c924d762de97000a85c74.jpg)

![f41d8ed5118bdbd48cb7d8d5ce78d33b83393b04c6252e480cc566ff023db498.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/f41d8ed5118bdbd48cb7d8d5ce78d33b83393b04c6252e480cc566ff023db498.jpg)

![f84813c53bea5172c5d87cf5735e407c7b89a442c35d309398879a2e946c249e.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/images/f84813c53bea5172c5d87cf5735e407c7b89a442c35d309398879a2e946c249e.jpg)

### Tables

![c86f3f71f7c92c16c4669e05086cf381f216a8fcc0f364c91e50c4c81b235c2e.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/tables/c86f3f71f7c92c16c4669e05086cf381f216a8fcc0f364c91e50c4c81b235c2e.jpg)

![c978dc801b60327b3d3a972f77d28d306530d1cb1c5f7796c331e7c032aa3440.jpg](../iclr_results/1538_Adversarial%20Search%20Engine%20Optimization%20for%20Large%20Language%20Models/tables/c978dc801b60327b3d3a972f77d28d306530d1cb1c5f7796c331e7c032aa3440.jpg)

## Causal Representation Learning from Multimodal Biomedical Observations


### Images

![0cf1b52ee5d3732a3bd988e227090c80964842df2bc5e89c6974837aee525c80.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/0cf1b52ee5d3732a3bd988e227090c80964842df2bc5e89c6974837aee525c80.jpg)

![30f5fff7800d582443cee3f095901ef8fcf970354b8b7b3a2fb3c37f4c30956c.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/30f5fff7800d582443cee3f095901ef8fcf970354b8b7b3a2fb3c37f4c30956c.jpg)

![352239bc9faadad7307a8b6d1ac0e08d64a6533db6efe587d9fc7aebd2c18146.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/352239bc9faadad7307a8b6d1ac0e08d64a6533db6efe587d9fc7aebd2c18146.jpg)

![4f3acd3409d9865588dc6c6eb91accb1413e76b3b365c9da335b3ba3d9c28c00.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/4f3acd3409d9865588dc6c6eb91accb1413e76b3b365c9da335b3ba3d9c28c00.jpg)

![56780cfe6332077e7350f081c4853e4f55ac6ccae692a3ad6272e0f9c5b47e12.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/56780cfe6332077e7350f081c4853e4f55ac6ccae692a3ad6272e0f9c5b47e12.jpg)

![7288ad48099104a13f24e90f4e7f5c1c35eeda8510cd449e597260429c42b229.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/7288ad48099104a13f24e90f4e7f5c1c35eeda8510cd449e597260429c42b229.jpg)

![91cf2cfd91f685c843b526723c57d438b4c3090866a291313ceba3b65e889a54.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/91cf2cfd91f685c843b526723c57d438b4c3090866a291313ceba3b65e889a54.jpg)

![c1da25b8d0b00dc9f134a4373e0b7e43b2ff6fdea765c2fb1b08f0a089b6b5a9.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/images/c1da25b8d0b00dc9f134a4373e0b7e43b2ff6fdea765c2fb1b08f0a089b6b5a9.jpg)

### Tables

![5a5f48b9a6e58da4a2fda141b700e67c9f4fb759eb9da8e8f7c7197aae512bb7.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/tables/5a5f48b9a6e58da4a2fda141b700e67c9f4fb759eb9da8e8f7c7197aae512bb7.jpg)

![906a942de037df0debb465de88e5bdd4b9e9aa0244de4960e292000e03d2fbec.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/tables/906a942de037df0debb465de88e5bdd4b9e9aa0244de4960e292000e03d2fbec.jpg)

![92571b9a6456312c2be1c28ed7cba3c63354ad80833c0b7d8ce2ee035a640b98.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/tables/92571b9a6456312c2be1c28ed7cba3c63354ad80833c0b7d8ce2ee035a640b98.jpg)

![c267a07686c2b838edb3908d26ede973a2fbae13e2af0a5ad91360fca7038868.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/tables/c267a07686c2b838edb3908d26ede973a2fbae13e2af0a5ad91360fca7038868.jpg)

![fca6ce84b123ea8604989485d22342877eeb52a5297d49b962b56b82d6b99017.jpg](../iclr_results/1539_Causal%20Representation%20Learning%20from%20Multimodal%20Biomedical%20Observations/tables/fca6ce84b123ea8604989485d22342877eeb52a5297d49b962b56b82d6b99017.jpg)

## Towards Robust Multimodal Open-set Test-time Adaptation via Adaptive Entropy-aware Optimization


### Images

![2973547133d6720719162949e9c8165a9645bcccf151cffc14e19657ff3cfee1.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/2973547133d6720719162949e9c8165a9645bcccf151cffc14e19657ff3cfee1.jpg)

![46f17d765caeed894bad375ca1f31c257e936344f7eaecf435a2081887921b60.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/46f17d765caeed894bad375ca1f31c257e936344f7eaecf435a2081887921b60.jpg)

![4e7f5157af5277b1d8eec496d3d1c53d5259e0663034e28c20589119dad9e083.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/4e7f5157af5277b1d8eec496d3d1c53d5259e0663034e28c20589119dad9e083.jpg)

![5b3c5da2a9e0f848bb4a5a7a0d367ab7b40e843bbdc1f7ef6a0340209a719942.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/5b3c5da2a9e0f848bb4a5a7a0d367ab7b40e843bbdc1f7ef6a0340209a719942.jpg)

![6fff965fea37d25706f56e4f9f9399c7efc6bd547d3c37f0a5f6d506b0811fb7.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/6fff965fea37d25706f56e4f9f9399c7efc6bd547d3c37f0a5f6d506b0811fb7.jpg)

![7c7359563910219c688512aad992428839cc0d9a29279caf53307ad1f356997e.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/7c7359563910219c688512aad992428839cc0d9a29279caf53307ad1f356997e.jpg)

![7dd6c15e873a41e0bf9c72fbd1db6b684d943702cdd34cd50075b80b909775f3.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/7dd6c15e873a41e0bf9c72fbd1db6b684d943702cdd34cd50075b80b909775f3.jpg)

![92f411ab4f8e19d5a2d455acbee8f5b012b0a81158c504c8204b324ebb2ef76f.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/92f411ab4f8e19d5a2d455acbee8f5b012b0a81158c504c8204b324ebb2ef76f.jpg)

![9cc4b742d331b21c72aa074f48249e33f1f3f8cba31355d12d7cfadb6ab98fb1.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/9cc4b742d331b21c72aa074f48249e33f1f3f8cba31355d12d7cfadb6ab98fb1.jpg)

![bd1c4fd687cc7f1daa1e736a8e34ae909d49e55093d7e5bd9dbad956e57b2aaf.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/bd1c4fd687cc7f1daa1e736a8e34ae909d49e55093d7e5bd9dbad956e57b2aaf.jpg)

![bf2dec7a9764792f8b636cc28a11725dcfcd9f84a07ceeaaa23df520eeec23da.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/bf2dec7a9764792f8b636cc28a11725dcfcd9f84a07ceeaaa23df520eeec23da.jpg)

![c5a3a7d707d97628b4a147dacb87fa5300d043f0da7765d19c55b82adb4cc0b8.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/c5a3a7d707d97628b4a147dacb87fa5300d043f0da7765d19c55b82adb4cc0b8.jpg)

![d5404ccbb91ea90a1315160b31c75322795df963e5efe66df26da3cabdf81fda.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/d5404ccbb91ea90a1315160b31c75322795df963e5efe66df26da3cabdf81fda.jpg)

![d95cd5c43bb467bd51ce29baf8c4c70f30c1555f0f9e6d7c9a3707c67ee295fb.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/d95cd5c43bb467bd51ce29baf8c4c70f30c1555f0f9e6d7c9a3707c67ee295fb.jpg)

![ebe2a9d1eac44a88ce97d21459189e34de333547a8535d6d9fbc2ec6ee9f69d3.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/images/ebe2a9d1eac44a88ce97d21459189e34de333547a8535d6d9fbc2ec6ee9f69d3.jpg)

### Tables

![02cf80cbb9086d0e281a153762ec18a5613fcccbbc87688936a2d7a8a7c0fb98.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/02cf80cbb9086d0e281a153762ec18a5613fcccbbc87688936a2d7a8a7c0fb98.jpg)

![05e2c09a08d24b9e50deeebbae8da2f0a88061f51bdf9ef61d47d37faf664d8d.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/05e2c09a08d24b9e50deeebbae8da2f0a88061f51bdf9ef61d47d37faf664d8d.jpg)

![07943b27bc9d51bcd12edf0e3c1bd56b178d42314e2ab16e5917ebf638840a80.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/07943b27bc9d51bcd12edf0e3c1bd56b178d42314e2ab16e5917ebf638840a80.jpg)

![2028f361267e4523ffa19c78c93b6aaf28a341af3d01c9bb442761a5e8321274.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/2028f361267e4523ffa19c78c93b6aaf28a341af3d01c9bb442761a5e8321274.jpg)

![2e41681a544c3dd6d23840069adb7f0e93cc2bca39d12382a9f034cce4c0243c.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/2e41681a544c3dd6d23840069adb7f0e93cc2bca39d12382a9f034cce4c0243c.jpg)

![3ff5b8a968f2d3ca3a8134176c183b6cf7cdbe3262bcb38e480e0c46c9b63596.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/3ff5b8a968f2d3ca3a8134176c183b6cf7cdbe3262bcb38e480e0c46c9b63596.jpg)

![435d3a6a85c081fc222d542ab4c18283477ec60ffe556cfa4e4e8e4928264ed7.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/435d3a6a85c081fc222d542ab4c18283477ec60ffe556cfa4e4e8e4928264ed7.jpg)

![46a1e5740d2f01940f7c8e19e27fdb4472b8eb3edbab639226ca3b87d7ca7331.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/46a1e5740d2f01940f7c8e19e27fdb4472b8eb3edbab639226ca3b87d7ca7331.jpg)

![64cc459735c7ebff3875612cd051ea2539bc73241e747769f5894a8e2b80c97d.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/64cc459735c7ebff3875612cd051ea2539bc73241e747769f5894a8e2b80c97d.jpg)

![6be4912e922d2de9128f9238b0fc47e59f8d4265996d2ebc9187432d8fda0685.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/6be4912e922d2de9128f9238b0fc47e59f8d4265996d2ebc9187432d8fda0685.jpg)

![7cbc412e2c34180368f1b2be9b5f0f6cc20985372198288c7b5916235929221d.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/7cbc412e2c34180368f1b2be9b5f0f6cc20985372198288c7b5916235929221d.jpg)

![7fe27e29dc07a3e4937562e265dd587ee4c8a759e0f758e29b12f8639c66f6d3.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/7fe27e29dc07a3e4937562e265dd587ee4c8a759e0f758e29b12f8639c66f6d3.jpg)

![8027dee8f1063517e11aaf6c6e38db5f59b3941bcb24e61db31da47bf8727b66.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/8027dee8f1063517e11aaf6c6e38db5f59b3941bcb24e61db31da47bf8727b66.jpg)

![8a4d40e60fadf3b443d26fd9ec81e1c1e755ebc07427b28ecd9396e4bdae0fdf.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/8a4d40e60fadf3b443d26fd9ec81e1c1e755ebc07427b28ecd9396e4bdae0fdf.jpg)

![a57d2070c6c5b99c956d3d19f31ccc77a1180faaeaa6dbe0246acd648e88b0f4.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/a57d2070c6c5b99c956d3d19f31ccc77a1180faaeaa6dbe0246acd648e88b0f4.jpg)

![b4f8be251e4113ecd867aef7575b28cabd0c7808ff4063f349131998c412b402.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/b4f8be251e4113ecd867aef7575b28cabd0c7808ff4063f349131998c412b402.jpg)

![c617fa10adf6ab919dcccc5b5341e14b75ebf1a7f8cf43269e0912723840701c.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/c617fa10adf6ab919dcccc5b5341e14b75ebf1a7f8cf43269e0912723840701c.jpg)

![c791291e309128b79c0ad1d62a055832d5407c8f0c6d75a3c5bdebd9d1eb008f.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/c791291e309128b79c0ad1d62a055832d5407c8f0c6d75a3c5bdebd9d1eb008f.jpg)

![dcc574fb60c0762dc41932051402bd41daa7d5eea7ff25d0894636d0f9960f07.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/dcc574fb60c0762dc41932051402bd41daa7d5eea7ff25d0894636d0f9960f07.jpg)

![e3081cbf1892934dbfd8a1cbb408f0ca800c3cf1114f24a62c114313f1c74160.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/e3081cbf1892934dbfd8a1cbb408f0ca800c3cf1114f24a62c114313f1c74160.jpg)

![e321edea571fb80844153b9abd6ae7bf0e2c53dc07e7825dc5f0877ac907c212.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/e321edea571fb80844153b9abd6ae7bf0e2c53dc07e7825dc5f0877ac907c212.jpg)

![e5b65e13e7a2ec9138fd57b9750825975870a02c4a14fe6f223ba8b03e1f7aab.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/e5b65e13e7a2ec9138fd57b9750825975870a02c4a14fe6f223ba8b03e1f7aab.jpg)

![e6a3e644e77860e2f7c26613f028e16425658896ea823cdb3a1850f620b35244.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/e6a3e644e77860e2f7c26613f028e16425658896ea823cdb3a1850f620b35244.jpg)

![e789153cf73ed3fa9b0b46c152bd7ec1f717b6822744748e021d8ca15b7a7767.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/e789153cf73ed3fa9b0b46c152bd7ec1f717b6822744748e021d8ca15b7a7767.jpg)

![ea9a110219656f101d6006e65dca2853a6526052f7d5dea99d0e3180ef3f4495.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/ea9a110219656f101d6006e65dca2853a6526052f7d5dea99d0e3180ef3f4495.jpg)

![f26f559fbe15b5ceb33a94c180ba125d3ffa0dd3becdb3627a3a90dd836a3e92.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/f26f559fbe15b5ceb33a94c180ba125d3ffa0dd3becdb3627a3a90dd836a3e92.jpg)

![fd0302c4d47e281e53ea6aba605e56cf50491905587fa7982cc4a612afb87ace.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/fd0302c4d47e281e53ea6aba605e56cf50491905587fa7982cc4a612afb87ace.jpg)

![ff4035dbf290273514fdc3093be59ebdacebfe2367e67826fba19778de228932.jpg](../iclr_results/1540_Towards%20Robust%20Multimodal%20Open-set%20Test-time%20Adaptation%20via%20Adaptive%20Entropy-aware%20Optimization/tables/ff4035dbf290273514fdc3093be59ebdacebfe2367e67826fba19778de228932.jpg)

## Mixture Compressor for Mixture-of-Experts LLMs Gains More


### Images

![0dad975f624d3f6b3415e76327ca405255f4de873ccc04d1f1f0b9921d541d7d.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/0dad975f624d3f6b3415e76327ca405255f4de873ccc04d1f1f0b9921d541d7d.jpg)

![347d6429236ff97fd7519162620db6ff32cff26fad35f86bfedba6b585b04a0d.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/347d6429236ff97fd7519162620db6ff32cff26fad35f86bfedba6b585b04a0d.jpg)

![53a3a1b1e0ebab14e6efa402daa72fe412e2e7859a38ed96dbb243dada46c8db.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/53a3a1b1e0ebab14e6efa402daa72fe412e2e7859a38ed96dbb243dada46c8db.jpg)

![55ef1e5493671b8b618c52fc1aeacc0cbe0cfd060d0884d5a55145cec2edd150.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/55ef1e5493671b8b618c52fc1aeacc0cbe0cfd060d0884d5a55145cec2edd150.jpg)

![5a0cd211453e91bef95d0593614d47d450fe44ba6e0a3ee90ac21b91cf389faa.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/5a0cd211453e91bef95d0593614d47d450fe44ba6e0a3ee90ac21b91cf389faa.jpg)

![6e965413de5a7bb174af6924ba7205f9a1d0f10baa3d47bb12fe55eadc7b54ef.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/6e965413de5a7bb174af6924ba7205f9a1d0f10baa3d47bb12fe55eadc7b54ef.jpg)

![b7a1bacbef245f3071a2e5f5b5d764039621406d661578ae307d8c2981ee720d.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/b7a1bacbef245f3071a2e5f5b5d764039621406d661578ae307d8c2981ee720d.jpg)

![da369f162034b47163d1fdd884bf522d25f511e88040e03fdf711e0f6632041f.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/da369f162034b47163d1fdd884bf522d25f511e88040e03fdf711e0f6632041f.jpg)

![e5008ff4d8e1634d964dcedf625b3ac68bff511cac05114c8333f56b4ce980af.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/e5008ff4d8e1634d964dcedf625b3ac68bff511cac05114c8333f56b4ce980af.jpg)

![f3e83464db310dc6a71b8f1bc8bf82aa3ccfa7dda8c7412ba29580a6fbf701bf.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/images/f3e83464db310dc6a71b8f1bc8bf82aa3ccfa7dda8c7412ba29580a6fbf701bf.jpg)

### Tables

![0a2d9c82e29cc1d2a498a20fbb29ce3723a6fb071fb140c42702bbfd08061a2d.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/0a2d9c82e29cc1d2a498a20fbb29ce3723a6fb071fb140c42702bbfd08061a2d.jpg)

![3585b89bf06c10676bf604c3fc1b8f3807fdccf30e3165fcad4cbac994cd8a50.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/3585b89bf06c10676bf604c3fc1b8f3807fdccf30e3165fcad4cbac994cd8a50.jpg)

![42ec625cf2e573085b9ac27de55d64a1e03089a33cd9511f76ae69ad653dbafe.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/42ec625cf2e573085b9ac27de55d64a1e03089a33cd9511f76ae69ad653dbafe.jpg)

![48867270ae1ed46cb2389964e5c0963fcd18c66567eaa5d9a51240ababc6335b.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/48867270ae1ed46cb2389964e5c0963fcd18c66567eaa5d9a51240ababc6335b.jpg)

![4e1f5fed8bc190f140ad39d127e1de42b24201d552c70b5b73c38234eba8d0ce.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/4e1f5fed8bc190f140ad39d127e1de42b24201d552c70b5b73c38234eba8d0ce.jpg)

![850b6dcce715197e2c9d09cc62d2b0cfc448f3ae05db9fb09dcf4320132dd68b.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/850b6dcce715197e2c9d09cc62d2b0cfc448f3ae05db9fb09dcf4320132dd68b.jpg)

![b4333a8ef982284305fffbb86a9e76bd2442e8fb89811e5a74b7ca0990bd1dd2.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/b4333a8ef982284305fffbb86a9e76bd2442e8fb89811e5a74b7ca0990bd1dd2.jpg)

![bbf4e4f5892e3c933059fabbdb491d12f9492310d36a095fb21443606106e0ca.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/bbf4e4f5892e3c933059fabbdb491d12f9492310d36a095fb21443606106e0ca.jpg)

![be15ba2ed77940b93269bac44c392df5393fafb0c76867f6bd220ef84a575a52.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/be15ba2ed77940b93269bac44c392df5393fafb0c76867f6bd220ef84a575a52.jpg)

![c6dcce5662fc9c139316b5b8f42e26e0720584a8ecf9d3fdd28137ff4adf6e90.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/c6dcce5662fc9c139316b5b8f42e26e0720584a8ecf9d3fdd28137ff4adf6e90.jpg)

![f264ab9536a1462796e086573d28eaa1d92aa8049bf9093bd95fffcaf5d09ee8.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/f264ab9536a1462796e086573d28eaa1d92aa8049bf9093bd95fffcaf5d09ee8.jpg)

![f9f1821bb527f8cf79e0e4a110ab9ae567fec68048cee4efa202d286273f8469.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/f9f1821bb527f8cf79e0e4a110ab9ae567fec68048cee4efa202d286273f8469.jpg)

![fc325432a830c684469f6d845c81244d131b40dfe1b92b5f82f0fa9c41e1dcbc.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/fc325432a830c684469f6d845c81244d131b40dfe1b92b5f82f0fa9c41e1dcbc.jpg)

![fecde55efb2ac974afcffc443196cad67215dc8ae920829b6adf1a4db4efd601.jpg](../iclr_results/1541_Mixture%20Compressor%20for%20Mixture-of-Experts%20LLMs%20Gains%20More/tables/fecde55efb2ac974afcffc443196cad67215dc8ae920829b6adf1a4db4efd601.jpg)

## Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction


### Images

![089038e1235e9ae41f8e54634f74ad66f3a96fa6c642fcfc04b1bc8aa0ea8f40.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/089038e1235e9ae41f8e54634f74ad66f3a96fa6c642fcfc04b1bc8aa0ea8f40.jpg)

![09ac3207bfd70a564005af80e28f3e36c9faaf1b78b265d3932c3056fe8f2a86.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/09ac3207bfd70a564005af80e28f3e36c9faaf1b78b265d3932c3056fe8f2a86.jpg)

![1f38ad2e5eee3515a747da6bf5a619003acf57f4cd74cbf575dc0bbed3ab0234.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/1f38ad2e5eee3515a747da6bf5a619003acf57f4cd74cbf575dc0bbed3ab0234.jpg)

![239f7160c828aa3aef71d222347c690a815ded0747075983fe46da072c8499b9.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/239f7160c828aa3aef71d222347c690a815ded0747075983fe46da072c8499b9.jpg)

![255036eb3518d97bd8f3c6f77b7a8fc7238a3420933c9860218fccca279b0448.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/255036eb3518d97bd8f3c6f77b7a8fc7238a3420933c9860218fccca279b0448.jpg)

![2fbd2d3cb337c0f3596751f3711077ea1be662e42e6ede14d1333d689fec9458.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/2fbd2d3cb337c0f3596751f3711077ea1be662e42e6ede14d1333d689fec9458.jpg)

![317e6a392a9fa188989064977ed2cff73e1463bc3730539cd14f1653933f99c6.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/317e6a392a9fa188989064977ed2cff73e1463bc3730539cd14f1653933f99c6.jpg)

![322e42ef5b4cc3400a1808ceb2dabc403267bd72118d99a90e52a158525dc227.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/322e42ef5b4cc3400a1808ceb2dabc403267bd72118d99a90e52a158525dc227.jpg)

![36f2dc8c229f86bc3c8e2fd529853b199edaa8873b6413d5adf276b46bc30542.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/36f2dc8c229f86bc3c8e2fd529853b199edaa8873b6413d5adf276b46bc30542.jpg)

![37c4ec5595504f9407d671ffc0e44dd4d2b3bb6bcf930a705beded31221cade9.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/37c4ec5595504f9407d671ffc0e44dd4d2b3bb6bcf930a705beded31221cade9.jpg)

![39ce19082d2a5232d8c7de17c9caea43bd3b5125e3405ad6906f9942efc1beb5.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/39ce19082d2a5232d8c7de17c9caea43bd3b5125e3405ad6906f9942efc1beb5.jpg)

![3e05b4aaed2d0134e77acac71e2bf93e2d6ce8340c3893a1c94c3986ebf30db6.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/3e05b4aaed2d0134e77acac71e2bf93e2d6ce8340c3893a1c94c3986ebf30db6.jpg)

![3e35145dffc20ce676afca1515e91c1da88989667902e4d35df572e561fbb43f.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/3e35145dffc20ce676afca1515e91c1da88989667902e4d35df572e561fbb43f.jpg)

![4752793a98a3bd01dbce6d79ce31c7f7a558f0f0622c26b788e8d72d7e324efe.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/4752793a98a3bd01dbce6d79ce31c7f7a558f0f0622c26b788e8d72d7e324efe.jpg)

![4b3aecd945dec0b4a1ae76368c220a3c58807a431a3ff7f809697e7c3ea3bb9c.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/4b3aecd945dec0b4a1ae76368c220a3c58807a431a3ff7f809697e7c3ea3bb9c.jpg)

![4c3ea47a4fd374e40d94ead370353da83357534f1d5aade75c1ca161cfcbc807.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/4c3ea47a4fd374e40d94ead370353da83357534f1d5aade75c1ca161cfcbc807.jpg)

![5a3faa909e473d7a8364b393ef272160e4ca8de091227a24ffd3562a6267a57e.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/5a3faa909e473d7a8364b393ef272160e4ca8de091227a24ffd3562a6267a57e.jpg)

![5c0c63becc130f2e4c5de665159f5efbd335795d5fe95b21a2f74d3d0ffae2d4.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/5c0c63becc130f2e4c5de665159f5efbd335795d5fe95b21a2f74d3d0ffae2d4.jpg)

![6100352e8c69f06feffb16c01aad9bbd1b9f13154e744440d849ababd440e1ae.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/6100352e8c69f06feffb16c01aad9bbd1b9f13154e744440d849ababd440e1ae.jpg)

![63ac908973148683dea5088689885de44fcf04edf10e2dd51c8c9a207e6689a0.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/63ac908973148683dea5088689885de44fcf04edf10e2dd51c8c9a207e6689a0.jpg)

![685a7d5fa0866c928dfe3ede48623a59d52d8ac3cf7c7952b0bed6750aec267e.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/685a7d5fa0866c928dfe3ede48623a59d52d8ac3cf7c7952b0bed6750aec267e.jpg)

![6a880b13dd71e218ca4828c623575d538cf69153600cdd41a7f6ad0e0ba3a439.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/6a880b13dd71e218ca4828c623575d538cf69153600cdd41a7f6ad0e0ba3a439.jpg)

![6f8fa0217055ab1144071dfb9f18042a4ceb5ec307b6236616045832e64e5041.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/6f8fa0217055ab1144071dfb9f18042a4ceb5ec307b6236616045832e64e5041.jpg)

![75d10eeadcc49bbf8449236064dd8d26923feb9db09751038021a6708870520f.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/75d10eeadcc49bbf8449236064dd8d26923feb9db09751038021a6708870520f.jpg)

![95bc96fb939a28302e68ace8d09639eb6fff05d6d0a336ec01854d2bf871c5fc.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/95bc96fb939a28302e68ace8d09639eb6fff05d6d0a336ec01854d2bf871c5fc.jpg)

![975619469120a609d0339d3cfe8dc3aa036c545954a6318ce0059aa27070bca4.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/975619469120a609d0339d3cfe8dc3aa036c545954a6318ce0059aa27070bca4.jpg)

![a36ef55b17c370e4cc2952d3f8170e64ec24dd00fdc0367c3dee9cd6ea4c8ef0.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/a36ef55b17c370e4cc2952d3f8170e64ec24dd00fdc0367c3dee9cd6ea4c8ef0.jpg)

![a726a06ef81f35fdab3b22173a0abc0ecb49ad11b30cfa0c443cdb1ab4177b62.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/a726a06ef81f35fdab3b22173a0abc0ecb49ad11b30cfa0c443cdb1ab4177b62.jpg)

![a750000646d979cbd6ccc3b80f7f07f7a616f8b3509089bfb9423feb30e80b57.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/a750000646d979cbd6ccc3b80f7f07f7a616f8b3509089bfb9423feb30e80b57.jpg)

![a9416a152ad7b8966d7494d955166122ba02e01b146200340e8bfa77f7e4fb33.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/a9416a152ad7b8966d7494d955166122ba02e01b146200340e8bfa77f7e4fb33.jpg)

![b7dbebe4b0fc78c55cefe4c812034b25801590b25efa18c8a06664f54bd97571.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/b7dbebe4b0fc78c55cefe4c812034b25801590b25efa18c8a06664f54bd97571.jpg)

![b97c6fa7639e3e95af85486036f2871f01cef2891a5f091b8bcb41dda0c63f6b.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/b97c6fa7639e3e95af85486036f2871f01cef2891a5f091b8bcb41dda0c63f6b.jpg)

![be1302746cc67ffd83a2c9ceb88223ee1356e45b90518accc84c5201a32f030f.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/be1302746cc67ffd83a2c9ceb88223ee1356e45b90518accc84c5201a32f030f.jpg)

![be69b1ff5fd5b5064cc6d746cac83f93101ea6ec896828733338c6b22e6b8114.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/be69b1ff5fd5b5064cc6d746cac83f93101ea6ec896828733338c6b22e6b8114.jpg)

![c494747bbe3d7abdc684755addefb166281ff6e18f4d78810106fe35987e47c2.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/c494747bbe3d7abdc684755addefb166281ff6e18f4d78810106fe35987e47c2.jpg)

![cb549a0726da7f95c51bfa7d931e4465278a232cbebf21a727c5aed2c367dd30.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/cb549a0726da7f95c51bfa7d931e4465278a232cbebf21a727c5aed2c367dd30.jpg)

![cb6727149d44e1259440a42ce082635f1054e8bf2b2fadb7c244d303e849e8a6.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/cb6727149d44e1259440a42ce082635f1054e8bf2b2fadb7c244d303e849e8a6.jpg)

![ced37049e706e2b14368cd40dbb9d092f4c73cef4ad99d13c6ac054a7a4534b5.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/ced37049e706e2b14368cd40dbb9d092f4c73cef4ad99d13c6ac054a7a4534b5.jpg)

![e64dc88904f2c5e601ac78993bd0f5cc9dec4c69f1a5a21f92f6878bc5a074c0.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/e64dc88904f2c5e601ac78993bd0f5cc9dec4c69f1a5a21f92f6878bc5a074c0.jpg)

![ee938bf2a368b18f834ffebf2644e0cff5b6b211ff0d1f9887a97a8a2dc9a3b4.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/ee938bf2a368b18f834ffebf2644e0cff5b6b211ff0d1f9887a97a8a2dc9a3b4.jpg)

![f241b8918f68270eae5af1f9afb8f50c5b95bba5c765cef1fba37c869f8bc54a.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/f241b8918f68270eae5af1f9afb8f50c5b95bba5c765cef1fba37c869f8bc54a.jpg)

![faddba125fdc6ab82a33742f55196c7f02c01bebaf7f0775018831ff156e0cd4.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/images/faddba125fdc6ab82a33742f55196c7f02c01bebaf7f0775018831ff156e0cd4.jpg)

### Tables

![26fd72929644ff9db60529f361457404b34da30adbb9ec4ab0d484d5df26e6a4.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/26fd72929644ff9db60529f361457404b34da30adbb9ec4ab0d484d5df26e6a4.jpg)

![3f23a118343bac7523c04fd0bf07708ec46fdd70420ee6cf862b39b3f3e92df1.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/3f23a118343bac7523c04fd0bf07708ec46fdd70420ee6cf862b39b3f3e92df1.jpg)

![45a45246415e107bf9d7177d65260cda112756c1752fff577cefb5632775374d.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/45a45246415e107bf9d7177d65260cda112756c1752fff577cefb5632775374d.jpg)

![4bed27ae30801caf69ced2bc211cb2bfa0c6f1b2a2a35465e5c73775c945264e.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/4bed27ae30801caf69ced2bc211cb2bfa0c6f1b2a2a35465e5c73775c945264e.jpg)

![6c13d93f34ddb1928da8eeb9d8196aa7f2f8729016af67a4d0af23c7b45d91cc.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/6c13d93f34ddb1928da8eeb9d8196aa7f2f8729016af67a4d0af23c7b45d91cc.jpg)

![73252a7a2515902943a0f6d2ee44e180c33216b46ef8f226dd2e04fc205ae12e.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/73252a7a2515902943a0f6d2ee44e180c33216b46ef8f226dd2e04fc205ae12e.jpg)

![b3145843aecffb1650073a667e406dbaae03d4ba696572002105b4206d6cda17.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/b3145843aecffb1650073a667e406dbaae03d4ba696572002105b4206d6cda17.jpg)

![cec1ebf22fc762dacf9fba3d3d1b3a3fe0613cf58aff6163e44a20f7bf594c69.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/cec1ebf22fc762dacf9fba3d3d1b3a3fe0613cf58aff6163e44a20f7bf594c69.jpg)

![ddb664aa32b0ca1631953f152ef7fd541eac2da2559258447b6909ba1cc3d608.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/ddb664aa32b0ca1631953f152ef7fd541eac2da2559258447b6909ba1cc3d608.jpg)

![e7e50453c45dd135695488c9c5a6a3bc9bcb2577ab38ebbf584ae585891262bc.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/e7e50453c45dd135695488c9c5a6a3bc9bcb2577ab38ebbf584ae585891262bc.jpg)

![fd378aa0026bd34f9914ae47afbc601ba8fab7c02a592d37a98645b33c3e8877.jpg](../iclr_results/1542_Efficient%20Exploration%20and%20Discriminative%20World%20Model%20Learning%20with%20an%20Object-Centric%20Abstraction/tables/fd378aa0026bd34f9914ae47afbc601ba8fab7c02a592d37a98645b33c3e8877.jpg)

## Watch Less, Do More: Implicit Skill Discovery for Video-Conditioned Policy


### Images

![bc4115a66b82aaf89679c6b62895a87ad21951b850030d798378ef1855f6a62f.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/images/bc4115a66b82aaf89679c6b62895a87ad21951b850030d798378ef1855f6a62f.jpg)

![c288658d64e90ee02d52e65af91f04f0751cf26ab013f6b36606f564afa439b1.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/images/c288658d64e90ee02d52e65af91f04f0751cf26ab013f6b36606f564afa439b1.jpg)

![feb99cdeca92152a5068fc5c609bc985c3663a38fb8a05ca2878689fe178a4ae.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/images/feb99cdeca92152a5068fc5c609bc985c3663a38fb8a05ca2878689fe178a4ae.jpg)

### Tables

![08be1ea5cd8121f36c0121aac9aaa26ff57f3e48533bcd1c9781760570bb701a.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/08be1ea5cd8121f36c0121aac9aaa26ff57f3e48533bcd1c9781760570bb701a.jpg)

![1dda2fc4cec3f87d2042f7b6e0eb107ddcdaf331ec348dcf17a424f6215eb041.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/1dda2fc4cec3f87d2042f7b6e0eb107ddcdaf331ec348dcf17a424f6215eb041.jpg)

![3687eb6c513216088baa98a92422c0cd05a02563b938e765b154990a7e24a28a.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/3687eb6c513216088baa98a92422c0cd05a02563b938e765b154990a7e24a28a.jpg)

![5a4dd7317ecbf8d635ba75eda77eacd060a75c9be82682bac4f0185be6c851d4.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/5a4dd7317ecbf8d635ba75eda77eacd060a75c9be82682bac4f0185be6c851d4.jpg)

![72a43946e31e16fdde82631604a134f6a9a55994ae99c9a27e5278152e5c3cf9.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/72a43946e31e16fdde82631604a134f6a9a55994ae99c9a27e5278152e5c3cf9.jpg)

![77744261acd2ea888a1b9d5946877040a81893a8e78c0fc805607ea6c6ef69f7.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/77744261acd2ea888a1b9d5946877040a81893a8e78c0fc805607ea6c6ef69f7.jpg)

![9a36850c2d83623b091d9b13331cbe515d24bae3425854ed9b64d6af06210406.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/9a36850c2d83623b091d9b13331cbe515d24bae3425854ed9b64d6af06210406.jpg)

![ed09cf2f0394fa7d9d822debf5687350ae2e108213961b64cd8792ae975a5294.jpg](../iclr_results/1543_Watch%20Less%2C%20Do%20More_%20Implicit%20Skill%20Discovery%20for%20Video-Conditioned%20Policy/tables/ed09cf2f0394fa7d9d822debf5687350ae2e108213961b64cd8792ae975a5294.jpg)

## SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation


### Images

![001755f57893149a343bfb5b0949cf05c8234585990ff363d869bdad76890aad.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/001755f57893149a343bfb5b0949cf05c8234585990ff363d869bdad76890aad.jpg)

![014bde5b23c6bbd23d41ff126cd7ce64a6476000d54bb346a98f9a79494ef986.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/014bde5b23c6bbd23d41ff126cd7ce64a6476000d54bb346a98f9a79494ef986.jpg)

![0a0ff2388f2c1206fd755873663abe54b151eff966370655925543b78ad63d46.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/0a0ff2388f2c1206fd755873663abe54b151eff966370655925543b78ad63d46.jpg)

![268d470eb883e288d581612c5226552bed8914848fa45912a3faede89f7c88fe.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/268d470eb883e288d581612c5226552bed8914848fa45912a3faede89f7c88fe.jpg)

![38bfa1154dc5e57fd498451c8729a58f6029230c04f664ce20bb5df7bee34099.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/38bfa1154dc5e57fd498451c8729a58f6029230c04f664ce20bb5df7bee34099.jpg)

![3ec99fdadb3bb01d79fc1ce9b5478dd003a78495e1c37ce18617083df50ffe3a.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/3ec99fdadb3bb01d79fc1ce9b5478dd003a78495e1c37ce18617083df50ffe3a.jpg)

![620f267cd91811a61d94414c907a998011e19bb06f62408631bdbab80abc8a41.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/620f267cd91811a61d94414c907a998011e19bb06f62408631bdbab80abc8a41.jpg)

![68d53ab74556d4d7512ad7c9858759b81686c5c96352a93bb9bf27df8ed09c16.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/68d53ab74556d4d7512ad7c9858759b81686c5c96352a93bb9bf27df8ed09c16.jpg)

![74e97b15c2b3a334303e05790554f44328a8c9753769e243ee68fa627529f46e.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/74e97b15c2b3a334303e05790554f44328a8c9753769e243ee68fa627529f46e.jpg)

![7bc232aea311f782b62e83bb732866b4c563c2359057ade6583990589352872e.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/7bc232aea311f782b62e83bb732866b4c563c2359057ade6583990589352872e.jpg)

![819b34b9331fd69354119e6f659abd80f03179f875c0a30b6a60fe412f66186e.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/819b34b9331fd69354119e6f659abd80f03179f875c0a30b6a60fe412f66186e.jpg)

![8467b7ef1783dfbae8cad8f8aed8f2469983b00d8c091939fde43e615d644603.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/8467b7ef1783dfbae8cad8f8aed8f2469983b00d8c091939fde43e615d644603.jpg)

![89950b240d771d57d4ac6d80cf63e1568613c1a598b84b34eddb4c666c720b55.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/89950b240d771d57d4ac6d80cf63e1568613c1a598b84b34eddb4c666c720b55.jpg)

![9c11e508638ab460b4e72c2ee52a64ce0ca61274fcab440439de0e2dc77d8777.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/9c11e508638ab460b4e72c2ee52a64ce0ca61274fcab440439de0e2dc77d8777.jpg)

![a03a6d903ae75c0e6c69de6882307160c99c38a82a12c41c7f43e4e0a42c1e9f.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/a03a6d903ae75c0e6c69de6882307160c99c38a82a12c41c7f43e4e0a42c1e9f.jpg)

![cf9b112dd62eff6c2f4a25359c457b9fb61e362b9da35dad7db601dd8ba6ad31.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/cf9b112dd62eff6c2f4a25359c457b9fb61e362b9da35dad7db601dd8ba6ad31.jpg)

![d25270d28cc2bedb4cfff8333493ab65a870532bbaba75f99138aa4836142c2d.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/d25270d28cc2bedb4cfff8333493ab65a870532bbaba75f99138aa4836142c2d.jpg)

![d31a9ae8fd5e55976730a3b4d0d016be2c61557f06f0e611a3827c31ce6c03d6.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/d31a9ae8fd5e55976730a3b4d0d016be2c61557f06f0e611a3827c31ce6c03d6.jpg)

![db9e21c07ca86cb5757a0fab8b749e93fc8d2ca8525f41d4290211df1e067ffd.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/db9e21c07ca86cb5757a0fab8b749e93fc8d2ca8525f41d4290211df1e067ffd.jpg)

![e609179385e3ce51d86826d0fc90358cf524462509424f67ded92c42785460f8.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/images/e609179385e3ce51d86826d0fc90358cf524462509424f67ded92c42785460f8.jpg)

### Tables

![08dfd94a42b7b9ff7bf70c76cf4fc84063bb8aa5d6eeb5792360bf98b8d95741.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/tables/08dfd94a42b7b9ff7bf70c76cf4fc84063bb8aa5d6eeb5792360bf98b8d95741.jpg)

![0d49891cfee3fda633696089b1fbce631f84a66e129f6e4b75a983b3acb695d3.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/tables/0d49891cfee3fda633696089b1fbce631f84a66e129f6e4b75a983b3acb695d3.jpg)

![392688378156d6bfc583af12a715b23ffcf6383d5798e9b654fbf226838d8553.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/tables/392688378156d6bfc583af12a715b23ffcf6383d5798e9b654fbf226838d8553.jpg)

![497fc4c857787757f4726922f01881efe172d1c6b3e27d046284cac7244c9f6b.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/tables/497fc4c857787757f4726922f01881efe172d1c6b3e27d046284cac7244c9f6b.jpg)

![bc29066a08151efd6de7b02696121f92ad06b4fd54f111b6a0a619ea5c5d2aad.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/tables/bc29066a08151efd6de7b02696121f92ad06b4fd54f111b6a0a619ea5c5d2aad.jpg)

![c472f5a2fe63de6c49d907dcdbf0489e484abefaea2724ae692f9b8c9c02a4f0.jpg](../iclr_results/1544_SAFREE_%20Training-Free%20and%20Adaptive%20Guard%20for%20Safe%20Text-to-Image%20And%20Video%20Generation/tables/c472f5a2fe63de6c49d907dcdbf0489e484abefaea2724ae692f9b8c9c02a4f0.jpg)

## Causal Graphical Models for Vision-Language Compositional Understanding


### Images

![0f04ca73aef134f93bf321d1e40ff1a7eeb305ce2247472c66fb5dfda8e2f34d.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/0f04ca73aef134f93bf321d1e40ff1a7eeb305ce2247472c66fb5dfda8e2f34d.jpg)

![18e7529a5eae27ebf22d12bb5827d322a573b9a6a1756dd85b4b65538710f6b2.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/18e7529a5eae27ebf22d12bb5827d322a573b9a6a1756dd85b4b65538710f6b2.jpg)

![192fe782eb38523ef4f2c7f8ed6a525dffea6fc55fc08a5baf32d40d3ce7a0f1.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/192fe782eb38523ef4f2c7f8ed6a525dffea6fc55fc08a5baf32d40d3ce7a0f1.jpg)

![25fc4a172b63dc6243279b5d6417478e1a0276368c1aba952c27020b6b32c0a1.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/25fc4a172b63dc6243279b5d6417478e1a0276368c1aba952c27020b6b32c0a1.jpg)

![38ea2c87ee5c42352027bcf19d0100f8ee24c65997c150e2b820db15320647d0.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/38ea2c87ee5c42352027bcf19d0100f8ee24c65997c150e2b820db15320647d0.jpg)

![4e36ab8cc8238edcc439462e2f3ce8bc0a30ede8c23b36cb94e57bf2b52e0d0a.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/4e36ab8cc8238edcc439462e2f3ce8bc0a30ede8c23b36cb94e57bf2b52e0d0a.jpg)

![601456318bde5f4f960bb19c24c8ae6610037f2b01124d84d1761f82235ec844.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/601456318bde5f4f960bb19c24c8ae6610037f2b01124d84d1761f82235ec844.jpg)

![617756085ca960196641f746840cb64b9b5e66563ca94f4ac0e30611bf6aebdc.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/617756085ca960196641f746840cb64b9b5e66563ca94f4ac0e30611bf6aebdc.jpg)

![6b63c5f3876035de64e1b59ed43e8aa1b95669bf07262a43988e28c50506ed2e.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/6b63c5f3876035de64e1b59ed43e8aa1b95669bf07262a43988e28c50506ed2e.jpg)

![966429caf1e4cf7cef953d8550bbddd6df5b172d6ebf40abb76c932faf6e5f12.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/966429caf1e4cf7cef953d8550bbddd6df5b172d6ebf40abb76c932faf6e5f12.jpg)

![9e995e27d55001b41d6bc24e03e6edfb8e8587dfffacf2d20d70d3e042732bc9.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/9e995e27d55001b41d6bc24e03e6edfb8e8587dfffacf2d20d70d3e042732bc9.jpg)

![a305eda3bbc8ec26febbbdf6f059747768951dcaf2415182f509846fe7f4cd92.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/a305eda3bbc8ec26febbbdf6f059747768951dcaf2415182f509846fe7f4cd92.jpg)

![acd20389b089e2523ded855ce36d6ea12a22840169a586da72aa0b97919b6e5d.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/acd20389b089e2523ded855ce36d6ea12a22840169a586da72aa0b97919b6e5d.jpg)

![b0f19a07c660aa170e861b79362fdeef34f694160b2d22bbd2b0d2e2edae2021.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/b0f19a07c660aa170e861b79362fdeef34f694160b2d22bbd2b0d2e2edae2021.jpg)

![bd5565a005b6f32e3f2868988b546be2cc9f9aa6e78c64d442e8e04171add6f4.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/bd5565a005b6f32e3f2868988b546be2cc9f9aa6e78c64d442e8e04171add6f4.jpg)

![c5b6c6385b94deca6dbebdddab24c052360ac1cd82d2939295c0f801bf6ac07d.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/c5b6c6385b94deca6dbebdddab24c052360ac1cd82d2939295c0f801bf6ac07d.jpg)

![c9aeb4e6f946ed1385bd28a8f168b1446b28d5a05217ee0aab0ad0c9040f8707.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/c9aeb4e6f946ed1385bd28a8f168b1446b28d5a05217ee0aab0ad0c9040f8707.jpg)

![d93533921b504b9ef61b223d213cb297ce3d2d26f0d0c886f50bfe3f69ee3f0d.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/d93533921b504b9ef61b223d213cb297ce3d2d26f0d0c886f50bfe3f69ee3f0d.jpg)

![fb7c051efb739a1dcff1080c663af214ae14870db3ed8d2fea144bf298ecb737.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/images/fb7c051efb739a1dcff1080c663af214ae14870db3ed8d2fea144bf298ecb737.jpg)

### Tables

![064e8f3949bb1f3ef0f47f53eb609f0d60aa5c87142c4c327d0081c2991bc011.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/064e8f3949bb1f3ef0f47f53eb609f0d60aa5c87142c4c327d0081c2991bc011.jpg)

![19162bd3de91514c10fa3021c5db88f446e06dcb7cedc9bdd97e8cee33c3fbae.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/19162bd3de91514c10fa3021c5db88f446e06dcb7cedc9bdd97e8cee33c3fbae.jpg)

![48ab516bb895a4a4d454587b8046a7964be2a6f575c389d7fe632b9db3f053f8.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/48ab516bb895a4a4d454587b8046a7964be2a6f575c389d7fe632b9db3f053f8.jpg)

![57a840f9181998b50ba7063a1982c1da63afc18f5aa4a181e87947d5df072adc.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/57a840f9181998b50ba7063a1982c1da63afc18f5aa4a181e87947d5df072adc.jpg)

![622378c9ee75d098a3d37ac18c9be7f404102253df3d98295221dbf4ad630ece.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/622378c9ee75d098a3d37ac18c9be7f404102253df3d98295221dbf4ad630ece.jpg)

![706f096fc269410e41d94bf1accf1f7a7c11a9dba11a63772c9443ab2883aaa6.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/706f096fc269410e41d94bf1accf1f7a7c11a9dba11a63772c9443ab2883aaa6.jpg)

![7310722e2c79181f32591573632cd2399c1b19b8b301d17f8b603878a8dfbe2e.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/7310722e2c79181f32591573632cd2399c1b19b8b301d17f8b603878a8dfbe2e.jpg)

![99468750a08ca0779c071a5b380ce199280adda5bf843a03458039c93f62842b.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/99468750a08ca0779c071a5b380ce199280adda5bf843a03458039c93f62842b.jpg)

![b14d6c18ca1794fe352e5408e47c73ee64af20b584688978fe4bd90c26d0a791.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/b14d6c18ca1794fe352e5408e47c73ee64af20b584688978fe4bd90c26d0a791.jpg)

![cfdf0a7210e33ebcbc220815d1f7f6dd6958561b1e09a0b7fdcd370416cf063d.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/cfdf0a7210e33ebcbc220815d1f7f6dd6958561b1e09a0b7fdcd370416cf063d.jpg)

![d97960309c3fdb77e72444ba0a1c3ec859b87d45f5047edbcc7a3f203e80289e.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/d97960309c3fdb77e72444ba0a1c3ec859b87d45f5047edbcc7a3f203e80289e.jpg)

![f4381b813ee72655b2a690e5f47c26bfe29a4b7f6711b8df9cfa841e166cd7d7.jpg](../iclr_results/1545_Causal%20Graphical%20Models%20for%20Vision-Language%20Compositional%20Understanding/tables/f4381b813ee72655b2a690e5f47c26bfe29a4b7f6711b8df9cfa841e166cd7d7.jpg)

## Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks


### Images

![0878b6808aa5ce1ca3255b8d7d96b4e9750387684d74c2cdb49b6eefa6a79d65.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/images/0878b6808aa5ce1ca3255b8d7d96b4e9750387684d74c2cdb49b6eefa6a79d65.jpg)

![0aaf42c876c813b75a3ede0ac670b3859416b34f58975b8846d30f3b97b1f0ed.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/images/0aaf42c876c813b75a3ede0ac670b3859416b34f58975b8846d30f3b97b1f0ed.jpg)

![794e240462d98a67d8f4c3445aa54fcbc270d5155e6e5b8cc38cdd16297c1aab.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/images/794e240462d98a67d8f4c3445aa54fcbc270d5155e6e5b8cc38cdd16297c1aab.jpg)

![7f6ec7adbcf3537f3659eadba857c65d88cf454145ca1318441bf4f7713ba76f.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/images/7f6ec7adbcf3537f3659eadba857c65d88cf454145ca1318441bf4f7713ba76f.jpg)

![cd142d40d403294711310d757798858a38dcc2ea8307824c2a1be355e4bc6ebb.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/images/cd142d40d403294711310d757798858a38dcc2ea8307824c2a1be355e4bc6ebb.jpg)

### Tables

![3d36b3923d2daadfe4a8bfb96ae31181a46046bf3ef81101b2375f70631fa643.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/3d36b3923d2daadfe4a8bfb96ae31181a46046bf3ef81101b2375f70631fa643.jpg)

![578fbabb85a38090141b1307a73605e7a7e04e07a0b6a91e926516f61fe715e9.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/578fbabb85a38090141b1307a73605e7a7e04e07a0b6a91e926516f61fe715e9.jpg)

![583b20a83550e71d0d9b82e292603ebb57bb9b5e9332eb0c18e989efddc486da.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/583b20a83550e71d0d9b82e292603ebb57bb9b5e9332eb0c18e989efddc486da.jpg)

![590523e52112e1350e6a8e727369b6dd4bf3d317dde4875b69cbba66ef5e4853.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/590523e52112e1350e6a8e727369b6dd4bf3d317dde4875b69cbba66ef5e4853.jpg)

![5efd00915c9d91cf803fc8e03bc06c6167fec85b3b8aa92c5590e68a6eec08f5.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/5efd00915c9d91cf803fc8e03bc06c6167fec85b3b8aa92c5590e68a6eec08f5.jpg)

![70656822f2d2e4ebbab920cd2a5502bd77e283827312b0514d28af5d4066ff78.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/70656822f2d2e4ebbab920cd2a5502bd77e283827312b0514d28af5d4066ff78.jpg)

![8a19a583dc090aa08676c3cdb746c8df81b6dca0b711d69e7f6f81e1a2a793f8.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/8a19a583dc090aa08676c3cdb746c8df81b6dca0b711d69e7f6f81e1a2a793f8.jpg)

![90b07e97bc3a5f17b1d082c480e3ecabafb576c0fe42d6670cb992e84cb59279.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/90b07e97bc3a5f17b1d082c480e3ecabafb576c0fe42d6670cb992e84cb59279.jpg)

![9ad2a6c4ca342940e8ca509e023519be7d8d642c9181be95ecd4d539df3f46b4.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/9ad2a6c4ca342940e8ca509e023519be7d8d642c9181be95ecd4d539df3f46b4.jpg)

![9baad191e7e146e9b8b2412fb402ffa03394cad769aaa6789543aa08d7712968.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/9baad191e7e146e9b8b2412fb402ffa03394cad769aaa6789543aa08d7712968.jpg)

![a28c9a763f43805c05d51d80b2b5305d7bc95e96cca3ea90225b354bdb2cacbc.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/a28c9a763f43805c05d51d80b2b5305d7bc95e96cca3ea90225b354bdb2cacbc.jpg)

![b8cfac2e17e4344575d7c4d863d6a1056e9dba6cc7fb3455a06f5fde3abc5852.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/b8cfac2e17e4344575d7c4d863d6a1056e9dba6cc7fb3455a06f5fde3abc5852.jpg)

![b90fba2d6204a32092389c64060ed3b84f2d1f477ebc5b095e167ed67aa101a3.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/b90fba2d6204a32092389c64060ed3b84f2d1f477ebc5b095e167ed67aa101a3.jpg)

![d713f71104dda03303671e7ac224d05e347a2813f08a52252aca19a7aa20cc89.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/d713f71104dda03303671e7ac224d05e347a2813f08a52252aca19a7aa20cc89.jpg)

![d816c7a8a896a41b78c590947c349a34f567736fdb430d9fcf4686e4232147a9.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/d816c7a8a896a41b78c590947c349a34f567736fdb430d9fcf4686e4232147a9.jpg)

![de985b00f9e6a8c13a82fc794ef73235e9b3abbfbab4dbb7f616549a581f9050.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/de985b00f9e6a8c13a82fc794ef73235e9b3abbfbab4dbb7f616549a581f9050.jpg)

![ef4704a7becc1706605bc7c2a821fdefc7674eef24bc977059838694216cec8f.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/ef4704a7becc1706605bc7c2a821fdefc7674eef24bc977059838694216cec8f.jpg)

![f319e37cde567fd4f8e7d71bb827fb99afaaba481b98781e082acd6349d875fb.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/f319e37cde567fd4f8e7d71bb827fb99afaaba481b98781e082acd6349d875fb.jpg)

![f6506166bbcf350963610f32b713b290e4842aa819dc322579e80e7115750d8d.jpg](../iclr_results/1546_Jailbreaking%20Leading%20Safety-Aligned%20LLMs%20with%20Simple%20Adaptive%20Attacks/tables/f6506166bbcf350963610f32b713b290e4842aa819dc322579e80e7115750d8d.jpg)

## Immunogenicity Prediction with Dual Attention Enables Vaccine Target Selection


### Images

![3dba77f2895f8798fe36d833d12a0a542dfdab199feea3afc1a00b8fd179a8bc.jpg](../iclr_results/1547_Immunogenicity%20Prediction%20with%20Dual%20Attention%20Enables%20Vaccine%20Target%20Selection/images/3dba77f2895f8798fe36d833d12a0a542dfdab199feea3afc1a00b8fd179a8bc.jpg)

![43c2cdb38bf7a47309e449274e9d22e05eac931a497888b4f8f4ea91f7a64482.jpg](../iclr_results/1547_Immunogenicity%20Prediction%20with%20Dual%20Attention%20Enables%20Vaccine%20Target%20Selection/images/43c2cdb38bf7a47309e449274e9d22e05eac931a497888b4f8f4ea91f7a64482.jpg)

![968eb5ace0bc2ab5cdb41f92544c432f53549199fcfe4803b1df0f95c49c3fcb.jpg](../iclr_results/1547_Immunogenicity%20Prediction%20with%20Dual%20Attention%20Enables%20Vaccine%20Target%20Selection/images/968eb5ace0bc2ab5cdb41f92544c432f53549199fcfe4803b1df0f95c49c3fcb.jpg)

![dae5da014402b52aa3042d2d0b4a5e83eb1034ffed1a97982ce5c39950c63f2c.jpg](../iclr_results/1547_Immunogenicity%20Prediction%20with%20Dual%20Attention%20Enables%20Vaccine%20Target%20Selection/images/dae5da014402b52aa3042d2d0b4a5e83eb1034ffed1a97982ce5c39950c63f2c.jpg)

![e98fd1e384d05064417984742352dc16beef9c264c292f7afdadf66d629eee35.jpg](../iclr_results/1547_Immunogenicity%20Prediction%20with%20Dual%20Attention%20Enables%20Vaccine%20Target%20Selection/images/e98fd1e384d05064417984742352dc16beef9c264c292f7afdadf66d629eee35.jpg)

### Tables

![5143ccd828c87f6729944471b9780c035e201bc1e59a647516d2c5e605d15d97.jpg](../iclr_results/1547_Immunogenicity%20Prediction%20with%20Dual%20Attention%20Enables%20Vaccine%20Target%20Selection/tables/5143ccd828c87f6729944471b9780c035e201bc1e59a647516d2c5e605d15d97.jpg)

## Privately Counting Partially Ordered Data


### Images

![2f3eabbab2e71008ee44363dac8d1ba316ef8f02baf9919e48516b34811b6829.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/images/2f3eabbab2e71008ee44363dac8d1ba316ef8f02baf9919e48516b34811b6829.jpg)

![5938610c72e7035f3d00ef594af346672cafa7b385d59971cb4c59e9171329c7.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/images/5938610c72e7035f3d00ef594af346672cafa7b385d59971cb4c59e9171329c7.jpg)

![60d60a79b26e2020c1585eac3d004789438ca5a77f6a6989ad6e1a55d2f6c58a.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/images/60d60a79b26e2020c1585eac3d004789438ca5a77f6a6989ad6e1a55d2f6c58a.jpg)

![8cbe1a49aeb49f840984c73dca70f00ff329c29b91d973ec49daed2626336197.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/images/8cbe1a49aeb49f840984c73dca70f00ff329c29b91d973ec49daed2626336197.jpg)

![a4384cbe9e1132a151fff110ef61faa305f5a45f5822af74ad3af5f6b38a429f.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/images/a4384cbe9e1132a151fff110ef61faa305f5a45f5822af74ad3af5f6b38a429f.jpg)

![b54820bbe2a0915c441ad60fc013fad5d5cfb02930c946e305f5a9a6e8b63be6.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/images/b54820bbe2a0915c441ad60fc013fad5d5cfb02930c946e305f5a9a6e8b63be6.jpg)

![efa0abb86430645cfaa847b64f34aeef1b388f4ffee7b915170aea950a0f2534.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/images/efa0abb86430645cfaa847b64f34aeef1b388f4ffee7b915170aea950a0f2534.jpg)

### Tables

![5dc27fc2dc04fcf663b34b42c130b55896c75672002a0247f02eb0cec3ed3bd5.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/tables/5dc27fc2dc04fcf663b34b42c130b55896c75672002a0247f02eb0cec3ed3bd5.jpg)

![9bc7aa76e758e66e8322220b88ccabe17a3b4812aa5526800936c229973ab1d3.jpg](../iclr_results/1548_Privately%20Counting%20Partially%20Ordered%20Data/tables/9bc7aa76e758e66e8322220b88ccabe17a3b4812aa5526800936c229973ab1d3.jpg)

## Mining your own secrets: Diffusion Classifier Scores for Continual Personalization of Text-to-Image Diffusion Models


### Images

![0ab9de68d750139e3653caccfff055428c40abc00c64fa9eb6243f2e375aeb8d.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/0ab9de68d750139e3653caccfff055428c40abc00c64fa9eb6243f2e375aeb8d.jpg)

![0d0668c40c3d2ef65975d2684a7b3ce01c03c2b30cb9cf6b3d40b932573b6c8d.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/0d0668c40c3d2ef65975d2684a7b3ce01c03c2b30cb9cf6b3d40b932573b6c8d.jpg)

![12786e1d025d397f7fe009d88b46cbfa7d31fe6e7af2794d0690b45ca1835f8d.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/12786e1d025d397f7fe009d88b46cbfa7d31fe6e7af2794d0690b45ca1835f8d.jpg)

![1b80d20c52aed591a509e6ab4047348e1800199639a4ec5f9f75dc30bff8047e.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/1b80d20c52aed591a509e6ab4047348e1800199639a4ec5f9f75dc30bff8047e.jpg)

![1dc0475290019662d821477aaeaad268b3d285c090c1e8c69f3f37169e17bbcf.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/1dc0475290019662d821477aaeaad268b3d285c090c1e8c69f3f37169e17bbcf.jpg)

![1ef30ba77f9be4c642539bcf9143f5e58cbf17962a01178a4c34a570fa24a8b9.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/1ef30ba77f9be4c642539bcf9143f5e58cbf17962a01178a4c34a570fa24a8b9.jpg)

![221bfe05d087a6b4f8a746f6918a24865a6c454609f1195c07f74d9659c6ca67.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/221bfe05d087a6b4f8a746f6918a24865a6c454609f1195c07f74d9659c6ca67.jpg)

![29663e585968a3173d24622b1ce2267864782ad6f169c27d2685b9f4fd3f01bc.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/29663e585968a3173d24622b1ce2267864782ad6f169c27d2685b9f4fd3f01bc.jpg)

![2ca84d5aae982ea63d19c82c276a5396bd31f89503c26b6cee13b4bb61865c52.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/2ca84d5aae982ea63d19c82c276a5396bd31f89503c26b6cee13b4bb61865c52.jpg)

![2f5bc7cecb1620452c64c6ae9f8aae88eb78058f37b62ae88d3307e121f30b81.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/2f5bc7cecb1620452c64c6ae9f8aae88eb78058f37b62ae88d3307e121f30b81.jpg)

![34b863fc92fe80de0c71a846d73f112221353f95fc019d3a7f69de07afe72899.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/34b863fc92fe80de0c71a846d73f112221353f95fc019d3a7f69de07afe72899.jpg)

![392e527c4722c27974e43ab53061f1ce0053077d535958d59c7b5c1c88f82497.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/392e527c4722c27974e43ab53061f1ce0053077d535958d59c7b5c1c88f82497.jpg)

![3a9cb371a8581028bb4f9545e0ae1a1ce5d2872f76cf6c86aacb282880c74e38.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/3a9cb371a8581028bb4f9545e0ae1a1ce5d2872f76cf6c86aacb282880c74e38.jpg)

![431594b010d761e5eea7ff9d52553999c9ebaa8052530184c29efc6c2fb07d98.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/431594b010d761e5eea7ff9d52553999c9ebaa8052530184c29efc6c2fb07d98.jpg)

![4e758110bba58b653148a338bb054e98578644e1a0a3fef60cc7ad424454b14b.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/4e758110bba58b653148a338bb054e98578644e1a0a3fef60cc7ad424454b14b.jpg)

![53e8d61ea9b706d427ef6b1e14d4480d2aa48742011fcf72aceb7aaa112ce568.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/53e8d61ea9b706d427ef6b1e14d4480d2aa48742011fcf72aceb7aaa112ce568.jpg)

![59f715a303f88d30670a70096e2ca52b40c5b76bdff01900a5748a4f85cee4ec.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/59f715a303f88d30670a70096e2ca52b40c5b76bdff01900a5748a4f85cee4ec.jpg)

![7d4e1241a9bb008e529121b1b0f45ac4b3c79324135dcdc11de2d3ebf383207b.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/7d4e1241a9bb008e529121b1b0f45ac4b3c79324135dcdc11de2d3ebf383207b.jpg)

![9b21b63e426fde0b48c87dcd44c917fedd5d60e33e1905aaa42174ccb3a73e0f.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/9b21b63e426fde0b48c87dcd44c917fedd5d60e33e1905aaa42174ccb3a73e0f.jpg)

![a66cc300f5d9ce8f98bbafc568b4906a90e1c8eb97a9234116c1e79ed497eb7f.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/a66cc300f5d9ce8f98bbafc568b4906a90e1c8eb97a9234116c1e79ed497eb7f.jpg)

![af88c8802e75ced8296f5462201940f861820775ec6a56b9737c6a0131fbd9ab.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/af88c8802e75ced8296f5462201940f861820775ec6a56b9737c6a0131fbd9ab.jpg)

![b1d4e87d41ba75205fd9a8574122c682f0fd9369b92d35538470ed97b1b0017a.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/b1d4e87d41ba75205fd9a8574122c682f0fd9369b92d35538470ed97b1b0017a.jpg)

![b99d647f44f0b3c5b4d4dc3356496b18f4b14459f80f6fb04a07023dce840010.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/b99d647f44f0b3c5b4d4dc3356496b18f4b14459f80f6fb04a07023dce840010.jpg)

![d113e48d4020659ae84096aca8cd7665b53296c8db331756e3ad730716ca1463.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/d113e48d4020659ae84096aca8cd7665b53296c8db331756e3ad730716ca1463.jpg)

![d82887c432421504a0c99de472ac36ff551dd9232244a3b8f16969a6ab894e8f.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/d82887c432421504a0c99de472ac36ff551dd9232244a3b8f16969a6ab894e8f.jpg)

![decd791b8d844895633de1286b0e1ba585adb2a4fd1440cc9eceb25dd4bc06d4.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/decd791b8d844895633de1286b0e1ba585adb2a4fd1440cc9eceb25dd4bc06d4.jpg)

![e76eaf7a2eca0677bfa44af8d3c18809bc7a25bb446e96f7346a7eccbb0c1d4a.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/e76eaf7a2eca0677bfa44af8d3c18809bc7a25bb446e96f7346a7eccbb0c1d4a.jpg)

![ed2a6d6e2fc0f7b28a839d289c28ccfdaca353d39da70ec2d684bfc35c083c32.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/ed2a6d6e2fc0f7b28a839d289c28ccfdaca353d39da70ec2d684bfc35c083c32.jpg)

![ee4fb3fb746a6c27e4d2b91c80b7c6f7ad81ff2e1ee80654158278583eae54c5.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/ee4fb3fb746a6c27e4d2b91c80b7c6f7ad81ff2e1ee80654158278583eae54c5.jpg)

![f6251e233bce55f9d26b31d875e56fb571e3c002a993b86c39d4e748db3dee64.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/images/f6251e233bce55f9d26b31d875e56fb571e3c002a993b86c39d4e748db3dee64.jpg)

### Tables

![12302787a890fdbe45dcdafdea973a42a114f32b89d65d5fae5dd17e1df04643.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/12302787a890fdbe45dcdafdea973a42a114f32b89d65d5fae5dd17e1df04643.jpg)

![130c1c26fb1b433a81ec7c8747720561adc17b9da0ec574b3728e8cb9ba75d14.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/130c1c26fb1b433a81ec7c8747720561adc17b9da0ec574b3728e8cb9ba75d14.jpg)

![1dd43f30ad93e62e240acbff0a31f3d4f86cdfd2e5ba58b67a1a56116aa8c063.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/1dd43f30ad93e62e240acbff0a31f3d4f86cdfd2e5ba58b67a1a56116aa8c063.jpg)

![303c8c1232504261c6b41abce6f9c0ce1a56bb31b3a1d4a7a46868c292c8eedb.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/303c8c1232504261c6b41abce6f9c0ce1a56bb31b3a1d4a7a46868c292c8eedb.jpg)

![351abbe357bc5e9dd3aa27e95da0771a170019317ba78482e1423d1fa15a8e76.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/351abbe357bc5e9dd3aa27e95da0771a170019317ba78482e1423d1fa15a8e76.jpg)

![48c254bf77db204511b644ded6c8db2fba160ab9a36b7d54c4da171f47194349.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/48c254bf77db204511b644ded6c8db2fba160ab9a36b7d54c4da171f47194349.jpg)

![4c02073bfdc2012cc0522977b7e780120ec5c7bc8eb998b63e85b12edb024698.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/4c02073bfdc2012cc0522977b7e780120ec5c7bc8eb998b63e85b12edb024698.jpg)

![8d3bec49dfb2d7ee0c44f07b9cd61294be60e954bbbc074c08cc4601ec6ee9ec.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/8d3bec49dfb2d7ee0c44f07b9cd61294be60e954bbbc074c08cc4601ec6ee9ec.jpg)

![9405b980577ca0d57f8cf1a5da897fc67716a86ba984843d30bf82b2bd3950e5.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/9405b980577ca0d57f8cf1a5da897fc67716a86ba984843d30bf82b2bd3950e5.jpg)

![c8b6125793f4619db2793042add0bcb1f38f3f713d52c25cdbd577a4dfe81ee3.jpg](../iclr_results/1549_Mining%20your%20own%20secrets_%20Diffusion%20Classifier%20Scores%20for%20Continual%20Personalization%20of%20Text-to-Image%20/tables/c8b6125793f4619db2793042add0bcb1f38f3f713d52c25cdbd577a4dfe81ee3.jpg)

## Large Language Models are Interpretable Learners


### Images

![0a7dbc7dab43da88309af2bec5b0123d5b018e94ac41c27ba3a009a92129567c.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/0a7dbc7dab43da88309af2bec5b0123d5b018e94ac41c27ba3a009a92129567c.jpg)

![26abf1fc726817953842ea28114b4d794f8612bf3ecf06ce878d6ff46fd6eac6.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/26abf1fc726817953842ea28114b4d794f8612bf3ecf06ce878d6ff46fd6eac6.jpg)

![3347d9f2d736b5ef7c5751fb463f30e512ef02ba4954084933c431ee054dd3c4.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/3347d9f2d736b5ef7c5751fb463f30e512ef02ba4954084933c431ee054dd3c4.jpg)

![3e9838db4f96539384392e45729854055369f1cb1665ed51caec7b52d2ad8c9e.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/3e9838db4f96539384392e45729854055369f1cb1665ed51caec7b52d2ad8c9e.jpg)

![4eced4c58cfba4434f8b79582588553389e1751674188700147f06f6090ef28c.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/4eced4c58cfba4434f8b79582588553389e1751674188700147f06f6090ef28c.jpg)

![6e518dcefb18986fe1c430a808b70f36b4526e9e4dfd0809ccb5c08aa7f9b5f8.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/6e518dcefb18986fe1c430a808b70f36b4526e9e4dfd0809ccb5c08aa7f9b5f8.jpg)

![728b5b00b1f4ff9d19cb4b7366b26dfdd647c898a2f5c9d04f4fa4cf226e6c31.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/728b5b00b1f4ff9d19cb4b7366b26dfdd647c898a2f5c9d04f4fa4cf226e6c31.jpg)

![a10666a3352bcf5bc862429514bff47ba076e6451feea8aa8d32d5bcff563fa3.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/images/a10666a3352bcf5bc862429514bff47ba076e6451feea8aa8d32d5bcff563fa3.jpg)

### Tables

![0995800744e461a2d4704711b5b0aa58b1d806b6385fd97e9982b7cef2137a6a.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/0995800744e461a2d4704711b5b0aa58b1d806b6385fd97e9982b7cef2137a6a.jpg)

![1788796e5b50ad32b5388ade2be84e86dca32b2a5fa1909bc9073a2d81d39682.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/1788796e5b50ad32b5388ade2be84e86dca32b2a5fa1909bc9073a2d81d39682.jpg)

![4e1cb83781b74d92e28e2f21b80a9296b348fdf5a3d669348d4694994e257172.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/4e1cb83781b74d92e28e2f21b80a9296b348fdf5a3d669348d4694994e257172.jpg)

![527118d04857bae000dec57bb27693549ea98f87d4a24345ac62cc4c46d26102.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/527118d04857bae000dec57bb27693549ea98f87d4a24345ac62cc4c46d26102.jpg)

![b6e6e28bcd70c9d5ea9519b642d15a69e42efdf4cf1cdbeb43fc986c23cadbb8.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/b6e6e28bcd70c9d5ea9519b642d15a69e42efdf4cf1cdbeb43fc986c23cadbb8.jpg)

![cce8179d68ce3c4f89f583e14a89396bd833934cf06ab14c5f3e795d1fac7026.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/cce8179d68ce3c4f89f583e14a89396bd833934cf06ab14c5f3e795d1fac7026.jpg)

![f24c33c81f345e709d33895512c3cbd361cb09e7badfcfb38077d9841d8fc108.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/f24c33c81f345e709d33895512c3cbd361cb09e7badfcfb38077d9841d8fc108.jpg)

![fee22083e35c4fde70b5ce500ebbe086765ca72f5f4bef2013a482883aa2d0a4.jpg](../iclr_results/1550_Large%20Language%20Models%20are%20Interpretable%20Learners/tables/fee22083e35c4fde70b5ce500ebbe086765ca72f5f4bef2013a482883aa2d0a4.jpg)

## No Location Left Behind: Measuring and Improving the Fairness of Implicit Representations for Earth Data


### Images

![018d9750ba0e70115c0f9b96e99dac78e4190480f0ee8cef2e77b8cef3ec523b.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/018d9750ba0e70115c0f9b96e99dac78e4190480f0ee8cef2e77b8cef3ec523b.jpg)

![1d15a624c8aeb6a4738db776b2a5e9189932bad142d602b3f855ee9341cff8f2.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/1d15a624c8aeb6a4738db776b2a5e9189932bad142d602b3f855ee9341cff8f2.jpg)

![353fe73ace1f5f6b6533557432147ebbdcc973c953a658643ca6d6e423f9870f.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/353fe73ace1f5f6b6533557432147ebbdcc973c953a658643ca6d6e423f9870f.jpg)

![3a20da60122662b1cc828717c48711dce15cffb2aed220387d8c7321aeff4791.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/3a20da60122662b1cc828717c48711dce15cffb2aed220387d8c7321aeff4791.jpg)

![3fdc48afb78d96bf1dde427dffd92304bf74dd67a66412a8e7cccfe8cd2ebf2d.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/3fdc48afb78d96bf1dde427dffd92304bf74dd67a66412a8e7cccfe8cd2ebf2d.jpg)

![446dcffbaa7c9858feea4dacb741a3320383409d374633f5c9fb9896042278d0.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/446dcffbaa7c9858feea4dacb741a3320383409d374633f5c9fb9896042278d0.jpg)

![44750bb9aadb4835187705207db600832c30b795cb9305f0b5b60fd42382a4cd.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/44750bb9aadb4835187705207db600832c30b795cb9305f0b5b60fd42382a4cd.jpg)

![6a26649ea27d123ed8d9e5d9d2fe668400c4c392629e46a1238b05e2d70fa272.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/6a26649ea27d123ed8d9e5d9d2fe668400c4c392629e46a1238b05e2d70fa272.jpg)

![6ac8d5f36ba17f81159f7b4e7c2276c842c83553490256db326af32515728181.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/6ac8d5f36ba17f81159f7b4e7c2276c842c83553490256db326af32515728181.jpg)

![74aeee19cb46ddc2a354b1a0cec35d98d0fe17f8ddf7ec6793c9254f678f4688.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/74aeee19cb46ddc2a354b1a0cec35d98d0fe17f8ddf7ec6793c9254f678f4688.jpg)

![756a01cd986b29f4903522fd88477192e4bef985ed3d36c69946050bdc5ffe86.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/756a01cd986b29f4903522fd88477192e4bef985ed3d36c69946050bdc5ffe86.jpg)

![76bccf1dc6c49603c53dd2b313cc55604ebbba6485d78ed16fd397e8b9fca89a.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/76bccf1dc6c49603c53dd2b313cc55604ebbba6485d78ed16fd397e8b9fca89a.jpg)

![76c8c6569285550e5e53624a2a105df9e923454757296186c12f0c1113170b83.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/76c8c6569285550e5e53624a2a105df9e923454757296186c12f0c1113170b83.jpg)

![78fa90cfdaf9a1d6b606f2d909f0b8cd6a9006e88390f03f064aac76a2cbcfa7.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/78fa90cfdaf9a1d6b606f2d909f0b8cd6a9006e88390f03f064aac76a2cbcfa7.jpg)

![7c9ef7d5829966c0074f1c7bc6444ce24ee26a6bfe55bddca1562b9dc78e66ea.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/7c9ef7d5829966c0074f1c7bc6444ce24ee26a6bfe55bddca1562b9dc78e66ea.jpg)

![8269ce51d2df0d2e151065e453e0b4d3251de96df6604e0c3c434e7087d7a87f.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/8269ce51d2df0d2e151065e453e0b4d3251de96df6604e0c3c434e7087d7a87f.jpg)

![88fd86847364d426771fd98ee78d94b66ef74d06dd9cc035c0e08b5d25287a6a.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/88fd86847364d426771fd98ee78d94b66ef74d06dd9cc035c0e08b5d25287a6a.jpg)

![8e2ac0c0acbc0832a06d70b2608260167e281022b9055a19a253688dba82ff74.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/8e2ac0c0acbc0832a06d70b2608260167e281022b9055a19a253688dba82ff74.jpg)

![8e3806abfd27fabaa6748642d4fd8050cd02be579cd6976902343eced178477c.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/8e3806abfd27fabaa6748642d4fd8050cd02be579cd6976902343eced178477c.jpg)

![985e86f3be0a0562ca2bc74ac9b434cbf0d23f19a52d7d9f2e626252d92de4e0.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/985e86f3be0a0562ca2bc74ac9b434cbf0d23f19a52d7d9f2e626252d92de4e0.jpg)

![b84d66c39e02cd38442a1e61ca92c9f2c7e22209afba9cd74e0745ac81de1430.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/b84d66c39e02cd38442a1e61ca92c9f2c7e22209afba9cd74e0745ac81de1430.jpg)

![d83bb25f6b7d331742f3206c8722758ffeb9b7e894f0de69dbf70506635bf7b8.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/d83bb25f6b7d331742f3206c8722758ffeb9b7e894f0de69dbf70506635bf7b8.jpg)

![e2dc21a8fb712ed75e46388feba02c6a42a6f314145b200eae9d32727481fb92.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/e2dc21a8fb712ed75e46388feba02c6a42a6f314145b200eae9d32727481fb92.jpg)

![ea13346de871c625574659904bd62defffa7b8191a86d753d65d533b6e4a0b09.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/ea13346de871c625574659904bd62defffa7b8191a86d753d65d533b6e4a0b09.jpg)

![fbbf99b45de0d9a40b299c7625c779d6f4aedcc76cb6fdb9776f6dd1e8425c5a.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/images/fbbf99b45de0d9a40b299c7625c779d6f4aedcc76cb6fdb9776f6dd1e8425c5a.jpg)

### Tables

![0adbfa134bcd9db510a070e18bba638460571fca936bc1fc052ddce99565f147.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/0adbfa134bcd9db510a070e18bba638460571fca936bc1fc052ddce99565f147.jpg)

![1f98301154454d892447c17d8b1223e6629d05a9a51785e6f7b874a18e92a3d5.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/1f98301154454d892447c17d8b1223e6629d05a9a51785e6f7b874a18e92a3d5.jpg)

![2b30d43d7e3938478641f59879d4e858be185c72f77a6e9c304375182e4ee44f.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/2b30d43d7e3938478641f59879d4e858be185c72f77a6e9c304375182e4ee44f.jpg)

![2ba876c34df60f51527794f4383e2a0677b1a578fc52c396f699a626498dabf0.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/2ba876c34df60f51527794f4383e2a0677b1a578fc52c396f699a626498dabf0.jpg)

![5476c8a345db78de9767c32e9a6f55725caa19aa79d8fd923d1fec2f4ace52e8.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/5476c8a345db78de9767c32e9a6f55725caa19aa79d8fd923d1fec2f4ace52e8.jpg)

![71657599e11bdd53821a80c1621d26de5bf565e79782a455e0f34efd3e5592e7.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/71657599e11bdd53821a80c1621d26de5bf565e79782a455e0f34efd3e5592e7.jpg)

![839473c592ca1f5a7f48197877611a96354ac1d1596accd70752d0346b029f99.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/839473c592ca1f5a7f48197877611a96354ac1d1596accd70752d0346b029f99.jpg)

![855fe87d40afa60aae94c5c7eee4ee2e2108182f64107266c6f9a8e07d6bf303.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/855fe87d40afa60aae94c5c7eee4ee2e2108182f64107266c6f9a8e07d6bf303.jpg)

![8c8b66a00b3b2639eb1e0404613d1053c9b591c8fd09033ef41bd061569e7281.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/8c8b66a00b3b2639eb1e0404613d1053c9b591c8fd09033ef41bd061569e7281.jpg)

![8ce4a55ef65a86fc3dca25a9ea41c8f83cae309ffae6df0980dbe099a13f33eb.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/8ce4a55ef65a86fc3dca25a9ea41c8f83cae309ffae6df0980dbe099a13f33eb.jpg)

![92a0392e75ad32aaab1f0e14ecd7685a4b7ca16cbacd17b6ebf6c20ac4519d24.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/92a0392e75ad32aaab1f0e14ecd7685a4b7ca16cbacd17b6ebf6c20ac4519d24.jpg)

![a1f6c6640b55eb94b960ac3086335244cdf4735c67871fa0fe092bdefd98677a.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/a1f6c6640b55eb94b960ac3086335244cdf4735c67871fa0fe092bdefd98677a.jpg)

![a264498fc4a0289a75060dca158288c9c4a1e8ae0eb3b2173e83a29aef256bae.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/a264498fc4a0289a75060dca158288c9c4a1e8ae0eb3b2173e83a29aef256bae.jpg)

![bb996976a9804f1c6e3b20a453d9109e80419729a4fc7995fdab2c4a75fa112f.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/bb996976a9804f1c6e3b20a453d9109e80419729a4fc7995fdab2c4a75fa112f.jpg)

![bbe86be86774f426eb467f56009e9b7a300b49845a2f312c879cff327e050461.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/bbe86be86774f426eb467f56009e9b7a300b49845a2f312c879cff327e050461.jpg)

![d6e4768b6e9daadd40c72c5ff030ed67e8a23d6ebeb75825a9bd6ee39755a356.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/d6e4768b6e9daadd40c72c5ff030ed67e8a23d6ebeb75825a9bd6ee39755a356.jpg)

![fdcbbef6d9a1a3ab59b5fd3268ab1cf16476c21ebb18e8269b56f6a95336150a.jpg](../iclr_results/1551_No%20Location%20Left%20Behind_%20Measuring%20and%20Improving%20the%20Fairness%20of%20Implicit%20Representations%20for%20Earth%20/tables/fdcbbef6d9a1a3ab59b5fd3268ab1cf16476c21ebb18e8269b56f6a95336150a.jpg)

## Bridging and Modeling Correlations in Pairwise Data for Direct Preference Optimization


### Images

![0ee30d28dc90fb0289917a354e562ca8ec87c97b4221cd35084345e0a5e7c752.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/0ee30d28dc90fb0289917a354e562ca8ec87c97b4221cd35084345e0a5e7c752.jpg)

![3128e51192262ad62afd64583f0b0e1c9bd28c2a22526a8b9efc34182e688b7a.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/3128e51192262ad62afd64583f0b0e1c9bd28c2a22526a8b9efc34182e688b7a.jpg)

![384a9086961ab8e4c9b8af4457e86be9ec108a8c5c9552a3dc29da75ab167c0e.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/384a9086961ab8e4c9b8af4457e86be9ec108a8c5c9552a3dc29da75ab167c0e.jpg)

![4743444a060d3435e22b871bb16f372910e727107477276eab7ad84d09d8c256.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/4743444a060d3435e22b871bb16f372910e727107477276eab7ad84d09d8c256.jpg)

![59e5249f5e6478ed4f3469b8830e0dd8d585024b693e8dc9ecd4fee174b460a0.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/59e5249f5e6478ed4f3469b8830e0dd8d585024b693e8dc9ecd4fee174b460a0.jpg)

![612bf4988d74ffb12edc10d908ccab6ff7d9c9637155cea93c72120424a5be0a.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/612bf4988d74ffb12edc10d908ccab6ff7d9c9637155cea93c72120424a5be0a.jpg)

![b0fe8daf9d30318739525f6bc544ca5d5eb094201785f43728169a74c8ea2736.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/b0fe8daf9d30318739525f6bc544ca5d5eb094201785f43728169a74c8ea2736.jpg)

![cd7608b167d983816d0dd22643bf15f390aa88f6a3a2f82f1c61e32cf0e5815c.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/cd7608b167d983816d0dd22643bf15f390aa88f6a3a2f82f1c61e32cf0e5815c.jpg)

![d24167a8faf795b20f15247c51ba2e26a5b69ad91feb65cbe1e91978c98eacb7.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/images/d24167a8faf795b20f15247c51ba2e26a5b69ad91feb65cbe1e91978c98eacb7.jpg)

### Tables

![19c5fb55001f77c841ceec4746e0aff8fa2709cce55cdf5a3d09714e8a1667c1.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/19c5fb55001f77c841ceec4746e0aff8fa2709cce55cdf5a3d09714e8a1667c1.jpg)

![2b252fca296349fa547a9fc16838b3781b9f229e78f1c01a27f8cb56f0f1079b.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/2b252fca296349fa547a9fc16838b3781b9f229e78f1c01a27f8cb56f0f1079b.jpg)

![54370809d9213fec52c9465f0fc12b0946bc45b39736db556d8be514716d4097.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/54370809d9213fec52c9465f0fc12b0946bc45b39736db556d8be514716d4097.jpg)

![57fa87b42277702c610b1a3bbb77c20a481e27c13582a8c9110e5e1c421bb9bd.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/57fa87b42277702c610b1a3bbb77c20a481e27c13582a8c9110e5e1c421bb9bd.jpg)

![5fd2527e9e0137acf05b91d3bd0eb01fbdabaa9cbb18a511a47d20e9c932eb6a.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/5fd2527e9e0137acf05b91d3bd0eb01fbdabaa9cbb18a511a47d20e9c932eb6a.jpg)

![72f4a9ec246ddc6f0c5acde9d0f6597fe9a69dbc6ffd877ebd0dff909bbed4b5.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/72f4a9ec246ddc6f0c5acde9d0f6597fe9a69dbc6ffd877ebd0dff909bbed4b5.jpg)

![787eeb1c453d1704ac7802a664d09c9ad8a320553f5c0dc5a67a7454674ac68a.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/787eeb1c453d1704ac7802a664d09c9ad8a320553f5c0dc5a67a7454674ac68a.jpg)

![966a1b15897a0458d2a9c762fd04e2b8a68d605e969fd729e60658948b857f1c.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/966a1b15897a0458d2a9c762fd04e2b8a68d605e969fd729e60658948b857f1c.jpg)

![b600774a3f51740ad36a4b18c3aebcfa674b9fd5e0208e83932ec1e7e441786e.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/b600774a3f51740ad36a4b18c3aebcfa674b9fd5e0208e83932ec1e7e441786e.jpg)

![c14e7d4c7a5376226bcbed9703759da079eb5341f8582575826130907e93749a.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/c14e7d4c7a5376226bcbed9703759da079eb5341f8582575826130907e93749a.jpg)

![f8354260544bb242a79a253061727af70529a6b671d85738c4f1c7320417f5df.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/f8354260544bb242a79a253061727af70529a6b671d85738c4f1c7320417f5df.jpg)

![fb2a13e62112c1ead42d3680c48cbbff0e551ccc243d2fbb2c4eef80b686f19c.jpg](../iclr_results/1552_Bridging%20and%20Modeling%20Correlations%20in%20Pairwise%20Data%20for%20Direct%20Preference%20Optimization/tables/fb2a13e62112c1ead42d3680c48cbbff0e551ccc243d2fbb2c4eef80b686f19c.jpg)

## DiTTo-TTS: Diffusion Transformers for Scalable Text-to-Speech without Domain-Specific Factors


### Images

![038cc48c246d989528fc9ed4fbb2d0efd3c809e21dcd351e97ed2881680cef67.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/038cc48c246d989528fc9ed4fbb2d0efd3c809e21dcd351e97ed2881680cef67.jpg)

![5db05a943cc4040923a8170fa23d985e98bc2e7d8297fc8f1de208c504c7008b.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/5db05a943cc4040923a8170fa23d985e98bc2e7d8297fc8f1de208c504c7008b.jpg)

![688156e723ddd5cfa71c214ba14701e1c5cc44c13f43211dd570bc1514ef0155.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/688156e723ddd5cfa71c214ba14701e1c5cc44c13f43211dd570bc1514ef0155.jpg)

![78261c51047746cb8dc98309875e10a678bca44cb798b88529ea60bf8a825435.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/78261c51047746cb8dc98309875e10a678bca44cb798b88529ea60bf8a825435.jpg)

![d1127134b7d51ab37c34d4f476b766477e660f00189190463ee73bfacb553584.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/d1127134b7d51ab37c34d4f476b766477e660f00189190463ee73bfacb553584.jpg)

![d29f1093de94ecd12b237d61cb9165c36f1bedc2f9f3dde51f815692303ab1fd.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/d29f1093de94ecd12b237d61cb9165c36f1bedc2f9f3dde51f815692303ab1fd.jpg)

![d862b8e14e3d03f5b0db075f5789eeed9fb1288eea92d1a6e1050ab529c75c2c.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/d862b8e14e3d03f5b0db075f5789eeed9fb1288eea92d1a6e1050ab529c75c2c.jpg)

![feb96905cc9571a3e3647d2d0a129a1b6dcb60c6388f613f0e3d0dbb9251d5d3.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/images/feb96905cc9571a3e3647d2d0a129a1b6dcb60c6388f613f0e3d0dbb9251d5d3.jpg)

### Tables

![0404d38481c84db2392a1e86426abf6d860f6851b6f95a97e6fa8a70493cdf6f.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/0404d38481c84db2392a1e86426abf6d860f6851b6f95a97e6fa8a70493cdf6f.jpg)

![0fc24e38ee49bbcdb15268cefe08480c72fb965b03516a21c31dd3519441db08.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/0fc24e38ee49bbcdb15268cefe08480c72fb965b03516a21c31dd3519441db08.jpg)

![1941c6f36c1d20e4393b8fa0e6b0688e65d8872810f7fbe428e395f68755cd70.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/1941c6f36c1d20e4393b8fa0e6b0688e65d8872810f7fbe428e395f68755cd70.jpg)

![1aeb8d2c074d58ee4a273ff9e6076118a1f36b25d75d8cf02ef9e50db7fc2aec.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/1aeb8d2c074d58ee4a273ff9e6076118a1f36b25d75d8cf02ef9e50db7fc2aec.jpg)

![2a3187bbdee9beffb72916ecd921d4e3c636a56f1cdf2edd1030213d96dff5db.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/2a3187bbdee9beffb72916ecd921d4e3c636a56f1cdf2edd1030213d96dff5db.jpg)

![341d4aac06d46d9b53f3dfde50d9221983896a7619717a52e99fb965df924826.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/341d4aac06d46d9b53f3dfde50d9221983896a7619717a52e99fb965df924826.jpg)

![4ea32894060cd48b8e91492b02d3a8fec6dbb9dc5c68e6bafef4afcfa0efbd3c.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/4ea32894060cd48b8e91492b02d3a8fec6dbb9dc5c68e6bafef4afcfa0efbd3c.jpg)

![5cab4fabab8e772cda45ea6f47c41f91c37e64e378107d7f6ba6a92873357206.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/5cab4fabab8e772cda45ea6f47c41f91c37e64e378107d7f6ba6a92873357206.jpg)

![7abfb1cfbb23cfef0bed75b65c984958cece30d40e98e0416a60e19567fbf77d.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/7abfb1cfbb23cfef0bed75b65c984958cece30d40e98e0416a60e19567fbf77d.jpg)

![91341dea3a108c6c6322e57904f4c4f3e10caa9aa5fd3d6eb555ef0dd117b733.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/91341dea3a108c6c6322e57904f4c4f3e10caa9aa5fd3d6eb555ef0dd117b733.jpg)

![994b4d60ac081f8ad8e83918d770007b976f66ab59f63c43c4f93a6b96e2bbaf.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/994b4d60ac081f8ad8e83918d770007b976f66ab59f63c43c4f93a6b96e2bbaf.jpg)

![9dfb155d9dea74d2dd23b9bda95d90b2a1fbbda38e35743977e6112abd97f223.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/9dfb155d9dea74d2dd23b9bda95d90b2a1fbbda38e35743977e6112abd97f223.jpg)

![b0e49c7849ff0a5e0810a9c486088a617c860037f477d9663747d1f7ade2dbbb.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/b0e49c7849ff0a5e0810a9c486088a617c860037f477d9663747d1f7ade2dbbb.jpg)

![b31ce9473904de0330ce35efdd9f6954d73dfa3199d4a9f01abc0b43400224e1.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/b31ce9473904de0330ce35efdd9f6954d73dfa3199d4a9f01abc0b43400224e1.jpg)

![b721da1381aaf0471848655c1920feb9af625ee27d0c1f1dedb1c93d476cb106.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/b721da1381aaf0471848655c1920feb9af625ee27d0c1f1dedb1c93d476cb106.jpg)

![bae3aef5c08248791d1d81c6eda2a55598d974782ddfe957fd626771a3e92e9a.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/bae3aef5c08248791d1d81c6eda2a55598d974782ddfe957fd626771a3e92e9a.jpg)

![c0eda74e85435aead294b199dc9583e741799c0306e0187f8a18acaa00530922.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/c0eda74e85435aead294b199dc9583e741799c0306e0187f8a18acaa00530922.jpg)

![d516a3e065b8df804104b64974f9360a1fa0df934bee8cc429054c1d73c0af54.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/d516a3e065b8df804104b64974f9360a1fa0df934bee8cc429054c1d73c0af54.jpg)

![d92da409f2f0ec4cdebc68d92aa36399d14db3b1bf45ee07e9317bbbc88dfd17.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/d92da409f2f0ec4cdebc68d92aa36399d14db3b1bf45ee07e9317bbbc88dfd17.jpg)

![e8480b16596f128964bf75bbc1f3bea1b13363067f29a87cf2c6eeb623f343dc.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/e8480b16596f128964bf75bbc1f3bea1b13363067f29a87cf2c6eeb623f343dc.jpg)

![e9e762bc660533ba0b9a109e232196c4c02226375009d812c262e93597b31552.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/e9e762bc660533ba0b9a109e232196c4c02226375009d812c262e93597b31552.jpg)

![f0ea6e3e76e6f350b3b66f0efc91dff030a15d8890ea60aef22a79e3415db1d3.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/f0ea6e3e76e6f350b3b66f0efc91dff030a15d8890ea60aef22a79e3415db1d3.jpg)

![f49201c74674b282625281ac8bbd6042aeb1c5c4ad603f329a5390754af47fbb.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/f49201c74674b282625281ac8bbd6042aeb1c5c4ad603f329a5390754af47fbb.jpg)

![f72bb3947901551a65027896108ab123387d28528f620cc506f99398dca3f946.jpg](../iclr_results/1553_DiTTo-TTS_%20Diffusion%20Transformers%20for%20Scalable%20Text-to-Speech%20without%20Domain-Specific%20Factors/tables/f72bb3947901551a65027896108ab123387d28528f620cc506f99398dca3f946.jpg)

## Uncertainty-Aware Decoding with Minimum Bayes Risk


### Images

![126960383feb70bbbfbfa67a8f7a8192eff8c3f01e662b0aa3816d5fe658a886.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/images/126960383feb70bbbfbfa67a8f7a8192eff8c3f01e662b0aa3816d5fe658a886.jpg)

![5e4d219c295dfe5bb0ca6673b20c936143edaade54614dcdf808969d18102199.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/images/5e4d219c295dfe5bb0ca6673b20c936143edaade54614dcdf808969d18102199.jpg)

![8cfcd9eaf19d74c1c5713c710ce135529e63d05884af8b9ba1da558ea8d5fc8f.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/images/8cfcd9eaf19d74c1c5713c710ce135529e63d05884af8b9ba1da558ea8d5fc8f.jpg)

### Tables

![27ed7581f53a23d1b76aa0cbbe867cf2153383f3ef88251d108340db00b401e8.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/27ed7581f53a23d1b76aa0cbbe867cf2153383f3ef88251d108340db00b401e8.jpg)

![79eaf629491a494c54ad19f351740f2f01d3e90c948330ccd28f2e833811cd5a.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/79eaf629491a494c54ad19f351740f2f01d3e90c948330ccd28f2e833811cd5a.jpg)

![aad33dcfa9127b9e0ba2fa844a826c66b643cb338a593d624fb7eef8f24332be.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/aad33dcfa9127b9e0ba2fa844a826c66b643cb338a593d624fb7eef8f24332be.jpg)

![ad2a5bf44eff6056d79586f66e17774d91af23cda081df3bd4b1103e49e636f4.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/ad2a5bf44eff6056d79586f66e17774d91af23cda081df3bd4b1103e49e636f4.jpg)

![afad70085aac05ab744b0056b095b588503b30baf49c215128bd5635dcd2dd5d.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/afad70085aac05ab744b0056b095b588503b30baf49c215128bd5635dcd2dd5d.jpg)

![cd1d4e2ea39ca80e2f4a62c3e10e274f91f9efd419ee4ebd822aee464dcff54a.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/cd1d4e2ea39ca80e2f4a62c3e10e274f91f9efd419ee4ebd822aee464dcff54a.jpg)

![cda7b7bb15c0cdb9ce0052f8d57c1874f5c3f09994dbfeee15fc7a29357913cb.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/cda7b7bb15c0cdb9ce0052f8d57c1874f5c3f09994dbfeee15fc7a29357913cb.jpg)

![ce149a5035ae162d4e7019ac8bd8f4dac2d0b9ab31602fa3c39f3f6dba173573.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/ce149a5035ae162d4e7019ac8bd8f4dac2d0b9ab31602fa3c39f3f6dba173573.jpg)

![f1580f5d2512b097a0f9890b655866c0495c8e9b6e49f1f6881d6ddfe5dd289e.jpg](../iclr_results/1554_Uncertainty-Aware%20Decoding%20with%20Minimum%20Bayes%20Risk/tables/f1580f5d2512b097a0f9890b655866c0495c8e9b6e49f1f6881d6ddfe5dd289e.jpg)

## GEVRM: Goal-Expressive Video Generation Model For Robust Visual Manipulation


### Images

![15910b7ec0ae49068a0088c0cd111cc01f6be9b95e9b8f076fa1fd20c2fa1621.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/15910b7ec0ae49068a0088c0cd111cc01f6be9b95e9b8f076fa1fd20c2fa1621.jpg)

![185c0552926135627746255c7f50f1782b5e40c3fb9ff456ca8a8cd2655635e4.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/185c0552926135627746255c7f50f1782b5e40c3fb9ff456ca8a8cd2655635e4.jpg)

![24ab8e48504899c49d98eafdd133228fb9da32dd2639df92b6c672d9cd25ac5f.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/24ab8e48504899c49d98eafdd133228fb9da32dd2639df92b6c672d9cd25ac5f.jpg)

![25071752c5ba71626bd3d12042558f0eb0f485e1226ab7dd00d021183905a3cf.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/25071752c5ba71626bd3d12042558f0eb0f485e1226ab7dd00d021183905a3cf.jpg)

![31e91cd21b98c6b2fff72d7b8bf169948510b18cc9b8a936d3026d88130e5412.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/31e91cd21b98c6b2fff72d7b8bf169948510b18cc9b8a936d3026d88130e5412.jpg)

![360e269d7346f28eec8ceaad63b265941fe37425e086e7582816f0f22b0fc254.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/360e269d7346f28eec8ceaad63b265941fe37425e086e7582816f0f22b0fc254.jpg)

![5850ca1fbbcfd58e4f26022738cd41dd49ba60f8fb613e580bf99be66d0c1137.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/5850ca1fbbcfd58e4f26022738cd41dd49ba60f8fb613e580bf99be66d0c1137.jpg)

![6f45019dd8d6677e7da7c43a6676850886b67c1dd317cf24fa300436a556201f.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/6f45019dd8d6677e7da7c43a6676850886b67c1dd317cf24fa300436a556201f.jpg)

![872183e21cb365c9578eb42b17720dc4cad3a9d2b1cffc1b043b4bbffaf93532.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/872183e21cb365c9578eb42b17720dc4cad3a9d2b1cffc1b043b4bbffaf93532.jpg)

![8e9f7611b746f745c54e7988952526688e6184ec80ff2f006faea46d91f963a4.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/8e9f7611b746f745c54e7988952526688e6184ec80ff2f006faea46d91f963a4.jpg)

![91feb959ada4fd46b1a2ff17e021bcbba55b328632a3f3bf0f4ca4a73b3846c5.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/91feb959ada4fd46b1a2ff17e021bcbba55b328632a3f3bf0f4ca4a73b3846c5.jpg)

![aa1faad5378cceab8f04edd4b21c6c54fbe1759b5ac73a222f290df98d4dc329.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/aa1faad5378cceab8f04edd4b21c6c54fbe1759b5ac73a222f290df98d4dc329.jpg)

![ab9bd72eddaef9fef34a8ab1716d6d4042eba485c0737c630691e89f48157da6.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/ab9bd72eddaef9fef34a8ab1716d6d4042eba485c0737c630691e89f48157da6.jpg)

![c613479e8a4786a37127779e1200e2b8792de6b3a66c7bf34d850077387addde.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/c613479e8a4786a37127779e1200e2b8792de6b3a66c7bf34d850077387addde.jpg)

![e654a858258ad848c6b2184044c3995a190407b128c7602d0e172284579a80c6.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/images/e654a858258ad848c6b2184044c3995a190407b128c7602d0e172284579a80c6.jpg)

### Tables

![2f0c39a6d9a08f66377ae326cb5e0b49f4a1926e5742f174ba14c3e6cc2acfbf.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/2f0c39a6d9a08f66377ae326cb5e0b49f4a1926e5742f174ba14c3e6cc2acfbf.jpg)

![494f08e405757a1a339cd3e2ae090d12543b4b6b484ed14d74ee401c5472a6d9.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/494f08e405757a1a339cd3e2ae090d12543b4b6b484ed14d74ee401c5472a6d9.jpg)

![502a478deab49d5c394cbfeaa22d4af42645095e62d29757a07998ccfdc20482.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/502a478deab49d5c394cbfeaa22d4af42645095e62d29757a07998ccfdc20482.jpg)

![52dd7718bf26baef57873e66f902219a52fc365d90b2498d1369532a1c753ade.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/52dd7718bf26baef57873e66f902219a52fc365d90b2498d1369532a1c753ade.jpg)

![9071dfb245ddd1eb1570e493ac1ed0a6b1ea77fa14db62e277de8dd264c96021.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/9071dfb245ddd1eb1570e493ac1ed0a6b1ea77fa14db62e277de8dd264c96021.jpg)

![92eed84c6e8869853c80e894a3f56fbfca6010fa69f917676092ce5c854288ab.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/92eed84c6e8869853c80e894a3f56fbfca6010fa69f917676092ce5c854288ab.jpg)

![99ed0e33e8240a00cbcc88e5c5048fdee025814d5e0d84e03f71fd7fbfa5972f.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/99ed0e33e8240a00cbcc88e5c5048fdee025814d5e0d84e03f71fd7fbfa5972f.jpg)

![a0b703dd22be2eb78d096e08fbb1413f7af666d7834e79bbfc85bfa53fcb4ec9.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/a0b703dd22be2eb78d096e08fbb1413f7af666d7834e79bbfc85bfa53fcb4ec9.jpg)

![fa0746a449c09e3e498e124902863f1456d88b78788b26a6e3924279244c4f6b.jpg](../iclr_results/1555_GEVRM_%20Goal-Expressive%20Video%20Generation%20Model%20For%20Robust%20Visual%20Manipulation/tables/fa0746a449c09e3e498e124902863f1456d88b78788b26a6e3924279244c4f6b.jpg)

## Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration


### Images

![14140f1d301670390d1829c6774d98e0520d4c712d7f5b54ef8f6c56e9829e13.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/14140f1d301670390d1829c6774d98e0520d4c712d7f5b54ef8f6c56e9829e13.jpg)

![150d546daf6057f6f03537aeb24a4d0c7ebe09155a0c6159f40e0d5d5b5c59ec.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/150d546daf6057f6f03537aeb24a4d0c7ebe09155a0c6159f40e0d5d5b5c59ec.jpg)

![280a8c01c6a479d30c03e50511ed50ceb0fd9bc1e31edb4cca32a10a8913bdbf.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/280a8c01c6a479d30c03e50511ed50ceb0fd9bc1e31edb4cca32a10a8913bdbf.jpg)

![32237d3e328331ba991d2b55cd69679a18234eb0e83eea4d517573d0f7ee573c.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/32237d3e328331ba991d2b55cd69679a18234eb0e83eea4d517573d0f7ee573c.jpg)

![3788554dc2d82507df514b261eb9f196a822a07bc3e156946f961c91896d7cd1.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/3788554dc2d82507df514b261eb9f196a822a07bc3e156946f961c91896d7cd1.jpg)

![4a2ee798c878f0f2fceca1fa177d0b0670dea3d1c32307ca448ac1d93d5b0417.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/4a2ee798c878f0f2fceca1fa177d0b0670dea3d1c32307ca448ac1d93d5b0417.jpg)

![56ce804702d102f095e99241cfaa47bde93af0ebc8a17387c82a3e99db150377.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/56ce804702d102f095e99241cfaa47bde93af0ebc8a17387c82a3e99db150377.jpg)

![640488e512c7de13556f11b47fecb0fdbfefc075c8d44a9b1e2879cf5288d103.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/640488e512c7de13556f11b47fecb0fdbfefc075c8d44a9b1e2879cf5288d103.jpg)

![89da81473f499763a1337e6d462cadd82b23b37709fe6c17d730e5b8cebdcdaf.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/89da81473f499763a1337e6d462cadd82b23b37709fe6c17d730e5b8cebdcdaf.jpg)

![9648672db5ab1170c70698d8bbc821394253eca4e6f3393d22273bfd1c610ca6.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/9648672db5ab1170c70698d8bbc821394253eca4e6f3393d22273bfd1c610ca6.jpg)

![c3d69882ef91e2c733c97b45966b8ef37272777d9af7d210c4843aeaa764dbdc.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/c3d69882ef91e2c733c97b45966b8ef37272777d9af7d210c4843aeaa764dbdc.jpg)

![d4958af745c9341d59b2ff568349cca945c45fc63abdf93f0eceeefc92ea490b.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/d4958af745c9341d59b2ff568349cca945c45fc63abdf93f0eceeefc92ea490b.jpg)

![d72ea8d11a8e7d980b51f729c6ec98c0536de5eebcc833e70e18de5d2e8b7237.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/d72ea8d11a8e7d980b51f729c6ec98c0536de5eebcc833e70e18de5d2e8b7237.jpg)

![e4d4459274540289b9fb68f614e79b0d5616da162e83869cef8a8c05275379e4.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/e4d4459274540289b9fb68f614e79b0d5616da162e83869cef8a8c05275379e4.jpg)

![eb2d9ff639979182d813d38f2b49937742bead85a2fa628e634e9cb744355de8.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/eb2d9ff639979182d813d38f2b49937742bead85a2fa628e634e9cb744355de8.jpg)

![fead4e842d04ffc39c3938430cc0acb272aa5a119744e0f21c006f52597002a7.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/images/fead4e842d04ffc39c3938430cc0acb272aa5a119744e0f21c006f52597002a7.jpg)

### Tables

![03aebb82a45a0c86e608b0cf0da9dcdf79937fdb411198d6b7ba8e147e1fbfa0.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/03aebb82a45a0c86e608b0cf0da9dcdf79937fdb411198d6b7ba8e147e1fbfa0.jpg)

![0e64860d1eb36e33b59d5cc50be49cdfd4cd4fd68b31525710f173f38abe312a.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/0e64860d1eb36e33b59d5cc50be49cdfd4cd4fd68b31525710f173f38abe312a.jpg)

![2333b3867f4859bfe3cdc985f3f6641948d096857ab0eb0130783aa72ddfc70a.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/2333b3867f4859bfe3cdc985f3f6641948d096857ab0eb0130783aa72ddfc70a.jpg)

![2d4a7804503e180662ea624448eacea0bbc31f75edbe1169ca347da381e35c8e.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/2d4a7804503e180662ea624448eacea0bbc31f75edbe1169ca347da381e35c8e.jpg)

![57f1c84b8ce9f2559a7bf2778cf3fea39d766dd40d5e54b30a072a49b7a313dc.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/57f1c84b8ce9f2559a7bf2778cf3fea39d766dd40d5e54b30a072a49b7a313dc.jpg)

![6fbf88d9d76fd0da4b47fab919940072167a8c17c1a4d3c9d23441a43cf2e4c5.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/6fbf88d9d76fd0da4b47fab919940072167a8c17c1a4d3c9d23441a43cf2e4c5.jpg)

![7403aa2034e8c8dbc7a2a5695039b111a921185d745cf2df4517ef3b6dfc3563.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/7403aa2034e8c8dbc7a2a5695039b111a921185d745cf2df4517ef3b6dfc3563.jpg)

![86b1f31c609fbdbb615eaa9b02c5b2d26452acb4b34828f7c60685462f1244ec.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/86b1f31c609fbdbb615eaa9b02c5b2d26452acb4b34828f7c60685462f1244ec.jpg)

![925782df88f875424b8dbad95e47a5b128dba57896b0c511e98de61e11d8d0b7.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/925782df88f875424b8dbad95e47a5b128dba57896b0c511e98de61e11d8d0b7.jpg)

![9eddf7fc699a5aabbb9998a989c1056bd572223e5d3439ae961d7b69ac44677b.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/9eddf7fc699a5aabbb9998a989c1056bd572223e5d3439ae961d7b69ac44677b.jpg)

![bad0a8d6d30f45251945174bb466f0b68228d5f0f6be08412fa03f0561de1dae.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/bad0a8d6d30f45251945174bb466f0b68228d5f0f6be08412fa03f0561de1dae.jpg)

![ead0bcf136f5b50362622d5be8e8cb787f56ab14b1bebe9cf2dc460dda19a991.jpg](../iclr_results/1556_Posterior-Mean%20Rectified%20Flow_%20Towards%20Minimum%20MSE%20Photo-Realistic%20Image%20Restoration/tables/ead0bcf136f5b50362622d5be8e8cb787f56ab14b1bebe9cf2dc460dda19a991.jpg)

## PWM: Policy Learning with Multi-Task World Models


### Images

![08af61ce59d1a344255bf366f64ab6defec9ce4e84c568dc2d5d11b3ba43c4ec.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/08af61ce59d1a344255bf366f64ab6defec9ce4e84c568dc2d5d11b3ba43c4ec.jpg)

![0f549fe9555ca239d8793e0c06dbe6d28be71972d856dfccd4c3a2bb10bc594a.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/0f549fe9555ca239d8793e0c06dbe6d28be71972d856dfccd4c3a2bb10bc594a.jpg)

![231ccc7ab2d5246caf173b05835af83880920ce734edd1b7451a5148e6dcbbe3.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/231ccc7ab2d5246caf173b05835af83880920ce734edd1b7451a5148e6dcbbe3.jpg)

![42fe1423f56f91db9d37f436d8350ae1e829fe59f93c6f7d52cf909d3cdb115d.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/42fe1423f56f91db9d37f436d8350ae1e829fe59f93c6f7d52cf909d3cdb115d.jpg)

![46a4403b6085a29aa5a8ed2f075724dc3556a6f616ffb6b78565dfa55907ab85.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/46a4403b6085a29aa5a8ed2f075724dc3556a6f616ffb6b78565dfa55907ab85.jpg)

![59ee5c2481c25ab47cadd30788963fd2c5e1e10e055dab7a3569da1072d0135b.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/59ee5c2481c25ab47cadd30788963fd2c5e1e10e055dab7a3569da1072d0135b.jpg)

![615eaf07d9b0b7caefef92a5b57004fb9bed7b5cdb8463675fcca2f8c1573d3b.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/615eaf07d9b0b7caefef92a5b57004fb9bed7b5cdb8463675fcca2f8c1573d3b.jpg)

![63f0ee2a9ee525dc5d6768234051649d05640483377509162cca06d4cc16a2a8.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/63f0ee2a9ee525dc5d6768234051649d05640483377509162cca06d4cc16a2a8.jpg)

![77e552418deadd516717b7a742620cff8f3f3820e6fa6d75fc2ab9a544cb6529.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/77e552418deadd516717b7a742620cff8f3f3820e6fa6d75fc2ab9a544cb6529.jpg)

![7957f60c75e6c25ee3c34e1b37df5c83067d43fd9e1a8a6d873f5a83c0750546.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/7957f60c75e6c25ee3c34e1b37df5c83067d43fd9e1a8a6d873f5a83c0750546.jpg)

![79adf77c24a920f71e16b2fedcdbe2f1c4c2316c45a73822d1c0eb9434149c33.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/79adf77c24a920f71e16b2fedcdbe2f1c4c2316c45a73822d1c0eb9434149c33.jpg)

![8f1b2c0619494c9a5107053c4315045ba63b66381078ff64a23b50a2d60fa730.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/8f1b2c0619494c9a5107053c4315045ba63b66381078ff64a23b50a2d60fa730.jpg)

![990201b626c2ebe264fa24a2a0b6d4da8a6a8e5a6d333e42f89fcb49338e9466.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/990201b626c2ebe264fa24a2a0b6d4da8a6a8e5a6d333e42f89fcb49338e9466.jpg)

![bb556b462b83287858d8b004aad3ec719656c4d0fb7b7ebe6c678613c4c84602.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/bb556b462b83287858d8b004aad3ec719656c4d0fb7b7ebe6c678613c4c84602.jpg)

![c85ebde130fda53d2c8dfb624591b4eeee2e50d5e4cf026347243717351b3ad3.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/c85ebde130fda53d2c8dfb624591b4eeee2e50d5e4cf026347243717351b3ad3.jpg)

![d6ab287c5fa68278182ef3bd0af022b6614f6b66a4a1a0f960f6746097360bd8.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/d6ab287c5fa68278182ef3bd0af022b6614f6b66a4a1a0f960f6746097360bd8.jpg)

![e89a586e8d6780fd5cdf7085b23d36bb05eac5aa5093f2008bf3a1d36f8c959d.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/e89a586e8d6780fd5cdf7085b23d36bb05eac5aa5093f2008bf3a1d36f8c959d.jpg)

![f56bb0bd9ea919e6cf44e71a06e93847075fc4c81b86a165de895411560f8285.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/f56bb0bd9ea919e6cf44e71a06e93847075fc4c81b86a165de895411560f8285.jpg)

![f685e8245fe7c32c0c85d87785f17dc3711340cb757af13d68dcd428e15d1171.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/images/f685e8245fe7c32c0c85d87785f17dc3711340cb757af13d68dcd428e15d1171.jpg)

### Tables

![1bf1ceafcf8a81a545d54a99d61665829b21e9d74ab698f89143f8db1f892e5b.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/tables/1bf1ceafcf8a81a545d54a99d61665829b21e9d74ab698f89143f8db1f892e5b.jpg)

![26849321328dcdf390255806e2018de3173c68c7a51bce8b177abd3f4efaab54.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/tables/26849321328dcdf390255806e2018de3173c68c7a51bce8b177abd3f4efaab54.jpg)

![437fc47f6feb6d75d284c2e75b8fc2df4dd068f8711512d1fa43671e606c00a3.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/tables/437fc47f6feb6d75d284c2e75b8fc2df4dd068f8711512d1fa43671e606c00a3.jpg)

![7bc1b394fd3f975bebaec1d8fee15ac00edccfb6df0a2a020950f0653e579661.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/tables/7bc1b394fd3f975bebaec1d8fee15ac00edccfb6df0a2a020950f0653e579661.jpg)

![d0833047b54f6c760bd1e42e621af4b64a3108d00ff6a266c187aca4ecf7f5d6.jpg](../iclr_results/1557_PWM_%20Policy%20Learning%20with%20Multi-Task%20World%20Models/tables/d0833047b54f6c760bd1e42e621af4b64a3108d00ff6a266c187aca4ecf7f5d6.jpg)

## SAGEPhos: Sage Bio-Coupled and Augmented Fusion for Phosphorylation Site Detection


### Images

![042ffd404b1871346f756169b86b52eb76668aef10c471d151142fd13ea3480d.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/images/042ffd404b1871346f756169b86b52eb76668aef10c471d151142fd13ea3480d.jpg)

![348db44b19c96efc9d12aa4b61af0fac2f46a2fdc9e5558ccb74ae527d064274.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/images/348db44b19c96efc9d12aa4b61af0fac2f46a2fdc9e5558ccb74ae527d064274.jpg)

![6ff61da19ccd3030b396939126265ad74c9648cb509557bb443aee8b70ee528a.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/images/6ff61da19ccd3030b396939126265ad74c9648cb509557bb443aee8b70ee528a.jpg)

![abac01f98e87b90137b75465d2b751f2380d08d82146f225adbd1a12e446e2a6.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/images/abac01f98e87b90137b75465d2b751f2380d08d82146f225adbd1a12e446e2a6.jpg)

![d9324f26474d2fdb5d2f07a2763f1bc1ac9215d8e3b5226b425ba5068e018af1.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/images/d9324f26474d2fdb5d2f07a2763f1bc1ac9215d8e3b5226b425ba5068e018af1.jpg)

### Tables

![43f142ed64d6fed44ae2763fdc45942dfc8563caf03e307160e041395bf0a27c.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/tables/43f142ed64d6fed44ae2763fdc45942dfc8563caf03e307160e041395bf0a27c.jpg)

![48ab8ca6798cc6b3a6edd8345fe0a47f1feef60cb1b7fe0e4c746ff2296e1030.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/tables/48ab8ca6798cc6b3a6edd8345fe0a47f1feef60cb1b7fe0e4c746ff2296e1030.jpg)

![57d231b7849e2b28f054cd64ef33f21bfb0c64756566499dcf0bbdec7650c723.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/tables/57d231b7849e2b28f054cd64ef33f21bfb0c64756566499dcf0bbdec7650c723.jpg)

![93593f912589ca9174d391c1ff25e0c9ac63855bd89549c97e73a835b4ff519d.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/tables/93593f912589ca9174d391c1ff25e0c9ac63855bd89549c97e73a835b4ff519d.jpg)

![ae8399893a71e5d7fda7b37d74321ebcb2768dc26c665630fdd28698dcd3baeb.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/tables/ae8399893a71e5d7fda7b37d74321ebcb2768dc26c665630fdd28698dcd3baeb.jpg)

![b1dcd1e80010b4f001934e7eba0d22d533ebbb5e0fdcfb5d86ce28afc23bfe7a.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/tables/b1dcd1e80010b4f001934e7eba0d22d533ebbb5e0fdcfb5d86ce28afc23bfe7a.jpg)

![cd06fc4ad38607d367b77c2dd037f198477593f12094eea7ce0abb883e293d1f.jpg](../iclr_results/1558_SAGEPhos_%20Sage%20Bio-Coupled%20and%20Augmented%20Fusion%20for%20Phosphorylation%20Site%20Detection/tables/cd06fc4ad38607d367b77c2dd037f198477593f12094eea7ce0abb883e293d1f.jpg)

## OBI-Bench: Can LMMs Aid in Study of Ancient Script on Oracle Bones?


### Images

![0aecd9e878ed5bd86b98c0eb23e79895bb731e6a3f948abb8c4ec6f95907c83a.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/0aecd9e878ed5bd86b98c0eb23e79895bb731e6a3f948abb8c4ec6f95907c83a.jpg)

![0e4ec2ced8b300eeccf59218b372b83626ef665ae28ed96a4e2d8e36a2c247aa.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/0e4ec2ced8b300eeccf59218b372b83626ef665ae28ed96a4e2d8e36a2c247aa.jpg)

![0f6cb724f0184d3df610211efffc2a1ac36928ad6815d623938a735b17b6f595.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/0f6cb724f0184d3df610211efffc2a1ac36928ad6815d623938a735b17b6f595.jpg)

![17e7a54aaacc5780c83c64b9410e7419ed7ec78bd448095e0f6822a3104a02f5.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/17e7a54aaacc5780c83c64b9410e7419ed7ec78bd448095e0f6822a3104a02f5.jpg)

![1bf525d54d89e17b15528930ecb5aef692d49675c7fc749f33fcd7372ab193cc.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/1bf525d54d89e17b15528930ecb5aef692d49675c7fc749f33fcd7372ab193cc.jpg)

![1c6450ae2bb0c75ff7a8e9c5d5fbc0acb13327ccead986ad3cda5a311457e304.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/1c6450ae2bb0c75ff7a8e9c5d5fbc0acb13327ccead986ad3cda5a311457e304.jpg)

![1d08638af83cc972ff55535745aceaecde5b80a08617b55a24cf30f6eaf0f058.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/1d08638af83cc972ff55535745aceaecde5b80a08617b55a24cf30f6eaf0f058.jpg)

![1f149146d189d754583f2de7dbf1b5c686e2e75867adf88b4758416c25cd6516.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/1f149146d189d754583f2de7dbf1b5c686e2e75867adf88b4758416c25cd6516.jpg)

![1f9b4cc5af61ff2eb72082431dd7be21d88008f9c139f0bcebf47f68ed115055.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/1f9b4cc5af61ff2eb72082431dd7be21d88008f9c139f0bcebf47f68ed115055.jpg)

![2c412cc38bac961324517c0996691e7409815d304f9fc925ec580acceb4214cb.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/2c412cc38bac961324517c0996691e7409815d304f9fc925ec580acceb4214cb.jpg)

![2e56c8caddc9f7d31c28f0889a6a6fe8df6af947b5643f7dbe31d603b91245ec.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/2e56c8caddc9f7d31c28f0889a6a6fe8df6af947b5643f7dbe31d603b91245ec.jpg)

![304cd02c2aadc2900bfa7430d76e2f4b692d5fe67ed2353ff97ed70fd02693a5.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/304cd02c2aadc2900bfa7430d76e2f4b692d5fe67ed2353ff97ed70fd02693a5.jpg)

![357175f526f6691fbc35efdd4e6673e29823d1477fecaf541e3a0e32f2fb8178.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/357175f526f6691fbc35efdd4e6673e29823d1477fecaf541e3a0e32f2fb8178.jpg)

![3725f6ad31e3b6aa40cd06109b18735a0a19af936725df8364da87d112119b23.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/3725f6ad31e3b6aa40cd06109b18735a0a19af936725df8364da87d112119b23.jpg)

![5582a69eb967f72ae0b2f45ddef1bc27fdf1b71d6c01d40470f5bdcd8e585c62.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/5582a69eb967f72ae0b2f45ddef1bc27fdf1b71d6c01d40470f5bdcd8e585c62.jpg)

![591b98da77bb25a90134a24e8867effb354e5dff7210a1da15f50f2c7c9cdda9.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/591b98da77bb25a90134a24e8867effb354e5dff7210a1da15f50f2c7c9cdda9.jpg)

![657856bda4591c5166501d4845988d7d17f9bbb6fd36c7ed60ae5a6d1c5a0b13.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/657856bda4591c5166501d4845988d7d17f9bbb6fd36c7ed60ae5a6d1c5a0b13.jpg)

![67c9ce4a2d7c93be77a797a96d6abc797ae2085f7857cea041f95784d3032ff4.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/67c9ce4a2d7c93be77a797a96d6abc797ae2085f7857cea041f95784d3032ff4.jpg)

![72d62c028666d718c49c4591cd0b49c6e288bada38218768d16dd1e4b6d5135b.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/72d62c028666d718c49c4591cd0b49c6e288bada38218768d16dd1e4b6d5135b.jpg)

![83e3bef00ce5f779a28b3097961db707134778944bdc42718fcbd4497019ed21.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/83e3bef00ce5f779a28b3097961db707134778944bdc42718fcbd4497019ed21.jpg)

![872ae40db3073e890dc87fc10203187c21994f25f691d0ee01c68d9862929468.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/872ae40db3073e890dc87fc10203187c21994f25f691d0ee01c68d9862929468.jpg)

![921a0328d50264d0c41453f092b4a201613d5b5771c6deb1d76ec92457601489.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/921a0328d50264d0c41453f092b4a201613d5b5771c6deb1d76ec92457601489.jpg)

![98134fcfa02d4bb489188f50a04b3a85f44614dfc48cc60ab2d4d0b09e13320d.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/98134fcfa02d4bb489188f50a04b3a85f44614dfc48cc60ab2d4d0b09e13320d.jpg)

![99f656f9b300b39b4427398f58f9dac1c67672861f5b447a3e56759bd6680859.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/99f656f9b300b39b4427398f58f9dac1c67672861f5b447a3e56759bd6680859.jpg)

![9ae17759e90af97631769b1416d12e1767c9ef35414a512cf1593e45f68279b1.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/9ae17759e90af97631769b1416d12e1767c9ef35414a512cf1593e45f68279b1.jpg)

![9ecc682af0f6adf820f87a2ebc316e7ddcdee7c159bdd8a864269b6f5a0d8564.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/9ecc682af0f6adf820f87a2ebc316e7ddcdee7c159bdd8a864269b6f5a0d8564.jpg)

![b627058f42640aac1c57f4b913f44cc5aaf09b0eb3000d5f7ce7b9b06be7dce3.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/b627058f42640aac1c57f4b913f44cc5aaf09b0eb3000d5f7ce7b9b06be7dce3.jpg)

![bb9211b53e2a8fc8a319162f765330dc53f775ef30b23052653a2751c3cbb771.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/bb9211b53e2a8fc8a319162f765330dc53f775ef30b23052653a2751c3cbb771.jpg)

![c4b0be0cbd749ad8f4eecf0b5ff8ac2b7768a91bcb426963a0037c04d67468fa.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/c4b0be0cbd749ad8f4eecf0b5ff8ac2b7768a91bcb426963a0037c04d67468fa.jpg)

![d3d68e20a0f221c74016aab8ae2bc2356ac48867221b96220462198439ac914e.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/d3d68e20a0f221c74016aab8ae2bc2356ac48867221b96220462198439ac914e.jpg)

![d4dfd42fc22cddf5d06cd64b61341b4271a108323e31e744da0cce9e3698fe33.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/d4dfd42fc22cddf5d06cd64b61341b4271a108323e31e744da0cce9e3698fe33.jpg)

![d6e00b46999d0480056b988564de4ff14fd453ed9e5c90c005a9949164cd43cb.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/d6e00b46999d0480056b988564de4ff14fd453ed9e5c90c005a9949164cd43cb.jpg)

![e09471ca73434f7c186c0ed21a6825fcc6cac6115d1c6d271b488c70fa6abc34.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/e09471ca73434f7c186c0ed21a6825fcc6cac6115d1c6d271b488c70fa6abc34.jpg)

![eefe4542e24e3c2c68ad208ae87dd693fe9b877c3562f00bd47350fe0cad349a.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/eefe4542e24e3c2c68ad208ae87dd693fe9b877c3562f00bd47350fe0cad349a.jpg)

![f0b0d18f1b5e21e37a05902b62ef0973a84ecec9c4052f662185bd7e7adaf87b.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/f0b0d18f1b5e21e37a05902b62ef0973a84ecec9c4052f662185bd7e7adaf87b.jpg)

![ff34fbd67b5f9a035016ee0ac05bd5e4803ba35c9db9f656cc855bf09e886c55.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/images/ff34fbd67b5f9a035016ee0ac05bd5e4803ba35c9db9f656cc855bf09e886c55.jpg)

### Tables

![197e48814ed282cdd0251334968e6c1195136400848b62151c52876e9e880da1.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/197e48814ed282cdd0251334968e6c1195136400848b62151c52876e9e880da1.jpg)

![20991dfd2b74b5707d574d77288d0ec57e74e0269951339c9d06d34277ffb8ed.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/20991dfd2b74b5707d574d77288d0ec57e74e0269951339c9d06d34277ffb8ed.jpg)

![3fae816e526ed5dc8a7843a6d3c037e1749c81588d5978383e215997d92c7a7d.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/3fae816e526ed5dc8a7843a6d3c037e1749c81588d5978383e215997d92c7a7d.jpg)

![446de4a67891c6ae0fae19a28605a7cd6170d5597536ee18ce0807470a85bf37.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/446de4a67891c6ae0fae19a28605a7cd6170d5597536ee18ce0807470a85bf37.jpg)

![44a849a2974e15c7bbbc077cb7ec6080fe84d781bebce1d34fd8f4be76bbd5f9.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/44a849a2974e15c7bbbc077cb7ec6080fe84d781bebce1d34fd8f4be76bbd5f9.jpg)

![6e9fb16fd12d0d709fdd5efb3fff74b7078112fa6d0b079e336a03c14d2e27f3.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/6e9fb16fd12d0d709fdd5efb3fff74b7078112fa6d0b079e336a03c14d2e27f3.jpg)

![6f4e650eb4251bc09a50dbd4b22b8658d022f81e8d13b7fa5dcf077b05426e84.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/6f4e650eb4251bc09a50dbd4b22b8658d022f81e8d13b7fa5dcf077b05426e84.jpg)

![934d8ceb2c32dc82c52b97e43e8b825e49cbf15775f50310ccdc57699212197d.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/934d8ceb2c32dc82c52b97e43e8b825e49cbf15775f50310ccdc57699212197d.jpg)

![bbe3ab14da363bd747d3f026329c961e140167082458a4c9904742258c1ae87d.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/bbe3ab14da363bd747d3f026329c961e140167082458a4c9904742258c1ae87d.jpg)

![c903a80eaa8ccd4e08520f42a5bf5464132f85ca6c67113dd8bbe4f1b763d1e7.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/c903a80eaa8ccd4e08520f42a5bf5464132f85ca6c67113dd8bbe4f1b763d1e7.jpg)

![d14f6801dba7b5c3951ee18193bb7ed8f8e5b32d0a9de5954527ddb4a192dbb6.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/d14f6801dba7b5c3951ee18193bb7ed8f8e5b32d0a9de5954527ddb4a192dbb6.jpg)

![d358336e26ffcf01266d6024efb0d972e64c95e1e3f0495b1d0d0fa925ffe406.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/d358336e26ffcf01266d6024efb0d972e64c95e1e3f0495b1d0d0fa925ffe406.jpg)

![e1a4f8e6d80948c054a5e0caef46aa9442de7db1d7b83b262519cfbc827de15c.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/e1a4f8e6d80948c054a5e0caef46aa9442de7db1d7b83b262519cfbc827de15c.jpg)

![fc8f1a62becd871bcc003467d7d663d5bb9d94c293d76739442fe972270f846d.jpg](../iclr_results/1559_OBI-Bench_%20Can%20LMMs%20Aid%20in%20Study%20of%20Ancient%20Script%20on%20Oracle%20Bones_/tables/fc8f1a62becd871bcc003467d7d663d5bb9d94c293d76739442fe972270f846d.jpg)

## NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics


### Images

![60fbd1716321c0c288bd3d37abe5573aa2273ed1cbeff3987b74067e1b50efd5.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/images/60fbd1716321c0c288bd3d37abe5573aa2273ed1cbeff3987b74067e1b50efd5.jpg)

![9f02f0182ac5af86dbbf76628aaddd299379753918b55f5156c3bcf30f2920b1.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/images/9f02f0182ac5af86dbbf76628aaddd299379753918b55f5156c3bcf30f2920b1.jpg)

![da58a7f0ecbd79d260cd563ce39d7042584c4964de1426cddf188ce215d5d984.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/images/da58a7f0ecbd79d260cd563ce39d7042584c4964de1426cddf188ce215d5d984.jpg)

### Tables

![2878d314dd9b125019fcceeb745af407f3c7ca5f1e2e9c8cc118ebcc6ef90fad.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/2878d314dd9b125019fcceeb745af407f3c7ca5f1e2e9c8cc118ebcc6ef90fad.jpg)

![2b2eb67839508aaf6c9561360f947e7078ebfded8a20019ff07d78779aace9db.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/2b2eb67839508aaf6c9561360f947e7078ebfded8a20019ff07d78779aace9db.jpg)

![2ddbe99228014b29fb577bd2d9b190e31319c9419685a3cb01cb9ac4cbef33ce.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/2ddbe99228014b29fb577bd2d9b190e31319c9419685a3cb01cb9ac4cbef33ce.jpg)

![30d91cc408fc6e4c9fe8784814325d8b8efb5c136b7b32bad6ad678bbb4d2dcd.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/30d91cc408fc6e4c9fe8784814325d8b8efb5c136b7b32bad6ad678bbb4d2dcd.jpg)

![697601dd4e6f035234a2172679278e1cc589686b2348548e2f2ceec9ca68019c.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/697601dd4e6f035234a2172679278e1cc589686b2348548e2f2ceec9ca68019c.jpg)

![804d8af84a95dc226a8de458a65affbd4d9bc13b79b337f9fdd0da9b59b0aa9e.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/804d8af84a95dc226a8de458a65affbd4d9bc13b79b337f9fdd0da9b59b0aa9e.jpg)

![8355b539245f0633fbb403361dbcdfdedc2a0529e17c89ae5a1afdd7261085c3.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/8355b539245f0633fbb403361dbcdfdedc2a0529e17c89ae5a1afdd7261085c3.jpg)

![9a29808f1dba2af5d5828809ba02c9c2420a4c2e020f29b719d03dba39751f9e.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/9a29808f1dba2af5d5828809ba02c9c2420a4c2e020f29b719d03dba39751f9e.jpg)

![c1ff81685fedd80d667f59c82184adb4024b90d3757e198855659cfb7b774980.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/c1ff81685fedd80d667f59c82184adb4024b90d3757e198855659cfb7b774980.jpg)

![e6e7873512b839f3d3cb5d0d397ea8d236fe7d24bad29a370e8e876b6b5df28a.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/e6e7873512b839f3d3cb5d0d397ea8d236fe7d24bad29a370e8e876b6b5df28a.jpg)

![fdba31a62a8ad18bdaabc67086f986428a273c7f32ab51ff3721d5c534aabfbb.jpg](../iclr_results/1560_NatureLM-audio_%20an%20Audio-Language%20Foundation%20Model%20for%20Bioacoustics/tables/fdba31a62a8ad18bdaabc67086f986428a273c7f32ab51ff3721d5c534aabfbb.jpg)

## GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering


### Images

![02ae932a7b23921f01a9a342700ec673659e89b390552f289ca62f9332d3f5aa.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/02ae932a7b23921f01a9a342700ec673659e89b390552f289ca62f9332d3f5aa.jpg)

![1aa7640af2d630e594d36355a7759d68fa54d21e56e14a9493bd350ab684ba57.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/1aa7640af2d630e594d36355a7759d68fa54d21e56e14a9493bd350ab684ba57.jpg)

![242c1bfce0b2ece56b2862f4dd1b8b4c71fb38e46375912b665c7f33a5059df1.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/242c1bfce0b2ece56b2862f4dd1b8b4c71fb38e46375912b665c7f33a5059df1.jpg)

![267ae8366bedb9950e4cc8cd7d04a3c7b41cf7a7d3d8c210e77cd9803a30e643.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/267ae8366bedb9950e4cc8cd7d04a3c7b41cf7a7d3d8c210e77cd9803a30e643.jpg)

![2c5a995d1ddb5261eba9493f734fe100a45aad71cc8c1abd5702fbaf003c2d66.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/2c5a995d1ddb5261eba9493f734fe100a45aad71cc8c1abd5702fbaf003c2d66.jpg)

![4a00af664393a816fca37822f3e28f4d3e2680ff246cbd020bddfb56da3d5d39.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/4a00af664393a816fca37822f3e28f4d3e2680ff246cbd020bddfb56da3d5d39.jpg)

![56b975b497c4ba35efb2128890769b912ac13a03f4b2061343f807a3d2ce51aa.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/56b975b497c4ba35efb2128890769b912ac13a03f4b2061343f807a3d2ce51aa.jpg)

![58ef5f30c1d030ca880d89085cf5dc4736d20d95bcc85869c79034668635ae16.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/58ef5f30c1d030ca880d89085cf5dc4736d20d95bcc85869c79034668635ae16.jpg)

![5aceac5851ae036922aabf34f5df0f7422aa54a09d5df241b65c875c46c403ed.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/5aceac5851ae036922aabf34f5df0f7422aa54a09d5df241b65c875c46c403ed.jpg)

![66701c9b7b89dfb582ae1321e97dc8eb6e5beb0a956c447ee0cd5db388bee3a6.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/66701c9b7b89dfb582ae1321e97dc8eb6e5beb0a956c447ee0cd5db388bee3a6.jpg)

![687f40d9d61429f7e22709ebaa64398f860adc765688dee7c3e72f0bd3bcfcff.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/687f40d9d61429f7e22709ebaa64398f860adc765688dee7c3e72f0bd3bcfcff.jpg)

![73d3fd6acab5d3fad3eb042ac4c6eef39a095a183ec2c426aaf0fe057dcc2bf6.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/73d3fd6acab5d3fad3eb042ac4c6eef39a095a183ec2c426aaf0fe057dcc2bf6.jpg)

![74f91de82c2e513603c19bb583c4a5eb19e4c3bd9ffe04656912550196ef95a7.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/74f91de82c2e513603c19bb583c4a5eb19e4c3bd9ffe04656912550196ef95a7.jpg)

![779c3535ecdf55293408c53ebfa7a8109b08534f00cc88e57312b1e7a3c89400.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/779c3535ecdf55293408c53ebfa7a8109b08534f00cc88e57312b1e7a3c89400.jpg)

![82eeb9155d66c368a849de9fc4e5717e77922cd9ea598387ab27e5a748640336.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/82eeb9155d66c368a849de9fc4e5717e77922cd9ea598387ab27e5a748640336.jpg)

![907d7ad64f761dfb86a144fce90a3f687dbd674e7cbd1b4100364eedbeb7c244.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/907d7ad64f761dfb86a144fce90a3f687dbd674e7cbd1b4100364eedbeb7c244.jpg)

![92b6b1549fe50e6956e3733fee73981ab2f43a55bcd36c609047a4e3641c8117.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/92b6b1549fe50e6956e3733fee73981ab2f43a55bcd36c609047a4e3641c8117.jpg)

![971e41b06253805db1a6057f2252f0f00333eb0a23cf5d759b05e2f10b96b508.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/971e41b06253805db1a6057f2252f0f00333eb0a23cf5d759b05e2f10b96b508.jpg)

![aaff5bc4de17b8aa5d7270c0d3edf8f50832198f494158a07049fed47e73e3ec.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/aaff5bc4de17b8aa5d7270c0d3edf8f50832198f494158a07049fed47e73e3ec.jpg)

![b0fea95a576594c442fdd1d888cd9c0c3ae687ce0fb5787a40811add101f95f7.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/b0fea95a576594c442fdd1d888cd9c0c3ae687ce0fb5787a40811add101f95f7.jpg)

![b395d5ca7a9807382d26aea9412cd02208302bc2a0489fc633aab342dec67def.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/b395d5ca7a9807382d26aea9412cd02208302bc2a0489fc633aab342dec67def.jpg)

![b78327ff0376f3c9e144248e4c3f692105e8f83939e4aa65c47a8bdec918f4d7.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/b78327ff0376f3c9e144248e4c3f692105e8f83939e4aa65c47a8bdec918f4d7.jpg)

![bfc4fe8bdd1a1767f22af149751c45b1bdc3c0fc8f5f7032d39a40cc45e5c88b.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/bfc4fe8bdd1a1767f22af149751c45b1bdc3c0fc8f5f7032d39a40cc45e5c88b.jpg)

![c2b0a57c2b6b2559e3440559d054f833027f39e4c0ebc5946569cbd5f14f7ba2.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/c2b0a57c2b6b2559e3440559d054f833027f39e4c0ebc5946569cbd5f14f7ba2.jpg)

![c35c41d8cb532b09f6533b4bed7957fdd702e4d212de1748e34e61ac5ab5f952.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/c35c41d8cb532b09f6533b4bed7957fdd702e4d212de1748e34e61ac5ab5f952.jpg)

![c86c0caddff28552997e810154360d5e969bf043fe04b6aeaa690710b101e9f9.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/c86c0caddff28552997e810154360d5e969bf043fe04b6aeaa690710b101e9f9.jpg)

![de10261fd65a6ae25a56978d9652f84678377383fa20f7c85f98b6a63b0fd7e1.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/de10261fd65a6ae25a56978d9652f84678377383fa20f7c85f98b6a63b0fd7e1.jpg)

![e07e166b82e2b156ceab7044abe650d05a07c48d5e1db745612c119a3329a758.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/e07e166b82e2b156ceab7044abe650d05a07c48d5e1db745612c119a3329a758.jpg)

![e7f08aad113d280a12ccbe4f10cbfba43ff256431e0261a211275eb043fa6985.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/e7f08aad113d280a12ccbe4f10cbfba43ff256431e0261a211275eb043fa6985.jpg)

![f063eb907ae98d14a4d39cf73b13f954f9445986d7a328e34e3d4f9db8bdb651.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/f063eb907ae98d14a4d39cf73b13f954f9445986d7a328e34e3d4f9db8bdb651.jpg)

![ffaae93c279d7a81c751eed670ab7a5413908374974279baa4e329055606cd4c.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/images/ffaae93c279d7a81c751eed670ab7a5413908374974279baa4e329055606cd4c.jpg)

### Tables

![01c1db6b6a3dca06e2298f4164146f1f28e937248b6f79dd4690d2fb2b0dff67.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/01c1db6b6a3dca06e2298f4164146f1f28e937248b6f79dd4690d2fb2b0dff67.jpg)

![333439fa9c3976382b7ea203c8f88ce7c1bb8ec630e286c634453c2e0f6cfbab.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/333439fa9c3976382b7ea203c8f88ce7c1bb8ec630e286c634453c2e0f6cfbab.jpg)

![4e6ecc65de48bc83ab41e778a85d853a03b2c5c9d0411a8a78bd12ef0e02178c.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/4e6ecc65de48bc83ab41e778a85d853a03b2c5c9d0411a8a78bd12ef0e02178c.jpg)

![64c71990b0460e8b49fae6bc077f372199a19d9c90394b6c14c0d3d4fb268252.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/64c71990b0460e8b49fae6bc077f372199a19d9c90394b6c14c0d3d4fb268252.jpg)

![694e5e723423075f73888700bc31780e39e68bf16b4e9a9c40ed6297e3a4dbbd.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/694e5e723423075f73888700bc31780e39e68bf16b4e9a9c40ed6297e3a4dbbd.jpg)

![b0df6049ae6947ff1ed4828668931ef74a060ecdf41faeba0385e4263f768201.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/b0df6049ae6947ff1ed4828668931ef74a060ecdf41faeba0385e4263f768201.jpg)

![d32ae2b6fc71354bf339ff2fa9177f30cdc7d85ff9742ef43e06aee352eafab9.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/d32ae2b6fc71354bf339ff2fa9177f30cdc7d85ff9742ef43e06aee352eafab9.jpg)

![f71cf3ed34afb32db91133f61e1583dec9c6bdcc5de0736edaa5e6bda3de775a.jpg](../iclr_results/1561_GI-GS_%20Global%20Illumination%20Decomposition%20on%20Gaussian%20Splatting%20for%20Inverse%20Rendering/tables/f71cf3ed34afb32db91133f61e1583dec9c6bdcc5de0736edaa5e6bda3de775a.jpg)

## Real-Time Video Generation with Pyramid Attention Broadcast


### Images

![00ca415dbbb34c270092cca817cc6fb73677f0ce21f3a4590f9dc96a8b753609.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/00ca415dbbb34c270092cca817cc6fb73677f0ce21f3a4590f9dc96a8b753609.jpg)

![1f555a969dc01124a26b4ff6bbadbf7139c6dd9b90c3049220138014fd7deac6.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/1f555a969dc01124a26b4ff6bbadbf7139c6dd9b90c3049220138014fd7deac6.jpg)

![20eaa64b1780e9db1e0eb514bf442fe9953232431975b47f39f8fbce0f963d8e.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/20eaa64b1780e9db1e0eb514bf442fe9953232431975b47f39f8fbce0f963d8e.jpg)

![216ba9dca19a537e856d58d90e88ef0cd3b2482a103e74c78fcd7f2a58402f48.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/216ba9dca19a537e856d58d90e88ef0cd3b2482a103e74c78fcd7f2a58402f48.jpg)

![3e8421ff69e83d8894f20e6149a5aa5f831a90e064f5513866092737c1007029.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/3e8421ff69e83d8894f20e6149a5aa5f831a90e064f5513866092737c1007029.jpg)

![4b93a1c7e68b25488a85a5450cd2086e36a87d3802849f5d74eaab9de4a6401c.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/4b93a1c7e68b25488a85a5450cd2086e36a87d3802849f5d74eaab9de4a6401c.jpg)

![4c6343d4842c8d8573c82e04b8f6ccac525e8c3538410e6a81f0e184f4be14e0.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/4c6343d4842c8d8573c82e04b8f6ccac525e8c3538410e6a81f0e184f4be14e0.jpg)

![4f3079fcd26c7cda66b77eb9b0b0534018c6e533dc376ea81964c2e99996ce39.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/4f3079fcd26c7cda66b77eb9b0b0534018c6e533dc376ea81964c2e99996ce39.jpg)

![5c2964f860bd9d3d3764980289e3999d8393a6ca6576f92188dab6bb060dfd7e.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/5c2964f860bd9d3d3764980289e3999d8393a6ca6576f92188dab6bb060dfd7e.jpg)

![79c1105af25e2c4782f07cc4335f4e934f7db87c4bd58ce43e4d5a3f43d54559.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/79c1105af25e2c4782f07cc4335f4e934f7db87c4bd58ce43e4d5a3f43d54559.jpg)

![7fa90951b6f35a0ed1f220d9e0a8d5dac7bd403a44724d50f355b3084dfd10fc.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/7fa90951b6f35a0ed1f220d9e0a8d5dac7bd403a44724d50f355b3084dfd10fc.jpg)

![83eb4c4caddbfc08a44f27814c87093815141e5827f3e4f9ce13f77d489a9094.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/83eb4c4caddbfc08a44f27814c87093815141e5827f3e4f9ce13f77d489a9094.jpg)

![8cdb7b3b96f2f5eee22b0ac545f8af3b5cb994d827f4d853fe3620d2369b470a.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/8cdb7b3b96f2f5eee22b0ac545f8af3b5cb994d827f4d853fe3620d2369b470a.jpg)

![c6ab4f789eeda6ae11559d6625f81f7f5c1e68f84cb6d527da6f60151f82eb7f.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/c6ab4f789eeda6ae11559d6625f81f7f5c1e68f84cb6d527da6f60151f82eb7f.jpg)

![cd3d39db9d702dd3ae64332e908467db8d523f0e99e8633743d2d9b9e6756a39.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/cd3d39db9d702dd3ae64332e908467db8d523f0e99e8633743d2d9b9e6756a39.jpg)

![d19e424907154f3d326b92878d65adad2db3c4daf165c16b3540d93fb0a15b89.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/d19e424907154f3d326b92878d65adad2db3c4daf165c16b3540d93fb0a15b89.jpg)

![e57166848cc66efa7519c69aad0b5fb27b8f9c156d6bf4d8e8d11d943af80da2.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/e57166848cc66efa7519c69aad0b5fb27b8f9c156d6bf4d8e8d11d943af80da2.jpg)

![e6f64544be3464ed9a97164d2c73fd71983da9899fbf9caae82bec2901185417.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/e6f64544be3464ed9a97164d2c73fd71983da9899fbf9caae82bec2901185417.jpg)

![e73a22290d0a808e73c7eee15eb7b5d3dac1ad0cf49af19fcc64c4ff284b4fc1.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/e73a22290d0a808e73c7eee15eb7b5d3dac1ad0cf49af19fcc64c4ff284b4fc1.jpg)

![ec9fcd389b6be13da4da3e3afc5f0d098ee6905efdd4dbf36d61cc5618bb1672.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/ec9fcd389b6be13da4da3e3afc5f0d098ee6905efdd4dbf36d61cc5618bb1672.jpg)

![ed7e00fdecc4cf357176222b0015996ef378cc5aa633132102450159a1752298.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/ed7e00fdecc4cf357176222b0015996ef378cc5aa633132102450159a1752298.jpg)

![fbc1dbacabe2db94042886d11c19fbd0c3158f8b5b091b5a27f1994cee24ca09.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/fbc1dbacabe2db94042886d11c19fbd0c3158f8b5b091b5a27f1994cee24ca09.jpg)

![fd329f3329150a394e681a25daee5fbe45ce84d24fd2854cda78d2ab3b7828ef.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/images/fd329f3329150a394e681a25daee5fbe45ce84d24fd2854cda78d2ab3b7828ef.jpg)

### Tables

![077c4e4fe784b2556aa43145e527e189a57c9e926c6e2570fc0cfebc00f93e61.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/077c4e4fe784b2556aa43145e527e189a57c9e926c6e2570fc0cfebc00f93e61.jpg)

![0fbd9bf0570a5520763b02dcf1f1808e469ab03f9ec5a14c7e199e659a18d9b3.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/0fbd9bf0570a5520763b02dcf1f1808e469ab03f9ec5a14c7e199e659a18d9b3.jpg)

![177644dbacdc6e37c4e4010e87404576daf351dce3511bd6078270acbb1a5912.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/177644dbacdc6e37c4e4010e87404576daf351dce3511bd6078270acbb1a5912.jpg)

![1aff0a36dd0203b467b0a281a14fcfe058031d530673a7864845f17ed4ab5618.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/1aff0a36dd0203b467b0a281a14fcfe058031d530673a7864845f17ed4ab5618.jpg)

![1d6337d26eff7d70b836a0557302c42a00aa4d8e37dfa5866c7e58ec63d7959d.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/1d6337d26eff7d70b836a0557302c42a00aa4d8e37dfa5866c7e58ec63d7959d.jpg)

![279d64925cca7140c89a72e4613f4cc036f9afc212605dcf80155b88b9b4cbf0.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/279d64925cca7140c89a72e4613f4cc036f9afc212605dcf80155b88b9b4cbf0.jpg)

![34b472fca51e2b5908a159085dde5b9a4c6eea0ed1bc6e67ce542616aba6834d.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/34b472fca51e2b5908a159085dde5b9a4c6eea0ed1bc6e67ce542616aba6834d.jpg)

![4788ccbd88f132f29bea026204a29d04dbe042e29f3c1b76a6f5f7b63e0553b0.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/4788ccbd88f132f29bea026204a29d04dbe042e29f3c1b76a6f5f7b63e0553b0.jpg)

![550f589d00d9f59b7d35904b2f5346d73c4e811c1fe22a19245729fc71b004b2.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/550f589d00d9f59b7d35904b2f5346d73c4e811c1fe22a19245729fc71b004b2.jpg)

![6a5151b523a760b9cd69068a4e76b13796e69d47b7ae772ff4d241f8fb43b1ba.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/6a5151b523a760b9cd69068a4e76b13796e69d47b7ae772ff4d241f8fb43b1ba.jpg)

![835440273d396759e4127f3fe7c3aff19e81a18250317a243e5d86cfa5d6ce84.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/835440273d396759e4127f3fe7c3aff19e81a18250317a243e5d86cfa5d6ce84.jpg)

![848d8ed5c65f881d3732d9c13509ba2f175d6e72977aa22899cb81b0dc98bc3f.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/848d8ed5c65f881d3732d9c13509ba2f175d6e72977aa22899cb81b0dc98bc3f.jpg)

![a9a236c086c232ea41e681e38b1d97a263467149fe8dab235ef142e91595bae6.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/a9a236c086c232ea41e681e38b1d97a263467149fe8dab235ef142e91595bae6.jpg)

![d0d7644b9ddd4401232d5c3bb4ffbe4fc984a7b1eff1eff828cad2af9a85f727.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/d0d7644b9ddd4401232d5c3bb4ffbe4fc984a7b1eff1eff828cad2af9a85f727.jpg)

![e2d63b718b74dc82fbe44bd75fb2ecc60471b6015387072ff46eebca76324daa.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/e2d63b718b74dc82fbe44bd75fb2ecc60471b6015387072ff46eebca76324daa.jpg)

![f4c151b8632899e9b462cf8ebfa7458e9a4f996dabbf62cfa70ffa931990f9d8.jpg](../iclr_results/1562_Real-Time%20Video%20Generation%20with%20Pyramid%20Attention%20Broadcast/tables/f4c151b8632899e9b462cf8ebfa7458e9a4f996dabbf62cfa70ffa931990f9d8.jpg)

## HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation


### Images

![073f3c786d2cd74a198e87b1f0613103546b619786564757435d3badd6c41446.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/073f3c786d2cd74a198e87b1f0613103546b619786564757435d3badd6c41446.jpg)

![17558ef6873e20adc4009bd6e87d90528153b49aaef4df3a5ca3acb502295409.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/17558ef6873e20adc4009bd6e87d90528153b49aaef4df3a5ca3acb502295409.jpg)

![23ca339ec573e18cf697078c083e7d086f96d68220f7d857952f57f6970d0bc9.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/23ca339ec573e18cf697078c083e7d086f96d68220f7d857952f57f6970d0bc9.jpg)

![352b9f6054bb879852f594b5bd4b8291985fbc8d14b648119d6aef50c400f4d3.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/352b9f6054bb879852f594b5bd4b8291985fbc8d14b648119d6aef50c400f4d3.jpg)

![458a2d54691267580576c3ac0f167736e17e7131dcd1472b93be86fdb948538a.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/458a2d54691267580576c3ac0f167736e17e7131dcd1472b93be86fdb948538a.jpg)

![55caf33c61f500950ffd3ba5ff2fe7377aa254723ee60b15adf626d6959fd27f.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/55caf33c61f500950ffd3ba5ff2fe7377aa254723ee60b15adf626d6959fd27f.jpg)

![7a0f8edb9b475a25fb59b017f2f744d90c52a70053cb0bcb1cb0edefc2e2f794.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/7a0f8edb9b475a25fb59b017f2f744d90c52a70053cb0bcb1cb0edefc2e2f794.jpg)

![82b5a327774d1dc1fd66ab5b38b731610282ba667f98c636180bfefe52256d46.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/82b5a327774d1dc1fd66ab5b38b731610282ba667f98c636180bfefe52256d46.jpg)

![9438bf246f8d839f9b43dbedb9436e77674cac4a3b68e80ea755a22dc2c8cabd.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/9438bf246f8d839f9b43dbedb9436e77674cac4a3b68e80ea755a22dc2c8cabd.jpg)

![9ec94cd279ca4630d9545eaabed178fb1bd42f14a2a8270465188834fd928bfb.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/9ec94cd279ca4630d9545eaabed178fb1bd42f14a2a8270465188834fd928bfb.jpg)

![abafc7c766f38c6d3de8338e5a7fc66d2210388d11b2dc846bd5bbb0ea0b3ac2.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/abafc7c766f38c6d3de8338e5a7fc66d2210388d11b2dc846bd5bbb0ea0b3ac2.jpg)

![b0c843ee01c88e35527ade903b6ee1af9477ef391642f758033e7de16c5c2a67.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/b0c843ee01c88e35527ade903b6ee1af9477ef391642f758033e7de16c5c2a67.jpg)

![c8c456530a3d7f8e77d8b9b5bf08ab8b23c37c65f1444257704bb4501fed9a5e.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/c8c456530a3d7f8e77d8b9b5bf08ab8b23c37c65f1444257704bb4501fed9a5e.jpg)

![eabe3c4d53280f4801187947740186036524deed4d6fe795f2dcdbed1d44677e.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/images/eabe3c4d53280f4801187947740186036524deed4d6fe795f2dcdbed1d44677e.jpg)

### Tables

![42cc0eb514b66a945564c0d8403f092a6a5b2c079e18cacdf90aa487b464944c.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/tables/42cc0eb514b66a945564c0d8403f092a6a5b2c079e18cacdf90aa487b464944c.jpg)

![4c338e9493f3c406a633a56ecd43968c08e0bf0df0e744481d0724021a4391cb.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/tables/4c338e9493f3c406a633a56ecd43968c08e0bf0df0e744481d0724021a4391cb.jpg)

![7494e9a05d9e9e08a915e266ec12e96d69dab04b8701ecf74d8f293b84b9caaf.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/tables/7494e9a05d9e9e08a915e266ec12e96d69dab04b8701ecf74d8f293b84b9caaf.jpg)

![7dc2feba9717f55eae594c884e0df02574f1ecd686e3d90532be507156f0c502.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/tables/7dc2feba9717f55eae594c884e0df02574f1ecd686e3d90532be507156f0c502.jpg)

![ddd037c59218882487d25bfb39308ba14fa1a7cf0e2700491a942803fdc31372.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/tables/ddd037c59218882487d25bfb39308ba14fa1a7cf0e2700491a942803fdc31372.jpg)

![e539aaf8e7385a94173588cd09b2a5910c42be359455f7870e89c4d737a20f1b.jpg](../iclr_results/1563_HAMSTER_%20Hierarchical%20Action%20Models%20for%20Open-World%20Robot%20Manipulation/tables/e539aaf8e7385a94173588cd09b2a5910c42be359455f7870e89c4d737a20f1b.jpg)

## Improving Complex Reasoning with Dynamic Prompt Corruption: A Soft Prompt Optimization Approach

### Images

![0ad64d650149733cf1adbd476ecc629c3f3c1b00bfcf77d99cdd77d77bfe9031.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/0ad64d650149733cf1adbd476ecc629c3f3c1b00bfcf77d99cdd77d77bfe9031.jpg)

![70a1d10726b37ed13ee5d01534f0eedb970a047cb3919646b0e62d54af44c8c3.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/70a1d10726b37ed13ee5d01534f0eedb970a047cb3919646b0e62d54af44c8c3.jpg)

![75bb62d9316560b01e10b523c1e8f9afcf5a09459e7be1bba09ae1b08f2280ea.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/75bb62d9316560b01e10b523c1e8f9afcf5a09459e7be1bba09ae1b08f2280ea.jpg)

![85d5f76336a7bf07946ff555b6258655d97d01f58dd0ce716344eb661fa70864.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/85d5f76336a7bf07946ff555b6258655d97d01f58dd0ce716344eb661fa70864.jpg)

![b10ff10fa07f7cbc0a60f637728cca1dee923c9eaeed8284bc4ca9e72c0b00f8.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/b10ff10fa07f7cbc0a60f637728cca1dee923c9eaeed8284bc4ca9e72c0b00f8.jpg)

![b9611f4e3ea830b4675cbf5cd0aa998649bd54b88b9c898f0e3e1bdec907d866.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/b9611f4e3ea830b4675cbf5cd0aa998649bd54b88b9c898f0e3e1bdec907d866.jpg)

![f787e3b797b2dc424d8aad582168fb200b8b7c629a1234fb1449d3e101474bbb.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/f787e3b797b2dc424d8aad582168fb200b8b7c629a1234fb1449d3e101474bbb.jpg)

### Tables

![bcf3c0b7f1d360e9b6b507a92662ef1d41bc30b451dfdfaed3d3de8480494335.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/bcf3c0b7f1d360e9b6b507a92662ef1d41bc30b451dfdfaed3d3de8480494335.jpg)

![c8af5dbe82fca565deda08ebcc76c60bd92242b7fdab74e08af950858eda890b.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/c8af5dbe82fca565deda08ebcc76c60bd92242b7fdab74e08af950858eda890b.jpg)

![d5967ebb8a2a4ef24da25dcfa38cb0cb75b6381a1296a2e485de98104698ac03.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/d5967ebb8a2a4ef24da25dcfa38cb0cb75b6381a1296a2e485de98104698ac03.jpg)

![dd26ad2cf6d6faf7a13d86a6ae0f02014cc9e6aea17dcd0969068dad7ef6e3a0.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/dd26ad2cf6d6faf7a13d86a6ae0f02014cc9e6aea17dcd0969068dad7ef6e3a0.jpg)
