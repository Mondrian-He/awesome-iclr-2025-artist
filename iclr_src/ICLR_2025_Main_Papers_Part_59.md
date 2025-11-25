# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 59 of 100

## 目录 (Table of Contents)

1. [FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"](#FaithEval-Can-Your-Language-Model-Stay-Faithful-to-Context-Even-If-The-Moon-is-Made-of-Marshmallows)
2. [A Transfer Attack to Image Watermarks](#A-Transfer-Attack-to-Image-Watermarks)
3. [Trajectory-LLM: A Language-based Data Generator for Trajectory Prediction in Autonomous Driving](#Trajectory-LLM-A-Language-based-Data-Generator-for-Trajectory-Prediction-in-Autonomous-Driving)
4. [Learning Hierarchical Polynomials of Multiple Nonlinear Features](#Learning-Hierarchical-Polynomials-of-Multiple-Nonlinear-Features)
5. [Generalizable Motion Planning via Operator Learning](#Generalizable-Motion-Planning-via-Operator-Learning)
6. [Efficient Causal Decision Making with One-sided Feedback](#Efficient-Causal-Decision-Making-with-One-sided-Feedback)
7. [Certifying Language Model Robustness with Fuzzed Randomized Smoothing: An Efficient Defense Against Backdoor Attacks](#Certifying-Language-Model-Robustness-with-Fuzzed-Randomized-Smoothing-An-Efficient-Defense-Against-Backdoor-Attacks)
8. [Circuit Representation Learning with Masked Gate Modeling and Verilog-AIG Alignment](#Circuit-Representation-Learning-with-Masked-Gate-Modeling-and-Verilog-AIG-Alignment)
9. [Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation](#Polyrating-A-Cost-Effective-and-Bias-Aware-Rating-System-for-LLM-Evaluation)
10. [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](#LLaVA-Mini-Efficient-Image-and-Video-Large-Multimodal-Models-with-One-Vision-Token)
11. [Self-Supervised Diffusion Models for Electron-Aware Molecular Representation Learning](#Self-Supervised-Diffusion-Models-for-Electron-Aware-Molecular-Representation-Learning)
12. [Exploring the Design Space of Visual Context Representation in Video MLLMs](#Exploring-the-Design-Space-of-Visual-Context-Representation-in-Video-MLLMs)
13. [Fair Submodular Cover](#Fair-Submodular-Cover)
14. [L3Ms — Lagrange Large Language Models](#L3Ms-Lagrange-Large-Language-Models)
15. [The 3D-PC: a benchmark for visual perspective taking in humans and machines](#The-3D-PC-a-benchmark-for-visual-perspective-taking-in-humans-and-machines)
16. [On the Optimal Memorization Capacity of Transformers](#On-the-Optimal-Memorization-Capacity-of-Transformers)
17. [InstantSwap: Fast Customized Concept Swapping across Sharp Shape Differences](#InstantSwap-Fast-Customized-Concept-Swapping-across-Sharp-Shape-Differences)
18. [Efficient Cross-Episode Meta-RL](#Efficient-Cross-Episode-Meta-RL)
19. [MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba](#MambaPEFT-Exploring-Parameter-Efficient-Fine-Tuning-for-Mamba)
20. [Adversaries With Incentives:  A Strategic Alternative to Adversarial Robustness](#Adversaries-With-Incentives-A-Strategic-Alternative-to-Adversarial-Robustness)
21. [A Formal Framework for Understanding Length Generalization in Transformers](#A-Formal-Framework-for-Understanding-Length-Generalization-in-Transformers)
22. [Mask in the Mirror: Implicit Sparsification](#Mask-in-the-Mirror-Implicit-Sparsification)
23. [Audio Large Language Models Can Be Descriptive Speech Quality Evaluators](#Audio-Large-Language-Models-Can-Be-Descriptive-Speech-Quality-Evaluators)
24. [Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents](#Perplexity-Trap-PLM-Based-Retrievers-Overrate-Low-Perplexity-Documents)
25. [Poisson-Dirac Neural Networks for Modeling Coupled Dynamical Systems across Domains](#Poisson-Dirac-Neural-Networks-for-Modeling-Coupled-Dynamical-Systems-across-Domains)
26. [GPS: A Probabilistic Distributional Similarity with Gumbel Priors for Set-to-Set Matching](#GPS-A-Probabilistic-Distributional-Similarity-with-Gumbel-Priors-for-Set-to-Set-Matching)
27. [Multi-Task Dense Predictions via Unleashing the Power of Diffusion](#Multi-Task-Dense-Predictions-via-Unleashing-the-Power-of-Diffusion)
28. [Discovering Group Structures via Unitary Representation Learning](#Discovering-Group-Structures-via-Unitary-Representation-Learning)
29. [Newton Meets Marchenko-Pastur: Massively Parallel Second-Order Optimization with Hessian Sketching and Debiasing](#Newton-Meets-Marchenko-Pastur-Massively-Parallel-Second-Order-Optimization-with-Hessian-Sketching-and-Debiasing)
30. [Grokking at the Edge of Numerical Stability](#Grokking-at-the-Edge-of-Numerical-Stability)
31. [MELODI: Exploring Memory Compression for Long Contexts](#MELODI-Exploring-Memory-Compression-for-Long-Contexts)
32. [MIND: Math Informed syNthetic Dialogues for Pretraining LLMs](#MIND-Math-Informed-syNthetic-Dialogues-for-Pretraining-LLMs)
33. [T2V2: A Unified Non-Autoregressive Model for Speech Recognition and Synthesis via Multitask Learning](#T2V2-A-Unified-Non-Autoregressive-Model-for-Speech-Recognition-and-Synthesis-via-Multitask-Learning)
34. [FreCaS: Efficient Higher-Resolution Image Generation via Frequency-aware Cascaded Sampling](#FreCaS-Efficient-Higher-Resolution-Image-Generation-via-Frequency-aware-Cascaded-Sampling)
35. [AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation](#AvatarGO-Zero-shot-4D-Human-Object-Interaction-Generation-and-Animation)
36. [MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding](#MuirBench-A-Comprehensive-Benchmark-for-Robust-Multi-image-Understanding)
37. [A Little Goes a Long Way: Efficient Long Context Training and Inference with Partial Contexts](#A-Little-Goes-a-Long-Way-Efficient-Long-Context-Training-and-Inference-with-Partial-Contexts)

---


## FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"

### Images

![0312f1faef49e72582e5033e38ac15c993a764195488615cfae0f38e0dc86760.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/0312f1faef49e72582e5033e38ac15c993a764195488615cfae0f38e0dc86760.jpg)

![07ea16d35f4a0e9d0b3497fa3e10b960698476287f91ed0809289611f681cf16.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/07ea16d35f4a0e9d0b3497fa3e10b960698476287f91ed0809289611f681cf16.jpg)

![19b9eb018f8e6f52e61ef1d0b8adca1ee700bbebf8c31199a32024ce825fc4b3.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/19b9eb018f8e6f52e61ef1d0b8adca1ee700bbebf8c31199a32024ce825fc4b3.jpg)

![3683975bcdbcb98e8830b8c619235059e22144e1d91bba3080267a1d38bbcc95.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/3683975bcdbcb98e8830b8c619235059e22144e1d91bba3080267a1d38bbcc95.jpg)

![538aa58f57368096c3e6d50d4f579d33d71ad78b0ff17efbeceb8333a44afb24.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/538aa58f57368096c3e6d50d4f579d33d71ad78b0ff17efbeceb8333a44afb24.jpg)

![b19c1521bdd6670f0c75ce65336d78ad46e423771ef4448e4bc09c702391c98a.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/b19c1521bdd6670f0c75ce65336d78ad46e423771ef4448e4bc09c702391c98a.jpg)

![ed6fc70602d4220fa6c84b35482cefea6b48fdee9695ff8bb29e1eba1dd08136.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/ed6fc70602d4220fa6c84b35482cefea6b48fdee9695ff8bb29e1eba1dd08136.jpg)

![f293a0b5cb9962f050f8863b0a4c009356653ea06ebd78a844f176361ea044df.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/images/f293a0b5cb9962f050f8863b0a4c009356653ea06ebd78a844f176361ea044df.jpg)

### Tables

![cf843aea74e9928822b577b32fdeffe452f43d31d8b901b756cd3d7a3f602e19.jpg](../iclr_results/2157_Uncertainty%20Herding_%20One%20Active%20Learning%20Method%20for%20All%20Label%20Budgets/tables/cf843aea74e9928822b577b32fdeffe452f43d31d8b901b756cd3d7a3f602e19.jpg)

## FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"


### Images

![1bb7fa9fc66099a33a8b91d3433fe7eb68d2bb70544051c86469988d9157281d.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/1bb7fa9fc66099a33a8b91d3433fe7eb68d2bb70544051c86469988d9157281d.jpg)

![205dd237eb953a852a216629173a9e8e0c47910b5167d6c84e70c91cc265a5a5.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/205dd237eb953a852a216629173a9e8e0c47910b5167d6c84e70c91cc265a5a5.jpg)

![2e9fbfe2e6837313b146df57e13a35e6a6a7fc728807d5fe174af10575ec1d7a.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/2e9fbfe2e6837313b146df57e13a35e6a6a7fc728807d5fe174af10575ec1d7a.jpg)

![3c49e8654de7b03f0068050dd8d9a0708d5b171cbf2a2c291f4d7673c7aaf053.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/3c49e8654de7b03f0068050dd8d9a0708d5b171cbf2a2c291f4d7673c7aaf053.jpg)

![463a6e808eb02b6bac3865c2ca7d87c206a43f623e05c7c5755ff99df0f6a5d8.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/463a6e808eb02b6bac3865c2ca7d87c206a43f623e05c7c5755ff99df0f6a5d8.jpg)

![47ed5bc010f9ef16fbe76ca62f14542e3b09bf538b77b205bb5b70404fc41dee.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/47ed5bc010f9ef16fbe76ca62f14542e3b09bf538b77b205bb5b70404fc41dee.jpg)

![6135e392b0143f321439f7fcc00335ac9fe93175b167b11d0af33077d2bebd0c.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/6135e392b0143f321439f7fcc00335ac9fe93175b167b11d0af33077d2bebd0c.jpg)

![63c4ad9c639ffb3db3c54f75673ed5cb2d20731970a705410708e1f028d714e7.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/63c4ad9c639ffb3db3c54f75673ed5cb2d20731970a705410708e1f028d714e7.jpg)

![73e16f65052376a46b643d665537194b6143df438d7e171bf41cbac6666a5c9d.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/73e16f65052376a46b643d665537194b6143df438d7e171bf41cbac6666a5c9d.jpg)

![7bc7648682b2cca45096314aa92344d788bad4c701abdc7d79e0f4004dde3b4f.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/7bc7648682b2cca45096314aa92344d788bad4c701abdc7d79e0f4004dde3b4f.jpg)

![86f5fe4fbe54ede31559a8961ef14715ad298f1bf36ac4171c4e8b3cf97f1e64.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/86f5fe4fbe54ede31559a8961ef14715ad298f1bf36ac4171c4e8b3cf97f1e64.jpg)

![89118330c6470f4a9c7d3e5cc146cd5eaefaf3f1df3c676b682d8a2441dd4b38.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/89118330c6470f4a9c7d3e5cc146cd5eaefaf3f1df3c676b682d8a2441dd4b38.jpg)

![9a4b943e7d9f317ce502e550afe1b9b96142f7d179100bb6aa64f8138626bbf8.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/9a4b943e7d9f317ce502e550afe1b9b96142f7d179100bb6aa64f8138626bbf8.jpg)

![9c4eeeee9e3cf911f2e5921b632aab0e32281e96e3ac6d7e7821ec1e99452f9e.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/9c4eeeee9e3cf911f2e5921b632aab0e32281e96e3ac6d7e7821ec1e99452f9e.jpg)

![a047277b484a3f91ac637af0d39e95ec09fd50f803b286617d4a0e45c9d8c44d.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/a047277b484a3f91ac637af0d39e95ec09fd50f803b286617d4a0e45c9d8c44d.jpg)

![a19d7006654aaa5b792bb74f8b967c452b520456cc88d4f64db6602821fc6ba7.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/a19d7006654aaa5b792bb74f8b967c452b520456cc88d4f64db6602821fc6ba7.jpg)

![ce15b385b97b49ccae37d7c3294ad41e2a83730239267dfdc5a0dda36490f365.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/ce15b385b97b49ccae37d7c3294ad41e2a83730239267dfdc5a0dda36490f365.jpg)

![e94b0175d0d96fe7dffddac6197a3481c8a603e46842ba742a7e42beb6d26775.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/images/e94b0175d0d96fe7dffddac6197a3481c8a603e46842ba742a7e42beb6d26775.jpg)

### Tables

![0f015eefa8fb33d8f8fa4d63f905800c217f282c9e7dce924a96485d492f284e.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/0f015eefa8fb33d8f8fa4d63f905800c217f282c9e7dce924a96485d492f284e.jpg)

![228bc9ca7dc97c86bcd6e14db67b6dac5642c6190b429494a3696dfae6526b72.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/228bc9ca7dc97c86bcd6e14db67b6dac5642c6190b429494a3696dfae6526b72.jpg)

![2ff89c2f3ffa29e06715e8af594f14014831d4636d97ec9e34d80933c53b6814.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/2ff89c2f3ffa29e06715e8af594f14014831d4636d97ec9e34d80933c53b6814.jpg)

![367578295180983eb69bb4913800e6ce40a894fb95004bf09d78b2b5735896bf.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/367578295180983eb69bb4913800e6ce40a894fb95004bf09d78b2b5735896bf.jpg)

![80fd305fc3d33870960b456813654a3376316b29845673be0580b2b9e8e44fb4.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/80fd305fc3d33870960b456813654a3376316b29845673be0580b2b9e8e44fb4.jpg)

![afc9d0df60dd9d268a20e2942a8468f38acb424cf347d192a739f9d69829dcf7.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/afc9d0df60dd9d268a20e2942a8468f38acb424cf347d192a739f9d69829dcf7.jpg)

![bf24adc524beca460549bd8b19d8dd414c173fe2163a49d27f6b2472d1cdee09.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/bf24adc524beca460549bd8b19d8dd414c173fe2163a49d27f6b2472d1cdee09.jpg)

![de02599791337067e54616e252cc511fc0ac950e654b75d9d494f3283ae053f3.jpg](../iclr_results/2158_FaithEval_%20Can%20Your%20Language%20Model%20Stay%20Faithful%20to%20Context%2C%20Even%20If%20_The%20Moon%20is%20Made%20of%20Marshmallo/tables/de02599791337067e54616e252cc511fc0ac950e654b75d9d494f3283ae053f3.jpg)

## A Transfer Attack to Image Watermarks


### Images

![0fac82a11e3f51a51e8e95ec45e72002d9a9446b3754166b28bf9a271bc6f1cc.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/0fac82a11e3f51a51e8e95ec45e72002d9a9446b3754166b28bf9a271bc6f1cc.jpg)

![1509c90159c333f88c1876596d0a2728795a7797c0186b1e1eb248be393f0180.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/1509c90159c333f88c1876596d0a2728795a7797c0186b1e1eb248be393f0180.jpg)

![1794837319a8b6c3ac7d544cce34432fef17db60614434addf05ff265229dbb7.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/1794837319a8b6c3ac7d544cce34432fef17db60614434addf05ff265229dbb7.jpg)

![21459a9755ddf01691100834815c43992116f11eba8cdbe6c4cf4f975261e3d0.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/21459a9755ddf01691100834815c43992116f11eba8cdbe6c4cf4f975261e3d0.jpg)

![40dafa0491cd503fdd00e691b921d61b869c6aacaa697a9f8b3c5f16a68dccf3.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/40dafa0491cd503fdd00e691b921d61b869c6aacaa697a9f8b3c5f16a68dccf3.jpg)

![521cb42c68e62d8abb005e319fc00a8f67181cdf79fe079875eedcde9351fe76.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/521cb42c68e62d8abb005e319fc00a8f67181cdf79fe079875eedcde9351fe76.jpg)

![8260648428a7a2742b6f9aecbceda3153890a53ed84da961bdcc1d0208815292.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/8260648428a7a2742b6f9aecbceda3153890a53ed84da961bdcc1d0208815292.jpg)

![9c2ca17ac79c955f87ad61886f897bb2d41dab52bdbfe9dfc400744a5e111636.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/9c2ca17ac79c955f87ad61886f897bb2d41dab52bdbfe9dfc400744a5e111636.jpg)

![9f25eaafbec1f029adb795bd2f98de024f96078c92e7037188a66cd4a7eead82.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/9f25eaafbec1f029adb795bd2f98de024f96078c92e7037188a66cd4a7eead82.jpg)

![a4113c9fe13cd11b3d49050d66d5775f596514f09ac83c5ea0ee16ccb8814c77.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/a4113c9fe13cd11b3d49050d66d5775f596514f09ac83c5ea0ee16ccb8814c77.jpg)

![dce5d7b64cd745e16a2bf8219e9fa538ce01eebbee98f243e2580e3817ff4b24.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/dce5d7b64cd745e16a2bf8219e9fa538ce01eebbee98f243e2580e3817ff4b24.jpg)

![ec086269d3f4a4bb06ff15cd7f254d15922d062acbce3c396ee0f440f2407cdf.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/ec086269d3f4a4bb06ff15cd7f254d15922d062acbce3c396ee0f440f2407cdf.jpg)

![f4079c28b23bf49a372227fc41020ef7acb1f5f3150ca56f185feedaa219dc20.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/f4079c28b23bf49a372227fc41020ef7acb1f5f3150ca56f185feedaa219dc20.jpg)

![fd3ca37ed19494b3467fa494aad92f4c1b1bd21ebf9d48ab724b6dc78f7ddc3b.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/fd3ca37ed19494b3467fa494aad92f4c1b1bd21ebf9d48ab724b6dc78f7ddc3b.jpg)

![ff916883bc7d49904e7373db9d35bc8726ad8b6262c919411a400a680082dfab.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/images/ff916883bc7d49904e7373db9d35bc8726ad8b6262c919411a400a680082dfab.jpg)

### Tables

![5738729763d63fc380e5c9483b70fba200572a2e29e0e2d984dc7ac32b01d7a2.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/tables/5738729763d63fc380e5c9483b70fba200572a2e29e0e2d984dc7ac32b01d7a2.jpg)

![f99f2b6a4eb57983ae796974323a53d833c28ce49e0f77e4006c07c47dba3d34.jpg](../iclr_results/2159_A%20Transfer%20Attack%20to%20Image%20Watermarks/tables/f99f2b6a4eb57983ae796974323a53d833c28ce49e0f77e4006c07c47dba3d34.jpg)

## Trajectory-LLM: A Language-based Data Generator for Trajectory Prediction in Autonomous Driving


### Images

![15ba102febbbb4271561d7327cdb16ad8e2aa6ec4a9b97ee23535f06bb54a8ab.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/images/15ba102febbbb4271561d7327cdb16ad8e2aa6ec4a9b97ee23535f06bb54a8ab.jpg)

![2879ad1007e3da1e982ba0d773ca63eaf88a2b494349c74ea9f10f51614b64d6.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/images/2879ad1007e3da1e982ba0d773ca63eaf88a2b494349c74ea9f10f51614b64d6.jpg)

![35f05adf7a1245ece22c317d0d33029f9f6766f82f4d69c26fed568b2ff4fac4.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/images/35f05adf7a1245ece22c317d0d33029f9f6766f82f4d69c26fed568b2ff4fac4.jpg)

![6c23efd1b881decb07395e93761cca2b732c7d13352331584f1dc30c9725a48d.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/images/6c23efd1b881decb07395e93761cca2b732c7d13352331584f1dc30c9725a48d.jpg)

![cb1966d80c29b8729ad2354cbc426ade684fdf1fc98e8000c3b68f2fd5458949.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/images/cb1966d80c29b8729ad2354cbc426ade684fdf1fc98e8000c3b68f2fd5458949.jpg)

![f787471f31dbb8a48ef45e4e820befe96423c804c452216b4bff176edc3d655f.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/images/f787471f31dbb8a48ef45e4e820befe96423c804c452216b4bff176edc3d655f.jpg)

### Tables

![3fdf5dcf47017038517a2af7add380f9c98fe9f493551c90076efd1344faae79.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/tables/3fdf5dcf47017038517a2af7add380f9c98fe9f493551c90076efd1344faae79.jpg)

![5fbcc9c5fd7b5900e8bf9e83f0cf11b96a054c7744e47239041fe7aafbaf7204.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/tables/5fbcc9c5fd7b5900e8bf9e83f0cf11b96a054c7744e47239041fe7aafbaf7204.jpg)

![e215d24add838f23e6a36ebb572c928f156d160e38e2cacf6def1bc43da593da.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/tables/e215d24add838f23e6a36ebb572c928f156d160e38e2cacf6def1bc43da593da.jpg)

![e89fa39a71120c6623a8c154567b775a9ccdd5bf76ffc84a7fa4735c8a5629a5.jpg](../iclr_results/2160_Trajectory-LLM_%20A%20Language-based%20Data%20Generator%20for%20Trajectory%20Prediction%20in%20Autonomous%20Driving/tables/e89fa39a71120c6623a8c154567b775a9ccdd5bf76ffc84a7fa4735c8a5629a5.jpg)

## Learning Hierarchical Polynomials of Multiple Nonlinear Features


### Images

![0b63d8350e2dc452426d83b803bff9aee198951addb133c9408ecaddebc840f0.jpg](../iclr_results/2161_Learning%20Hierarchical%20Polynomials%20of%20Multiple%20Nonlinear%20Features/images/0b63d8350e2dc452426d83b803bff9aee198951addb133c9408ecaddebc840f0.jpg)

![7d62066eb3d66b2939619db1816b577656671ffa65cf174de3a92ce5ecfa828f.jpg](../iclr_results/2161_Learning%20Hierarchical%20Polynomials%20of%20Multiple%20Nonlinear%20Features/images/7d62066eb3d66b2939619db1816b577656671ffa65cf174de3a92ce5ecfa828f.jpg)

![beeb8ad68d1da24ecdbb4085de6b5dae66e6337581a12dc44706e6186e94148f.jpg](../iclr_results/2161_Learning%20Hierarchical%20Polynomials%20of%20Multiple%20Nonlinear%20Features/images/beeb8ad68d1da24ecdbb4085de6b5dae66e6337581a12dc44706e6186e94148f.jpg)

![c0bbf250eef5163d6a5c68b53f56bd72c3bad066304a1e8114da8b496ed11ad3.jpg](../iclr_results/2161_Learning%20Hierarchical%20Polynomials%20of%20Multiple%20Nonlinear%20Features/images/c0bbf250eef5163d6a5c68b53f56bd72c3bad066304a1e8114da8b496ed11ad3.jpg)

## Generalizable Motion Planning via Operator Learning


### Images

![0329c78c92f5d0939fe56832bccac2900dac5c232d1f9fb594dee66605f5a2a8.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/0329c78c92f5d0939fe56832bccac2900dac5c232d1f9fb594dee66605f5a2a8.jpg)

![058a87e8aa3a07ef5395efceee1ff18460e72463cf959947d74db039a66aa78a.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/058a87e8aa3a07ef5395efceee1ff18460e72463cf959947d74db039a66aa78a.jpg)

![08988b3bc85753915935fb2d1a3acecc1e1036a2e0ecdbccbdb434e1df3be22c.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/08988b3bc85753915935fb2d1a3acecc1e1036a2e0ecdbccbdb434e1df3be22c.jpg)

![0cd1f8818fc1e6a39692a00bad85d6d3205e0eeb41929eb1f3c3bdffef3c27dc.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/0cd1f8818fc1e6a39692a00bad85d6d3205e0eeb41929eb1f3c3bdffef3c27dc.jpg)

![21a04cd636fe6d213cbdaabe7dd91c1d255d3f323333efe401646c6aa1fe28e0.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/21a04cd636fe6d213cbdaabe7dd91c1d255d3f323333efe401646c6aa1fe28e0.jpg)

![35ba0243f49903efafc0e8a190fd0b26304b0804c071239280c20f121b3eb2c9.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/35ba0243f49903efafc0e8a190fd0b26304b0804c071239280c20f121b3eb2c9.jpg)

![88f0c0663b9ce52ad4dcb3a055ffac6bcd12fd211e554f17b9d51187efb059f1.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/88f0c0663b9ce52ad4dcb3a055ffac6bcd12fd211e554f17b9d51187efb059f1.jpg)

![9dd8ff6c41679a5ca42f1421419b9313ae7ca4b3e8765af71c0979f82d5b3f77.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/9dd8ff6c41679a5ca42f1421419b9313ae7ca4b3e8765af71c0979f82d5b3f77.jpg)

![c750af3a7a9f255b3701ae3c0a4763c5cc2f082ed236b1068405f0462e12cee4.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/c750af3a7a9f255b3701ae3c0a4763c5cc2f082ed236b1068405f0462e12cee4.jpg)

![dbf04bf1a561fe3a5332604894b18a473a969273bc154f56d6f84acf08ad6b1a.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/dbf04bf1a561fe3a5332604894b18a473a969273bc154f56d6f84acf08ad6b1a.jpg)

![e6eae1de0823803994de2cad560f78a09a1917d7c8ca79ce1395f23840a11f3a.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/images/e6eae1de0823803994de2cad560f78a09a1917d7c8ca79ce1395f23840a11f3a.jpg)

### Tables

![0b4bacff8450f3139a820c63e3f44e96489a15b437ceeb9f2e5e6d9a613866a0.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/tables/0b4bacff8450f3139a820c63e3f44e96489a15b437ceeb9f2e5e6d9a613866a0.jpg)

![4b31e10854d5588bf7cb74da67e03c10a4e637c082c1f7df6609812103632f08.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/tables/4b31e10854d5588bf7cb74da67e03c10a4e637c082c1f7df6609812103632f08.jpg)

![554036cc46c60f6006cf00d0d3062878fc98b69549f0c1c5f8b71e8ef64d55af.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/tables/554036cc46c60f6006cf00d0d3062878fc98b69549f0c1c5f8b71e8ef64d55af.jpg)

![75e65e762916399276c67d87328c271efb362587e19f666c7f2c666dca622d1a.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/tables/75e65e762916399276c67d87328c271efb362587e19f666c7f2c666dca622d1a.jpg)

![b3f578f343b2f8a6927f2b702bf87894d9f638163485ebb57c08577526d9c04a.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/tables/b3f578f343b2f8a6927f2b702bf87894d9f638163485ebb57c08577526d9c04a.jpg)

![c3a5f1019988be8943de8021e1f0d152ed1aaae8041dd5d58b89eb93d94a553c.jpg](../iclr_results/2162_Generalizable%20Motion%20Planning%20via%20Operator%20Learning/tables/c3a5f1019988be8943de8021e1f0d152ed1aaae8041dd5d58b89eb93d94a553c.jpg)

## Efficient Causal Decision Making with One-sided Feedback


### Images

![618571644b71f5ced3299fd0980c91efaced31ca8731056fb572d2e486b5638f.jpg](../iclr_results/2163_Efficient%20Causal%20Decision%20Making%20with%20One-sided%20Feedback/images/618571644b71f5ced3299fd0980c91efaced31ca8731056fb572d2e486b5638f.jpg)

![d2c626b06f83440f8b3c19ff5d1a47c2115593001416973f2dd350460fd6a9da.jpg](../iclr_results/2163_Efficient%20Causal%20Decision%20Making%20with%20One-sided%20Feedback/images/d2c626b06f83440f8b3c19ff5d1a47c2115593001416973f2dd350460fd6a9da.jpg)

### Tables

![66b813472f3f425428f8413ce69df21da49e49c2f6a264661493de6df4654742.jpg](../iclr_results/2163_Efficient%20Causal%20Decision%20Making%20with%20One-sided%20Feedback/tables/66b813472f3f425428f8413ce69df21da49e49c2f6a264661493de6df4654742.jpg)

![6e84ed83c63eb700204a35a6f7ca8e5dd4a27c2e3d9fd4a1878e813fd129d3b9.jpg](../iclr_results/2163_Efficient%20Causal%20Decision%20Making%20with%20One-sided%20Feedback/tables/6e84ed83c63eb700204a35a6f7ca8e5dd4a27c2e3d9fd4a1878e813fd129d3b9.jpg)

## Certifying Language Model Robustness with Fuzzed Randomized Smoothing: An Efficient Defense Against Backdoor Attacks


### Images

![201de20c2d81ac55234595bcc85cd857641c89ef71613e4f746dcfa06c725efd.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/images/201de20c2d81ac55234595bcc85cd857641c89ef71613e4f746dcfa06c725efd.jpg)

### Tables

![10dcd60fb275b3db2c999b5ebf3a8688ac89c52d52cc25fe0fc8a46e2fc77fe3.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/10dcd60fb275b3db2c999b5ebf3a8688ac89c52d52cc25fe0fc8a46e2fc77fe3.jpg)

![194bde431a63ce2cd536acc6f8c99b62e857e81a71f6f039dc6a8d33e7f8f9a1.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/194bde431a63ce2cd536acc6f8c99b62e857e81a71f6f039dc6a8d33e7f8f9a1.jpg)

![2b73b2d7816a0883485630215c021db5d2e21ae2e049abe8f38cf69deef2d811.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/2b73b2d7816a0883485630215c021db5d2e21ae2e049abe8f38cf69deef2d811.jpg)

![312fadac099c41355e32aab90b42c0b516de348655c65ad346002118221725f2.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/312fadac099c41355e32aab90b42c0b516de348655c65ad346002118221725f2.jpg)

![4272c93d50819e2e6020cd802367235db5b85d9107b6541b72038e191c50fbf6.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/4272c93d50819e2e6020cd802367235db5b85d9107b6541b72038e191c50fbf6.jpg)

![70285475935f8a023ce03176c6f02047847adb8e7e5c1780a41795c667ebd100.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/70285475935f8a023ce03176c6f02047847adb8e7e5c1780a41795c667ebd100.jpg)

![862d513f7407b51136d78690bc9c8ee4c5ba2bc7b3ba3bcee91f1a99ae5a4d84.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/862d513f7407b51136d78690bc9c8ee4c5ba2bc7b3ba3bcee91f1a99ae5a4d84.jpg)

![9c678c00f0785a533d8c750d4fd0d215857d1e8574cc288726b221909306b785.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/9c678c00f0785a533d8c750d4fd0d215857d1e8574cc288726b221909306b785.jpg)

![a5315db455fdeb4ddf23710766ae0357afbe201826ad6b3b5ed2b080310a0817.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/a5315db455fdeb4ddf23710766ae0357afbe201826ad6b3b5ed2b080310a0817.jpg)

![e7f3240f9573250219f1c529b57f276c4b081db50f4a98d5c416d2af00a77dc1.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/e7f3240f9573250219f1c529b57f276c4b081db50f4a98d5c416d2af00a77dc1.jpg)

![e9aab2cbbfaba0da01f3e3dad4ec26b3b3d77e58542326fb96c4fcdeb2f11250.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/e9aab2cbbfaba0da01f3e3dad4ec26b3b3d77e58542326fb96c4fcdeb2f11250.jpg)

![f83c8163b76735fcedbbca5e1f4970c9cee8e1cf7d53395534357db351a20038.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/f83c8163b76735fcedbbca5e1f4970c9cee8e1cf7d53395534357db351a20038.jpg)

![fc76e8d7358fb7767678be12349966fa542ffb72ad0f74ef6b9986e60296ac17.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/fc76e8d7358fb7767678be12349966fa542ffb72ad0f74ef6b9986e60296ac17.jpg)

![ff0eb1c6e88d6d243628bb5113765da299ebe92e25db050b62695ef4d9552823.jpg](../iclr_results/2164_Certifying%20Language%20Model%20Robustness%20with%20Fuzzed%20Randomized%20Smoothing_%20An%20Efficient%20Defense%20Against%20/tables/ff0eb1c6e88d6d243628bb5113765da299ebe92e25db050b62695ef4d9552823.jpg)

## Circuit Representation Learning with Masked Gate Modeling and Verilog-AIG Alignment


### Images

![0f469655fde0bb32f9e7b33175b4e21dee633b2b9d13e154d79ad26f4e77b294.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/images/0f469655fde0bb32f9e7b33175b4e21dee633b2b9d13e154d79ad26f4e77b294.jpg)

![1057b494e1e9555181fafc972859daa7108a4374e519aa80cac360e2dfc4ba79.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/images/1057b494e1e9555181fafc972859daa7108a4374e519aa80cac360e2dfc4ba79.jpg)

![806d58a04ceb70b5781f9e428b368d5f47ed49886c007f4118c941f8b78c96cc.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/images/806d58a04ceb70b5781f9e428b368d5f47ed49886c007f4118c941f8b78c96cc.jpg)

![af33347963b1c87a7fce54abe2f6a5488b1d2f5695721cfbd2e89b4c5b3ad671.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/images/af33347963b1c87a7fce54abe2f6a5488b1d2f5695721cfbd2e89b4c5b3ad671.jpg)

![d58cc931ecec74f6082063a1c7e95e417cf102ef15bebfd8a32d15e22b9550cb.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/images/d58cc931ecec74f6082063a1c7e95e417cf102ef15bebfd8a32d15e22b9550cb.jpg)

### Tables

![2a5d3a0cd3cd6274a47706aad5203593f6656ea530fe7fd1c48ddf822aca8070.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/2a5d3a0cd3cd6274a47706aad5203593f6656ea530fe7fd1c48ddf822aca8070.jpg)

![3556232e52b5f68937f24ef30e999a9228f3f8b3508363d50e8cf37fcc2f057a.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/3556232e52b5f68937f24ef30e999a9228f3f8b3508363d50e8cf37fcc2f057a.jpg)

![4c34ade14c48dadea0b693267352eb8256c3ddf2674e0088c4110d7690432b08.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/4c34ade14c48dadea0b693267352eb8256c3ddf2674e0088c4110d7690432b08.jpg)

![68e9dfdf117f9518c7707767bb46c3b976bc3d655b3261db3df16dce38028fab.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/68e9dfdf117f9518c7707767bb46c3b976bc3d655b3261db3df16dce38028fab.jpg)

![700852aa3e9592e97e1e5fcb11adbff38c773fdb89a872542f6b5a3a842e103d.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/700852aa3e9592e97e1e5fcb11adbff38c773fdb89a872542f6b5a3a842e103d.jpg)

![8398dd3690d1428a0a45d829768e7daa47585682272c34b806d20f1beb3089e4.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/8398dd3690d1428a0a45d829768e7daa47585682272c34b806d20f1beb3089e4.jpg)

![90baad310fa140682fb202b2c50a321079da880f51e7d6fc1950d823ddd12bd0.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/90baad310fa140682fb202b2c50a321079da880f51e7d6fc1950d823ddd12bd0.jpg)

![db4f0d0d644a71f687064d1e2f1fb7fd83c42f9fc8540ffbe4ea931f37590868.jpg](../iclr_results/2165_Circuit%20Representation%20Learning%20with%20Masked%20Gate%20Modeling%20and%20Verilog-AIG%20Alignment/tables/db4f0d0d644a71f687064d1e2f1fb7fd83c42f9fc8540ffbe4ea931f37590868.jpg)

## Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation


### Images

![0319389b4ac8a2ee29b4abf43f44628228be124638d177241def3ce922e080cb.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/images/0319389b4ac8a2ee29b4abf43f44628228be124638d177241def3ce922e080cb.jpg)

![3e262775fac552a7f537721333f91058cd8d2fe33eae7729463f2c7368f6a058.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/images/3e262775fac552a7f537721333f91058cd8d2fe33eae7729463f2c7368f6a058.jpg)

![63a3539228d055d73bc4bea24a9c0286cf8e5d157b2662f4f8f211c593a91c8d.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/images/63a3539228d055d73bc4bea24a9c0286cf8e5d157b2662f4f8f211c593a91c8d.jpg)

![69b522c63005c04e949a6f85080276541104daebc0ae60db9c8ec907fdd35bbc.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/images/69b522c63005c04e949a6f85080276541104daebc0ae60db9c8ec907fdd35bbc.jpg)

![dec7599a48272254fb0fce1d9080addec1b55b24e3d327ec038805dd58b7bc8b.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/images/dec7599a48272254fb0fce1d9080addec1b55b24e3d327ec038805dd58b7bc8b.jpg)

### Tables

![04407bc0859afa116e8107e64b3236d1e7d9d397752b97b8d28c3ae69906e9de.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/04407bc0859afa116e8107e64b3236d1e7d9d397752b97b8d28c3ae69906e9de.jpg)

![248bbf746bdced9db88a0e71a6e63fea1fb8dd528205ea820a65ca89d4e74049.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/248bbf746bdced9db88a0e71a6e63fea1fb8dd528205ea820a65ca89d4e74049.jpg)

![4970ee9206dbea20ab47bc950892b55c2d7b7d0c4e3d36d3f0add49593b31520.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/4970ee9206dbea20ab47bc950892b55c2d7b7d0c4e3d36d3f0add49593b31520.jpg)

![4c62cacc509af342540163eff476ee1bae51c8b72f1e6ce369da6607f8e8ad1c.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/4c62cacc509af342540163eff476ee1bae51c8b72f1e6ce369da6607f8e8ad1c.jpg)

![5490fcbbeca6d11d4b882466f5c9792d826506c8505e08fc57921969c73c16d1.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/5490fcbbeca6d11d4b882466f5c9792d826506c8505e08fc57921969c73c16d1.jpg)

![5b2292df619f09a4be0415485f3bb4a564ef355fd37acf1cfe6a28d936360e16.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/5b2292df619f09a4be0415485f3bb4a564ef355fd37acf1cfe6a28d936360e16.jpg)

![60e2a5dab69110fdde51fa2b8e9eb2bb6b3ca01e85eb59b451ec4f1058bd4657.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/60e2a5dab69110fdde51fa2b8e9eb2bb6b3ca01e85eb59b451ec4f1058bd4657.jpg)

![76488689e872ead7168f54ae255d531afc92be537207947317d60e476415d7eb.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/76488689e872ead7168f54ae255d531afc92be537207947317d60e476415d7eb.jpg)

![85500aa04b4736d3700c93107eabc11ab10539166cb1ec999963ab04d4287991.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/85500aa04b4736d3700c93107eabc11ab10539166cb1ec999963ab04d4287991.jpg)

![856cab541991057a8357410c37f3d0db792f0f5927e16254bf81cc8716895712.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/856cab541991057a8357410c37f3d0db792f0f5927e16254bf81cc8716895712.jpg)

![9cf3656786a810b04dabcf6db543cfbd0779825c48142f096137b0271e4ae0a8.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/9cf3656786a810b04dabcf6db543cfbd0779825c48142f096137b0271e4ae0a8.jpg)

![b55ed35e032d7ae0928d488c6efa21590990cc9685a897f5c2e6e486714125ed.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/b55ed35e032d7ae0928d488c6efa21590990cc9685a897f5c2e6e486714125ed.jpg)

![cc4a5611c98ab603b5fdc9ef643264d859a2cb2d7336e831a490a91c7d0d5ee8.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/cc4a5611c98ab603b5fdc9ef643264d859a2cb2d7336e831a490a91c7d0d5ee8.jpg)

![d219a1f77d3836e9449a0dca56a32a2bb6bc85edbb012b3bf3bfc4b7d53295a5.jpg](../iclr_results/2166_Polyrating_%20A%20Cost-Effective%20and%20Bias-Aware%20Rating%20System%20for%20LLM%20Evaluation/tables/d219a1f77d3836e9449a0dca56a32a2bb6bc85edbb012b3bf3bfc4b7d53295a5.jpg)

## LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token


### Images

![105acc131dce50c232c480ecce5669c13fc0eff9635eefe4f24de4d05a929a59.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/105acc131dce50c232c480ecce5669c13fc0eff9635eefe4f24de4d05a929a59.jpg)

![10746a43a305886644c51974e819f7d3ca729674a561f23a6df85f1053958e23.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/10746a43a305886644c51974e819f7d3ca729674a561f23a6df85f1053958e23.jpg)

![13d29f5543c0c948168ee187e47c98bda28a23845d998767e6446a95b846c1c4.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/13d29f5543c0c948168ee187e47c98bda28a23845d998767e6446a95b846c1c4.jpg)

![253a4fb5a574acdd963b04c5690bb2ebdb963a3599f12788af10761c945027e0.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/253a4fb5a574acdd963b04c5690bb2ebdb963a3599f12788af10761c945027e0.jpg)

![28cb93de6ee95638e4fc37e1ecb2e3a3e16b72e4c5e78f3e12c78648fb63a104.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/28cb93de6ee95638e4fc37e1ecb2e3a3e16b72e4c5e78f3e12c78648fb63a104.jpg)

![31fa11440b4487db764019a720d97e4e23fa33b439e65abc5f4c88161baa22c7.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/31fa11440b4487db764019a720d97e4e23fa33b439e65abc5f4c88161baa22c7.jpg)

![4089fe9f4da00175c2f9f658b2c8ac99d1d16992111981820550b2d51595e24b.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/4089fe9f4da00175c2f9f658b2c8ac99d1d16992111981820550b2d51595e24b.jpg)

![475202978a647c4860d3ef50e712615ebd5066d208171d5e933353725ff6ca52.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/475202978a647c4860d3ef50e712615ebd5066d208171d5e933353725ff6ca52.jpg)

![5ed0aea388a8d01e3c78c6b4efec903c1553c43bbd8b2860f2b015ac99c0064e.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/5ed0aea388a8d01e3c78c6b4efec903c1553c43bbd8b2860f2b015ac99c0064e.jpg)

![608aa56ce9127bdf1e6627006bf41c4ba7b7d085cf99c64f3cee3daf6f2ebbe4.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/608aa56ce9127bdf1e6627006bf41c4ba7b7d085cf99c64f3cee3daf6f2ebbe4.jpg)

![7af5df8424cac3423e23f4527a7808e44808d69cb93657d383fa1f7b1a5ea90c.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/7af5df8424cac3423e23f4527a7808e44808d69cb93657d383fa1f7b1a5ea90c.jpg)

![8223d4be98e27257b92aef34f7514ccc060c3b9b8e64b0709af08e2ddd868203.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/8223d4be98e27257b92aef34f7514ccc060c3b9b8e64b0709af08e2ddd868203.jpg)

![b2db0ac0e7a7221df1cfba1164e3a301a27500dde91f1f3b401c2cd0d948c5a6.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/b2db0ac0e7a7221df1cfba1164e3a301a27500dde91f1f3b401c2cd0d948c5a6.jpg)

![b623c3729da855a4f51bb20950827116b5198ec356e36c6ae73176d4ae951860.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/b623c3729da855a4f51bb20950827116b5198ec356e36c6ae73176d4ae951860.jpg)

![bb7ecb5a5aafdac32c5c37af8344c29348b26ca3fe8f0e4595681c42f52855fa.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/bb7ecb5a5aafdac32c5c37af8344c29348b26ca3fe8f0e4595681c42f52855fa.jpg)

![bd76ceaadfcc0327cba7d7e2e87c50783181609c2db945c18fda71aa10ee0c86.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/bd76ceaadfcc0327cba7d7e2e87c50783181609c2db945c18fda71aa10ee0c86.jpg)

![cd33823a8b664cda74a36d6d9454244b6bd53a592f6551116bc76fc1a93bdf68.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/cd33823a8b664cda74a36d6d9454244b6bd53a592f6551116bc76fc1a93bdf68.jpg)

![cea4433dba1ef351fcc0aac32624071a8588cb77a7f80388af932ce3cb0a9e91.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/cea4433dba1ef351fcc0aac32624071a8588cb77a7f80388af932ce3cb0a9e91.jpg)

![e0075a4a72e7ef5f0903e75257efb3ab0593b8b44d392d3bb8e4072e31172b87.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/e0075a4a72e7ef5f0903e75257efb3ab0593b8b44d392d3bb8e4072e31172b87.jpg)

![e268bb589b6cb13fede75cd84ca570be5d1a2e276d0c3b370f56a8505b2af77c.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/e268bb589b6cb13fede75cd84ca570be5d1a2e276d0c3b370f56a8505b2af77c.jpg)

![f74be7d72c45be8794865577679239b8dc36509c8b2f3793f028e06055b7f11a.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/images/f74be7d72c45be8794865577679239b8dc36509c8b2f3793f028e06055b7f11a.jpg)

### Tables

![0b17664abdd5e52a08534e661164020234b18474bf7bae8f13ce99bfb205bb32.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/0b17664abdd5e52a08534e661164020234b18474bf7bae8f13ce99bfb205bb32.jpg)

![0f1e6ae413a36f92c50b00e8f6366f62b17ede1667a54ffb6c9decba9ffa6ae9.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/0f1e6ae413a36f92c50b00e8f6366f62b17ede1667a54ffb6c9decba9ffa6ae9.jpg)

![28ec832fad76853aadfcc683810ec74033b99df89cb2ba28776f0bee1b32783b.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/28ec832fad76853aadfcc683810ec74033b99df89cb2ba28776f0bee1b32783b.jpg)

![3aab9c08c5e8267612edfe74dff0f75a74fd3bc29b5410111c3915633461b728.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/3aab9c08c5e8267612edfe74dff0f75a74fd3bc29b5410111c3915633461b728.jpg)

![3bcdb64132324863d0a1d7af1af1ec9f243e8ddb3a38634f4e3bb88c3ed8aae8.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/3bcdb64132324863d0a1d7af1af1ec9f243e8ddb3a38634f4e3bb88c3ed8aae8.jpg)

![3c92f07c54d879c91ef3f033ac60a4f2ca59737cbffbfcaaad00478eeea6d9cb.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/3c92f07c54d879c91ef3f033ac60a4f2ca59737cbffbfcaaad00478eeea6d9cb.jpg)

![4f331efac87ca9bc85d86edaf95ec7c4fc5f3ce14c600e54a5e0b41fd5f2c557.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/4f331efac87ca9bc85d86edaf95ec7c4fc5f3ce14c600e54a5e0b41fd5f2c557.jpg)

![67d140c6d68daf57eb19c57bd8441a527d83b2edf4222823f82d7a685fef9988.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/67d140c6d68daf57eb19c57bd8441a527d83b2edf4222823f82d7a685fef9988.jpg)

![6b9a088a1b087f2c27d5b785bbb7ce16f8a548fff83ae43a6b5bf6f444dbebf2.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/6b9a088a1b087f2c27d5b785bbb7ce16f8a548fff83ae43a6b5bf6f444dbebf2.jpg)

![6e3e3c444dedb24b4f2082ca666fd7295a162ec5eddb3c1fee9cf5d09b34c702.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/6e3e3c444dedb24b4f2082ca666fd7295a162ec5eddb3c1fee9cf5d09b34c702.jpg)

![7a1790ebdca3b3907944ca2a25f9320d1d65ef8e6ac4087e8c31ebdac96eb660.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/7a1790ebdca3b3907944ca2a25f9320d1d65ef8e6ac4087e8c31ebdac96eb660.jpg)

![87b65c349856ad6009e85d40ea9ab0f8fbc069bd249b33128ed7b359217e8f73.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/87b65c349856ad6009e85d40ea9ab0f8fbc069bd249b33128ed7b359217e8f73.jpg)

![9d1ff045c23de8b12db4d4c7392882da718f8741610b8a6a48cb75876982a65d.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/9d1ff045c23de8b12db4d4c7392882da718f8741610b8a6a48cb75876982a65d.jpg)

![a81459df5d8a9b7e86e4826db6b89cd2cbf6a8ffbd51e7d4ed4a40d383706ce4.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/a81459df5d8a9b7e86e4826db6b89cd2cbf6a8ffbd51e7d4ed4a40d383706ce4.jpg)

![eb6ab5c2c503e3836270975462b7a5e287a29aab5d60e21a2df7acd5a53557f3.jpg](../iclr_results/2167_LLaVA-Mini_%20Efficient%20Image%20and%20Video%20Large%20Multimodal%20Models%20with%20One%20Vision%20Token/tables/eb6ab5c2c503e3836270975462b7a5e287a29aab5d60e21a2df7acd5a53557f3.jpg)

## Self-Supervised Diffusion Models for Electron-Aware Molecular Representation Learning


### Images

![028839b1331c8c3c5fb64bd7722c136a86e134bc42d157e0758660d4e3ef7ad4.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/images/028839b1331c8c3c5fb64bd7722c136a86e134bc42d157e0758660d4e3ef7ad4.jpg)

![24335f4f1d91f06d293992342adc74d11a4a19237d737045bd0764aa2e22be45.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/images/24335f4f1d91f06d293992342adc74d11a4a19237d737045bd0764aa2e22be45.jpg)

![3dd3d46ef260869183faf1b9ba1b8c4b19d26108fd64fab5c9e1737d3ba5aadd.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/images/3dd3d46ef260869183faf1b9ba1b8c4b19d26108fd64fab5c9e1737d3ba5aadd.jpg)

![c5f5c350bcfbda8561babdcc11f69f6a3c246318794f19efadf7d7dbbe010c07.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/images/c5f5c350bcfbda8561babdcc11f69f6a3c246318794f19efadf7d7dbbe010c07.jpg)

![db0f6613d4e236da67628112efae8192555b3c3a1007254c11a9c59ac38911bc.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/images/db0f6613d4e236da67628112efae8192555b3c3a1007254c11a9c59ac38911bc.jpg)

![f02c8c644e5b12c82ace4b1fb80756a38fe6ac5bf05e63793ddd9da80a69644b.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/images/f02c8c644e5b12c82ace4b1fb80756a38fe6ac5bf05e63793ddd9da80a69644b.jpg)

### Tables

![03d5c627bc72774426a66b7455de3446261a8d1a0e4f6aed225cd958caa9631d.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/03d5c627bc72774426a66b7455de3446261a8d1a0e4f6aed225cd958caa9631d.jpg)

![0fd3d273090333351ebfb81b91ac159784834564af83cf4a63402bfa58bb8714.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/0fd3d273090333351ebfb81b91ac159784834564af83cf4a63402bfa58bb8714.jpg)

![17561ded989db0491702d2363f6ad9ac642a6ea29b31bf47dc1799f1aa6d6621.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/17561ded989db0491702d2363f6ad9ac642a6ea29b31bf47dc1799f1aa6d6621.jpg)

![2bd1231c63f03822d7912e4a196578da1b85e2eae5162407f117192221e7fd1f.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/2bd1231c63f03822d7912e4a196578da1b85e2eae5162407f117192221e7fd1f.jpg)

![3245a7ef0bdf6806795950eb36ea1ac515c4c9b3b4c73a7545ae72a9f422515a.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/3245a7ef0bdf6806795950eb36ea1ac515c4c9b3b4c73a7545ae72a9f422515a.jpg)

![3f820a7a399e40cfed6aabee5869f7bedd923394bf2e2097c8b1e7a080bd9f79.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/3f820a7a399e40cfed6aabee5869f7bedd923394bf2e2097c8b1e7a080bd9f79.jpg)

![4356032470ee1aef389b076368cc0e0b903758e826336ac120c7fcb8cdba4623.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/4356032470ee1aef389b076368cc0e0b903758e826336ac120c7fcb8cdba4623.jpg)

![4dc6d1418ca7a1f0a1dd17d44a362203a754b328c5c5b7ed91187e3c319e0f92.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/4dc6d1418ca7a1f0a1dd17d44a362203a754b328c5c5b7ed91187e3c319e0f92.jpg)

![6f462b685a4eb6afd06f0ec9ef6ac43d33593a277b82444f303f57113cb042a1.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/6f462b685a4eb6afd06f0ec9ef6ac43d33593a277b82444f303f57113cb042a1.jpg)

![7da1c51f291bc183daaa99e5db658702aac5332ac1add2feb957fcb46ea3ad9a.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/7da1c51f291bc183daaa99e5db658702aac5332ac1add2feb957fcb46ea3ad9a.jpg)

![8b8f123092a0edad3953dc481e10a0f3ca7e2dc3bf35d19342456ad8bfdd2ed9.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/8b8f123092a0edad3953dc481e10a0f3ca7e2dc3bf35d19342456ad8bfdd2ed9.jpg)

![9e84990e2cc1ca8d35fa4518103470558f25557b339a932a68ddb51d3f5176fe.jpg](../iclr_results/2168_Self-Supervised%20Diffusion%20Models%20for%20Electron-Aware%20Molecular%20Representation%20Learning/tables/9e84990e2cc1ca8d35fa4518103470558f25557b339a932a68ddb51d3f5176fe.jpg)

## Exploring the Design Space of Visual Context Representation in Video MLLMs


### Images

![03dfdd70db66d22f30d013c3775c9813b6ff44189f7f1f4eb90c41c24379407f.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/images/03dfdd70db66d22f30d013c3775c9813b6ff44189f7f1f4eb90c41c24379407f.jpg)

![4441cbe744e81f05e83d9fab4311a6d9e88a2c05cdf05daf7c437f5207044f32.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/images/4441cbe744e81f05e83d9fab4311a6d9e88a2c05cdf05daf7c437f5207044f32.jpg)

![45377573013ceb90241a710f8c0613e504c2515d1fa1c185e15b8121c78df459.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/images/45377573013ceb90241a710f8c0613e504c2515d1fa1c185e15b8121c78df459.jpg)

![521a3e7b6bd2e5e30060f7eb34f183f167ebb64e09b95117d50c67ab16f7d56e.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/images/521a3e7b6bd2e5e30060f7eb34f183f167ebb64e09b95117d50c67ab16f7d56e.jpg)

![ba007119c1a55822532c26953ab39d66873309a78b68880ae545df6b5fbb6b9c.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/images/ba007119c1a55822532c26953ab39d66873309a78b68880ae545df6b5fbb6b9c.jpg)

![f04a61bfd790ff84ef1ef40b3ae69a7573bc8be5574d79227cd85e207e46152f.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/images/f04a61bfd790ff84ef1ef40b3ae69a7573bc8be5574d79227cd85e207e46152f.jpg)

### Tables

![048ec06c80e1256272996cb10e76b7975375a528439995cba89d1ce54d4b3303.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/048ec06c80e1256272996cb10e76b7975375a528439995cba89d1ce54d4b3303.jpg)

![04f0e61005cdfa8c8e0c5729769c1ca988caf730cb0794c46474d52a5b7fb74b.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/04f0e61005cdfa8c8e0c5729769c1ca988caf730cb0794c46474d52a5b7fb74b.jpg)

![08b075e6fe4e837e9fbb1c906f8413ac9a25340924b6bb6ca349cd23f4ad7e45.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/08b075e6fe4e837e9fbb1c906f8413ac9a25340924b6bb6ca349cd23f4ad7e45.jpg)

![0e1606959b1f67bf7d2c91938700631494913220b82163781427530ac0d6da75.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/0e1606959b1f67bf7d2c91938700631494913220b82163781427530ac0d6da75.jpg)

![0ec22709f8e7094fdf17a31ae15ad5fe9c869cb44e3b2d10c27da8abb329144c.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/0ec22709f8e7094fdf17a31ae15ad5fe9c869cb44e3b2d10c27da8abb329144c.jpg)

![3a600758276d784665702b9e009cdd5aa0f2faeaac3fb3db32660edbe0068cdc.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/3a600758276d784665702b9e009cdd5aa0f2faeaac3fb3db32660edbe0068cdc.jpg)

![3df9ceb57c4a4634efca6d45f5a76a0c677720bf0e4473181c5795bdcf3084c0.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/3df9ceb57c4a4634efca6d45f5a76a0c677720bf0e4473181c5795bdcf3084c0.jpg)

![3e99ed75d556f292b3e43848336fe62d80e3dc6a52a0f81b386a35799b3cf6ce.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/3e99ed75d556f292b3e43848336fe62d80e3dc6a52a0f81b386a35799b3cf6ce.jpg)

![42275a3697e52a05710bd2976363f1f6b9a540b417ca92a78a2eb1b79f0f6df0.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/42275a3697e52a05710bd2976363f1f6b9a540b417ca92a78a2eb1b79f0f6df0.jpg)

![60101053108751125503a7dd786b4d308987a899cc7d6b2641b7576ef188eee0.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/60101053108751125503a7dd786b4d308987a899cc7d6b2641b7576ef188eee0.jpg)

![8afbb83aaae850bc2955b787c84f974db5467dbec2e3592d497d8d78e06f8a35.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/8afbb83aaae850bc2955b787c84f974db5467dbec2e3592d497d8d78e06f8a35.jpg)

![8ba7ca4d695caa9cd65f8c999f2a70523567bc96706cbb55491f57e24b782d5f.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/8ba7ca4d695caa9cd65f8c999f2a70523567bc96706cbb55491f57e24b782d5f.jpg)

![a5c4c54c3771ccb45f1d73373496b228e746c2fb89692040553eade1ceb142c2.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/a5c4c54c3771ccb45f1d73373496b228e746c2fb89692040553eade1ceb142c2.jpg)

![ea4a2f038766b9412ed95c39a8a66bdfecccc52977cbc6e101e9f7460a9dbafa.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/ea4a2f038766b9412ed95c39a8a66bdfecccc52977cbc6e101e9f7460a9dbafa.jpg)

![ea6404477cca3bdb5bcb505b53fc0be92c8e3ab381917bc39c7e6e9332cf8545.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/ea6404477cca3bdb5bcb505b53fc0be92c8e3ab381917bc39c7e6e9332cf8545.jpg)

![f3d17a5fcd09b6b52815e1064340427953c6f58739b722d1192ae93dcc0cfb69.jpg](../iclr_results/2169_Exploring%20the%20Design%20Space%20of%20Visual%20Context%20Representation%20in%20Video%20MLLMs/tables/f3d17a5fcd09b6b52815e1064340427953c6f58739b722d1192ae93dcc0cfb69.jpg)

## Fair Submodular Cover


### Images

![043555b304dff7b7ab5fc3163568f1ff9415cbcb6e8ba9028385ae8aa4702601.jpg](../iclr_results/2170_Fair%20Submodular%20Cover/images/043555b304dff7b7ab5fc3163568f1ff9415cbcb6e8ba9028385ae8aa4702601.jpg)

![7ef499eefaf32a4db0a3a788b45428fb97dcc30c2f85c24a92c557131f78049c.jpg](../iclr_results/2170_Fair%20Submodular%20Cover/images/7ef499eefaf32a4db0a3a788b45428fb97dcc30c2f85c24a92c557131f78049c.jpg)

![e3609c1c7db894fed8191ad44ba6412ae30602e60953495fbcb25013450b2cf2.jpg](../iclr_results/2170_Fair%20Submodular%20Cover/images/e3609c1c7db894fed8191ad44ba6412ae30602e60953495fbcb25013450b2cf2.jpg)

## L3Ms — Lagrange Large Language Models


### Images

![49a799743394f8d47aaa63dd347f4fa32558337e51dd42f5fc6a837bcc602b6c.jpg](../iclr_results/2171_L3Ms%20%E2%80%94%20Lagrange%20Large%20Language%20Models/images/49a799743394f8d47aaa63dd347f4fa32558337e51dd42f5fc6a837bcc602b6c.jpg)

![99503765ca7a01d5bc90cde1d8993abbe19d4d3263aa4896fff6449977e8edf5.jpg](../iclr_results/2171_L3Ms%20%E2%80%94%20Lagrange%20Large%20Language%20Models/images/99503765ca7a01d5bc90cde1d8993abbe19d4d3263aa4896fff6449977e8edf5.jpg)

![b42703ab44f8a81c69dfb4d70f1eac8d3dc50887dd4bcbaaf398fd3b60389355.jpg](../iclr_results/2171_L3Ms%20%E2%80%94%20Lagrange%20Large%20Language%20Models/images/b42703ab44f8a81c69dfb4d70f1eac8d3dc50887dd4bcbaaf398fd3b60389355.jpg)

### Tables

![0c35942131ff67f5da55f21a1ec0f32083597232d25ef79e9a304c2c9cad6d4f.jpg](../iclr_results/2171_L3Ms%20%E2%80%94%20Lagrange%20Large%20Language%20Models/tables/0c35942131ff67f5da55f21a1ec0f32083597232d25ef79e9a304c2c9cad6d4f.jpg)

![7e2d6a50f2b126fe1e8484ddfd9770d9f372a97c8c5a339147c8e8603bb925f1.jpg](../iclr_results/2171_L3Ms%20%E2%80%94%20Lagrange%20Large%20Language%20Models/tables/7e2d6a50f2b126fe1e8484ddfd9770d9f372a97c8c5a339147c8e8603bb925f1.jpg)

![e829a7f9cbfcc3a3c82f6ae3945857a5beea36960270847e9a4d4ddb76761ae6.jpg](../iclr_results/2171_L3Ms%20%E2%80%94%20Lagrange%20Large%20Language%20Models/tables/e829a7f9cbfcc3a3c82f6ae3945857a5beea36960270847e9a4d4ddb76761ae6.jpg)

## The 3D-PC: a benchmark for visual perspective taking in humans and machines


### Images

![0354545bd2b725b6c79d45ae18a8dd1ce07ad6d7f18bbfaca56d98cab5653f9c.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/0354545bd2b725b6c79d45ae18a8dd1ce07ad6d7f18bbfaca56d98cab5653f9c.jpg)

![14d4976a3039006b1728b4b94514c5d01284616162dbeec1844d8e93252e20bf.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/14d4976a3039006b1728b4b94514c5d01284616162dbeec1844d8e93252e20bf.jpg)

![167ba214887215d733bddd1b7c5c8f6a78cfcaac6f51a441c5aeeb0939e5f970.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/167ba214887215d733bddd1b7c5c8f6a78cfcaac6f51a441c5aeeb0939e5f970.jpg)

![16c425606f75cefea7286c0b1bbfd119bcd0e9b6034de828c7003bb87f6bca20.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/16c425606f75cefea7286c0b1bbfd119bcd0e9b6034de828c7003bb87f6bca20.jpg)

![1bd4b23e1797e1ea746a237a4beff4fe6df7222e2413080c20e7ae4a9c23779c.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/1bd4b23e1797e1ea746a237a4beff4fe6df7222e2413080c20e7ae4a9c23779c.jpg)

![368ed715b17a1eeddcf86e91dc831c9e57f4a995f3fa8e280d4625a821846314.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/368ed715b17a1eeddcf86e91dc831c9e57f4a995f3fa8e280d4625a821846314.jpg)

![44d1db43c8216a793d69b0f31ae1b00dd881a945b4e9962cbc801e489cf517c9.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/44d1db43c8216a793d69b0f31ae1b00dd881a945b4e9962cbc801e489cf517c9.jpg)

![58565fe9557112a96ab0de33ea15dc8edd08e961251af32ba92e0da0ab29526e.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/58565fe9557112a96ab0de33ea15dc8edd08e961251af32ba92e0da0ab29526e.jpg)

![69805cff90fd8d6736988e0e6b9fd9c164da25bf40c54057822f34607ec70b11.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/69805cff90fd8d6736988e0e6b9fd9c164da25bf40c54057822f34607ec70b11.jpg)

![6e472d36ad96443f3caed7b4e66fc7f240878267ad4ba7ef626b1c00d3319b4a.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/6e472d36ad96443f3caed7b4e66fc7f240878267ad4ba7ef626b1c00d3319b4a.jpg)

![988ffa889d9a734c90d62080790c296fa97d9c6be27f2655b55fb877113ae525.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/988ffa889d9a734c90d62080790c296fa97d9c6be27f2655b55fb877113ae525.jpg)

![e78ebbd177d862158e117ef711cacd97bd9f977fec5075d80e2d10c0fc402d38.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/images/e78ebbd177d862158e117ef711cacd97bd9f977fec5075d80e2d10c0fc402d38.jpg)

### Tables

![ebcdbfb0caada1ca6a401ed148018981030e71ab01bb8e4ffe35da579e297764.jpg](../iclr_results/2172_The%203D-PC_%20a%20benchmark%20for%20visual%20perspective%20taking%20in%20humans%20and%20machines/tables/ebcdbfb0caada1ca6a401ed148018981030e71ab01bb8e4ffe35da579e297764.jpg)

## On the Optimal Memorization Capacity of Transformers


### Images

![a3b1b737e9eca7a76c9c7507244eb1d8d3d9c79a2a46218f5d2e0f8dc5698f98.jpg](../iclr_results/2173_On%20the%20Optimal%20Memorization%20Capacity%20of%20Transformers/images/a3b1b737e9eca7a76c9c7507244eb1d8d3d9c79a2a46218f5d2e0f8dc5698f98.jpg)

![be168466bb84a6c9a0ad0ab32844766e768eb3987a5a05086455c76eb0e453dc.jpg](../iclr_results/2173_On%20the%20Optimal%20Memorization%20Capacity%20of%20Transformers/images/be168466bb84a6c9a0ad0ab32844766e768eb3987a5a05086455c76eb0e453dc.jpg)

![f776f05bcf93f45ce6d21184da8da0e326165a9acf1417261d23b8a2663ffa9c.jpg](../iclr_results/2173_On%20the%20Optimal%20Memorization%20Capacity%20of%20Transformers/images/f776f05bcf93f45ce6d21184da8da0e326165a9acf1417261d23b8a2663ffa9c.jpg)

### Tables

![47f624d58adf9148dcbc10831e5d9ace4b68e4830a318ad833b1d3c9a439a476.jpg](../iclr_results/2173_On%20the%20Optimal%20Memorization%20Capacity%20of%20Transformers/tables/47f624d58adf9148dcbc10831e5d9ace4b68e4830a318ad833b1d3c9a439a476.jpg)

## InstantSwap: Fast Customized Concept Swapping across Sharp Shape Differences


### Images

![056d5ac4c74992ba531f596f8f598ae6ea6ac10361ace38bf1e8d7e26a6966d9.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/056d5ac4c74992ba531f596f8f598ae6ea6ac10361ace38bf1e8d7e26a6966d9.jpg)

![108ca23bf1576f4f6f04e6d86c4db6ffcd934fe05e30e0fb64264e4acfd67a3f.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/108ca23bf1576f4f6f04e6d86c4db6ffcd934fe05e30e0fb64264e4acfd67a3f.jpg)

![14f4e8d98754176c8e82585face19f867a1b75d8b5d19a1d93933c9a79ee8c72.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/14f4e8d98754176c8e82585face19f867a1b75d8b5d19a1d93933c9a79ee8c72.jpg)

![1b061cfcc2c7df6f7cb21796923a6b3b637241a3826ad1c18da359ae3749b3d3.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/1b061cfcc2c7df6f7cb21796923a6b3b637241a3826ad1c18da359ae3749b3d3.jpg)

![28c35b321a68ccd953d4af12ca677896948aa4edf0ddf7f5baa624959a6dc307.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/28c35b321a68ccd953d4af12ca677896948aa4edf0ddf7f5baa624959a6dc307.jpg)

![2b0e32983c82ef7b25a94f53030f4fe4d7ac3cda6d285dc989d14e331f5fbd2b.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/2b0e32983c82ef7b25a94f53030f4fe4d7ac3cda6d285dc989d14e331f5fbd2b.jpg)

![2b615f528415fadcfa3ca0e4526fb6131f1fed37dbbd16eeedb969003c068ef7.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/2b615f528415fadcfa3ca0e4526fb6131f1fed37dbbd16eeedb969003c068ef7.jpg)

![36747fb6206a3b58f9bf4e92b8632ba5dc9fd40309a7853c207f7c475f6d3d10.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/36747fb6206a3b58f9bf4e92b8632ba5dc9fd40309a7853c207f7c475f6d3d10.jpg)

![38cd335bc8c54649e8794cec12c4bd012677dabb5e1f41a8345e09b716324cbe.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/38cd335bc8c54649e8794cec12c4bd012677dabb5e1f41a8345e09b716324cbe.jpg)

![40e9ac21f5debcb90fe8b84b36b9b2b2b731d81fa4a17062fb4188e02a95a358.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/40e9ac21f5debcb90fe8b84b36b9b2b2b731d81fa4a17062fb4188e02a95a358.jpg)

![58c04d8b857525c3750b3d150e708aa56b9a7cdfe1dc7e6fa9a94fb724993861.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/58c04d8b857525c3750b3d150e708aa56b9a7cdfe1dc7e6fa9a94fb724993861.jpg)

![60484f39d2599e3f3a164aae5a355d25484298107f7da464712fc03f1fdb596f.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/60484f39d2599e3f3a164aae5a355d25484298107f7da464712fc03f1fdb596f.jpg)

![623cb53cc6aebece384cb428096291b34da3290df5d96feb20b3b11beacb627d.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/623cb53cc6aebece384cb428096291b34da3290df5d96feb20b3b11beacb627d.jpg)

![640226bc8e0984200d1631356ed3f598e50adc4384b647c4cc1ab26edb78d625.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/640226bc8e0984200d1631356ed3f598e50adc4384b647c4cc1ab26edb78d625.jpg)

![673ce0a98a6a8953c2f1acfcea93674cf277a9e3217bbc559c2c1778116d00e5.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/673ce0a98a6a8953c2f1acfcea93674cf277a9e3217bbc559c2c1778116d00e5.jpg)

![7537cf4992e880778e799f8d2db8fbc90cfe5c5e435abeda5a7c76a438bc27ac.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/7537cf4992e880778e799f8d2db8fbc90cfe5c5e435abeda5a7c76a438bc27ac.jpg)

![7bac9bad904beaba834d7f0c51f920f0eb8d4a89c0dd5661ff129ff21053d9dc.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/7bac9bad904beaba834d7f0c51f920f0eb8d4a89c0dd5661ff129ff21053d9dc.jpg)

![7d8e7be03ba14894ebde66f968c66e9ca1acf417a3bf17d7f4370984436ec488.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/7d8e7be03ba14894ebde66f968c66e9ca1acf417a3bf17d7f4370984436ec488.jpg)

![7e8ac8cfff348dd2cf276e84f9ba9f781602b7cefe529073e844deedb28d449b.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/7e8ac8cfff348dd2cf276e84f9ba9f781602b7cefe529073e844deedb28d449b.jpg)

![7fa43d83603b21ad22b657ccc85ada1a275b3638e34c941b273b73a3b2113b91.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/7fa43d83603b21ad22b657ccc85ada1a275b3638e34c941b273b73a3b2113b91.jpg)

![a258db9e71e8da6dbeee5603fddf8dbe31f1ab904b6a0dcee0ec6eff2c71f1ba.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/a258db9e71e8da6dbeee5603fddf8dbe31f1ab904b6a0dcee0ec6eff2c71f1ba.jpg)

![a29e7289b9c701004b7d4401487530b3b84aa17dd0fe2d20b6be17dfe417cb8f.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/a29e7289b9c701004b7d4401487530b3b84aa17dd0fe2d20b6be17dfe417cb8f.jpg)

![a632205b962961ae1305fafddd8679324f50ca0710ae864c96057a48690afee1.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/a632205b962961ae1305fafddd8679324f50ca0710ae864c96057a48690afee1.jpg)

![a7edd8e320b928efb5913d5b2cfbab079336b9e8fac130cba7395f94edaef76f.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/a7edd8e320b928efb5913d5b2cfbab079336b9e8fac130cba7395f94edaef76f.jpg)

![aadb2709ae7954d6e6997969e99d44ead0d851c6947799a994ebd7f5864726c8.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/aadb2709ae7954d6e6997969e99d44ead0d851c6947799a994ebd7f5864726c8.jpg)

![bb5ce31303050f50a875b356d8b32add1797194586aba4cdb0db43fbf6e20b52.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/bb5ce31303050f50a875b356d8b32add1797194586aba4cdb0db43fbf6e20b52.jpg)

![bdf763e9f773c5f8519ca4dd19fa7c11a9aed2bcf1167e9058ba941250991297.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/bdf763e9f773c5f8519ca4dd19fa7c11a9aed2bcf1167e9058ba941250991297.jpg)

![bfb660938d8b080b16218ac4baec46a4867f5d3c3d6512fd9e9f518ca2a22d26.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/bfb660938d8b080b16218ac4baec46a4867f5d3c3d6512fd9e9f518ca2a22d26.jpg)

![c01233cb3eb2b884fe4b03678bdf6ede33cfeeb1af5e1ffe1cbc3cfbd28b6c34.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/c01233cb3eb2b884fe4b03678bdf6ede33cfeeb1af5e1ffe1cbc3cfbd28b6c34.jpg)

![c32e14bd4c30301e3d4752e66b22ff97b343dfc48b24fc381922d2840ec75068.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/c32e14bd4c30301e3d4752e66b22ff97b343dfc48b24fc381922d2840ec75068.jpg)

![c462777964e55abe5ed3c7591751194d1bb00ec41a4b51592fca7fa4097b030a.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/c462777964e55abe5ed3c7591751194d1bb00ec41a4b51592fca7fa4097b030a.jpg)

![c5b232629ee0f79143e20ad822fb2c86784564bd8f576ba4b47a5d3cf47ef0a0.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/c5b232629ee0f79143e20ad822fb2c86784564bd8f576ba4b47a5d3cf47ef0a0.jpg)

![ca8cd2fbdc3755eb88389aa45420d02d732445b5e27642ecd75e52d5ff034a6e.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/ca8cd2fbdc3755eb88389aa45420d02d732445b5e27642ecd75e52d5ff034a6e.jpg)

![d0052605b1dd316bb7780f208ab9df8e9c4f0dfe1bfb7226f077711ed6aa096f.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/d0052605b1dd316bb7780f208ab9df8e9c4f0dfe1bfb7226f077711ed6aa096f.jpg)

![d0ca905c59043fc9c9a1d8bc36e9cc88522e04c42d3bd35c0711092203fc6307.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/d0ca905c59043fc9c9a1d8bc36e9cc88522e04c42d3bd35c0711092203fc6307.jpg)

![ead34b0add8a217cd216f92ff3607e8e4b6db7b5e62beb87442c67ac19d545e4.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/ead34b0add8a217cd216f92ff3607e8e4b6db7b5e62beb87442c67ac19d545e4.jpg)

![eadff040952e2735832bd8d74776560b5251fdec26c8acd08260af78d34780ef.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/eadff040952e2735832bd8d74776560b5251fdec26c8acd08260af78d34780ef.jpg)

![ece5823402fff8921712ef1280c2ef88d340bfe1a28fb6e608ddcc8180079ecd.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/ece5823402fff8921712ef1280c2ef88d340bfe1a28fb6e608ddcc8180079ecd.jpg)

![f3257c70398b0def643567baec10e4450df92c0e7d4ee3beb776f1ea0fed553c.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/f3257c70398b0def643567baec10e4450df92c0e7d4ee3beb776f1ea0fed553c.jpg)

![f7824fe7896194bfbe61035997dbfb8c07c244dc47d01189b133a92ae3d9394c.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/f7824fe7896194bfbe61035997dbfb8c07c244dc47d01189b133a92ae3d9394c.jpg)

![ffdc0a8f360eb7f342d248033ae1fdc8afb2616d8899207b6204830ba3d7d64f.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/images/ffdc0a8f360eb7f342d248033ae1fdc8afb2616d8899207b6204830ba3d7d64f.jpg)

### Tables

![027473e72d10a9832d99577906f89f6f30ce6c04536e6b04a00b41291a1c04f1.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/027473e72d10a9832d99577906f89f6f30ce6c04536e6b04a00b41291a1c04f1.jpg)

![56e5a0d3c87a000aef09b7d20cbf90c57ec35757e7b6c2eaf41f75da7766cd66.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/56e5a0d3c87a000aef09b7d20cbf90c57ec35757e7b6c2eaf41f75da7766cd66.jpg)

![763494d4fffb6873c5284a6e8b7142a00abe690e4d8b6e3999d17eba318a428c.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/763494d4fffb6873c5284a6e8b7142a00abe690e4d8b6e3999d17eba318a428c.jpg)

![76e3e545d103b8bb3f23e8517870becad09b68f9c531e874876d66bbb6e30f86.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/76e3e545d103b8bb3f23e8517870becad09b68f9c531e874876d66bbb6e30f86.jpg)

![7d20f82a44ab01bfc3c19a84328ba0a53101abc8050ff9abda5c0c6080369097.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/7d20f82a44ab01bfc3c19a84328ba0a53101abc8050ff9abda5c0c6080369097.jpg)

![853a24644948388c242b094ef7894476508ef2aa7c78333438282be4316e94c2.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/853a24644948388c242b094ef7894476508ef2aa7c78333438282be4316e94c2.jpg)

![b74225986d8c85b37959a314a1c7838fbc999e28537a26bd6573cbc1e5801edc.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/b74225986d8c85b37959a314a1c7838fbc999e28537a26bd6573cbc1e5801edc.jpg)

![bd6c436ecfbbfb2fab5eb6a3dd515b122d80bc70195a539acb6e61c93de8e6ca.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/bd6c436ecfbbfb2fab5eb6a3dd515b122d80bc70195a539acb6e61c93de8e6ca.jpg)

![c6df8a32dd201569feb1013c97824b18208187ce5b584508bed163a1f14f9d5d.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/c6df8a32dd201569feb1013c97824b18208187ce5b584508bed163a1f14f9d5d.jpg)

![fd7430f6a87656437a2584f964b59616f965150106286a6379ecfa2fbef746cc.jpg](../iclr_results/2174_InstantSwap_%20Fast%20Customized%20Concept%20Swapping%20across%20Sharp%20Shape%20Differences/tables/fd7430f6a87656437a2584f964b59616f965150106286a6379ecfa2fbef746cc.jpg)

## Efficient Cross-Episode Meta-RL


### Images

![06ea9a53033404a577a9fc23d26557cfb48f67ee0850171f6a1d00edc2c2a810.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/06ea9a53033404a577a9fc23d26557cfb48f67ee0850171f6a1d00edc2c2a810.jpg)

![087d90e47526798159c222329c49a83f740f28fa86557b744b4a4188c22b3461.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/087d90e47526798159c222329c49a83f740f28fa86557b744b4a4188c22b3461.jpg)

![0a75cdb42308efb04fab9058bc9766f6b27dd068e55e97cf5408f8052c95ebf8.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/0a75cdb42308efb04fab9058bc9766f6b27dd068e55e97cf5408f8052c95ebf8.jpg)

![11d961cf1cb7dbae18a4951d2d1ce68f85a29ed93fe19fec1154889c03b3abc9.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/11d961cf1cb7dbae18a4951d2d1ce68f85a29ed93fe19fec1154889c03b3abc9.jpg)

![12ed99facfdaa8c7d51f168e220d57d9f0f790e5df7f10f021cc285465c713fb.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/12ed99facfdaa8c7d51f168e220d57d9f0f790e5df7f10f021cc285465c713fb.jpg)

![160cec6b68830a68eae36bdb14a6ceb68185fe1160e14f2e445ed4fa8df6798a.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/160cec6b68830a68eae36bdb14a6ceb68185fe1160e14f2e445ed4fa8df6798a.jpg)

![1d486750f642a4525a478eac4b646008c5c5b40301378516d288d55dcc9db7ec.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/1d486750f642a4525a478eac4b646008c5c5b40301378516d288d55dcc9db7ec.jpg)

![2361efe9f1b8edc93d84ec25725b9fd3e3cf3fb1d42a47771754f99e7d221fc5.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/2361efe9f1b8edc93d84ec25725b9fd3e3cf3fb1d42a47771754f99e7d221fc5.jpg)

![305500bab286c826c397031d33a48ec8b3f26a266187e1f2fd3964cb8b33c8a0.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/305500bab286c826c397031d33a48ec8b3f26a266187e1f2fd3964cb8b33c8a0.jpg)

![33216ba779f0f19ff1ff156b5001b71740885a01f0b53f996626ac6db1015efc.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/33216ba779f0f19ff1ff156b5001b71740885a01f0b53f996626ac6db1015efc.jpg)

![3cc0a4fbd01a19ebed0e3ad578af7dde6f07d302e97e838c76ad476ecd64eead.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/3cc0a4fbd01a19ebed0e3ad578af7dde6f07d302e97e838c76ad476ecd64eead.jpg)

![43a7677d3a0022ba2bf929c588d48c6a0374cdb5ac43256b0b7c11500da597d9.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/43a7677d3a0022ba2bf929c588d48c6a0374cdb5ac43256b0b7c11500da597d9.jpg)

![495e0abb90a59601a787324282676937fcb44b9d57e4ca375ccbd463c0996201.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/495e0abb90a59601a787324282676937fcb44b9d57e4ca375ccbd463c0996201.jpg)

![4e755fb696ab1d8ee5ddb0cf083c1cf75900fd68ed1460eb825bde741a621dce.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/4e755fb696ab1d8ee5ddb0cf083c1cf75900fd68ed1460eb825bde741a621dce.jpg)

![54178cd54c0b0ddc3a071f793a92a3bdd16ee40614b9b46350fff52a002d845e.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/54178cd54c0b0ddc3a071f793a92a3bdd16ee40614b9b46350fff52a002d845e.jpg)

![67d28271fde098090e32c771559bfb98f24055687f813c614fa345dae09b2470.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/67d28271fde098090e32c771559bfb98f24055687f813c614fa345dae09b2470.jpg)

![67eb74664ec2cfb6cb43e3d2cc127e2a67aa7d04bf7ace190b972969ef958fca.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/67eb74664ec2cfb6cb43e3d2cc127e2a67aa7d04bf7ace190b972969ef958fca.jpg)

![699f65a5b1af431a733c202cf6ad9cc00455b312f57a784a9b0ef35453572d16.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/699f65a5b1af431a733c202cf6ad9cc00455b312f57a784a9b0ef35453572d16.jpg)

![6e15aa3a27f1173220ae6f338741fa87aa786dd8212a3ffd947515d73fbacfff.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/6e15aa3a27f1173220ae6f338741fa87aa786dd8212a3ffd947515d73fbacfff.jpg)

![7bcf7f97310e026b923766b66df3c3f2f1442d4cac30df42a1523d1d146c83dc.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/7bcf7f97310e026b923766b66df3c3f2f1442d4cac30df42a1523d1d146c83dc.jpg)

![7ea06e0e728f0a891720d270e552dff1179c7280b471316a72d84d375703c8c7.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/7ea06e0e728f0a891720d270e552dff1179c7280b471316a72d84d375703c8c7.jpg)

![7fab500cd9b863942ecdcd7f671fab28624692605033ecb7bc6ec42985646791.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/7fab500cd9b863942ecdcd7f671fab28624692605033ecb7bc6ec42985646791.jpg)

![82c882ff80d9ca6a870d3c0a4a661cd88a8455e2eedaee023eb51492ddbdb657.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/82c882ff80d9ca6a870d3c0a4a661cd88a8455e2eedaee023eb51492ddbdb657.jpg)

![87c69c3509d5862a7df6a4c685b9977fe33acd9f227b0f229e0d45f3d4e74387.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/87c69c3509d5862a7df6a4c685b9977fe33acd9f227b0f229e0d45f3d4e74387.jpg)

![8a66a8ed60f3658ff1eb1c582123503802c8b2a2aa0699105b801cb6f20e8c9c.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/8a66a8ed60f3658ff1eb1c582123503802c8b2a2aa0699105b801cb6f20e8c9c.jpg)

![8ac670546c3e366771d62b1dc08ac0365b4aecfbb41774ed353422d6f083d2d3.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/8ac670546c3e366771d62b1dc08ac0365b4aecfbb41774ed353422d6f083d2d3.jpg)

![8b8ba49977f853afcf2f2c63706b1348d27a9d08fd4b4b5bc744949fdb4f1f4e.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/8b8ba49977f853afcf2f2c63706b1348d27a9d08fd4b4b5bc744949fdb4f1f4e.jpg)

![8cd309a576715dbc4d144721d07c6abc67abd558f4598f24d1a949094d339b17.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/8cd309a576715dbc4d144721d07c6abc67abd558f4598f24d1a949094d339b17.jpg)

![94a9602881eda9e9d48cda64144b27145bd6ad850908f02bff41829bb87a8ef7.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/94a9602881eda9e9d48cda64144b27145bd6ad850908f02bff41829bb87a8ef7.jpg)

![96967246beb3e1713bf6b527aafca2a37ac8bd8e9e204c49f22ec2de372eddd5.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/96967246beb3e1713bf6b527aafca2a37ac8bd8e9e204c49f22ec2de372eddd5.jpg)

![9c7b6868f64c22db7baf0fcea66a0b9ed5aaf957b45d6db6551b67e10134f9b9.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/9c7b6868f64c22db7baf0fcea66a0b9ed5aaf957b45d6db6551b67e10134f9b9.jpg)

![a57364d7329f250792bc4b79d91cf5236db4592d1fe24686f96dc16c3818d729.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/a57364d7329f250792bc4b79d91cf5236db4592d1fe24686f96dc16c3818d729.jpg)

![ac5a52d5eefd32ce696c4db7c27129ef746e4da5abbd4b3e5cc752101735512f.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/ac5a52d5eefd32ce696c4db7c27129ef746e4da5abbd4b3e5cc752101735512f.jpg)

![c402687626426e7670a95f077d06185d150fcc133f0701aeebcad7c122c299c0.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/c402687626426e7670a95f077d06185d150fcc133f0701aeebcad7c122c299c0.jpg)

![d59d85389704b036ee5b50bb23fe18d724a0da39a6cc8904ca04dd2b45aa28ac.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/d59d85389704b036ee5b50bb23fe18d724a0da39a6cc8904ca04dd2b45aa28ac.jpg)

![dca78d09ed2d1eb371657000eb3cd9769c1dbea6e2b2e6ab679e63844ca43dae.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/dca78d09ed2d1eb371657000eb3cd9769c1dbea6e2b2e6ab679e63844ca43dae.jpg)

![f2890f8fd5d867b628484f84680df7a87d04eddf6209046a1260d5d9d4a97ff9.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/f2890f8fd5d867b628484f84680df7a87d04eddf6209046a1260d5d9d4a97ff9.jpg)

![fc3a55d949acc6d8c05ba843be8151b8a718a3821bcc66bf14ef499fcdb9b7db.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/fc3a55d949acc6d8c05ba843be8151b8a718a3821bcc66bf14ef499fcdb9b7db.jpg)

![fd895ab2bf9470cce93afb44e12690439d3d09c306d69c3fb8da4320994da8ba.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/images/fd895ab2bf9470cce93afb44e12690439d3d09c306d69c3fb8da4320994da8ba.jpg)

### Tables

![0cd41353e50c1dce11e0ad7c852c778d39078f2e84e2a0ec17ec983420ef2859.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/tables/0cd41353e50c1dce11e0ad7c852c778d39078f2e84e2a0ec17ec983420ef2859.jpg)

![c23ab2d587acc6aa9bec7995db1ae9aed60403fcdbae8b2196e4c72645bbda71.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/tables/c23ab2d587acc6aa9bec7995db1ae9aed60403fcdbae8b2196e4c72645bbda71.jpg)

![f15af2f4ef4af02deb7499716201af7c7f0abadd48ac5ee3164a7e85b29ae95f.jpg](../iclr_results/2175_Efficient%20Cross-Episode%20Meta-RL/tables/f15af2f4ef4af02deb7499716201af7c7f0abadd48ac5ee3164a7e85b29ae95f.jpg)

## MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba


### Images

![2668c390394aa579eb327404581be78c7546b4e1d8b0dfa4bfcf5e4d0623899f.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/images/2668c390394aa579eb327404581be78c7546b4e1d8b0dfa4bfcf5e4d0623899f.jpg)

![392ed8fb04a6057d840f3aeebe02e6fedd2f6d0888286dc6c015b219f574a20a.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/images/392ed8fb04a6057d840f3aeebe02e6fedd2f6d0888286dc6c015b219f574a20a.jpg)

![7d88add71817e2a3fc34a6583e5428669cc4389fba1539d54f7ce5ee8d158eb5.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/images/7d88add71817e2a3fc34a6583e5428669cc4389fba1539d54f7ce5ee8d158eb5.jpg)

![9943c1022008e9c7f052cd4530a2a79d3ca831358bf365ba21fa2e1d3899a445.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/images/9943c1022008e9c7f052cd4530a2a79d3ca831358bf365ba21fa2e1d3899a445.jpg)

![e265b59fb025b7f32b546a63cb89d3cd2f2feac634bc8e153fbfdcf6c9e12f70.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/images/e265b59fb025b7f32b546a63cb89d3cd2f2feac634bc8e153fbfdcf6c9e12f70.jpg)

![e71157ce74190e3f382e168fccd929f54e83f87ed107019ae6f8e7f0956c313f.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/images/e71157ce74190e3f382e168fccd929f54e83f87ed107019ae6f8e7f0956c313f.jpg)

![ec1dd13bf9917b9f5a7893296bcfb048112959274079b587426160dc9f036bfd.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/images/ec1dd13bf9917b9f5a7893296bcfb048112959274079b587426160dc9f036bfd.jpg)

### Tables

![333388079c102c1f2efa2be0a009ae673232d8886a94b3255b2f15baceaa2c4b.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/333388079c102c1f2efa2be0a009ae673232d8886a94b3255b2f15baceaa2c4b.jpg)

![6d75d2e57026cf8a363887dd210e24c17f9972d1c6435d2ee5ce6f772460af93.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/6d75d2e57026cf8a363887dd210e24c17f9972d1c6435d2ee5ce6f772460af93.jpg)

![779c0e460081b4799d45a35beddccc834725669879e4d888bd15a5e629b3b997.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/779c0e460081b4799d45a35beddccc834725669879e4d888bd15a5e629b3b997.jpg)

![7b2d181dce322720bf38af49dc95759858bc2a7696d98b1c5ffd997645d8dd96.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/7b2d181dce322720bf38af49dc95759858bc2a7696d98b1c5ffd997645d8dd96.jpg)

![ae20b05d4a1c622ce56e4ecf656d5ecb9f21baab6f2ba49a4b4ada07e2b77091.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/ae20b05d4a1c622ce56e4ecf656d5ecb9f21baab6f2ba49a4b4ada07e2b77091.jpg)

![b021c306e7cc6abd3d962f39a11156fb7a95ca332619fc596fd72fbb6e81f1ac.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/b021c306e7cc6abd3d962f39a11156fb7a95ca332619fc596fd72fbb6e81f1ac.jpg)

![b92e72f56a2feb67b53d439b8f9b9fcae9073cd206904d8722211bc96e48e946.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/b92e72f56a2feb67b53d439b8f9b9fcae9073cd206904d8722211bc96e48e946.jpg)

![bd520dd6b9ec493523dc583015d501c8caaada649c31d23405b5bbca7b48b8be.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/bd520dd6b9ec493523dc583015d501c8caaada649c31d23405b5bbca7b48b8be.jpg)

![cc23287a5941d1ccaef0d1ce5625ff259d3479be1be9daf3c05fc2d884863575.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/cc23287a5941d1ccaef0d1ce5625ff259d3479be1be9daf3c05fc2d884863575.jpg)

![e85a59044efed07969fde39c542d6c36a37d34285ef94fbcff47925e19a0afb7.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/e85a59044efed07969fde39c542d6c36a37d34285ef94fbcff47925e19a0afb7.jpg)

![ea51ef0371fd9aea8490e739bb0ce346b5b2aa4cbcc567d178451ce74c449a1a.jpg](../iclr_results/2176_MambaPEFT_%20Exploring%20Parameter-Efficient%20Fine-Tuning%20for%20Mamba/tables/ea51ef0371fd9aea8490e739bb0ce346b5b2aa4cbcc567d178451ce74c449a1a.jpg)

## Adversaries With Incentives:  A Strategic Alternative to Adversarial Robustness


### Images

![0bad17333f4df07550b6632cfb124c984114283a9b0fea92f66791a0f52f28ea.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/0bad17333f4df07550b6632cfb124c984114283a9b0fea92f66791a0f52f28ea.jpg)

![3ae5d0b8f7225126d72493a4d8cbcf3827619c908eaa72547a3900e74fc923b4.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/3ae5d0b8f7225126d72493a4d8cbcf3827619c908eaa72547a3900e74fc923b4.jpg)

![477e02cac17f9b7e1772a9e036dc42ccbd9df3e725491dd28100dca875ad185c.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/477e02cac17f9b7e1772a9e036dc42ccbd9df3e725491dd28100dca875ad185c.jpg)

![4bf4439a97ab16764e6cd14d81ca33efa9c0a34d74610a2367690e0550a2e851.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/4bf4439a97ab16764e6cd14d81ca33efa9c0a34d74610a2367690e0550a2e851.jpg)

![5742b1357f2dccffae2921d05ccf7649056dc7dd619548c83139ff00331d7416.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/5742b1357f2dccffae2921d05ccf7649056dc7dd619548c83139ff00331d7416.jpg)

![6f2ca5190eab33775de1dfe272bf2aa94caf0c9567a0acc6088675ecb2834ba5.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/6f2ca5190eab33775de1dfe272bf2aa94caf0c9567a0acc6088675ecb2834ba5.jpg)

![b062a6c31a1f3a3c8b5b94ee33c4d99fb8d29d16c2db8be0ef5776ee02c0d3c4.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/b062a6c31a1f3a3c8b5b94ee33c4d99fb8d29d16c2db8be0ef5776ee02c0d3c4.jpg)

![b859f2a91cfcd0316252aa1b38a66cecc35ee411729fb79ebed2ab3ab4a0f797.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/b859f2a91cfcd0316252aa1b38a66cecc35ee411729fb79ebed2ab3ab4a0f797.jpg)

![cc2cf7e4e7c4f774b2fc8561e76acf83d5e49be02edab228e24556d4877dd1e4.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/cc2cf7e4e7c4f774b2fc8561e76acf83d5e49be02edab228e24556d4877dd1e4.jpg)

![f314ecd9943935371a3a1f152bc62984e41e44351500b74c226dfd42190f076c.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/images/f314ecd9943935371a3a1f152bc62984e41e44351500b74c226dfd42190f076c.jpg)

### Tables

![0204dedd53a859fe65504359a160b2e20dac26e0d33dd7b5aba064406e11fb9b.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/0204dedd53a859fe65504359a160b2e20dac26e0d33dd7b5aba064406e11fb9b.jpg)

![21f4608d7230ff184e1e8a242e11be45cb04e5b21e378c6becfc1f6aa0b6c013.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/21f4608d7230ff184e1e8a242e11be45cb04e5b21e378c6becfc1f6aa0b6c013.jpg)

![2535107c4af35179c2e0d01f27ff603ea95d59212477414f5f01e340367d7750.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/2535107c4af35179c2e0d01f27ff603ea95d59212477414f5f01e340367d7750.jpg)

![36a50b69be1bddb6d848ce2917c0179f17276aa5960b96ded38271bc3749c3b1.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/36a50b69be1bddb6d848ce2917c0179f17276aa5960b96ded38271bc3749c3b1.jpg)

![49d7a6c94412a406e4d1f3bc53ec4a0c2e2a958ddbb62a29ca4d7873fad0b70b.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/49d7a6c94412a406e4d1f3bc53ec4a0c2e2a958ddbb62a29ca4d7873fad0b70b.jpg)

![4f41d7ad99ccced91bf941730c5751bb8209f9d196673dc737f81aed925539c4.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/4f41d7ad99ccced91bf941730c5751bb8209f9d196673dc737f81aed925539c4.jpg)

![5a110816f5b284dba5bfdf26b6253f020048e70b0e777d2ed5391f1e6b053a69.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/5a110816f5b284dba5bfdf26b6253f020048e70b0e777d2ed5391f1e6b053a69.jpg)

![62f436cb5368d5830f698b5e5a9be9adf090289690d773c8ac4d990bc22518c5.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/62f436cb5368d5830f698b5e5a9be9adf090289690d773c8ac4d990bc22518c5.jpg)

![d600c546d07714b737a9245a5b54381c84dae4e35d9cba2ee7d70aaba852cf24.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/d600c546d07714b737a9245a5b54381c84dae4e35d9cba2ee7d70aaba852cf24.jpg)

![e6d91dee23a10fd1dd13ac5318aca51f8acf1d16fe3a3561243aecb1c8a18217.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/e6d91dee23a10fd1dd13ac5318aca51f8acf1d16fe3a3561243aecb1c8a18217.jpg)

![fff0102fd0146e39f8bebe54df7848e0f3a03267db7fd368cc481722a13ff30a.jpg](../iclr_results/2177_Adversaries%20With%20Incentives_%20%20A%20Strategic%20Alternative%20to%20Adversarial%20Robustness/tables/fff0102fd0146e39f8bebe54df7848e0f3a03267db7fd368cc481722a13ff30a.jpg)

## A Formal Framework for Understanding Length Generalization in Transformers


### Images

![057ba728674a87b7940c77940e67cc2d62daf04bc5f5b985566a73d91d064137.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/057ba728674a87b7940c77940e67cc2d62daf04bc5f5b985566a73d91d064137.jpg)

![118877e2315a39846d1dbb90893315ca0c83465792822aa14cfc64decc6c591f.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/118877e2315a39846d1dbb90893315ca0c83465792822aa14cfc64decc6c591f.jpg)

![19448bd73d59f5449984fb8e07ac7e1ddd0338e332acce5e67143c2871c132b4.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/19448bd73d59f5449984fb8e07ac7e1ddd0338e332acce5e67143c2871c132b4.jpg)

![39ea0a47c9b40153e399038323d0f83cefe5dc8a89c299dade0b70960eb1de12.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/39ea0a47c9b40153e399038323d0f83cefe5dc8a89c299dade0b70960eb1de12.jpg)

![4974cfbcce481aeb2aa2030efb3b17d587e2c79d9f652c362c508304eacdb8e1.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/4974cfbcce481aeb2aa2030efb3b17d587e2c79d9f652c362c508304eacdb8e1.jpg)

![49d96f14f89ed81f620bd144eb0190e3b6608ef033cfda860cff73e03602cda9.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/49d96f14f89ed81f620bd144eb0190e3b6608ef033cfda860cff73e03602cda9.jpg)

![527e1c524973f6c3c55f761bd7abc7c59642a8c4e6c47dcd8bbc8344ae37120f.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/527e1c524973f6c3c55f761bd7abc7c59642a8c4e6c47dcd8bbc8344ae37120f.jpg)

![9c45f6563df9eb1ee12eb27110bb180635e1320645dda5bd7547ec958ad0705f.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/9c45f6563df9eb1ee12eb27110bb180635e1320645dda5bd7547ec958ad0705f.jpg)

![9da75d1b70a578c556f356f062d86dd20d56998b807a36b805edaec2cd7e9a49.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/9da75d1b70a578c556f356f062d86dd20d56998b807a36b805edaec2cd7e9a49.jpg)

![b77c8b5511109cebdc38231dd53190553b36e81bf671b21d97ee444885472307.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/b77c8b5511109cebdc38231dd53190553b36e81bf671b21d97ee444885472307.jpg)

![d54f29070ff8a44d889cc42dbec41f3622384e1d182ebba22992289f8a5bd04e.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/d54f29070ff8a44d889cc42dbec41f3622384e1d182ebba22992289f8a5bd04e.jpg)

![e9c3fbc08f33e268164f52bffd31ce7a3aaa0ec0e422afc240525593addd509b.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/images/e9c3fbc08f33e268164f52bffd31ce7a3aaa0ec0e422afc240525593addd509b.jpg)

### Tables

![12e7b0b12b7b5f343a7adab964a502e12ad8f12e04108cc05fbee3d6e13ebfe6.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/12e7b0b12b7b5f343a7adab964a502e12ad8f12e04108cc05fbee3d6e13ebfe6.jpg)

![20c7897758f181f818c1e41d8530ab663d43000f1e8f7242917167c604cb9972.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/20c7897758f181f818c1e41d8530ab663d43000f1e8f7242917167c604cb9972.jpg)

![351bbe1c07af65270a2d8979e23776dee78230ebb24b16c9701fbbf0d03b3715.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/351bbe1c07af65270a2d8979e23776dee78230ebb24b16c9701fbbf0d03b3715.jpg)

![46806c4164bb5df14030aafdab5a1cb67815fd615609ab0bd5b24688d19e94e3.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/46806c4164bb5df14030aafdab5a1cb67815fd615609ab0bd5b24688d19e94e3.jpg)

![52568f9ec9651920cc822d2a9a0106eb5b2c6ac1c0a1ed34d61429285ad8c235.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/52568f9ec9651920cc822d2a9a0106eb5b2c6ac1c0a1ed34d61429285ad8c235.jpg)

![52687c95af59c22c8b049d688f997d6fe8c42733caa3f2c30ec5b99493c6624d.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/52687c95af59c22c8b049d688f997d6fe8c42733caa3f2c30ec5b99493c6624d.jpg)

![5c2365bd6a2c9d8c6bd6da6094556a5952160b58ccb42d345bf3bec99c0df69c.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/5c2365bd6a2c9d8c6bd6da6094556a5952160b58ccb42d345bf3bec99c0df69c.jpg)

![8f9550510cd721fcf9f84b635665da6eee934c35286463f5d5e4a449d7569c24.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/8f9550510cd721fcf9f84b635665da6eee934c35286463f5d5e4a449d7569c24.jpg)

![b36ff07dbd0aa2896811b4ab33fd2974bbb69939b765ae61860050e1ed28797f.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/b36ff07dbd0aa2896811b4ab33fd2974bbb69939b765ae61860050e1ed28797f.jpg)

![b4df85c12b188965993028be37834ce055c122d3ffdfbe01b53ccf3900858957.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/b4df85c12b188965993028be37834ce055c122d3ffdfbe01b53ccf3900858957.jpg)

![d37403aff4b8aa8289f91c8b49bcf3facd206db0be5f52a0ca9991e2150d115a.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/d37403aff4b8aa8289f91c8b49bcf3facd206db0be5f52a0ca9991e2150d115a.jpg)

![d6ee9cce97973397981af48927c3bf822570513b1038c6d36731eaae64e0681c.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/d6ee9cce97973397981af48927c3bf822570513b1038c6d36731eaae64e0681c.jpg)

![dde90b2542f6190fe92496285c102aa7c5040e0d8a07767a3480c50f0c689308.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/dde90b2542f6190fe92496285c102aa7c5040e0d8a07767a3480c50f0c689308.jpg)

![ea0fea25e4fd2e18ff2fe89ea630907cecf3231d881f17439e0aacc50dfec562.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/ea0fea25e4fd2e18ff2fe89ea630907cecf3231d881f17439e0aacc50dfec562.jpg)

![efeb257a9001c6021da5123d84c19d27aaa7f6f24017ba5642534b5df510ca2e.jpg](../iclr_results/2178_A%20Formal%20Framework%20for%20Understanding%20Length%20Generalization%20in%20Transformers/tables/efeb257a9001c6021da5123d84c19d27aaa7f6f24017ba5642534b5df510ca2e.jpg)

## Mask in the Mirror: Implicit Sparsification


### Images

![53db41e3273a2df2c355c6a253b68dddfab64f4ee3c7501bee036c08abdb10ca.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/images/53db41e3273a2df2c355c6a253b68dddfab64f4ee3c7501bee036c08abdb10ca.jpg)

![6093ec805b2cfad86e5a439f84011d070406cab6d0af7b565aef70e3a594bb95.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/images/6093ec805b2cfad86e5a439f84011d070406cab6d0af7b565aef70e3a594bb95.jpg)

![a501d99f81d9908e720e746c37c4064967caff9196d4ca360ec2ebf13a9f6fde.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/images/a501d99f81d9908e720e746c37c4064967caff9196d4ca360ec2ebf13a9f6fde.jpg)

![e1c320a37f1fa1401bd48f9dd24e8f24e2ca3127773da13792353d3920ab49cc.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/images/e1c320a37f1fa1401bd48f9dd24e8f24e2ca3127773da13792353d3920ab49cc.jpg)

![e2de3cbc587a6f3db272bef0e392bf8b668db3ef044a68c0f27122063721658a.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/images/e2de3cbc587a6f3db272bef0e392bf8b668db3ef044a68c0f27122063721658a.jpg)

![e74b24a127d2301fd6acd80fae804c866923e6cda18931104851d21fe1ea2c3d.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/images/e74b24a127d2301fd6acd80fae804c866923e6cda18931104851d21fe1ea2c3d.jpg)

### Tables

![042a59c3c100d9a4313596d759fed9c0b3d229c4aa1d827ce8cb4170ab79218a.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/042a59c3c100d9a4313596d759fed9c0b3d229c4aa1d827ce8cb4170ab79218a.jpg)

![23ce35c4ee6d32205206beeced93d6b4b95d03f4e39bea34f9c4c713719ace8a.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/23ce35c4ee6d32205206beeced93d6b4b95d03f4e39bea34f9c4c713719ace8a.jpg)

![3d6a481891f7bfc6f08002d879d8a0b56121cdcf3980705a5276246fbe6a5f28.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/3d6a481891f7bfc6f08002d879d8a0b56121cdcf3980705a5276246fbe6a5f28.jpg)

![3f4a8094e0b7e2696a85861335317e074c8a425ca7418af36673832c198da0f4.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/3f4a8094e0b7e2696a85861335317e074c8a425ca7418af36673832c198da0f4.jpg)

![4878c48fe2f302255ac9a4990e6e8f65ca99b0dbe06c1e07d9a06fee70c61d70.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/4878c48fe2f302255ac9a4990e6e8f65ca99b0dbe06c1e07d9a06fee70c61d70.jpg)

![58efbd7c2c08a72c9a138ce8b55ecd3dcc0f98cbd7d82deca39cc2155831fc3e.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/58efbd7c2c08a72c9a138ce8b55ecd3dcc0f98cbd7d82deca39cc2155831fc3e.jpg)

![6f0fa13afb8b202e06574a597c13819d53e38e519e1c3e1f72c36ff1731dfb92.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/6f0fa13afb8b202e06574a597c13819d53e38e519e1c3e1f72c36ff1731dfb92.jpg)

![a3585cd89b87604cfbaf5fb67347dd388a15690af4dc74ce1056430b069a53a2.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/a3585cd89b87604cfbaf5fb67347dd388a15690af4dc74ce1056430b069a53a2.jpg)

![b580bbf82d7cf75f5ad83be11a63dbcd2828b9e2cb7dc15fae3b5fd40004745f.jpg](../iclr_results/2179_Mask%20in%20the%20Mirror_%20Implicit%20Sparsification/tables/b580bbf82d7cf75f5ad83be11a63dbcd2828b9e2cb7dc15fae3b5fd40004745f.jpg)

## Audio Large Language Models Can Be Descriptive Speech Quality Evaluators


### Images

![4a738754bb3006eee04d9972960e20f9975907b69b54b7703e7d495f204fe38b.jpg](../iclr_results/2180_Audio%20Large%20Language%20Models%20Can%20Be%20Descriptive%20Speech%20Quality%20Evaluators/images/4a738754bb3006eee04d9972960e20f9975907b69b54b7703e7d495f204fe38b.jpg)

![6e3d29082f3381f614e5523eb8ae47f352762d1571e3b2d618065fb4e8eb5366.jpg](../iclr_results/2180_Audio%20Large%20Language%20Models%20Can%20Be%20Descriptive%20Speech%20Quality%20Evaluators/images/6e3d29082f3381f614e5523eb8ae47f352762d1571e3b2d618065fb4e8eb5366.jpg)

![7da118f84d9cc34969a6a077b641eeff90236fe1042bb7e9ab36489d267885c7.jpg](../iclr_results/2180_Audio%20Large%20Language%20Models%20Can%20Be%20Descriptive%20Speech%20Quality%20Evaluators/images/7da118f84d9cc34969a6a077b641eeff90236fe1042bb7e9ab36489d267885c7.jpg)

### Tables

![b9a75c016ebce490792843ccc70f7d6314cee0a0264d14a1d23fc5c6e4f0c34b.jpg](../iclr_results/2180_Audio%20Large%20Language%20Models%20Can%20Be%20Descriptive%20Speech%20Quality%20Evaluators/tables/b9a75c016ebce490792843ccc70f7d6314cee0a0264d14a1d23fc5c6e4f0c34b.jpg)

![d05cb3229c77807ac599ff667ef0d1b0d27f3178c5e252c7fed91cbebf03104f.jpg](../iclr_results/2180_Audio%20Large%20Language%20Models%20Can%20Be%20Descriptive%20Speech%20Quality%20Evaluators/tables/d05cb3229c77807ac599ff667ef0d1b0d27f3178c5e252c7fed91cbebf03104f.jpg)

![d362a0306a00c28babf2fa37ba83c1558369b431bbae779f9a75c0cde59059a6.jpg](../iclr_results/2180_Audio%20Large%20Language%20Models%20Can%20Be%20Descriptive%20Speech%20Quality%20Evaluators/tables/d362a0306a00c28babf2fa37ba83c1558369b431bbae779f9a75c0cde59059a6.jpg)

![d57d11310e140e31fcc8cac0a79f51104c2a162d89cc9df15a6866f0b6f7705f.jpg](../iclr_results/2180_Audio%20Large%20Language%20Models%20Can%20Be%20Descriptive%20Speech%20Quality%20Evaluators/tables/d57d11310e140e31fcc8cac0a79f51104c2a162d89cc9df15a6866f0b6f7705f.jpg)

## Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents


### Images

![2be36223bd58ecf5d858ce051069ff03a9f1ca6a9f690375884370e052ad105e.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/images/2be36223bd58ecf5d858ce051069ff03a9f1ca6a9f690375884370e052ad105e.jpg)

![a07ca4487c84d15ebc5b2576780779950edddace76f03e758bc28ac7d15a4afb.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/images/a07ca4487c84d15ebc5b2576780779950edddace76f03e758bc28ac7d15a4afb.jpg)

![b212aa36466d3c015cc2fa4339633d75d6eab6c4ff54916226eb71b17260c5c3.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/images/b212aa36466d3c015cc2fa4339633d75d6eab6c4ff54916226eb71b17260c5c3.jpg)

![c184d17bdb9466da785f9b3f82fd3d7d737d8bec7d13f6cfbfb61163a5222f7a.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/images/c184d17bdb9466da785f9b3f82fd3d7d737d8bec7d13f6cfbfb61163a5222f7a.jpg)

![c667297865cddb7ddee571b3638b31be41b8bd9169742611f44939ae224421c8.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/images/c667297865cddb7ddee571b3638b31be41b8bd9169742611f44939ae224421c8.jpg)

![d0b0008a63256a7a8cd7676a800d2d1c88a03b63499a28969bf2963a44fcd475.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/images/d0b0008a63256a7a8cd7676a800d2d1c88a03b63499a28969bf2963a44fcd475.jpg)

### Tables

![0b63b74949061d3c5a4eae10b124fc62975f2cb701e7108e49fc7972f6842981.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/tables/0b63b74949061d3c5a4eae10b124fc62975f2cb701e7108e49fc7972f6842981.jpg)

![2ee364e88e425e9c518fd177648df1993a1db7151de5dcabc778ee5278a62c67.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/tables/2ee364e88e425e9c518fd177648df1993a1db7151de5dcabc778ee5278a62c67.jpg)

![56c6a32cc86533d039657773cf804e0700411caebb97f561847290602084db66.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/tables/56c6a32cc86533d039657773cf804e0700411caebb97f561847290602084db66.jpg)

![5a70f26ba7d4f0baa41e8918d24ffc8490666b23f257787cc496186ce4b7fa13.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/tables/5a70f26ba7d4f0baa41e8918d24ffc8490666b23f257787cc496186ce4b7fa13.jpg)

![6bbe2594cf41728f09506a374395a07d36f2f2cb5160ad61b5c53f467c7743ce.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/tables/6bbe2594cf41728f09506a374395a07d36f2f2cb5160ad61b5c53f467c7743ce.jpg)

![7696f8ab46dd1f50a10149e114771981e6e9ae0077518bc5e8349f37e01a33d1.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/tables/7696f8ab46dd1f50a10149e114771981e6e9ae0077518bc5e8349f37e01a33d1.jpg)

![896d5c389eb17c7db8ed6a2e5a00b24da51824ba4bd945c763523c89b9df6bc1.jpg](../iclr_results/2181_Perplexity%20Trap_%20PLM-Based%20Retrievers%20Overrate%20Low%20Perplexity%20Documents/tables/896d5c389eb17c7db8ed6a2e5a00b24da51824ba4bd945c763523c89b9df6bc1.jpg)

## Poisson-Dirac Neural Networks for Modeling Coupled Dynamical Systems across Domains


### Images

![1afc590d8852364f76123c283e854099bb5cac9b6a5b3d4c02fd20ba63b6857d.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/images/1afc590d8852364f76123c283e854099bb5cac9b6a5b3d4c02fd20ba63b6857d.jpg)

![4e9bb95eb65b17acb175d94ccc41659d74a0da1f9a726e23dfe8f6a7ee7dddd2.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/images/4e9bb95eb65b17acb175d94ccc41659d74a0da1f9a726e23dfe8f6a7ee7dddd2.jpg)

![532038a8ec4a4fc929cc6f46d4da0687f39686bf8863e663132fd850bbeb59a8.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/images/532038a8ec4a4fc929cc6f46d4da0687f39686bf8863e663132fd850bbeb59a8.jpg)

![6011aa1a0d97e1727b763555d11fe5f1ebb897323830cf2cdadcf46eef2a1507.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/images/6011aa1a0d97e1727b763555d11fe5f1ebb897323830cf2cdadcf46eef2a1507.jpg)

![c560f891a675aa7e8e4ad987b286a6fed044146fe4335854a3e751a7c1a5d637.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/images/c560f891a675aa7e8e4ad987b286a6fed044146fe4335854a3e751a7c1a5d637.jpg)

### Tables

![2fd03f042ab4bf57dfef3d3ec9c6b334486ed7a0cc06e3c5b8e9c7fa6f2b357f.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/tables/2fd03f042ab4bf57dfef3d3ec9c6b334486ed7a0cc06e3c5b8e9c7fa6f2b357f.jpg)

![65a8123d556df4fb905b3e4296d5e46d54879f4ca7fdc14177be00d40aa06ede.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/tables/65a8123d556df4fb905b3e4296d5e46d54879f4ca7fdc14177be00d40aa06ede.jpg)

![b402fc3baa7bb7b50385ae59ef04626aebc0a262a379aa7b6c4c8e10eb44f459.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/tables/b402fc3baa7bb7b50385ae59ef04626aebc0a262a379aa7b6c4c8e10eb44f459.jpg)

![da355eb09fd2c16d84b543cfa265ad854a9c14cbd8cc037b3b86a26ecea3c221.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/tables/da355eb09fd2c16d84b543cfa265ad854a9c14cbd8cc037b3b86a26ecea3c221.jpg)

![f24169a03ed8f8a830f38105abb325da53c3e984b6ae0da4f708546264fda731.jpg](../iclr_results/2182_Poisson-Dirac%20Neural%20Networks%20for%20Modeling%20Coupled%20Dynamical%20Systems%20across%20Domains/tables/f24169a03ed8f8a830f38105abb325da53c3e984b6ae0da4f708546264fda731.jpg)

## GPS: A Probabilistic Distributional Similarity with Gumbel Priors for Set-to-Set Matching


### Images

![3dabe878d7eac2d99b2b629a683e7a37fea4dc68e8f24fdf899672a4d0af1100.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/3dabe878d7eac2d99b2b629a683e7a37fea4dc68e8f24fdf899672a4d0af1100.jpg)

![3ef62f8dcdb2d4ac346f857ae1034de54d436345a0830bf37a60b4e06de45bd2.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/3ef62f8dcdb2d4ac346f857ae1034de54d436345a0830bf37a60b4e06de45bd2.jpg)

![460ea526118003e3f916e640378b2a1f82dcb82bc14493fb763826fafa037147.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/460ea526118003e3f916e640378b2a1f82dcb82bc14493fb763826fafa037147.jpg)

![59817578b6bd8f6bf3c599a34702e5b613265d7f7618527d428d65bb5c1e8d82.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/59817578b6bd8f6bf3c599a34702e5b613265d7f7618527d428d65bb5c1e8d82.jpg)

![5decfd3cfee67725620bb244dcfeef01fb0bb584016380d030a59f898e0cfe5f.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/5decfd3cfee67725620bb244dcfeef01fb0bb584016380d030a59f898e0cfe5f.jpg)

![8114c5c24b358b84ab7cfdad45892472de614b1b537e953d7a4b248d2027aff9.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/8114c5c24b358b84ab7cfdad45892472de614b1b537e953d7a4b248d2027aff9.jpg)

![ac208bd931ae30a896a8b4bc8e0a9b06469f92cc3dc19cc0ad14118f99ccf2ba.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/ac208bd931ae30a896a8b4bc8e0a9b06469f92cc3dc19cc0ad14118f99ccf2ba.jpg)

![d7b9b7da37997ff0c8dbf58ba563532982362e0a470ad526a0a7c8926cff053d.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/images/d7b9b7da37997ff0c8dbf58ba563532982362e0a470ad526a0a7c8926cff053d.jpg)

### Tables

![1f3d51fcec6ddef835e3b292734310133ca58017e0979a2c3b772d936fcc49f8.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/1f3d51fcec6ddef835e3b292734310133ca58017e0979a2c3b772d936fcc49f8.jpg)

![207937218d5424e903011fba105973632ba5b086729b50f3d8835e1cd220bdd4.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/207937218d5424e903011fba105973632ba5b086729b50f3d8835e1cd220bdd4.jpg)

![273994c13a94a2e331b4c9af61589a6997cfcdaff279193b0f2b4cc8447e30a4.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/273994c13a94a2e331b4c9af61589a6997cfcdaff279193b0f2b4cc8447e30a4.jpg)

![4a867e4919057d1728e4968a5c2705488a641e08ddcc9f8f51e88c23b64dc3f4.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/4a867e4919057d1728e4968a5c2705488a641e08ddcc9f8f51e88c23b64dc3f4.jpg)

![6697c1e640e9b8c6af76fd8b328838eb05ce3ffb2487fcb977f2551cb2fd6e97.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/6697c1e640e9b8c6af76fd8b328838eb05ce3ffb2487fcb977f2551cb2fd6e97.jpg)

![6bda36d6401ca72155dc5a482dc17c1532d1e42f18fdafccc826a87d73e8fee3.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/6bda36d6401ca72155dc5a482dc17c1532d1e42f18fdafccc826a87d73e8fee3.jpg)

![7f108fe064cf1a91a41a5b22e7ebff3dd5cd0895fac46826730918aa7260a213.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/7f108fe064cf1a91a41a5b22e7ebff3dd5cd0895fac46826730918aa7260a213.jpg)

![8dbe98d05e4bbb627db8295d3a70ae4c2cfb4f919f0a410b26b9dbed3937ed65.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/8dbe98d05e4bbb627db8295d3a70ae4c2cfb4f919f0a410b26b9dbed3937ed65.jpg)

![b2ae7c3ff25007406fbec4b9cc7446240d53b76abd393173de4c1808148d3381.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/b2ae7c3ff25007406fbec4b9cc7446240d53b76abd393173de4c1808148d3381.jpg)

![c0f53000a2e8f093b9ccea633db9e5d3a425e99b88b76685a4d1829c156830a9.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/c0f53000a2e8f093b9ccea633db9e5d3a425e99b88b76685a4d1829c156830a9.jpg)

![e15bea6e4f717eb395a9d6874ae31e689d5e86e42d2659c5fdca105cf2dedcd7.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/e15bea6e4f717eb395a9d6874ae31e689d5e86e42d2659c5fdca105cf2dedcd7.jpg)

![f7cda93b9aa15f3489b0cd738d8ea46dbe3e65ac69f516a53b7e94dea48dd554.jpg](../iclr_results/2183_GPS_%20A%20Probabilistic%20Distributional%20Similarity%20with%20Gumbel%20Priors%20for%20Set-to-Set%20Matching/tables/f7cda93b9aa15f3489b0cd738d8ea46dbe3e65ac69f516a53b7e94dea48dd554.jpg)

## Multi-Task Dense Predictions via Unleashing the Power of Diffusion


### Images

![1f923e83ee48be645887089f9dd4357c5110f3af8b6f03fafa1e6014c6dfc6a2.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/images/1f923e83ee48be645887089f9dd4357c5110f3af8b6f03fafa1e6014c6dfc6a2.jpg)

![51e8470328d9f4f7dfb8d464df4a5fc03ed09262582168a996f503c2a74e3fd6.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/images/51e8470328d9f4f7dfb8d464df4a5fc03ed09262582168a996f503c2a74e3fd6.jpg)

![84c4500a62915ec17b6fcd3b56096c9dbb076deddd25ef9a392cbe2d97cc9524.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/images/84c4500a62915ec17b6fcd3b56096c9dbb076deddd25ef9a392cbe2d97cc9524.jpg)

![9c1bb6b58122b4dde0972682ef555e95f8461e9d1ae9b6a07a13e9a0cfd6ab7c.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/images/9c1bb6b58122b4dde0972682ef555e95f8461e9d1ae9b6a07a13e9a0cfd6ab7c.jpg)

![a2081605c24af704cff7905e46c40f628673083c25f05b2412744e1d8c9dfdd1.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/images/a2081605c24af704cff7905e46c40f628673083c25f05b2412744e1d8c9dfdd1.jpg)

![eb452d277fdfe81ba32b2ab633ce63e53c064d91d44f7a5916e7e358fb39da76.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/images/eb452d277fdfe81ba32b2ab633ce63e53c064d91d44f7a5916e7e358fb39da76.jpg)

### Tables

![18086cea3b878209df8b83aa2f0e2802d5d71e57db754927848811f3cb2d6b09.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/18086cea3b878209df8b83aa2f0e2802d5d71e57db754927848811f3cb2d6b09.jpg)

![3fda9c1ed60e16e2fcccde03a2b01f2db2e2865e273f33a755aec15f66433c94.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/3fda9c1ed60e16e2fcccde03a2b01f2db2e2865e273f33a755aec15f66433c94.jpg)

![509c5d4dad636abf4cd762adf642557f5d5deeb6b3ec9ecfb0cc3234bf0d205a.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/509c5d4dad636abf4cd762adf642557f5d5deeb6b3ec9ecfb0cc3234bf0d205a.jpg)

![5d7fe0077a7ae334d5b368aee5e81bcf891e1273e43f2d08829436176f4e218f.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/5d7fe0077a7ae334d5b368aee5e81bcf891e1273e43f2d08829436176f4e218f.jpg)

![8e9c0c4bdc6227d96621e20d6db1e0a795697e3480b64f9fd41d68b42a761c95.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/8e9c0c4bdc6227d96621e20d6db1e0a795697e3480b64f9fd41d68b42a761c95.jpg)

![9913b101a39cd4006b7bb0b6b09a9c634ce489b5914982233c2517a14e94173c.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/9913b101a39cd4006b7bb0b6b09a9c634ce489b5914982233c2517a14e94173c.jpg)

![cfa035964c5e8f6c508646b260322ed2f5e8c9bf72906150a17cd7f7f85169db.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/cfa035964c5e8f6c508646b260322ed2f5e8c9bf72906150a17cd7f7f85169db.jpg)

![d7f44902ff6d83f81f614248dc71f949ca1c614be4dba1586de3a234b01cb951.jpg](../iclr_results/2184_Multi-Task%20Dense%20Predictions%20via%20Unleashing%20the%20Power%20of%20Diffusion/tables/d7f44902ff6d83f81f614248dc71f949ca1c614be4dba1586de3a234b01cb951.jpg)

## Discovering Group Structures via Unitary Representation Learning


### Images

![11e3e54b119675b44201dfb250b8b4e55410f8ac45111b08dbeb526fdae550c2.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/11e3e54b119675b44201dfb250b8b4e55410f8ac45111b08dbeb526fdae550c2.jpg)

![286a154b9802249dc870982e87172114d5824e542a491736bafc4d786bfd2a5c.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/286a154b9802249dc870982e87172114d5824e542a491736bafc4d786bfd2a5c.jpg)

![444ad53375d92dda968a8aa9d505f675e860273ab3be0a5a5ccaeb2c4f37987f.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/444ad53375d92dda968a8aa9d505f675e860273ab3be0a5a5ccaeb2c4f37987f.jpg)

![4daac86100ac2a8aa0f9a880d798500f81c785ece07ea908dd91cc03b8eb860e.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/4daac86100ac2a8aa0f9a880d798500f81c785ece07ea908dd91cc03b8eb860e.jpg)

![566af6bd59cb6dc4ecfc4920b28f9c8ca2f284cecb35a56d86e0fedae0221a0d.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/566af6bd59cb6dc4ecfc4920b28f9c8ca2f284cecb35a56d86e0fedae0221a0d.jpg)

![7e81e91be7f48dea1ca6a4dcb3869c440353561b2a239f425983e9adc287ced6.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/7e81e91be7f48dea1ca6a4dcb3869c440353561b2a239f425983e9adc287ced6.jpg)

![9437fb26d17093e1879adc138f0ffc1399cb7526a0fbd17cef38ab55e1b068f8.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/9437fb26d17093e1879adc138f0ffc1399cb7526a0fbd17cef38ab55e1b068f8.jpg)

![96ae9f2dd1a967c67f3590b848ce6f5ce41498f9e658f5b8db626e640554e557.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/96ae9f2dd1a967c67f3590b848ce6f5ce41498f9e658f5b8db626e640554e557.jpg)

![a41521ba7f690a1e4b07aca9ab57e901a2dc1726be6f4b3d116a57683aad0290.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/a41521ba7f690a1e4b07aca9ab57e901a2dc1726be6f4b3d116a57683aad0290.jpg)

![a725e9b80c2e15133d0e8160c409a65fdd7b4d3dce43070de2a00ddf5834120f.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/a725e9b80c2e15133d0e8160c409a65fdd7b4d3dce43070de2a00ddf5834120f.jpg)

![a822b157f2c0dbc230d188bddb283c0562fe0ac30d43a05a1b3389b4a58d1177.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/a822b157f2c0dbc230d188bddb283c0562fe0ac30d43a05a1b3389b4a58d1177.jpg)

![a9f7d1ba1fc33e7bff81b934e9a9aedf50e073e169924e2c854ad8841e0295d7.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/a9f7d1ba1fc33e7bff81b934e9a9aedf50e073e169924e2c854ad8841e0295d7.jpg)

![b7d29b2fd847e00e5e1c2b1013ab0ee8900e56809d51766c7d972342a32c8e73.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/b7d29b2fd847e00e5e1c2b1013ab0ee8900e56809d51766c7d972342a32c8e73.jpg)

![bb6803d05891b96b339e98b0b38bce4cc9083ba1e5235129e48f080b866ab593.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/bb6803d05891b96b339e98b0b38bce4cc9083ba1e5235129e48f080b866ab593.jpg)

![c18b7367c7847963a83730cd5feba74093d635cb0deb1aebe84bb129ea6d4bb3.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/c18b7367c7847963a83730cd5feba74093d635cb0deb1aebe84bb129ea6d4bb3.jpg)

![caf5b13661e45ac2c8fccd7f27ccc88cd0c1146deadaca4e3de14a1ab0f926db.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/caf5b13661e45ac2c8fccd7f27ccc88cd0c1146deadaca4e3de14a1ab0f926db.jpg)

![d160176f19736163ddbe19e3abc8c6299064d8724338040b462d8d38163f30ae.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/d160176f19736163ddbe19e3abc8c6299064d8724338040b462d8d38163f30ae.jpg)

![d164d0a8a98cbea151f06ba7071cbfe4631d65750c9d48cf5455e5e1c4027266.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/d164d0a8a98cbea151f06ba7071cbfe4631d65750c9d48cf5455e5e1c4027266.jpg)

![d66938f58f73da5e9c66895f44052d00b047b9f359ad4eeee594f6f90145fcd5.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/d66938f58f73da5e9c66895f44052d00b047b9f359ad4eeee594f6f90145fcd5.jpg)

![ff22239b8e6b9707d4641585888821571be92a095ce2fa50273603f20555182f.jpg](../iclr_results/2185_Discovering%20Group%20Structures%20via%20Unitary%20Representation%20Learning/images/ff22239b8e6b9707d4641585888821571be92a095ce2fa50273603f20555182f.jpg)

## Newton Meets Marchenko-Pastur: Massively Parallel Second-Order Optimization with Hessian Sketching and Debiasing


### Images

![12ed342a7cf8a445c179e3ab2bdbc2b7ae5b19c9c4419efdb0f9fb7deba5876e.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/images/12ed342a7cf8a445c179e3ab2bdbc2b7ae5b19c9c4419efdb0f9fb7deba5876e.jpg)

![63798640001b85e635719a1cd30137924656dedc71c42fe3a5070d35d95672e5.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/images/63798640001b85e635719a1cd30137924656dedc71c42fe3a5070d35d95672e5.jpg)

![7618d77a71e354445767258a756f2f8a957b6fca5b6e39e0b810060a0804655e.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/images/7618d77a71e354445767258a756f2f8a957b6fca5b6e39e0b810060a0804655e.jpg)

![b602637efe97b4f21ef67eb14cf4bff2e69da9a78339ad6392a7921d5af8c205.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/images/b602637efe97b4f21ef67eb14cf4bff2e69da9a78339ad6392a7921d5af8c205.jpg)

![bc3fa9439e0fb026da96f19345569f575d2641dfe16984c766ed40394117fa58.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/images/bc3fa9439e0fb026da96f19345569f575d2641dfe16984c766ed40394117fa58.jpg)

![d654baf3955de2fd16dc4617f07794f6e7095c8409718ffb7040a79cec4e51e8.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/images/d654baf3955de2fd16dc4617f07794f6e7095c8409718ffb7040a79cec4e51e8.jpg)

### Tables

![43d0d11f91eccf61d538f17dfd18f42a44fe48eb6310c6f66e4e3d0e4b20aed5.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/tables/43d0d11f91eccf61d538f17dfd18f42a44fe48eb6310c6f66e4e3d0e4b20aed5.jpg)

![e88a8660cae1df8ebd0d3d5d7fca1d8ac372149c3befb25c97ad5e905d04c1eb.jpg](../iclr_results/2186_Newton%20Meets%20Marchenko-Pastur_%20Massively%20Parallel%20Second-Order%20Optimization%20with%20Hessian%20Sketching%20a/tables/e88a8660cae1df8ebd0d3d5d7fca1d8ac372149c3befb25c97ad5e905d04c1eb.jpg)

## Grokking at the Edge of Numerical Stability


### Images

![0a69efb68c16de28452338ef433ebac941cd0e140b65dfc7d556ff4a615d71f7.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/0a69efb68c16de28452338ef433ebac941cd0e140b65dfc7d556ff4a615d71f7.jpg)

![0ca2aadb8b544eedcfeb60b37263ba8f91d0edd4f345aab31605c5aebd92ae64.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/0ca2aadb8b544eedcfeb60b37263ba8f91d0edd4f345aab31605c5aebd92ae64.jpg)

![13606233eaf14ac6e6707adffa5c70a95535b98fc233f85a66e2361e352ec580.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/13606233eaf14ac6e6707adffa5c70a95535b98fc233f85a66e2361e352ec580.jpg)

![277a73c94e3b3cb6b475a630455ea74d5f5b68d3c40314b0a06aee584ad17a3d.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/277a73c94e3b3cb6b475a630455ea74d5f5b68d3c40314b0a06aee584ad17a3d.jpg)

![2d5cdf41133e8450cfdffbee061496c87765b38bf78189316b8cb45fd9252135.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/2d5cdf41133e8450cfdffbee061496c87765b38bf78189316b8cb45fd9252135.jpg)

![4085875f94bc5402cc0ec3304122da336e0d70f876d472393a1378c0b6c5a77d.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/4085875f94bc5402cc0ec3304122da336e0d70f876d472393a1378c0b6c5a77d.jpg)

![573cfc7c2c7ced2503a29ca106d090a113f0379decc13021ccf7d9407d092d94.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/573cfc7c2c7ced2503a29ca106d090a113f0379decc13021ccf7d9407d092d94.jpg)

![66e6171d277557c0396d3fb11980a7453a8b701aedb13acc84cfb029f786ed4d.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/66e6171d277557c0396d3fb11980a7453a8b701aedb13acc84cfb029f786ed4d.jpg)

![835d51c6c89ab4b3992752417162ca0046a5e561cd5dd86ff0f85702e5c0d7c5.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/835d51c6c89ab4b3992752417162ca0046a5e561cd5dd86ff0f85702e5c0d7c5.jpg)

![986803c516bd12f17cac52fd55cbc495baf8648046aaae980187f7b282be50e0.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/986803c516bd12f17cac52fd55cbc495baf8648046aaae980187f7b282be50e0.jpg)

![a71459ec292310d9baaed3177d08c6024a3da90d505b7f53b11f3022c3a46de3.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/a71459ec292310d9baaed3177d08c6024a3da90d505b7f53b11f3022c3a46de3.jpg)

![bae2a37c2f65bbdf2d070ca954f01e1114a78d7682c2f6afc367917178fdee43.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/bae2a37c2f65bbdf2d070ca954f01e1114a78d7682c2f6afc367917178fdee43.jpg)

![c210830af136391bc745387ca734e7dd78c020d7d1289e647e431791dbea622c.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/c210830af136391bc745387ca734e7dd78c020d7d1289e647e431791dbea622c.jpg)

![c4d47e0b0bd1813a0718def50cc3380c09271fd4fd04bbf59b939275e3365850.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/c4d47e0b0bd1813a0718def50cc3380c09271fd4fd04bbf59b939275e3365850.jpg)

![d66b0da1c3a09b80cb1b5e63b3f5db398d7e3e8d330ac481e0d07aa4c8c67c0d.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/d66b0da1c3a09b80cb1b5e63b3f5db398d7e3e8d330ac481e0d07aa4c8c67c0d.jpg)

![f22c3cc042add38fb65790c2df5a5ef882a97da69d1a0929e0acfd86a3a9c1d9.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/f22c3cc042add38fb65790c2df5a5ef882a97da69d1a0929e0acfd86a3a9c1d9.jpg)

![fba2356bf8b3daa9d39218c93c4a619f15ab3af1b5df0b4c0fd4525c890c2787.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/images/fba2356bf8b3daa9d39218c93c4a619f15ab3af1b5df0b4c0fd4525c890c2787.jpg)

### Tables

![342f135f8932a51f39e29f0b5fe8bfd0325f7c7f76b3363959cfe99c20798340.jpg](../iclr_results/2187_Grokking%20at%20the%20Edge%20of%20Numerical%20Stability/tables/342f135f8932a51f39e29f0b5fe8bfd0325f7c7f76b3363959cfe99c20798340.jpg)

## MELODI: Exploring Memory Compression for Long Contexts


### Images

![01cf7f1cdb8b291765c3aee91097bbe82ab17f6a1fef91b51cf62d283503247e.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/01cf7f1cdb8b291765c3aee91097bbe82ab17f6a1fef91b51cf62d283503247e.jpg)

![09ed3bdbba7f93f13bba0883050705bac779490e61ce7f3e07514ddc159b34c5.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/09ed3bdbba7f93f13bba0883050705bac779490e61ce7f3e07514ddc159b34c5.jpg)

![16d1e256ababf398833501a16ae3b58de01a5a610569cce298f6116c51b5e1a4.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/16d1e256ababf398833501a16ae3b58de01a5a610569cce298f6116c51b5e1a4.jpg)

![1f3ef0e1e61612e2e88956d764dd079ff51bcc5a37f5517dc7c7631680b0819e.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/1f3ef0e1e61612e2e88956d764dd079ff51bcc5a37f5517dc7c7631680b0819e.jpg)

![4d61db041e9059abad9ba5b8ce69b236574679629f4036ea63928cafad0d820a.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/4d61db041e9059abad9ba5b8ce69b236574679629f4036ea63928cafad0d820a.jpg)

![90b8565f6426381d53c4c26ce974ffc4e8f6d582688dc7e6eef44cd710a2788a.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/90b8565f6426381d53c4c26ce974ffc4e8f6d582688dc7e6eef44cd710a2788a.jpg)

![b002236997a04adb28ca43ea15c9b5404b1868518dcbd8b2668c7a8255567d94.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/b002236997a04adb28ca43ea15c9b5404b1868518dcbd8b2668c7a8255567d94.jpg)

![f4319c46943cc2d6386f87a789f5605e8ae03807606967890a5fca2c6442c78c.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/images/f4319c46943cc2d6386f87a789f5605e8ae03807606967890a5fca2c6442c78c.jpg)

### Tables

![14047ffd9608de93371cf3410a6d08c1f6ab53fed9895c9b9a96d2194203d7e8.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/14047ffd9608de93371cf3410a6d08c1f6ab53fed9895c9b9a96d2194203d7e8.jpg)

![1a5012938ec7d3ad780e1ae0e289a8fb5648ae8e179a65e02c957e6534460e84.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/1a5012938ec7d3ad780e1ae0e289a8fb5648ae8e179a65e02c957e6534460e84.jpg)

![332d8b48516526cfa713e52a239682af2f5e439b36a29fec136b8589cfbba38c.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/332d8b48516526cfa713e52a239682af2f5e439b36a29fec136b8589cfbba38c.jpg)

![5093b5a4ceec2595d733aa514521b48b181599291017ce58422ea0f45107cdda.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/5093b5a4ceec2595d733aa514521b48b181599291017ce58422ea0f45107cdda.jpg)

![5e39ad15e3c870c90a737691bed0680b06fa86083e2d6f9b729a4e41e015ebf3.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/5e39ad15e3c870c90a737691bed0680b06fa86083e2d6f9b729a4e41e015ebf3.jpg)

![6e0c68b01647c0a3868e99d35e2708a5585941091f8d345cdee3d8448fe7154a.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/6e0c68b01647c0a3868e99d35e2708a5585941091f8d345cdee3d8448fe7154a.jpg)

![93b70ceada631bd80fbd37d519455ce357c8170f29dd21f531ab10155fe94633.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/93b70ceada631bd80fbd37d519455ce357c8170f29dd21f531ab10155fe94633.jpg)

![b4085f9996f3bc24511c150cb60cecc372af6d9864f5c03e2ce7c46c1a04e60a.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/b4085f9996f3bc24511c150cb60cecc372af6d9864f5c03e2ce7c46c1a04e60a.jpg)

![bc9b233d31b5bdcc8412f4fb57b9f6ac58f6da0d36f29d2dc02111b72a64c7f9.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/bc9b233d31b5bdcc8412f4fb57b9f6ac58f6da0d36f29d2dc02111b72a64c7f9.jpg)

![c50e4043c45f1cba4dcdaa9b68016e962fa8080a2263c47bc872fe022f103482.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/c50e4043c45f1cba4dcdaa9b68016e962fa8080a2263c47bc872fe022f103482.jpg)

![c634d678f815f7a722a1b97c9fe6d24c251c3ebf29bb132ebbab4aea03bf7f4b.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/c634d678f815f7a722a1b97c9fe6d24c251c3ebf29bb132ebbab4aea03bf7f4b.jpg)

![c9d88436f00156a20b122b37e849188dd5d9b7f549e09a7f11586777319b42e0.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/c9d88436f00156a20b122b37e849188dd5d9b7f549e09a7f11586777319b42e0.jpg)

![dedf1cfaf2d332b4e57fcf4bec5091b5223019ae23fce1a187ea84dcec396aee.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/dedf1cfaf2d332b4e57fcf4bec5091b5223019ae23fce1a187ea84dcec396aee.jpg)

![e23a064d7c6e1bf6340f6372f6573a812bf1f4656b67a37a3df987ddbf7d394a.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/e23a064d7c6e1bf6340f6372f6573a812bf1f4656b67a37a3df987ddbf7d394a.jpg)

![e4545e18fbb231b72a969d76d0be9441dfb5002cd1ed1815735ade4d24a59619.jpg](../iclr_results/2188_MELODI_%20Exploring%20Memory%20Compression%20for%20Long%20Contexts/tables/e4545e18fbb231b72a969d76d0be9441dfb5002cd1ed1815735ade4d24a59619.jpg)

## MIND: Math Informed syNthetic Dialogues for Pretraining LLMs


### Images

![2a59ada460154cc19033d8a964ff5d3907fe98a9ff13af9a4522cf21d1830b7c.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/images/2a59ada460154cc19033d8a964ff5d3907fe98a9ff13af9a4522cf21d1830b7c.jpg)

![3ad7394541daefcacbd6c0796e5bd881ec51967160cc79dbcc848d54d60d798c.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/images/3ad7394541daefcacbd6c0796e5bd881ec51967160cc79dbcc848d54d60d798c.jpg)

![5ebd91e4f50480ad1feb26e9930bd15b0910403306b2fb0d50b3dc7581c50c69.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/images/5ebd91e4f50480ad1feb26e9930bd15b0910403306b2fb0d50b3dc7581c50c69.jpg)

![975c56bb9281567ee19b0fdf410b68d1a541c90557a65a9c28d8f109db83462d.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/images/975c56bb9281567ee19b0fdf410b68d1a541c90557a65a9c28d8f109db83462d.jpg)

![af578e5fe37eaa9a6cc8709c5a3de444f3255d193224b15d284132575c458e75.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/images/af578e5fe37eaa9a6cc8709c5a3de444f3255d193224b15d284132575c458e75.jpg)

### Tables

![1a68c6c8d953f24a76cb89cf5360d6f572768b33cea45b1ba772b5a082395a5e.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/1a68c6c8d953f24a76cb89cf5360d6f572768b33cea45b1ba772b5a082395a5e.jpg)

![2a088398538491b988d918d27ed9530ec7aa2351910f29b26eea3b5fe3772ad3.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/2a088398538491b988d918d27ed9530ec7aa2351910f29b26eea3b5fe3772ad3.jpg)

![491317e0d04b1ea8fd89d7a4538018afba39188cf17abe3479a7165782cab6e9.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/491317e0d04b1ea8fd89d7a4538018afba39188cf17abe3479a7165782cab6e9.jpg)

![50aef77ef52591784b71153c6801ce93a7609dda79eff6319841edd3c5542756.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/50aef77ef52591784b71153c6801ce93a7609dda79eff6319841edd3c5542756.jpg)

![7fe97612cca6cfa9125b6b4a3f2d06c6222c979c9a4df921fa91ccc614fdbb50.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/7fe97612cca6cfa9125b6b4a3f2d06c6222c979c9a4df921fa91ccc614fdbb50.jpg)

![87f1a7c8e89148143c8c216754a11f3bef58992d87f927d151ec72268c9fa6a0.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/87f1a7c8e89148143c8c216754a11f3bef58992d87f927d151ec72268c9fa6a0.jpg)

![b4ed2a34c35718d31d391219842a6c42c96b593c2ed8d75894ca5ca7faf6f6ec.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/b4ed2a34c35718d31d391219842a6c42c96b593c2ed8d75894ca5ca7faf6f6ec.jpg)

![b8ea9b138e4bbeb2a34df04b7ec0364da4deaa86043bfd6f926bd90ecf410296.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/b8ea9b138e4bbeb2a34df04b7ec0364da4deaa86043bfd6f926bd90ecf410296.jpg)

![c108d56b2608e413b84453f975c855fed626e8628dfce4ca244f318d06daa8d6.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/c108d56b2608e413b84453f975c855fed626e8628dfce4ca244f318d06daa8d6.jpg)

![ca5dd620c6b0331c33b802ef6e9d2dd369a48198b4c68480582005e9d695f148.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/ca5dd620c6b0331c33b802ef6e9d2dd369a48198b4c68480582005e9d695f148.jpg)

![d060c817d866501193dea811be657f17dd710de91ae8237fb1a01c1c5a369015.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/d060c817d866501193dea811be657f17dd710de91ae8237fb1a01c1c5a369015.jpg)

![d7360ebd46b30f700214738974d9410693780546b4d77896898d9a82b4f43d6b.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/d7360ebd46b30f700214738974d9410693780546b4d77896898d9a82b4f43d6b.jpg)

![e409db9df70d58472eb7f81c0bd5fd27f8bf6012a16d5a03f4d6966fc72e7ddb.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/e409db9df70d58472eb7f81c0bd5fd27f8bf6012a16d5a03f4d6966fc72e7ddb.jpg)

![e7648724ab71437948b16821907ff216c197da46e735d7f38e2451d853faf819.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/e7648724ab71437948b16821907ff216c197da46e735d7f38e2451d853faf819.jpg)

![f03882c2ede5edf3b2a21fa4a90db3a08b3b4c26e59952a3648e07e602449e7f.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/f03882c2ede5edf3b2a21fa4a90db3a08b3b4c26e59952a3648e07e602449e7f.jpg)

![f1861ddb4f0712db776f64b3e0dd7ad7429de56aece851e44b74d89c23265c9b.jpg](../iclr_results/2189_MIND_%20Math%20Informed%20syNthetic%20Dialogues%20for%20Pretraining%20LLMs/tables/f1861ddb4f0712db776f64b3e0dd7ad7429de56aece851e44b74d89c23265c9b.jpg)

## T2V2: A Unified Non-Autoregressive Model for Speech Recognition and Synthesis via Multitask Learning


### Images

![d5980a04789f09da3ce611a728b17d6d06a406025b85e4f02e2ea701b2740d0c.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/images/d5980a04789f09da3ce611a728b17d6d06a406025b85e4f02e2ea701b2740d0c.jpg)

### Tables

![1cab1ee39631332cade96055c8868b18ab046870e2b22a21d443c1d98d0a1459.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/1cab1ee39631332cade96055c8868b18ab046870e2b22a21d443c1d98d0a1459.jpg)

![361eef9c84672a420ec3eab857ea1a0f191eaa00b2de7a59aa4ce55f1480b43c.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/361eef9c84672a420ec3eab857ea1a0f191eaa00b2de7a59aa4ce55f1480b43c.jpg)

![498c862d2c4219c39b58e95019d9020ca1bb07b28199cf9d460d4ddc66992d5d.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/498c862d2c4219c39b58e95019d9020ca1bb07b28199cf9d460d4ddc66992d5d.jpg)

![75f57c65440fb4a8822d246387454f2879d320e9f6aa22810fec30a86a8ff660.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/75f57c65440fb4a8822d246387454f2879d320e9f6aa22810fec30a86a8ff660.jpg)

![7b896a0ffd3e4ebd312cf16b0149b16d234e23da84730020b65f32509b19ff0b.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/7b896a0ffd3e4ebd312cf16b0149b16d234e23da84730020b65f32509b19ff0b.jpg)

![814332f71a2bc5431962c26e1cec51e56df26d74146bc8683c7dfce86982687d.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/814332f71a2bc5431962c26e1cec51e56df26d74146bc8683c7dfce86982687d.jpg)

![9fde795d1a6b50d3a296cf72587b10d2d78955e3a9162c5e6c7c070e8a86e442.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/9fde795d1a6b50d3a296cf72587b10d2d78955e3a9162c5e6c7c070e8a86e442.jpg)

![b3c89529ab3eb99ec945a8f5b6c5d876ce07f54da1683cba5a089d04d8447733.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/b3c89529ab3eb99ec945a8f5b6c5d876ce07f54da1683cba5a089d04d8447733.jpg)

![bb823fd7396e0b27f9ccea9843f46cd13a5b82ad5da4362134c78b822cda676f.jpg](../iclr_results/2190_T2V2_%20A%20Unified%20Non-Autoregressive%20Model%20for%20Speech%20Recognition%20and%20Synthesis%20via%20Multitask%20Learning/tables/bb823fd7396e0b27f9ccea9843f46cd13a5b82ad5da4362134c78b822cda676f.jpg)

## FreCaS: Efficient Higher-Resolution Image Generation via Frequency-aware Cascaded Sampling


### Images

![1bdae394adc0ad53cf5d6387bc2138b62246a6af618bc0ff93c885aa7f651f1c.jpg](../iclr_results/2191_FreCaS_%20Efficient%20Higher-Resolution%20Image%20Generation%20via%20Frequency-aware%20Cascaded%20Sampling/images/1bdae394adc0ad53cf5d6387bc2138b62246a6af618bc0ff93c885aa7f651f1c.jpg)

![467eb0394e5be97a30ecd50fabd14d706b40e20f50811c4f718146c6d95680dd.jpg](../iclr_results/2191_FreCaS_%20Efficient%20Higher-Resolution%20Image%20Generation%20via%20Frequency-aware%20Cascaded%20Sampling/images/467eb0394e5be97a30ecd50fabd14d706b40e20f50811c4f718146c6d95680dd.jpg)

![53fffc5f9079d52146d9c0c98b9db93aebdbc107415f5efdf9f864ce43f31ca9.jpg](../iclr_results/2191_FreCaS_%20Efficient%20Higher-Resolution%20Image%20Generation%20via%20Frequency-aware%20Cascaded%20Sampling/images/53fffc5f9079d52146d9c0c98b9db93aebdbc107415f5efdf9f864ce43f31ca9.jpg)

![9a9c1380ab0ffe4f4f58e26e7d97b1a43f88260de04f0165a28d5ccd3425df8c.jpg](../iclr_results/2191_FreCaS_%20Efficient%20Higher-Resolution%20Image%20Generation%20via%20Frequency-aware%20Cascaded%20Sampling/images/9a9c1380ab0ffe4f4f58e26e7d97b1a43f88260de04f0165a28d5ccd3425df8c.jpg)

![a207bee0b02da3a328a7f12ff4a560791a64f33e298c22520fef9e4407e7ff94.jpg](../iclr_results/2191_FreCaS_%20Efficient%20Higher-Resolution%20Image%20Generation%20via%20Frequency-aware%20Cascaded%20Sampling/images/a207bee0b02da3a328a7f12ff4a560791a64f33e298c22520fef9e4407e7ff94.jpg)

![f930c540b81be382534193534c033b24afac285439580f8bdca31124c4ebd6ad.jpg](../iclr_results/2191_FreCaS_%20Efficient%20Higher-Resolution%20Image%20Generation%20via%20Frequency-aware%20Cascaded%20Sampling/images/f930c540b81be382534193534c033b24afac285439580f8bdca31124c4ebd6ad.jpg)

### Tables

![5592bc6d7aff720b8709eb08f75ab46300b6c80d8c72784ae3fd4a6886e95c0a.jpg](../iclr_results/2191_FreCaS_%20Efficient%20Higher-Resolution%20Image%20Generation%20via%20Frequency-aware%20Cascaded%20Sampling/tables/5592bc6d7aff720b8709eb08f75ab46300b6c80d8c72784ae3fd4a6886e95c0a.jpg)

## AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation


### Images

![05b8f994d33b235d4b1e6452b43370b8b797d49b3e15bb29ef01e48a17454c43.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/images/05b8f994d33b235d4b1e6452b43370b8b797d49b3e15bb29ef01e48a17454c43.jpg)

![1b78e7eb5e4d68fc9d59132e8d3776028e0bbde9e88f4071b8660bc3f4662fd7.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/images/1b78e7eb5e4d68fc9d59132e8d3776028e0bbde9e88f4071b8660bc3f4662fd7.jpg)

![1eb58d4da1e3b0f1acd3ef83bfb5307c25ed8e6a55eebfedf5c644a4a59138d1.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/images/1eb58d4da1e3b0f1acd3ef83bfb5307c25ed8e6a55eebfedf5c644a4a59138d1.jpg)

![3914e1801d1a0a711dd44b52eea3ed98fea26c9ffadca83c2182582bcd4b1bef.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/images/3914e1801d1a0a711dd44b52eea3ed98fea26c9ffadca83c2182582bcd4b1bef.jpg)

![a9a7ead2e4d0b13e726b51f9184b81b9df1cfa204bac1145a545045523586fef.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/images/a9a7ead2e4d0b13e726b51f9184b81b9df1cfa204bac1145a545045523586fef.jpg)

![f686b4e37cc764d34fcface7bd8fb0dc82c578ecc0a9aadecbbc0a89148763f8.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/images/f686b4e37cc764d34fcface7bd8fb0dc82c578ecc0a9aadecbbc0a89148763f8.jpg)

### Tables

![0a4aef40e69020b9419e6603afe9c5fd67b23af08c01bd9d4c551a5200971d4d.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/tables/0a4aef40e69020b9419e6603afe9c5fd67b23af08c01bd9d4c551a5200971d4d.jpg)

![eb74a62d7ca9f897c01307a67c0d143f724c336dcbb5b9857a7454bd7979b901.jpg](../iclr_results/2192_AvatarGO_%20Zero-shot%204D%20Human-Object%20Interaction%20Generation%20and%20Animation/tables/eb74a62d7ca9f897c01307a67c0d143f724c336dcbb5b9857a7454bd7979b901.jpg)

## MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding


### Images

![0eed7ddf3dc7e60c9c287e112c6aec7a47555cab3a8a911a84b630b75d2d72cd.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/0eed7ddf3dc7e60c9c287e112c6aec7a47555cab3a8a911a84b630b75d2d72cd.jpg)

![271e07e35c00907a654fdb4e8ed42476b09d0b58a3d5bf52ebef4f9b24867e91.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/271e07e35c00907a654fdb4e8ed42476b09d0b58a3d5bf52ebef4f9b24867e91.jpg)

![273ed8cff5f9095ffefa60e08ca2092c064213a04169d252aa7a462437755af7.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/273ed8cff5f9095ffefa60e08ca2092c064213a04169d252aa7a462437755af7.jpg)

![2c7b85b4e0aa3453bc46c316f72e9cabf2a204b03cff4d665c0a2e94343e2f43.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/2c7b85b4e0aa3453bc46c316f72e9cabf2a204b03cff4d665c0a2e94343e2f43.jpg)

![4678a56218d885402940d2f3b12ff0c0883b16a17e7fa5107f525bf9fd624565.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/4678a56218d885402940d2f3b12ff0c0883b16a17e7fa5107f525bf9fd624565.jpg)

![539bcd0fbccc00820341e654397c2c0b819709e1c7e3c60c91d48eb8a8df07e4.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/539bcd0fbccc00820341e654397c2c0b819709e1c7e3c60c91d48eb8a8df07e4.jpg)

![7f7b31867d2f6f8522358130b13c5243b50ad2ac4bfe352b604a0e27c9216ec2.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/7f7b31867d2f6f8522358130b13c5243b50ad2ac4bfe352b604a0e27c9216ec2.jpg)

![88886b11b4c2805ddb8e92f486ceb1a8a435413abee1379f4fcade35d364e622.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/88886b11b4c2805ddb8e92f486ceb1a8a435413abee1379f4fcade35d364e622.jpg)

![8b7eb37443481b1de4f24cb34b06db9299f6c1a7294a5627e17b6eb04e06bd50.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/8b7eb37443481b1de4f24cb34b06db9299f6c1a7294a5627e17b6eb04e06bd50.jpg)

![979a799e56df0d334e4288c737c4ee5c28ffee1ff2acc632ad1cd68f9ed095b0.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/979a799e56df0d334e4288c737c4ee5c28ffee1ff2acc632ad1cd68f9ed095b0.jpg)

![b62ec07f839abfd5be282a2598a32b147883e6a5c424356fe23e73f76cc82a3a.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/b62ec07f839abfd5be282a2598a32b147883e6a5c424356fe23e73f76cc82a3a.jpg)

![ca69eb1c00acea5a869e3fbc682a35b648ba91f24f434137e87f7cd090ad40b5.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/ca69eb1c00acea5a869e3fbc682a35b648ba91f24f434137e87f7cd090ad40b5.jpg)

![e2a9b396d62471dd648e8429eadf2ecb06f78b9015e2d4ae27d090a4b9b2fbd8.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/e2a9b396d62471dd648e8429eadf2ecb06f78b9015e2d4ae27d090a4b9b2fbd8.jpg)

![fa8a337db30842e8172d5845c4d6abe61c8f4a36652db4c66a8e0809f85f1dc6.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/images/fa8a337db30842e8172d5845c4d6abe61c8f4a36652db4c66a8e0809f85f1dc6.jpg)

### Tables

![02d705654da09f040b46c4bebb6d8dd998d61d5475dc509477381e3a9b288178.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/02d705654da09f040b46c4bebb6d8dd998d61d5475dc509477381e3a9b288178.jpg)

![2de29c26c6ab890df1dcd30714257262e94312c2b650ab3226e207aa73117562.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/2de29c26c6ab890df1dcd30714257262e94312c2b650ab3226e207aa73117562.jpg)

![32a4bef581003e649f02f88e92c22ad93c69377d4a045134ffedaeb0a56dbd92.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/32a4bef581003e649f02f88e92c22ad93c69377d4a045134ffedaeb0a56dbd92.jpg)

![9d093dc1e6e82e8f2c96f7e6c81eb93a3bf3cb368c2392ceb601107040886186.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/9d093dc1e6e82e8f2c96f7e6c81eb93a3bf3cb368c2392ceb601107040886186.jpg)

![c1dd7acd62f487678f1c8fdd3392af4244dc8d11cb0c4571a9e1a88c10926e7f.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/c1dd7acd62f487678f1c8fdd3392af4244dc8d11cb0c4571a9e1a88c10926e7f.jpg)

![de7c7e44c3eaa974b51c2bec04492d55644d38c2759a379c4869fbfa090cfc1c.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/de7c7e44c3eaa974b51c2bec04492d55644d38c2759a379c4869fbfa090cfc1c.jpg)

![ea93369e2f51a588af0937d7194eb79093a6b5e2364c8ad732bbc08062e8bc33.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/ea93369e2f51a588af0937d7194eb79093a6b5e2364c8ad732bbc08062e8bc33.jpg)

![ed3d909896ae69be022c98e8160f4c9f7736e43244b7a669ce5af93a5c6c8d6a.jpg](../iclr_results/2193_MuirBench_%20A%20Comprehensive%20Benchmark%20for%20Robust%20Multi-image%20Understanding/tables/ed3d909896ae69be022c98e8160f4c9f7736e43244b7a669ce5af93a5c6c8d6a.jpg)

## A Little Goes a Long Way: Efficient Long Context Training and Inference with Partial Contexts

### Images

![3eed5ab150d95716223df81173f252f0ca8a01b8e100ccd3bfa7c80fc938682c.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/images/3eed5ab150d95716223df81173f252f0ca8a01b8e100ccd3bfa7c80fc938682c.jpg)

![7e3b7dbee2f9a1553569bc10f13bcb04735b8f0b2a4d5edd7b6a503dca6e59ab.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/images/7e3b7dbee2f9a1553569bc10f13bcb04735b8f0b2a4d5edd7b6a503dca6e59ab.jpg)

![a7f719e84a93f4f2258b1ce4bba1c59a40306970f564f56b852774923503f77b.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/images/a7f719e84a93f4f2258b1ce4bba1c59a40306970f564f56b852774923503f77b.jpg)

### Tables

![17cc9ca8775ea7159522867111cfec663fe5c59b6a22e158fb29c1f0b2f5db16.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/tables/17cc9ca8775ea7159522867111cfec663fe5c59b6a22e158fb29c1f0b2f5db16.jpg)

![286e3aacf1cf06843919cba0bdeb722b384159e03c6141ea84e42dc4d6872fe2.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/tables/286e3aacf1cf06843919cba0bdeb722b384159e03c6141ea84e42dc4d6872fe2.jpg)

![2b077cdfe08a2aaa7d9b693bf4013cfa33ff34433da8dcc615e991e54348c705.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/tables/2b077cdfe08a2aaa7d9b693bf4013cfa33ff34433da8dcc615e991e54348c705.jpg)

![453bba83c260019f314ba54d4ca425dd551751f83afe2997871ebba532209ec9.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/tables/453bba83c260019f314ba54d4ca425dd551751f83afe2997871ebba532209ec9.jpg)

![577190ed655f9dc404709cef4358d5a19b34164c83d9ccacba888ff5960e7764.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/tables/577190ed655f9dc404709cef4358d5a19b34164c83d9ccacba888ff5960e7764.jpg)

![9bbc4466e4c9c4e6900f4ea5a3978388ce70fd45001f156357341677f7bb9444.jpg](../iclr_results/2194_A%20Little%20Goes%20a%20Long%20Way_%20Efficient%20Long%20Context%20Training%20and%20Inference%20with%20Partial%20Contexts/tables/9bbc4466e4c9c4e6900f4ea5a3978388ce70fd45001f156357341677f7bb9444.jpg)
