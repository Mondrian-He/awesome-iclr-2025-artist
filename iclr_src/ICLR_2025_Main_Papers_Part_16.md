# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 16 of 100

## 目录 (Table of Contents)

1. [NetMoE: Accelerating MoE Training through Dynamic Sample Placement](#NetMoE-Accelerating-MoE-Training-through-Dynamic-Sample-Placement)
2. [Bayesian Optimization via Continual Variational Last Layer Training](#Bayesian-Optimization-via-Continual-Variational-Last-Layer-Training)
3. [Fast Uncovering of Protein Sequence Diversity from Structure](#Fast-Uncovering-of-Protein-Sequence-Diversity-from-Structure)
4. [Century: A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images](#Century-A-Framework-and-Dataset-for-Evaluating-Historical-Contextualisation-of-Sensitive-Images)
5. [MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code](#MathCoder2-Better-Math-Reasoning-from-Continued-Pretraining-on-Model-translated-Mathematical-Code)
6. [OmniRe: Omni Urban Scene Reconstruction](#OmniRe-Omni-Urban-Scene-Reconstruction)
7. [In vivo cell-type and brain region classification via multimodal contrastive learning](#In-vivo-cell-type-and-brain-region-classification-via-multimodal-contrastive-learning)
8. [Monitoring Latent World States in Language Models with Propositional Probes](#Monitoring-Latent-World-States-in-Language-Models-with-Propositional-Probes)
9. [Universal generalization guarantees for Wasserstein distributionally robust models](#Universal-generalization-guarantees-for-Wasserstein-distributionally-robust-models)
10. [PETRA: Parallel End-to-end Training with Reversible Architectures](#PETRA-Parallel-End-to-end-Training-with-Reversible-Architectures)
11. [Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage](#Multi-modal-Agent-Tuning-Building-a-VLM-Driven-Agent-for-Efficient-Tool-Usage)
12. [Holistically Evaluating the Environmental Impact of Creating Language Models](#Holistically-Evaluating-the-Environmental-Impact-of-Creating-Language-Models)
13. [Adaptive Gradient Clipping for Robust Federated Learning](#Adaptive-Gradient-Clipping-for-Robust-Federated-Learning)
14. [Re-Imagining Multimodal Instruction Tuning: A Representation View](#Re-Imagining-Multimodal-Instruction-Tuning-A-Representation-View)
15. [Inverse decision-making using neural amortized Bayesian actors](#Inverse-decision-making-using-neural-amortized-Bayesian-actors)
16. [Designing Concise ConvNets with Columnar Stages](#Designing-Concise-ConvNets-with-Columnar-Stages)
17. [MVTokenFlow: High-quality 4D Content Generation using Multiview Token Flow](#MVTokenFlow-High-quality-4D-Content-Generation-using-Multiview-Token-Flow)
18. [Let the Code LLM Edit Itself When You Edit the Code](#Let-the-Code-LLM-Edit-Itself-When-You-Edit-the-Code)
19. [Fewer May Be Better: Enhancing Offline Reinforcement Learning with Reduced Dataset](#Fewer-May-Be-Better-Enhancing-Offline-Reinforcement-Learning-with-Reduced-Dataset)
20. [Generalizing Reasoning Problems to Longer Lengths](#Generalizing-Reasoning-Problems-to-Longer-Lengths)
21. [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](#Physics-of-Language-Models-Part-22-How-to-Learn-From-Mistakes-on-Grade-School-Math-Problems)
22. [Understanding the Stability-based Generalization of Personalized Federated Learning](#Understanding-the-Stability-based-Generalization-of-Personalized-Federated-Learning)
23. [IgGM: A Generative Model for Functional Antibody and Nanobody Design](#IgGM-A-Generative-Model-for-Functional-Antibody-and-Nanobody-Design)
24. [MMTEB: Massive Multilingual Text Embedding Benchmark](#MMTEB-Massive-Multilingual-Text-Embedding-Benchmark)
25. [Ultra-Sparse Memory Network](#Ultra-Sparse-Memory-Network)
26. [Lines of Thought in Large Language Models](#Lines-of-Thought-in-Large-Language-Models)
27. [Do You Keep an Eye on What I Ask? Mitigating Multimodal Hallucination via Attention-Guided Ensemble Decoding](#Do-You-Keep-an-Eye-on-What-I-Ask-Mitigating-Multimodal-Hallucination-via-Attention-Guided-Ensemble-Decoding)
28. [Exact Community Recovery under Side Information: Optimality of Spectral Algorithms](#Exact-Community-Recovery-under-Side-Information-Optimality-of-Spectral-Algorithms)
29. [Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data](#Context-Clues-Evaluating-Long-Context-Models-for-Clinical-Prediction-Tasks-on-EHR-Data)
30. [Deconstructing What Makes a Good Optimizer for Autoregressive Language Models](#Deconstructing-What-Makes-a-Good-Optimizer-for-Autoregressive-Language-Models)
31. [Dataset Ownership Verification in Contrastive Pre-trained Models](#Dataset-Ownership-Verification-in-Contrastive-Pre-trained-Models)
32. [System 1.x: Learning to Balance Fast and Slow Planning with Language Models](#System-1x-Learning-to-Balance-Fast-and-Slow-Planning-with-Language-Models)
33. [Time-to-Event Pretraining for 3D Medical Imaging](#Time-to-Event-Pretraining-for-3D-Medical-Imaging)
34. [Semialgebraic Neural Networks: From roots to representations](#Semialgebraic-Neural-Networks-From-roots-to-representations)
35. [h4rm3l: A Language for Composable Jailbreak Attack Synthesis](#h4rm3l-A-Language-for-Composable-Jailbreak-Attack-Synthesis)
36. [Synthesizing Realistic fMRI: A Physiological Dynamics-Driven Hierarchical Diffusion Model for Efficient fMRI Acquisition](#Synthesizing-Realistic-fMRI-A-Physiological-Dynamics-Driven-Hierarchical-Diffusion-Model-for-Efficient-fMRI-Acquisition)

---


## NetMoE: Accelerating MoE Training through Dynamic Sample Placement

### Images

![71432a5dd7cc9b2b5545f0d2d6283d70fb9d1e8466cd7f8b272a8bb368812b31.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/images/71432a5dd7cc9b2b5545f0d2d6283d70fb9d1e8466cd7f8b272a8bb368812b31.jpg)

![929bb0d1bcdbbb848c7d9889e24134e0b31984cc16c60db6ad94f18c97c81bdc.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/images/929bb0d1bcdbbb848c7d9889e24134e0b31984cc16c60db6ad94f18c97c81bdc.jpg)

![bafe6f9f4c53f143d590555ac3992561ab1c884a4482f493a8deec1eeb6c0414.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/images/bafe6f9f4c53f143d590555ac3992561ab1c884a4482f493a8deec1eeb6c0414.jpg)

![c731b26df1013372eb3a5249621d4fb19e96231b941204023b8db6c203c15eb9.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/images/c731b26df1013372eb3a5249621d4fb19e96231b941204023b8db6c203c15eb9.jpg)

![d321c86e30bd5331cb5b108ceed5dacbacc1443bfb23ea94dfa36ad1521eceba.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/images/d321c86e30bd5331cb5b108ceed5dacbacc1443bfb23ea94dfa36ad1521eceba.jpg)

![e9df1a100d86e117bfb8c7df9300ec41ecc320e3705aebffa27e1013638f8f06.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/images/e9df1a100d86e117bfb8c7df9300ec41ecc320e3705aebffa27e1013638f8f06.jpg)

### Tables

![23366f834c8e6a591a3ebae957bc2eec59028a0fb19c0a7f6066f3cde2fb8007.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/tables/23366f834c8e6a591a3ebae957bc2eec59028a0fb19c0a7f6066f3cde2fb8007.jpg)

![fca167094bae9583e7515c9fe79f693edd8790f162a9770fd199630b8a2f0d81.jpg](../iclr_results/578_High-dimensional Analysis of Knowledge Distillation_ Weak-to-Strong Generalization and Scaling Laws/tables/fca167094bae9583e7515c9fe79f693edd8790f162a9770fd199630b8a2f0d81.jpg)

## NetMoE: Accelerating MoE Training through Dynamic Sample Placement


### Images

![5d60326e17d3e88dd9f9096685d52a4bc5eb178084a27c9900bb2d4680fb1929.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/5d60326e17d3e88dd9f9096685d52a4bc5eb178084a27c9900bb2d4680fb1929.jpg)

![7be2647dd724605a8d5ea7f3e578a6da4b06715fd170eb04d69965ed1adca537.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/7be2647dd724605a8d5ea7f3e578a6da4b06715fd170eb04d69965ed1adca537.jpg)

![8da88076786fe499979db7e32fd1c50ed21e7edfa516c3881ed9a09db4430d64.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/8da88076786fe499979db7e32fd1c50ed21e7edfa516c3881ed9a09db4430d64.jpg)

![920e6585c7f361d14a12fa68fe9746998352a6fa56fe541bbbe5c47b0ae65733.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/920e6585c7f361d14a12fa68fe9746998352a6fa56fe541bbbe5c47b0ae65733.jpg)

![97b2c4f60d0b9e095debf497a4b14da63500a88fc412aea4681c9a63d9337f53.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/97b2c4f60d0b9e095debf497a4b14da63500a88fc412aea4681c9a63d9337f53.jpg)

![99d0d7160e22ca7e2be0a38850c040802ab5d52d672d969280faf46cad9f8cb8.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/99d0d7160e22ca7e2be0a38850c040802ab5d52d672d969280faf46cad9f8cb8.jpg)

![a0a9a5c8dda47c0a1fede798be805581c9102a57d97b5b10f9784d3f6d49a9b1.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/a0a9a5c8dda47c0a1fede798be805581c9102a57d97b5b10f9784d3f6d49a9b1.jpg)

![a10c9ef346efd64f739f71659f0103dc4f411771bd3c3a0267105856189b6346.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/a10c9ef346efd64f739f71659f0103dc4f411771bd3c3a0267105856189b6346.jpg)

![a3e82271d82754905bb3443dd8c35126abcaf905865d3a676f7c9510ec9aad7c.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/a3e82271d82754905bb3443dd8c35126abcaf905865d3a676f7c9510ec9aad7c.jpg)

![af1ff744be15c959bf83b92fc291955dfba0eec698a378983145b5cb0507dcce.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/images/af1ff744be15c959bf83b92fc291955dfba0eec698a378983145b5cb0507dcce.jpg)

### Tables

![4b9f95eee0c21d18503d62d97c7da294748470cc709005138b11048399198788.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/tables/4b9f95eee0c21d18503d62d97c7da294748470cc709005138b11048399198788.jpg)

![74a04bf586742550d2716ad93c0e3e5fcf2076301b86414154750284dce38090.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/tables/74a04bf586742550d2716ad93c0e3e5fcf2076301b86414154750284dce38090.jpg)

![99f7b2a1d6466a3fb90ddff77cda8e1d34d5c7eebea6f7cf6ee175c354e8fbd4.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/tables/99f7b2a1d6466a3fb90ddff77cda8e1d34d5c7eebea6f7cf6ee175c354e8fbd4.jpg)

![cf7ba3f0279b39ef4b3aa125a96a5275523ebc87ee7106474650e344ffac8a1f.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/tables/cf7ba3f0279b39ef4b3aa125a96a5275523ebc87ee7106474650e344ffac8a1f.jpg)

![f4d95e8d325fe430d6412b000019f1a152b70f8e2c3d94c9eccda3c0d7e4871f.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/tables/f4d95e8d325fe430d6412b000019f1a152b70f8e2c3d94c9eccda3c0d7e4871f.jpg)

![ffc57d1db74613144cb73576b8c9651945950203b22cbb58247f9fe98174ebe6.jpg](../iclr_results/579_NetMoE_ Accelerating MoE Training through Dynamic Sample Placement/tables/ffc57d1db74613144cb73576b8c9651945950203b22cbb58247f9fe98174ebe6.jpg)

## Bayesian Optimization via Continual Variational Last Layer Training


### Images

![10bd3930939948158e8e922e8112ca04f6498d931641b8f273d72fb74da4c069.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/10bd3930939948158e8e922e8112ca04f6498d931641b8f273d72fb74da4c069.jpg)

![1476f66b8315e959965a63e0a2a51126b7c2755e14714450f4308a057b5da76d.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/1476f66b8315e959965a63e0a2a51126b7c2755e14714450f4308a057b5da76d.jpg)

![303a5c8cbfcabf14ee47d83649a6fe89efdfaa3a323897223c7a295bbb101e6e.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/303a5c8cbfcabf14ee47d83649a6fe89efdfaa3a323897223c7a295bbb101e6e.jpg)

![30b0f5d6f291d62fdc91d80298d6c8ff27c27de3b082e1717991947706de9178.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/30b0f5d6f291d62fdc91d80298d6c8ff27c27de3b082e1717991947706de9178.jpg)

![35ccd72e631110f2f4df522a8610418cdca7ec3799ee53b2583232ec53068371.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/35ccd72e631110f2f4df522a8610418cdca7ec3799ee53b2583232ec53068371.jpg)

![59b79a2caf2a6dfde17ffaca078cb84435d424c1de7e0669ca0d1d5405a988f0.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/59b79a2caf2a6dfde17ffaca078cb84435d424c1de7e0669ca0d1d5405a988f0.jpg)

![5ad483bb4c1a7a665c6eb81ea7dd779f706f064976974154ca7c731a3e975950.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/5ad483bb4c1a7a665c6eb81ea7dd779f706f064976974154ca7c731a3e975950.jpg)

![5b388f5edafac3d569d3e78f98afcdde2af0132b28428f64f5d486e9b41c75b0.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/5b388f5edafac3d569d3e78f98afcdde2af0132b28428f64f5d486e9b41c75b0.jpg)

![5cf72d56f4b156aa9b44c483c1c1fdf2b9e1067f93f1b45d81c1bb3732934d7f.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/5cf72d56f4b156aa9b44c483c1c1fdf2b9e1067f93f1b45d81c1bb3732934d7f.jpg)

![5d19e5e504be73fb9a6e4770e8ddf4bbffb042477c3dc3eef46a3dd4440669cd.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/5d19e5e504be73fb9a6e4770e8ddf4bbffb042477c3dc3eef46a3dd4440669cd.jpg)

![5fae8e7c9f20c164ad1f780367ed22b40f8e072908248273c899891fe10dfb82.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/5fae8e7c9f20c164ad1f780367ed22b40f8e072908248273c899891fe10dfb82.jpg)

![6f38b603ab541b4c19424e01929a9468a78f0aa2dbfa7638d865ccbde2a3997a.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/6f38b603ab541b4c19424e01929a9468a78f0aa2dbfa7638d865ccbde2a3997a.jpg)

![71d26784430e1e54844ca009929f69d0023fcacce98bc817e609dd1a98c2738d.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/71d26784430e1e54844ca009929f69d0023fcacce98bc817e609dd1a98c2738d.jpg)

![7ffee7248b4eb250ecb688a7f8d88436b142783235fd4944e1c4084c56714ff5.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/7ffee7248b4eb250ecb688a7f8d88436b142783235fd4944e1c4084c56714ff5.jpg)

![9be1a3568822d5aee55e4296f10416d01f6dd4968eeb0b8952256bb68cc2172c.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/9be1a3568822d5aee55e4296f10416d01f6dd4968eeb0b8952256bb68cc2172c.jpg)

![9bedae638faae14ebb99c77e3465e4136f4345372f32a0f755760af83d21102a.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/9bedae638faae14ebb99c77e3465e4136f4345372f32a0f755760af83d21102a.jpg)

![af13270173294c1b890bd132bfe6385906f800be96a85f337440c080fc821308.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/af13270173294c1b890bd132bfe6385906f800be96a85f337440c080fc821308.jpg)

![b2648f448136ffb0baaede4725a7d62f91831bb98fd4f4410b1d9653f4c2e352.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/b2648f448136ffb0baaede4725a7d62f91831bb98fd4f4410b1d9653f4c2e352.jpg)

![b53ff65584d74dc8cf015462810e39ba47d0a910ddab01437cde0d17681f0448.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/b53ff65584d74dc8cf015462810e39ba47d0a910ddab01437cde0d17681f0448.jpg)

![d267f656b8cce95d71e3994aaae197e05a4747046df96b80dbc24491e2568a7a.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/d267f656b8cce95d71e3994aaae197e05a4747046df96b80dbc24491e2568a7a.jpg)

![d84af66192dace75a199839834fae6e95b1bb9184c66045910f64d8714525905.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/d84af66192dace75a199839834fae6e95b1bb9184c66045910f64d8714525905.jpg)

![e10581f5cfacd96bc10caff23f349ae11077ec39cc6d489625c4db074df3c51c.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/images/e10581f5cfacd96bc10caff23f349ae11077ec39cc6d489625c4db074df3c51c.jpg)

### Tables

![accb19019398d784fc239bddeff82600e930d6c31e2775b1f3401a231d969a81.jpg](../iclr_results/580_Bayesian Optimization via Continual Variational Last Layer Training/tables/accb19019398d784fc239bddeff82600e930d6c31e2775b1f3401a231d969a81.jpg)

## Fast Uncovering of Protein Sequence Diversity from Structure


### Images

![0d44cedf1d0cd3d1bd9e314f197810ed55e404dfc3a964c56f912be72afd1248.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/0d44cedf1d0cd3d1bd9e314f197810ed55e404dfc3a964c56f912be72afd1248.jpg)

![153b915f1a09f104682d7f05ce17e9d45ef4b1d5fc603e140f052034a38c903c.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/153b915f1a09f104682d7f05ce17e9d45ef4b1d5fc603e140f052034a38c903c.jpg)

![2499a4e0ec54efc598047d219838a4ffc294ba8bb2a729bb9580547ec07b9dc8.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/2499a4e0ec54efc598047d219838a4ffc294ba8bb2a729bb9580547ec07b9dc8.jpg)

![2ce64a322cfffb27ad3d5fefa1c4f4ae0112fd7d66451b623cc9a99e1a377293.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/2ce64a322cfffb27ad3d5fefa1c4f4ae0112fd7d66451b623cc9a99e1a377293.jpg)

![469c254f80be1555bff56c8d77387e62813890662807d5b18f9270be2ab927c7.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/469c254f80be1555bff56c8d77387e62813890662807d5b18f9270be2ab927c7.jpg)

![5500784d6383e95cd7668bc3dfccb00715fbe45536483e8c60d766afe93585f7.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/5500784d6383e95cd7668bc3dfccb00715fbe45536483e8c60d766afe93585f7.jpg)

![604be6c8f89b2070bf85752c5bad1c809f22677cf98cb0398dca1a06a82b3661.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/604be6c8f89b2070bf85752c5bad1c809f22677cf98cb0398dca1a06a82b3661.jpg)

![6221b6926a109438f03788e67b74b7cf2bdd400349f400a423c7e09dd719c974.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/6221b6926a109438f03788e67b74b7cf2bdd400349f400a423c7e09dd719c974.jpg)

![6acea3aec9c20068078a252ecefb4a90df81c2d186feb51a8d1c8c5e7660ad61.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/6acea3aec9c20068078a252ecefb4a90df81c2d186feb51a8d1c8c5e7660ad61.jpg)

![724e604fa167d77fb42cb451cdf1f984e92bb7e2a5d38c6f0b310551e0d83bae.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/724e604fa167d77fb42cb451cdf1f984e92bb7e2a5d38c6f0b310551e0d83bae.jpg)

![78a733dcc7f1d0e0ce3f5902ab837f972f4a2e5882556909fd8cd893a71be22b.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/78a733dcc7f1d0e0ce3f5902ab837f972f4a2e5882556909fd8cd893a71be22b.jpg)

![82de8a0a510c6b335eb5120efef26478bffeee2366a3d3fd811738a1c31642ba.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/82de8a0a510c6b335eb5120efef26478bffeee2366a3d3fd811738a1c31642ba.jpg)

![8c755aee690333081f41ce2140ac2b8a1e3514ac73a9ffa55843b9bce5292ace.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/8c755aee690333081f41ce2140ac2b8a1e3514ac73a9ffa55843b9bce5292ace.jpg)

![972729d903e49eb9d5836f27fa027a0ba44b3dfb448bc06226eae12514aa649f.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/972729d903e49eb9d5836f27fa027a0ba44b3dfb448bc06226eae12514aa649f.jpg)

![9c8632c904783a217fbf638ba4f50fe7928df9a459e1e9854af3d1f68e86877d.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/9c8632c904783a217fbf638ba4f50fe7928df9a459e1e9854af3d1f68e86877d.jpg)

![a90d41f275557638c9ba73b5ce49801118b01c868dd45dfaa1cc165d2f6e83eb.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/a90d41f275557638c9ba73b5ce49801118b01c868dd45dfaa1cc165d2f6e83eb.jpg)

![aa6ed9966e784aa9a73b832c6235451125a54f9fd8611ff91208f9460c9572fd.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/aa6ed9966e784aa9a73b832c6235451125a54f9fd8611ff91208f9460c9572fd.jpg)

![c16010b536db2256c9139e0746d1d81d912b87b9bcae826859ad0aa3dfec33b9.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/c16010b536db2256c9139e0746d1d81d912b87b9bcae826859ad0aa3dfec33b9.jpg)

![d3543da87958499477c44c8710f60590cf039fb84cd62a6c5aa245bf06e1d139.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/d3543da87958499477c44c8710f60590cf039fb84cd62a6c5aa245bf06e1d139.jpg)

![e1bdc93d010a0564f6e5d701866778df3867dcbcfba584a5701ecad16d9f9b5c.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/e1bdc93d010a0564f6e5d701866778df3867dcbcfba584a5701ecad16d9f9b5c.jpg)

![ed2a27315a0794a37b6374140803b4cef6b72d9f7c0b83119e38f0e7c8d686db.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/ed2a27315a0794a37b6374140803b4cef6b72d9f7c0b83119e38f0e7c8d686db.jpg)

![ee6f1db6b8f3d53d367be77c17cd36f9735945c811811b242c68a28bc3a96caa.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/ee6f1db6b8f3d53d367be77c17cd36f9735945c811811b242c68a28bc3a96caa.jpg)

![eea6585504b717110826486c25a6c79b0f8480396ced48807cd805f6dcf3aff9.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/eea6585504b717110826486c25a6c79b0f8480396ced48807cd805f6dcf3aff9.jpg)

![fc6cd54e44a6a768325961dbef75a44d8f15fe9a148040f1de8d2bc35a25b4bf.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/images/fc6cd54e44a6a768325961dbef75a44d8f15fe9a148040f1de8d2bc35a25b4bf.jpg)

### Tables

![6528c018f5ae26c53576fabf8cf1807d32c0cdf7ee9eace221e91e0cc784dc5a.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/tables/6528c018f5ae26c53576fabf8cf1807d32c0cdf7ee9eace221e91e0cc784dc5a.jpg)

![99fbca9d03d22a56e4c2acbcc2d444a9e0d90edc153fb93d42f960155aa64c15.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/tables/99fbca9d03d22a56e4c2acbcc2d444a9e0d90edc153fb93d42f960155aa64c15.jpg)

![c0f2cf9102cb9613e774574f8bd44e82587a66dfc9ac777781c3d3495708859c.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/tables/c0f2cf9102cb9613e774574f8bd44e82587a66dfc9ac777781c3d3495708859c.jpg)

![d79d2ea9dc885bc3c500140152f4a53ecc93f7d2b0b38d9f3f14aff4f16490b1.jpg](../iclr_results/581_Fast Uncovering of Protein Sequence Diversity from Structure/tables/d79d2ea9dc885bc3c500140152f4a53ecc93f7d2b0b38d9f3f14aff4f16490b1.jpg)

## Century: A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images


### Images

![0579968824855e5134c2293e30c2e6d15497c160eaf9f399ab28dffd59d2181d.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/0579968824855e5134c2293e30c2e6d15497c160eaf9f399ab28dffd59d2181d.jpg)

![0fb0fe3566c7b004ee9fe39063a4734c4213e86c7530661423ba35c9b4d00127.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/0fb0fe3566c7b004ee9fe39063a4734c4213e86c7530661423ba35c9b4d00127.jpg)

![175f56d28191cd25a5694cd0bfdeda9ecf4c52c5c698ca2a34d3335773335ec3.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/175f56d28191cd25a5694cd0bfdeda9ecf4c52c5c698ca2a34d3335773335ec3.jpg)

![1cc14af8ce83d383500adce4185ecda9a7af09a5af94ad7a8be0f7b44bf4673e.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/1cc14af8ce83d383500adce4185ecda9a7af09a5af94ad7a8be0f7b44bf4673e.jpg)

![3e81478c7d8cd477285afc1d1c445eacced90b4968db544db240bfec966c3ffd.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/3e81478c7d8cd477285afc1d1c445eacced90b4968db544db240bfec966c3ffd.jpg)

![3f0fb96fe7a007a6dffd9ba7c01d78de65c8efed8bf645e134be59fd25d74957.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/3f0fb96fe7a007a6dffd9ba7c01d78de65c8efed8bf645e134be59fd25d74957.jpg)

![4e2e6c377721f0d2aeb1b1f35594b35d2350f5622638b12d63285096feed81db.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/4e2e6c377721f0d2aeb1b1f35594b35d2350f5622638b12d63285096feed81db.jpg)

![671b9c64a1fec278fdd2df67381a692ccfce1264f0d47a4850d5775a4ee93a47.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/671b9c64a1fec278fdd2df67381a692ccfce1264f0d47a4850d5775a4ee93a47.jpg)

![6b57723618c0498d9de0b320f2625869b3a519915627818a40dffa4aff50047e.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/6b57723618c0498d9de0b320f2625869b3a519915627818a40dffa4aff50047e.jpg)

![8617c94da482d6bbdf36a5e3a32b06541fb749ced42dd62e8a9eb1f7df1b5694.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/8617c94da482d6bbdf36a5e3a32b06541fb749ced42dd62e8a9eb1f7df1b5694.jpg)

![91b848cf0ea46c2e507a6593398ca8248fbd4765888fdcc1c03a341044fb0a3e.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/91b848cf0ea46c2e507a6593398ca8248fbd4765888fdcc1c03a341044fb0a3e.jpg)

![91fe7bb5147e72e4c104463d4022cfbec3a66f2a72b744e7901e6b794f41d006.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/91fe7bb5147e72e4c104463d4022cfbec3a66f2a72b744e7901e6b794f41d006.jpg)

![9a10f714cae58bab57c732a6f0dbd7e125321fc99c0bc9403f92cde88f043c2d.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/9a10f714cae58bab57c732a6f0dbd7e125321fc99c0bc9403f92cde88f043c2d.jpg)

![a3349b3d6605d03da2fb331f14b2cddc1ac156c434ba7c3c7d5faa4a89b1f1ca.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/a3349b3d6605d03da2fb331f14b2cddc1ac156c434ba7c3c7d5faa4a89b1f1ca.jpg)

![a92cfc12274113af7a4f78520d82c6dac73b47653f53755dc766e66fdbf2eb59.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/a92cfc12274113af7a4f78520d82c6dac73b47653f53755dc766e66fdbf2eb59.jpg)

![b36de30bfb2d6c730059c49d8a97e0bbc0730c2c2ad8e083d471bb618f20f907.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/b36de30bfb2d6c730059c49d8a97e0bbc0730c2c2ad8e083d471bb618f20f907.jpg)

![bb91d730e70bdde3950d68afb39d7921efed7100f0e88e1df554fd218d50fd68.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/bb91d730e70bdde3950d68afb39d7921efed7100f0e88e1df554fd218d50fd68.jpg)

![bf0d404bc19ca22eef444638130f4fc66aa50404d15952c2f911a48f7eddd182.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/bf0d404bc19ca22eef444638130f4fc66aa50404d15952c2f911a48f7eddd182.jpg)

![c6a69ab01b7454c4e191ba86c041557f6644bb269b5c08ab7d163d4530c36bac.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/c6a69ab01b7454c4e191ba86c041557f6644bb269b5c08ab7d163d4530c36bac.jpg)

![c966eea222dad07db72029b3f0ebeb047c8440c78e516acda33b563af9330bf6.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/c966eea222dad07db72029b3f0ebeb047c8440c78e516acda33b563af9330bf6.jpg)

![d377d4481ea123ceb7628b55772f098b79cdcdd891ca15fb9b38dbb53f471a84.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/d377d4481ea123ceb7628b55772f098b79cdcdd891ca15fb9b38dbb53f471a84.jpg)

![dbd5b3342c76270cafe493f1e36637de1224be277130dd7f22bd5342ef463b7a.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/dbd5b3342c76270cafe493f1e36637de1224be277130dd7f22bd5342ef463b7a.jpg)

![de19fe47ba3e3cb12a1ffd3e91993d47c8f49c0b8dd5f7c56a432d790227788e.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/de19fe47ba3e3cb12a1ffd3e91993d47c8f49c0b8dd5f7c56a432d790227788e.jpg)

![e5987002c8a88d43e4f8ebae7fecccba2613d88fe64e87abe0838f8cee95f5ba.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/images/e5987002c8a88d43e4f8ebae7fecccba2613d88fe64e87abe0838f8cee95f5ba.jpg)

### Tables

![0426785f2d0c5465d816cda81c51da85f86cfa19b04f91991ebabb5476f40856.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/0426785f2d0c5465d816cda81c51da85f86cfa19b04f91991ebabb5476f40856.jpg)

![111bb6119129e24e99b07fd7b59ae8d197788716102c96a39e548209b1fc8389.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/111bb6119129e24e99b07fd7b59ae8d197788716102c96a39e548209b1fc8389.jpg)

![15f474ebd1652943ab3690a4b3be859de1ec64e01b468e1c6f4192dd9455f09b.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/15f474ebd1652943ab3690a4b3be859de1ec64e01b468e1c6f4192dd9455f09b.jpg)

![3054590d030b5103443f43bcf981fb711e283da70a732af4e236228d8a11471e.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/3054590d030b5103443f43bcf981fb711e283da70a732af4e236228d8a11471e.jpg)

![3e3bc708cbfe3b2066aedf31aba117ab4ef0580e1dcae13afed508a0c663c2dd.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/3e3bc708cbfe3b2066aedf31aba117ab4ef0580e1dcae13afed508a0c663c2dd.jpg)

![52898fa8c2690653588cec6498e9b6edef5d1fccae9a372d8988174efd5f4851.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/52898fa8c2690653588cec6498e9b6edef5d1fccae9a372d8988174efd5f4851.jpg)

![5dd1e753cb6581fd21b2819e0565366e16cbbfd9f2f7debee3aaf820dd80b7bd.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/5dd1e753cb6581fd21b2819e0565366e16cbbfd9f2f7debee3aaf820dd80b7bd.jpg)

![6a4548508e45a16813f63b55065538ec1747fab92678c754c675dd0bf8322e0a.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/6a4548508e45a16813f63b55065538ec1747fab92678c754c675dd0bf8322e0a.jpg)

![8298f90bca465adfbba0abd705f76d6c5f640e59ea3d7526513d73a2f55470e1.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/8298f90bca465adfbba0abd705f76d6c5f640e59ea3d7526513d73a2f55470e1.jpg)

![99ce50b5a9e20651f50e94ae62639bf79151763ce0b9f2c850cd918342ad78aa.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/99ce50b5a9e20651f50e94ae62639bf79151763ce0b9f2c850cd918342ad78aa.jpg)

![a7117883a883c678eaddddb6455e162702bfa4e136045122cef646183c4ee79b.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/a7117883a883c678eaddddb6455e162702bfa4e136045122cef646183c4ee79b.jpg)

![aa023c191dc9b1d3f06f27e49e6c001895894d1fdaf64bd9f695daa09dd26f36.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/aa023c191dc9b1d3f06f27e49e6c001895894d1fdaf64bd9f695daa09dd26f36.jpg)

![d03ef20ef7d00a77782b6dbd745e58db0a8dff87b227af01e69e7cf572cd562d.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/d03ef20ef7d00a77782b6dbd745e58db0a8dff87b227af01e69e7cf572cd562d.jpg)

![d8ccdb9889bc6f577ea7f919cbb095c8c954d21c3cbcc30b381418c915097919.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/d8ccdb9889bc6f577ea7f919cbb095c8c954d21c3cbcc30b381418c915097919.jpg)

![ef95f8467e28e52caea023b9b1eda93d13bfd8c87b8b843301401a82e57eac8e.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/ef95f8467e28e52caea023b9b1eda93d13bfd8c87b8b843301401a82e57eac8e.jpg)

![efe071dea6729b471e4fd0424caf2078e2bca2cc03340a596003d210c3ee991b.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/efe071dea6729b471e4fd0424caf2078e2bca2cc03340a596003d210c3ee991b.jpg)

![f04790ac6c400f36dfbce28227727a917468a49cbf8bd1ed8e007ec46326f152.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/f04790ac6c400f36dfbce28227727a917468a49cbf8bd1ed8e007ec46326f152.jpg)

![f7ecac16323564c5647920e051699faacc71cfd08f96e67d15dcd6ea6a31668d.jpg](../iclr_results/582_Century_ A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images/tables/f7ecac16323564c5647920e051699faacc71cfd08f96e67d15dcd6ea6a31668d.jpg)

## MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code


### Images

![4dce28d6b5b5d5644bfcccec89d1dd84fcdff05f84e392ce4d9c507b852eddcc.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/images/4dce28d6b5b5d5644bfcccec89d1dd84fcdff05f84e392ce4d9c507b852eddcc.jpg)

![600a90139e7c1b90af05455bc8debbeab665ea0d0d3af409bc5470f481ad3b17.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/images/600a90139e7c1b90af05455bc8debbeab665ea0d0d3af409bc5470f481ad3b17.jpg)

![8987c706e7d0e0355432f667ecea994932ca62024f688f070abee01a54cc4bf5.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/images/8987c706e7d0e0355432f667ecea994932ca62024f688f070abee01a54cc4bf5.jpg)

### Tables

![2795a3e10022453fce360e82798a5e88b06bf621a990e3efd76346b07659adfb.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/2795a3e10022453fce360e82798a5e88b06bf621a990e3efd76346b07659adfb.jpg)

![3f944b8055548dc0e801c4052c4753d9a25fbbd1cef74b2c77098c3fa31ac40d.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/3f944b8055548dc0e801c4052c4753d9a25fbbd1cef74b2c77098c3fa31ac40d.jpg)

![5ff72dc663a1af2a7c8c88bbf46db6f2fdef05b66dfecf5bac1c6bbbb171d758.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/5ff72dc663a1af2a7c8c88bbf46db6f2fdef05b66dfecf5bac1c6bbbb171d758.jpg)

![745cf3f0304346849b95e8c3a9eede9cfcc566ce8df99af82c009a3d2a5bc1d5.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/745cf3f0304346849b95e8c3a9eede9cfcc566ce8df99af82c009a3d2a5bc1d5.jpg)

![89ca232b14a301ac2c3aa126c46404d418be274d5433076e1746ae4a1e67f4ba.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/89ca232b14a301ac2c3aa126c46404d418be274d5433076e1746ae4a1e67f4ba.jpg)

![9561b97296e7208e5c3ca78f9e78f7bfb2470dd1b578d536a67cfb7fc42a65ba.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/9561b97296e7208e5c3ca78f9e78f7bfb2470dd1b578d536a67cfb7fc42a65ba.jpg)

![addebd45dee0a591014183bf9cf4ca7b1f10bafab1c1ee0be31021a7e9a7e4fe.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/addebd45dee0a591014183bf9cf4ca7b1f10bafab1c1ee0be31021a7e9a7e4fe.jpg)

![afd975e82b0eaae3b7e947f71c243a36ca6ae95d0b524aa13044bc61e71e780a.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/afd975e82b0eaae3b7e947f71c243a36ca6ae95d0b524aa13044bc61e71e780a.jpg)

![dbf3c2b93992c7503f8f47197cca98ef4555564359801a5f64125b50ce81be16.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/dbf3c2b93992c7503f8f47197cca98ef4555564359801a5f64125b50ce81be16.jpg)

![ea11766bff8c88706f2ca7baec0b2f6425442f9a38d7246bacc59c526f9f7de0.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/ea11766bff8c88706f2ca7baec0b2f6425442f9a38d7246bacc59c526f9f7de0.jpg)

![ea8d4ad7164f3287e75d3077152a17e3599d6e254e688df9b9674a7e0d55ec04.jpg](../iclr_results/583_MathCoder2_ Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code/tables/ea8d4ad7164f3287e75d3077152a17e3599d6e254e688df9b9674a7e0d55ec04.jpg)

## OmniRe: Omni Urban Scene Reconstruction


### Images

![00d60c3dc5ebc77e5b8888f013dbb351b620c573a7dfd319ee57837557fbfb40.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/00d60c3dc5ebc77e5b8888f013dbb351b620c573a7dfd319ee57837557fbfb40.jpg)

![16b1bb647bc02fd1da7b66693c91606df47ba95507b1da21f2ff096b3ff3b155.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/16b1bb647bc02fd1da7b66693c91606df47ba95507b1da21f2ff096b3ff3b155.jpg)

![27d824fb740e4ef07cee6a1681053d7b25f5c46dd4e2bf5f9cbbf92c7fb905c9.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/27d824fb740e4ef07cee6a1681053d7b25f5c46dd4e2bf5f9cbbf92c7fb905c9.jpg)

![6a098a64f4c663632678961672e2830e567b0e4521e21b32f455560485cb88c8.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/6a098a64f4c663632678961672e2830e567b0e4521e21b32f455560485cb88c8.jpg)

![7748f2d0c5a1bad059ebb4ab0ef476c47c48a0ebf96c11a9d63ad173f93cce1e.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/7748f2d0c5a1bad059ebb4ab0ef476c47c48a0ebf96c11a9d63ad173f93cce1e.jpg)

![995eda197c5ee5627b2573048db2653697494aff3d9daf4a0d7965456562ed30.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/995eda197c5ee5627b2573048db2653697494aff3d9daf4a0d7965456562ed30.jpg)

![a58aa2e292347d1a096f6012e513ccc09db4a445f4f992c8e230fa770f9dbc5d.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/a58aa2e292347d1a096f6012e513ccc09db4a445f4f992c8e230fa770f9dbc5d.jpg)

![b77f68836e86e2acdddc4b3bdab20c4c13a220a141bb1b2681eec509a24a780a.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/b77f68836e86e2acdddc4b3bdab20c4c13a220a141bb1b2681eec509a24a780a.jpg)

![d60b3bc9071f4407fb753a96dd6b373bd8b8e14e6b254f7604a73fe4706bb0b8.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/d60b3bc9071f4407fb753a96dd6b373bd8b8e14e6b254f7604a73fe4706bb0b8.jpg)

![d838400dce39d20aba4b334d48c3b94e0ac5771d56bcef47fa83bb83a076741b.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/d838400dce39d20aba4b334d48c3b94e0ac5771d56bcef47fa83bb83a076741b.jpg)

![f20535b63eb5de8b4d1b7ad9477274cc613cb0fc7025f9e2902052b3a80ad079.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/f20535b63eb5de8b4d1b7ad9477274cc613cb0fc7025f9e2902052b3a80ad079.jpg)

![fccd93a4169eaa0a74c189db06cad77c2df0e449b87624c1db218a851244b0cc.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/images/fccd93a4169eaa0a74c189db06cad77c2df0e449b87624c1db218a851244b0cc.jpg)

### Tables

![1b6eb03ed87a0b698a6856fd9c4d1cecdea7c80e60b0799701c34906e3d5c8c3.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/1b6eb03ed87a0b698a6856fd9c4d1cecdea7c80e60b0799701c34906e3d5c8c3.jpg)

![474fc82f271c1b9c25ceec3a549c80d2448c86ee665165ceae0a238ec7d67d14.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/474fc82f271c1b9c25ceec3a549c80d2448c86ee665165ceae0a238ec7d67d14.jpg)

![553ee98bc489f96bba4b3ae2e970341ba08f6652dd694783825fb2306c8ab2c3.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/553ee98bc489f96bba4b3ae2e970341ba08f6652dd694783825fb2306c8ab2c3.jpg)

![5eafaee1861299439fbaf097d2d1922dd58e28df374aa2a1be1edff3bfa9ab35.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/5eafaee1861299439fbaf097d2d1922dd58e28df374aa2a1be1edff3bfa9ab35.jpg)

![6fbb55322f442f1adead0ef00358c1c2396f9385f085164596fe40da35ce366a.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/6fbb55322f442f1adead0ef00358c1c2396f9385f085164596fe40da35ce366a.jpg)

![7d09e70ca63e8315ae753b4e9a5811ea5609f92e552e611e9b7963ca88ddf1a0.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/7d09e70ca63e8315ae753b4e9a5811ea5609f92e552e611e9b7963ca88ddf1a0.jpg)

![875286e9a929e81c66adae587c21326a92ee3509c12740d2ad0b302c59e8910c.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/875286e9a929e81c66adae587c21326a92ee3509c12740d2ad0b302c59e8910c.jpg)

![91c2ac53ba6ae45df585b46d3a3385929ce4cea49580059ff14e8be8ef19e0bc.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/91c2ac53ba6ae45df585b46d3a3385929ce4cea49580059ff14e8be8ef19e0bc.jpg)

![9729a77bd36d5152f256cad5352c6208c5a923010ccab3ee9728d8db41c66cd7.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/9729a77bd36d5152f256cad5352c6208c5a923010ccab3ee9728d8db41c66cd7.jpg)

![a2994cf93060df46395ba75f8a1fbf641606b0f032aa1bfd50f15d197b2cd4ab.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/a2994cf93060df46395ba75f8a1fbf641606b0f032aa1bfd50f15d197b2cd4ab.jpg)

![d1fa74c1e07401b8070878a7c738a4f2912e4f2ceec25bd289c7129775059028.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/d1fa74c1e07401b8070878a7c738a4f2912e4f2ceec25bd289c7129775059028.jpg)

![d5bde4781f72a61acf7eb30cafc0a66f77f97c0e372983bd8dea16b6d363b86e.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/d5bde4781f72a61acf7eb30cafc0a66f77f97c0e372983bd8dea16b6d363b86e.jpg)

![df52936636aa6ecd07e24a8db8570885e53d67b8e5688db9305afb45ac65e239.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/df52936636aa6ecd07e24a8db8570885e53d67b8e5688db9305afb45ac65e239.jpg)

![e23c54b461a67a017734aaac59c8a7b6f9de6707312d239fce35be3ab826110c.jpg](../iclr_results/584_OmniRe_ Omni Urban Scene Reconstruction/tables/e23c54b461a67a017734aaac59c8a7b6f9de6707312d239fce35be3ab826110c.jpg)

## In vivo cell-type and brain region classification via multimodal contrastive learning


### Images

![0d1c1577676dc115c007c27dea46a926483d0781adce71bd7c22071adce2e001.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/0d1c1577676dc115c007c27dea46a926483d0781adce71bd7c22071adce2e001.jpg)

![10cc402de1fb08a2d6019ca3b139c2ea997f339c5884c3fd1477afa001a4644f.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/10cc402de1fb08a2d6019ca3b139c2ea997f339c5884c3fd1477afa001a4644f.jpg)

![2138ad2358b857dd2fcf83483f80ae743e4ea44b2b66490a9699cc3d30b5b7c0.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/2138ad2358b857dd2fcf83483f80ae743e4ea44b2b66490a9699cc3d30b5b7c0.jpg)

![2305746fd23638832fb1ff375ac7b9e767a6d23632bf46691d05f00e529ff9f1.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/2305746fd23638832fb1ff375ac7b9e767a6d23632bf46691d05f00e529ff9f1.jpg)

![26cdf36da878ef51940e21ef825255fada48b664bfbe41390857498e560c1232.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/26cdf36da878ef51940e21ef825255fada48b664bfbe41390857498e560c1232.jpg)

![2ab8c8b9f2dfce54af27b8201c86f276a371bce0604c72f2191bb16a7092bf26.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/2ab8c8b9f2dfce54af27b8201c86f276a371bce0604c72f2191bb16a7092bf26.jpg)

![30afbe624ce2f15e52e487e010831af251b0e2d1e6d28bbcf73edbf7f58537a7.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/30afbe624ce2f15e52e487e010831af251b0e2d1e6d28bbcf73edbf7f58537a7.jpg)

![4e9b62d468d714ed32e6b66eed463b2bb56c9e6e5fcaa01c9775e33027799d97.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/4e9b62d468d714ed32e6b66eed463b2bb56c9e6e5fcaa01c9775e33027799d97.jpg)

![51cbb763922ba922a06ea8843478e57738676448448110d5e94b5566e4a530ca.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/51cbb763922ba922a06ea8843478e57738676448448110d5e94b5566e4a530ca.jpg)

![533a939bad4e3c11a1d4dea30edc77940f4ac478dc390f7ccddbfca7152f3895.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/533a939bad4e3c11a1d4dea30edc77940f4ac478dc390f7ccddbfca7152f3895.jpg)

![5da339daf2f025f7c27562e226accb6b76c600ecf7845484e40ee464f7557e5f.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/5da339daf2f025f7c27562e226accb6b76c600ecf7845484e40ee464f7557e5f.jpg)

![5dec3efa166c9961c56f48df084c3c83195148eb492f9577ad8a0f3b3776cd3d.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/5dec3efa166c9961c56f48df084c3c83195148eb492f9577ad8a0f3b3776cd3d.jpg)

![6e15a6632265ea314ec2921fa6ce2f37e999f159c758ecc7c28800cbcc7903ee.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/6e15a6632265ea314ec2921fa6ce2f37e999f159c758ecc7c28800cbcc7903ee.jpg)

![7284cd169826e64fe9d24767cf97131872d92926ef4b8a52c6fa770b18701c4a.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/7284cd169826e64fe9d24767cf97131872d92926ef4b8a52c6fa770b18701c4a.jpg)

![758e7154f748237c73fc892521233f02245cc46fbbac6337021cd9fec6964bd2.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/758e7154f748237c73fc892521233f02245cc46fbbac6337021cd9fec6964bd2.jpg)

![785d4e8fdb2a6b62ece0027e2ad581ab13d2688711727f7283ca0f565bf17408.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/785d4e8fdb2a6b62ece0027e2ad581ab13d2688711727f7283ca0f565bf17408.jpg)

![864c8467d20cd5e416c7f31af5a70849be013c892ec163eb165815b40d9eeaf9.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/864c8467d20cd5e416c7f31af5a70849be013c892ec163eb165815b40d9eeaf9.jpg)

![94b198fe1ee626aee4aa568463cf2ca8cecda68ef77bd12fb558513804eac034.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/94b198fe1ee626aee4aa568463cf2ca8cecda68ef77bd12fb558513804eac034.jpg)

![95919b8f586f54b0d5780af1ba23f779124c4971142854f945aae3d32f2e41bb.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/95919b8f586f54b0d5780af1ba23f779124c4971142854f945aae3d32f2e41bb.jpg)

![a83d2b7fec1aabbdaceaecfbe2eb6a29e0dd9f0f93aaffa3d14a86fd70454c63.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/a83d2b7fec1aabbdaceaecfbe2eb6a29e0dd9f0f93aaffa3d14a86fd70454c63.jpg)

![a850813a1d65265d19222a8637f13eede8a466bd06abce41c8add4acec6e7f43.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/a850813a1d65265d19222a8637f13eede8a466bd06abce41c8add4acec6e7f43.jpg)

![a9451decec5022a8e68f8998f757b2562b6cfda09346f1d5924aa1299755d793.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/a9451decec5022a8e68f8998f757b2562b6cfda09346f1d5924aa1299755d793.jpg)

![a9a63aee596f948044599d0d798c0ebc0aa8968d5d3dd6888b9383ab4082377b.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/a9a63aee596f948044599d0d798c0ebc0aa8968d5d3dd6888b9383ab4082377b.jpg)

![b241b304c3a8dfc99488e687d1caa3db29b6530cfffda5d75e620ae6a51325f6.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/b241b304c3a8dfc99488e687d1caa3db29b6530cfffda5d75e620ae6a51325f6.jpg)

![c71906ea5558badbd5e3f926c319aeebfac30e48b3141835980aac5d2ca04565.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/c71906ea5558badbd5e3f926c319aeebfac30e48b3141835980aac5d2ca04565.jpg)

![c9d1c34683bd03d044ab8bbe40ff76c97f6062112046f9b2317abb7363ba9329.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/c9d1c34683bd03d044ab8bbe40ff76c97f6062112046f9b2317abb7363ba9329.jpg)

![d0a6ed8aca638643a165d1f06183482ae3b4755e269865bdc2e26f314027f722.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/d0a6ed8aca638643a165d1f06183482ae3b4755e269865bdc2e26f314027f722.jpg)

![e8662221e29e55e1c3c00313ea6181aa501932368735700f738e03d4c89101aa.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/e8662221e29e55e1c3c00313ea6181aa501932368735700f738e03d4c89101aa.jpg)

![e99f15a3c0cfbeb6da0e56a4607313b916e2dd56dafd1a0d86996bae49882c14.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/e99f15a3c0cfbeb6da0e56a4607313b916e2dd56dafd1a0d86996bae49882c14.jpg)

![f645672cce802e55116d93ab987d3ac763f36785ff612f998c682ee7b54f8068.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/images/f645672cce802e55116d93ab987d3ac763f36785ff612f998c682ee7b54f8068.jpg)

### Tables

![0bbe92b1f6363abe7b7119afa01b5f341517c1ff3c708d27db2f0748167b0af3.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/0bbe92b1f6363abe7b7119afa01b5f341517c1ff3c708d27db2f0748167b0af3.jpg)

![25b7e5afd5c6384ecf595935261ea518542c41076c3e41c4789c0e86c9544f30.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/25b7e5afd5c6384ecf595935261ea518542c41076c3e41c4789c0e86c9544f30.jpg)

![284ca0ad137ec1c04e2c5d58286660696077b62632770c66d6014ae62c087887.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/284ca0ad137ec1c04e2c5d58286660696077b62632770c66d6014ae62c087887.jpg)

![5a631a12321300e6c841fb77b1a8638a3b94d24f8d9056043ad8526c45a6310c.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/5a631a12321300e6c841fb77b1a8638a3b94d24f8d9056043ad8526c45a6310c.jpg)

![5b2419e185afaf5123bdec9d99cd793634025e9fd351d871d49a1198d750e4c2.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/5b2419e185afaf5123bdec9d99cd793634025e9fd351d871d49a1198d750e4c2.jpg)

![6810be95cc19817d9f421348768fed7c0a00cd465679251b0b898cadfbca7256.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/6810be95cc19817d9f421348768fed7c0a00cd465679251b0b898cadfbca7256.jpg)

![68a63082b6ce51df45ed9118f7761b42bea362e22cd4606142265efcbdfb483b.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/68a63082b6ce51df45ed9118f7761b42bea362e22cd4606142265efcbdfb483b.jpg)

![9769741bafcf81431fc7a9a7c66268c13a3355b856aeaca826d03e8bcf106f58.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/9769741bafcf81431fc7a9a7c66268c13a3355b856aeaca826d03e8bcf106f58.jpg)

![9fb07a8834e1330cd51557e173a6b22349ec9996089ce99ae48c03cdc672affc.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/9fb07a8834e1330cd51557e173a6b22349ec9996089ce99ae48c03cdc672affc.jpg)

![b8f04a859aabfc897a89bdcfc304e64a4e2117ac5b26b94320bbad4f17fe8f59.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/b8f04a859aabfc897a89bdcfc304e64a4e2117ac5b26b94320bbad4f17fe8f59.jpg)

![cde484c6daaa603b51e1367970b52fcec4b34eae3f696806df4ef49006e9e8ab.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/cde484c6daaa603b51e1367970b52fcec4b34eae3f696806df4ef49006e9e8ab.jpg)

![d50d36568ddf4e65533ba861695032c9307254bdef779fa50f6014d265fa1dc8.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/d50d36568ddf4e65533ba861695032c9307254bdef779fa50f6014d265fa1dc8.jpg)

![d9f2a461a828ec3c3f1ced1e9c7fe18c0cf2c2ce1ba26e331fd2df313ec2feed.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/d9f2a461a828ec3c3f1ced1e9c7fe18c0cf2c2ce1ba26e331fd2df313ec2feed.jpg)

![e5d2e4514c532c424b1d913b244ac142924f7f6f1ca2dd6323f4137d106ef727.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/e5d2e4514c532c424b1d913b244ac142924f7f6f1ca2dd6323f4137d106ef727.jpg)

![f264f42acd2131f3f04cb9f7e7d4411668ddd260db1102e5003aeeb8ea23b9d0.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/f264f42acd2131f3f04cb9f7e7d4411668ddd260db1102e5003aeeb8ea23b9d0.jpg)

![fe8d12254a3a5623dc20a10de583a281390f94c311a729d481c22d58055996e1.jpg](../iclr_results/585_In vivo cell-type and brain region classification via multimodal contrastive learning/tables/fe8d12254a3a5623dc20a10de583a281390f94c311a729d481c22d58055996e1.jpg)

## Monitoring Latent World States in Language Models with Propositional Probes


### Images

![00382adac5d5c0001c3c2c920e83db94971e33bbd76ec9181f8212ccfe8a04c8.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/00382adac5d5c0001c3c2c920e83db94971e33bbd76ec9181f8212ccfe8a04c8.jpg)

![0120a3f5512725c3f55e18aef760cb5f36e698e2ce46b752e41d5d3e31ca623f.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/0120a3f5512725c3f55e18aef760cb5f36e698e2ce46b752e41d5d3e31ca623f.jpg)

![12faac9ba564e82f83767bcef8c6c8747b64b2b5b57df288da4bc6d42a815232.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/12faac9ba564e82f83767bcef8c6c8747b64b2b5b57df288da4bc6d42a815232.jpg)

![2bbae1a7e2a4211f227cc3e9e0aa61a7dff43c8a2eb463051d51b8e4c1838a05.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/2bbae1a7e2a4211f227cc3e9e0aa61a7dff43c8a2eb463051d51b8e4c1838a05.jpg)

![34b76a415704c1e57a67fa73d2a4a0517af8752d9bc275a95d829e64d317ed44.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/34b76a415704c1e57a67fa73d2a4a0517af8752d9bc275a95d829e64d317ed44.jpg)

![6c2ce3a0fa982721f2f8e986d57baa7398412a211b56b9738d8a785c82dd69d1.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/6c2ce3a0fa982721f2f8e986d57baa7398412a211b56b9738d8a785c82dd69d1.jpg)

![81637e96ca2e3311c1394f1cc1ae06e06f76672a5844e2700f457df32bd24c7a.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/81637e96ca2e3311c1394f1cc1ae06e06f76672a5844e2700f457df32bd24c7a.jpg)

![864c981072dad7c010f9bbb643f14c0f2eaec398eb7dee03b48d84fbff0b2097.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/864c981072dad7c010f9bbb643f14c0f2eaec398eb7dee03b48d84fbff0b2097.jpg)

![8aa0615e0adbaefc385ae80c2bee609648490b920eb6f9775085a48adc34bfea.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/8aa0615e0adbaefc385ae80c2bee609648490b920eb6f9775085a48adc34bfea.jpg)

![9b5ae3d0196c7969be156cb8eb709aea3f5312b81e014164dac5be9695d0af81.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/9b5ae3d0196c7969be156cb8eb709aea3f5312b81e014164dac5be9695d0af81.jpg)

![9bd16e7ec1bc7a6325134c8c9b3a73f3e8a7b7df5123c81f10e360448fec554f.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/9bd16e7ec1bc7a6325134c8c9b3a73f3e8a7b7df5123c81f10e360448fec554f.jpg)

![a309d6713da88266b630dbaa299d9fb8c538d92a7d44ae8616abea1c8549983a.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/a309d6713da88266b630dbaa299d9fb8c538d92a7d44ae8616abea1c8549983a.jpg)

![a83a4217d46c0c42b695bed00cad110e5a1ea64fa0b9d1c4c8d87fb41fae3c9b.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/a83a4217d46c0c42b695bed00cad110e5a1ea64fa0b9d1c4c8d87fb41fae3c9b.jpg)

![b0b9084e8f40a8feae82f226148668810e51e724befa6bc77b094421630ca43e.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/b0b9084e8f40a8feae82f226148668810e51e724befa6bc77b094421630ca43e.jpg)

![b81b8917f2fa9c1323b4c45985295bd1ef9fc5cfa88ef09438222a62853dfe4e.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/b81b8917f2fa9c1323b4c45985295bd1ef9fc5cfa88ef09438222a62853dfe4e.jpg)

![bb6bda360c5ba97ea10f28e553eb36b826e6ef0ce98498db19202921f2019d0a.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/bb6bda360c5ba97ea10f28e553eb36b826e6ef0ce98498db19202921f2019d0a.jpg)

![c74197363fc1a5c314b73632ad217d644c9eeff743e0da0621b69c2124c560cf.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/c74197363fc1a5c314b73632ad217d644c9eeff743e0da0621b69c2124c560cf.jpg)

![e59c8868b20493c6c60fa4f8c897629de2f4e76081d6185fdfa6def61f6ea14c.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/e59c8868b20493c6c60fa4f8c897629de2f4e76081d6185fdfa6def61f6ea14c.jpg)

![fba8c0263f6c11a5385eaac4d1cfbd25094a5164b45f49fa19e9e43a912336a1.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/images/fba8c0263f6c11a5385eaac4d1cfbd25094a5164b45f49fa19e9e43a912336a1.jpg)

### Tables

![0dcefde4edc97424d7960dca80fde29d1c6ce3483ac2f9137c03314a63c185f7.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/tables/0dcefde4edc97424d7960dca80fde29d1c6ce3483ac2f9137c03314a63c185f7.jpg)

![af911a3d4ef6371e52b438592af89444343504ee52369a1e4664a351325c2b3e.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/tables/af911a3d4ef6371e52b438592af89444343504ee52369a1e4664a351325c2b3e.jpg)

![d96c5b476e60d83565c5ea462f04dbaa73ef10304e6e74fef998fffab55726d9.jpg](../iclr_results/586_Monitoring Latent World States in Language Models with Propositional Probes/tables/d96c5b476e60d83565c5ea462f04dbaa73ef10304e6e74fef998fffab55726d9.jpg)

## Universal generalization guarantees for Wasserstein distributionally robust models


### Images

![f24f2c52be1f531899f8e2de3ac44f449c14c10ab7f5b80112297a076c0a500a.jpg](../iclr_results/587_Universal generalization guarantees for Wasserstein distributionally robust models/images/f24f2c52be1f531899f8e2de3ac44f449c14c10ab7f5b80112297a076c0a500a.jpg)

![fd8abf05e43e0f0b6f48adbed42006db22810dd1b1e0a97598be21ca555757b9.jpg](../iclr_results/587_Universal generalization guarantees for Wasserstein distributionally robust models/images/fd8abf05e43e0f0b6f48adbed42006db22810dd1b1e0a97598be21ca555757b9.jpg)

## PETRA: Parallel End-to-end Training with Reversible Architectures


### Images

![31e656ba173da5d3d875e91a8403e141906200065b92a4d51fb6e83b20053c83.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/31e656ba173da5d3d875e91a8403e141906200065b92a4d51fb6e83b20053c83.jpg)

![42fd084357fd86e54f88c18b122c0cad206af92193b036b48559840c023a3097.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/42fd084357fd86e54f88c18b122c0cad206af92193b036b48559840c023a3097.jpg)

![47da7b00b0383609a36146f0be9adc122ed83b5aed5cb307943b4306f2ff829a.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/47da7b00b0383609a36146f0be9adc122ed83b5aed5cb307943b4306f2ff829a.jpg)

![55e3d1b5a227a3c879eb9a948ebf2eefa455991d1bdf27947a3879ef4ad25151.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/55e3d1b5a227a3c879eb9a948ebf2eefa455991d1bdf27947a3879ef4ad25151.jpg)

![581ba2aa59166430e30c41e2954511ead9dcf5e236337a4dd91c4eb675565d3b.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/581ba2aa59166430e30c41e2954511ead9dcf5e236337a4dd91c4eb675565d3b.jpg)

![62f54c7b46192a9e59e96cd786898d0fb6248d8b4fae7ab09c57cf33367f12ab.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/62f54c7b46192a9e59e96cd786898d0fb6248d8b4fae7ab09c57cf33367f12ab.jpg)

![67844e7ebee1ec14e2a425e36db8b5e83e708014c1eef7de5f2030bd5a6956e8.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/67844e7ebee1ec14e2a425e36db8b5e83e708014c1eef7de5f2030bd5a6956e8.jpg)

![a2897a99f70a24a2b18d88a877087d116fe875f5e6ac6e0d2972eac13ae4eb33.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/a2897a99f70a24a2b18d88a877087d116fe875f5e6ac6e0d2972eac13ae4eb33.jpg)

![c92536ce0e5da799a3ae180ef6086cac3b2329c57ad502b701a7bc6753c50c60.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/c92536ce0e5da799a3ae180ef6086cac3b2329c57ad502b701a7bc6753c50c60.jpg)

![d5a9be8a8bce87801c9faa812e73d99c1530b68cd7a0acab917534fa68f67457.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/d5a9be8a8bce87801c9faa812e73d99c1530b68cd7a0acab917534fa68f67457.jpg)

![ee4d2a5b9a10e1505f9d3e0e8c333a22036747820b9d1ac5a0b64d52dfd59271.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/ee4d2a5b9a10e1505f9d3e0e8c333a22036747820b9d1ac5a0b64d52dfd59271.jpg)

![f31ae603d3ccf689de76e21672ede12d0336a470eb9dd585b74dc46ac997703e.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/images/f31ae603d3ccf689de76e21672ede12d0336a470eb9dd585b74dc46ac997703e.jpg)

### Tables

![0ee0e1cb4a411644b6de76a1228968e058584558c44bfd25775fd103432294a3.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/tables/0ee0e1cb4a411644b6de76a1228968e058584558c44bfd25775fd103432294a3.jpg)

![45c5f7e864f8167f084d299d86678383769804f92d8a4e435db976b5deedf4b0.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/tables/45c5f7e864f8167f084d299d86678383769804f92d8a4e435db976b5deedf4b0.jpg)

![5186dc0448ee9286ccaa6e7dea9c3c2fe9a02f0481aa8ba0d612d5f417543069.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/tables/5186dc0448ee9286ccaa6e7dea9c3c2fe9a02f0481aa8ba0d612d5f417543069.jpg)

![81819f015aa25e7f81988fb26152c0198b97d99ba84681b73714320538d8177f.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/tables/81819f015aa25e7f81988fb26152c0198b97d99ba84681b73714320538d8177f.jpg)

![83b8f6e24c7276259dc856ab354fc46a4dede2e2ed42de4eb2e68911e06386ef.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/tables/83b8f6e24c7276259dc856ab354fc46a4dede2e2ed42de4eb2e68911e06386ef.jpg)

![da4c3ef962a2f8172c4fb6902f188632c746d70d1000f759ad5d3c3526318878.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/tables/da4c3ef962a2f8172c4fb6902f188632c746d70d1000f759ad5d3c3526318878.jpg)

![e91c6947090667931cf4089447f637d3d30107a90d18f150e4e5691fbdc56407.jpg](../iclr_results/588_PETRA_ Parallel End-to-end Training with Reversible Architectures/tables/e91c6947090667931cf4089447f637d3d30107a90d18f150e4e5691fbdc56407.jpg)

## Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage


### Images

![03677b183cc10c2aaf1339b41417f8b1238c4bf88e970fc7f15f59187221f856.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/03677b183cc10c2aaf1339b41417f8b1238c4bf88e970fc7f15f59187221f856.jpg)

![1763f10a809fc078627eb257ccdb38931083bae02cf706ae6b0e477e618a6baa.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/1763f10a809fc078627eb257ccdb38931083bae02cf706ae6b0e477e618a6baa.jpg)

![20285bd2ea075962366dac5db54c5c839c9d4becb75dbf5537f15f1cbb5c1382.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/20285bd2ea075962366dac5db54c5c839c9d4becb75dbf5537f15f1cbb5c1382.jpg)

![37d80d564938a1d090a2e08d7f992e7d96bdb087ddf8c81c0fa087389efb59fb.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/37d80d564938a1d090a2e08d7f992e7d96bdb087ddf8c81c0fa087389efb59fb.jpg)

![626452f7afe09c70bddcc8c7639acaa2e5fe9b03e2b613f51b40b9ca26d49de2.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/626452f7afe09c70bddcc8c7639acaa2e5fe9b03e2b613f51b40b9ca26d49de2.jpg)

![7705c1fe23f86ebb9deaf348e854d93ed2eed0f8750851a5de534704579736ca.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/7705c1fe23f86ebb9deaf348e854d93ed2eed0f8750851a5de534704579736ca.jpg)

![b41fcc8ee84907bbe1e56fab6582fcc425899391183673085d415fd5b3b9b874.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/b41fcc8ee84907bbe1e56fab6582fcc425899391183673085d415fd5b3b9b874.jpg)

![be912a680ab4288f59e8e9b4e5d23510cf0c63bbe98689b0a656b4efa58140b4.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/be912a680ab4288f59e8e9b4e5d23510cf0c63bbe98689b0a656b4efa58140b4.jpg)

![c7b2ecb58c2bf52c5b5ceed99f72b9649e726625dd9669e156c88775ed8a63e6.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/c7b2ecb58c2bf52c5b5ceed99f72b9649e726625dd9669e156c88775ed8a63e6.jpg)

![ca7fdc99fd57aca60f4eed2a272a5a9dda6755dde6a950d9e66c4dae3eeb15ec.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/ca7fdc99fd57aca60f4eed2a272a5a9dda6755dde6a950d9e66c4dae3eeb15ec.jpg)

![cbd38634171a6c7c75f9bbd8f644b3ec73789c729bf96714d381ce6a3f63b3d4.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/cbd38634171a6c7c75f9bbd8f644b3ec73789c729bf96714d381ce6a3f63b3d4.jpg)

![d2fe94b6480e2207726a399bd94760911dbd54d3938b473f16db23e0e7bd02c7.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/d2fe94b6480e2207726a399bd94760911dbd54d3938b473f16db23e0e7bd02c7.jpg)

![e7c7b78532ba4ddd7048605399b0bcc9b83180ae1e00777e32f0b4341ddaa280.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/images/e7c7b78532ba4ddd7048605399b0bcc9b83180ae1e00777e32f0b4341ddaa280.jpg)

### Tables

![052e81db36f43fa567c91c15b556f5313f2ceed8113d3ff4341f55f9d2eb2441.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/052e81db36f43fa567c91c15b556f5313f2ceed8113d3ff4341f55f9d2eb2441.jpg)

![0fb39d41a72b5265fcb9f0c9004d0cbc89be4dfd20a6419cc09c6fbd95dba324.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/0fb39d41a72b5265fcb9f0c9004d0cbc89be4dfd20a6419cc09c6fbd95dba324.jpg)

![19b4e227ee1efebe3c32fcf266567c8dc2d19eb061af38dc82e06ddac1d7788a.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/19b4e227ee1efebe3c32fcf266567c8dc2d19eb061af38dc82e06ddac1d7788a.jpg)

![1acd5325e7607bae89a8727bd6480514c566b6c6226e137f95379ef0238b7332.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/1acd5325e7607bae89a8727bd6480514c566b6c6226e137f95379ef0238b7332.jpg)

![2485410e7ccc1cf933e91bba8f2a14d84f0cf4fd31a5918b9a63373edb4b6d2b.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/2485410e7ccc1cf933e91bba8f2a14d84f0cf4fd31a5918b9a63373edb4b6d2b.jpg)

![3933e2c8ce311574ce6ca5699441362219091219037ae09f97f799bd540819bb.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/3933e2c8ce311574ce6ca5699441362219091219037ae09f97f799bd540819bb.jpg)

![494beed986af3d377e58d96ad58b6c72c296b2bdbe844ee559450957ad757204.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/494beed986af3d377e58d96ad58b6c72c296b2bdbe844ee559450957ad757204.jpg)

![57159596ef101f11765a0e2ea79aeea136a63ab73db7ad0b9d87e3fef84fdfb4.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/57159596ef101f11765a0e2ea79aeea136a63ab73db7ad0b9d87e3fef84fdfb4.jpg)

![95fd81ec8785012655c7b5031cc76f84d99612981439fb7ce453ee26c404ab19.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/95fd81ec8785012655c7b5031cc76f84d99612981439fb7ce453ee26c404ab19.jpg)

![d1e91d7b04db11ab42578ab2c962eebf2974467b1518ac5aebaadefd8c834c99.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/d1e91d7b04db11ab42578ab2c962eebf2974467b1518ac5aebaadefd8c834c99.jpg)

![dfab3f9f1877bf0cf9a4bdf53d6865476e29940e96a54682a54ce8e20ebbc2e7.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/dfab3f9f1877bf0cf9a4bdf53d6865476e29940e96a54682a54ce8e20ebbc2e7.jpg)

![e557fa7a1ee2f1e5e11f09a5ca1c7ceb36a48a6d3fbfa01204bc9198391cbd8c.jpg](../iclr_results/590_Multi-modal Agent Tuning_ Building a VLM-Driven Agent for Efficient Tool Usage/tables/e557fa7a1ee2f1e5e11f09a5ca1c7ceb36a48a6d3fbfa01204bc9198391cbd8c.jpg)

## Holistically Evaluating the Environmental Impact of Creating Language Models


### Images

![af6afca81e80bae10ba51fd251bd13f7f4af9a5c8559d724c22da22aea0171e4.jpg](../iclr_results/591_Holistically Evaluating the Environmental Impact of Creating Language Models/images/af6afca81e80bae10ba51fd251bd13f7f4af9a5c8559d724c22da22aea0171e4.jpg)

![ca32abcace37e22c9a2bc945c8a3a17764d952cea0b7df2a3a2fd5e76ba3bf7e.jpg](../iclr_results/591_Holistically Evaluating the Environmental Impact of Creating Language Models/images/ca32abcace37e22c9a2bc945c8a3a17764d952cea0b7df2a3a2fd5e76ba3bf7e.jpg)

### Tables

![2cb0910f98f9418ba9dd1c8e714e6acb675fe6f27a1787378f532210d5ce0a61.jpg](../iclr_results/591_Holistically Evaluating the Environmental Impact of Creating Language Models/tables/2cb0910f98f9418ba9dd1c8e714e6acb675fe6f27a1787378f532210d5ce0a61.jpg)

![a81dec0258a5a711ddc34606d95ebea24f76ffc239984d16033a2ba5f06b5677.jpg](../iclr_results/591_Holistically Evaluating the Environmental Impact of Creating Language Models/tables/a81dec0258a5a711ddc34606d95ebea24f76ffc239984d16033a2ba5f06b5677.jpg)

![afd92b6343d74004b399c34a4fdffb41cb7fcfe9153f68579cf9e142415a0cec.jpg](../iclr_results/591_Holistically Evaluating the Environmental Impact of Creating Language Models/tables/afd92b6343d74004b399c34a4fdffb41cb7fcfe9153f68579cf9e142415a0cec.jpg)

![b04ddcfea3b61e83d8338475d39f1b0faf0fe7701a2c4e0cf8d565a001e5acd0.jpg](../iclr_results/591_Holistically Evaluating the Environmental Impact of Creating Language Models/tables/b04ddcfea3b61e83d8338475d39f1b0faf0fe7701a2c4e0cf8d565a001e5acd0.jpg)

## Adaptive Gradient Clipping for Robust Federated Learning


### Images

![02ee545b4788cdc667316cac111fc2e2f196c576032e8021ab79e52526a14123.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/02ee545b4788cdc667316cac111fc2e2f196c576032e8021ab79e52526a14123.jpg)

![03d7436e2e2c2d7f32bc872b026e444ceef2140a32a39a867727a1d270b77786.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/03d7436e2e2c2d7f32bc872b026e444ceef2140a32a39a867727a1d270b77786.jpg)

![083265e5f9940a5053a071efa76717e09eac219605a4000389528a4040ba0637.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/083265e5f9940a5053a071efa76717e09eac219605a4000389528a4040ba0637.jpg)

![104ad214d47f9520b7f2adcd11e5439c049083b189f66adef443d6b122cdf535.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/104ad214d47f9520b7f2adcd11e5439c049083b189f66adef443d6b122cdf535.jpg)

![1feaff50cb2fb5d0890c56198c062502587db0452accede3283a8dbf2eaa3670.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/1feaff50cb2fb5d0890c56198c062502587db0452accede3283a8dbf2eaa3670.jpg)

![24efb479c192aa455f3743a4af7db58d7d8daf00a6242d0aba236b6c29d47ec4.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/24efb479c192aa455f3743a4af7db58d7d8daf00a6242d0aba236b6c29d47ec4.jpg)

![473c05f61a17856f164c903b3a57ea4cc41532c885e5b47271978116b3b87272.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/473c05f61a17856f164c903b3a57ea4cc41532c885e5b47271978116b3b87272.jpg)

![49a28c94df3067a102fc6d8ed507f29939347a21006cc9b2c4a672342f042229.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/49a28c94df3067a102fc6d8ed507f29939347a21006cc9b2c4a672342f042229.jpg)

![59effbb3a4db4ca3520b0fc8f037937a05effc111d815703acb5ed0e36c5cc8e.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/59effbb3a4db4ca3520b0fc8f037937a05effc111d815703acb5ed0e36c5cc8e.jpg)

![5d068a408513848c9231ea4e14440afcd778e6852aca9d649a7385ecf68213e8.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/5d068a408513848c9231ea4e14440afcd778e6852aca9d649a7385ecf68213e8.jpg)

![627f66e98d73a736f9085538c0921ad458fbc0543dd19637f5e7b8dd6b5ae9a9.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/627f66e98d73a736f9085538c0921ad458fbc0543dd19637f5e7b8dd6b5ae9a9.jpg)

![673f2fe2939cb1541b395a76331ae86846da9aa07ba17417ecab3b524fcd20b7.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/673f2fe2939cb1541b395a76331ae86846da9aa07ba17417ecab3b524fcd20b7.jpg)

![740f5ad090ae11e9f2093effc18eaf2e6030c760b39d94434015145c56f2b269.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/740f5ad090ae11e9f2093effc18eaf2e6030c760b39d94434015145c56f2b269.jpg)

![7e4bfdd428462674f84bca8005c4bb9a6f728728db7ca73c581b580072792aee.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/7e4bfdd428462674f84bca8005c4bb9a6f728728db7ca73c581b580072792aee.jpg)

![7fb1d24b26421cd98304b0d5aa484866fefb67e1ef0d37c2504b3e2af4649607.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/7fb1d24b26421cd98304b0d5aa484866fefb67e1ef0d37c2504b3e2af4649607.jpg)

![8a1d69eae3d7f431e9de5f19e3dce203813ea381337ab05fe452a90db54709a3.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/8a1d69eae3d7f431e9de5f19e3dce203813ea381337ab05fe452a90db54709a3.jpg)

![a31237df371f8e299023dc619292b9b1fc0625f13c08083e709f237c9a5a37cb.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/a31237df371f8e299023dc619292b9b1fc0625f13c08083e709f237c9a5a37cb.jpg)

![a3faa64ec6c8cee1fb954eac422ea00692b119f6b0ea979305b06b6831b8d254.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/a3faa64ec6c8cee1fb954eac422ea00692b119f6b0ea979305b06b6831b8d254.jpg)

![a812975e144cb744013d91edbd6b3db8325516b8755939f4b29c8e08322f5b85.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/a812975e144cb744013d91edbd6b3db8325516b8755939f4b29c8e08322f5b85.jpg)

![b05b209775ecfdff352dedfc9e6f118965bd5f098a0fbd6aa9e0f4e053cfc829.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/b05b209775ecfdff352dedfc9e6f118965bd5f098a0fbd6aa9e0f4e053cfc829.jpg)

![bc95ad0a6dc89ad4532f7b33dc14935145dadc8b1e112e22586ab5e7384befce.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/bc95ad0a6dc89ad4532f7b33dc14935145dadc8b1e112e22586ab5e7384befce.jpg)

![c125c7827817d9389ed5d8b8606b253f8f0e5815bc77c0d83a09fe6c88730fd5.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/c125c7827817d9389ed5d8b8606b253f8f0e5815bc77c0d83a09fe6c88730fd5.jpg)

![d08a9096374c4bc07343e8e224b00de06d056acdd1c615d2c2c536e23ab8c7c2.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/d08a9096374c4bc07343e8e224b00de06d056acdd1c615d2c2c536e23ab8c7c2.jpg)

![d153d07567a771ae733264683627b8a995320a26b0b2034de099bc552894c120.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/d153d07567a771ae733264683627b8a995320a26b0b2034de099bc552894c120.jpg)

![d1708d84c549766b4f8a9858c9293f21bf43bb431bf39848c8e8ef51bda56679.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/d1708d84c549766b4f8a9858c9293f21bf43bb431bf39848c8e8ef51bda56679.jpg)

![d2cb8936e802e253cd16ad628864cdea8441fba2d02131f524d81ab8bf0c7f1e.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/d2cb8936e802e253cd16ad628864cdea8441fba2d02131f524d81ab8bf0c7f1e.jpg)

![f94daff88f45dcf227503b346eec9bc2c09f93d33998fb5c2df2d71367b582d4.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/images/f94daff88f45dcf227503b346eec9bc2c09f93d33998fb5c2df2d71367b582d4.jpg)

### Tables

![04cc6ccebdbf22f291d7e2fa9357c1e427a3234c6551a697a11f19214aa63481.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/tables/04cc6ccebdbf22f291d7e2fa9357c1e427a3234c6551a697a11f19214aa63481.jpg)

![2cc6d43b00a474ba86857947fd792429fcaf082399fd7b3a9c71b316ab985971.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/tables/2cc6d43b00a474ba86857947fd792429fcaf082399fd7b3a9c71b316ab985971.jpg)

![2d8f6135792b130722d4ae7994a54b161d97ee4493998d91766daa8855a99fff.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/tables/2d8f6135792b130722d4ae7994a54b161d97ee4493998d91766daa8855a99fff.jpg)

![67434d19f6d89ea4ab18ec00576aa122d91007b91959942ac1a7bd65319a6ad2.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/tables/67434d19f6d89ea4ab18ec00576aa122d91007b91959942ac1a7bd65319a6ad2.jpg)

![776d783358a1113f5b30ffd9311d4afce25b18820cd67f6f1fa9d5387550a90a.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/tables/776d783358a1113f5b30ffd9311d4afce25b18820cd67f6f1fa9d5387550a90a.jpg)

![c7604aea19b8d81c0043e1f879f0bd89b1c8cbac75a66f0404a966bb68b49263.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/tables/c7604aea19b8d81c0043e1f879f0bd89b1c8cbac75a66f0404a966bb68b49263.jpg)

![cf147e538a077615c070859835c5d68c95eca5e090f845a3863be21e726b43f2.jpg](../iclr_results/592_Adaptive Gradient Clipping for Robust Federated Learning/tables/cf147e538a077615c070859835c5d68c95eca5e090f845a3863be21e726b43f2.jpg)

## Re-Imagining Multimodal Instruction Tuning: A Representation View


### Images

![2205ef7cdbe34e371df96527dedec16e550288405da47b04ab39c0ac4bd56d0b.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/2205ef7cdbe34e371df96527dedec16e550288405da47b04ab39c0ac4bd56d0b.jpg)

![270ea1dcc6b5b522177ff550426eb297a48c3f1e48055d5bfd387926e7b7df18.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/270ea1dcc6b5b522177ff550426eb297a48c3f1e48055d5bfd387926e7b7df18.jpg)

![58bf63304385dd9eb423eeb20f45c80b11fec280cfe54907f2a9aca2d986a3e1.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/58bf63304385dd9eb423eeb20f45c80b11fec280cfe54907f2a9aca2d986a3e1.jpg)

![75715e430155f6ecf998cab27ffd1a50f14b7767c4fc8ba52dfbb5ecb9501027.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/75715e430155f6ecf998cab27ffd1a50f14b7767c4fc8ba52dfbb5ecb9501027.jpg)

![868f4ac4750396ac521a4621c215284abe24dc10587cd74ff98c29abc805b7b2.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/868f4ac4750396ac521a4621c215284abe24dc10587cd74ff98c29abc805b7b2.jpg)

![92bf76e1864c8eb2111e9c864b257a8c0d506abd07d9871484b4ef543221d014.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/92bf76e1864c8eb2111e9c864b257a8c0d506abd07d9871484b4ef543221d014.jpg)

![c1d11429791f5f9fb07c7c5150f0b91e00ce10282a5e81872a1f3d837a6cb52b.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/c1d11429791f5f9fb07c7c5150f0b91e00ce10282a5e81872a1f3d837a6cb52b.jpg)

![c23d77015f52c853caba92d3019b20cd7383d99d2f3e4d7cca06d41a3f5b6ada.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/c23d77015f52c853caba92d3019b20cd7383d99d2f3e4d7cca06d41a3f5b6ada.jpg)

![ddf70cf18af71bcde99a88da5de67f365340e848f0c6b3e95c31337d24491524.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/images/ddf70cf18af71bcde99a88da5de67f365340e848f0c6b3e95c31337d24491524.jpg)

### Tables

![057e17c7b8ff0000f9a156a030bde843c586138ba35eea259d86ecf6f775cf7d.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/057e17c7b8ff0000f9a156a030bde843c586138ba35eea259d86ecf6f775cf7d.jpg)

![223718410e54364f205a3ef7256aada4e333c67ddc0ce8041e5b475fecf0bd4a.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/223718410e54364f205a3ef7256aada4e333c67ddc0ce8041e5b475fecf0bd4a.jpg)

![24fb599d0c90153b4f9de2517d30521f807f5616768245e6eda520c80f5c29ea.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/24fb599d0c90153b4f9de2517d30521f807f5616768245e6eda520c80f5c29ea.jpg)

![26e84c3545939f44bc3afda770384878d5e498e0ed5a63497e374a5bf405bd64.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/26e84c3545939f44bc3afda770384878d5e498e0ed5a63497e374a5bf405bd64.jpg)

![42f1b5a95babe6653ac8c62d4ca336dff2792146c9983d0ca3d9606a30e51c14.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/42f1b5a95babe6653ac8c62d4ca336dff2792146c9983d0ca3d9606a30e51c14.jpg)

![5f4aa1ab57fffd7bcb24ba9fdc9462c1b9c778f5b3855919131bc2f9e5fb4ec1.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/5f4aa1ab57fffd7bcb24ba9fdc9462c1b9c778f5b3855919131bc2f9e5fb4ec1.jpg)

![67607dfd07476fd1e17df5947568f8a4bcb455ca280e819cfb3f75d561671dbd.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/67607dfd07476fd1e17df5947568f8a4bcb455ca280e819cfb3f75d561671dbd.jpg)

![7310ba98cfa99737388fb1e4c3d0b40e8b6b0159fb01ceb002122a17cd0bd12f.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/7310ba98cfa99737388fb1e4c3d0b40e8b6b0159fb01ceb002122a17cd0bd12f.jpg)

![90635c7ef2c60ea4aeda7db3b1fce74b6be432c88774458f45c6c4642cf11c1c.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/90635c7ef2c60ea4aeda7db3b1fce74b6be432c88774458f45c6c4642cf11c1c.jpg)

![c1a1ac79980cb35673a4284eb821be32ef96ff1f1639ef9c41baf0ff54accae2.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/c1a1ac79980cb35673a4284eb821be32ef96ff1f1639ef9c41baf0ff54accae2.jpg)

![c5d8ddd616a1dbe2a72815a5e22b003683c2772c1d4e170ceb60583ed1558b31.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/c5d8ddd616a1dbe2a72815a5e22b003683c2772c1d4e170ceb60583ed1558b31.jpg)

![ca7d8b6bcb33b3f6f9f3a5267f9939d73cfb3ae6889f6d4e08334cc9d3df515d.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/ca7d8b6bcb33b3f6f9f3a5267f9939d73cfb3ae6889f6d4e08334cc9d3df515d.jpg)

![eeb02927d25a885ebf734a49ee1cbdcaffa40a310abd7f60b7b85fabf4b9bd5d.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/eeb02927d25a885ebf734a49ee1cbdcaffa40a310abd7f60b7b85fabf4b9bd5d.jpg)

![f4e973d0536ae0e8191ad895e77074f55d8959aa35944ee0cd95797bf3e27dfa.jpg](../iclr_results/594_Re-Imagining Multimodal Instruction Tuning_ A Representation View/tables/f4e973d0536ae0e8191ad895e77074f55d8959aa35944ee0cd95797bf3e27dfa.jpg)

## Inverse decision-making using neural amortized Bayesian actors


### Images

![5df12ef7061dc629c473f147288eecb8f8acc5a291869d958d72c6d602e789ef.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/5df12ef7061dc629c473f147288eecb8f8acc5a291869d958d72c6d602e789ef.jpg)

![7a5b13ff5116f30a43d421d7e914b703c34e2de210d108476aae87de93596afe.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/7a5b13ff5116f30a43d421d7e914b703c34e2de210d108476aae87de93596afe.jpg)

![91980472e386ff8a1f30c76a5e6c6bc94a0a4bbeadeed94f7f8c2cc1c2ea9e1a.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/91980472e386ff8a1f30c76a5e6c6bc94a0a4bbeadeed94f7f8c2cc1c2ea9e1a.jpg)

![bc577177a2a28ac980fb9f18e442d8c783b3f9a84088a81bda22d97846902d1a.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/bc577177a2a28ac980fb9f18e442d8c783b3f9a84088a81bda22d97846902d1a.jpg)

![c18ac4e2f68f5d5d767cc61092939521624e1606bf821f370898a39eff8cd31d.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/c18ac4e2f68f5d5d767cc61092939521624e1606bf821f370898a39eff8cd31d.jpg)

![c63a754e68b88c1e2fd0bc41d554ca51b34a90878eefa713161865706133dd1a.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/c63a754e68b88c1e2fd0bc41d554ca51b34a90878eefa713161865706133dd1a.jpg)

![e685894059de70b6839266bab30a9dd197ba93d6f1a3184f788dc7dbe10b4468.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/e685894059de70b6839266bab30a9dd197ba93d6f1a3184f788dc7dbe10b4468.jpg)

![ec941f366807065b2e3ad5b19a4706f93d1220d3e437b9a54502d74355502fe4.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/ec941f366807065b2e3ad5b19a4706f93d1220d3e437b9a54502d74355502fe4.jpg)

![f03ba9b76cd6800ad9e4c69446058ad3641b4ccad95090c3540bdb345077ab11.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/images/f03ba9b76cd6800ad9e4c69446058ad3641b4ccad95090c3540bdb345077ab11.jpg)

### Tables

![6a5f533f1d3c237083fb3be1ac783a8f2fd51aada6441ba11a43dadf9ad094a7.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/tables/6a5f533f1d3c237083fb3be1ac783a8f2fd51aada6441ba11a43dadf9ad094a7.jpg)

![926fe0da0a22655df6215a27d567c5e3ab3903481fe4013d37a33b9a35645efc.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/tables/926fe0da0a22655df6215a27d567c5e3ab3903481fe4013d37a33b9a35645efc.jpg)

![e0ef52165d876068f6d49d549dc6cd1b72ab79fa49c1a3f8bd249eaa78927cc7.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/tables/e0ef52165d876068f6d49d549dc6cd1b72ab79fa49c1a3f8bd249eaa78927cc7.jpg)

![fe4d3b56d323f21dbb5da88560423613af496a70cd9e9d3a6e5ae3a9b275cf97.jpg](../iclr_results/595_Inverse decision-making using neural amortized Bayesian actors/tables/fe4d3b56d323f21dbb5da88560423613af496a70cd9e9d3a6e5ae3a9b275cf97.jpg)

## Designing Concise ConvNets with Columnar Stages


### Images

![0652de361ebe33ce87e1c17575b8f6f5677de6a5316e9b0b2be2796b807c9e51.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/0652de361ebe33ce87e1c17575b8f6f5677de6a5316e9b0b2be2796b807c9e51.jpg)

![086d05b921cb31b8244a8b3dc1010ae60c0ce16d245e712351efae6430c74265.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/086d05b921cb31b8244a8b3dc1010ae60c0ce16d245e712351efae6430c74265.jpg)

![3dddc0d42625f2be2945c42deb2915fbff1642aa1fe69ba1efc9f6570b1b3482.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/3dddc0d42625f2be2945c42deb2915fbff1642aa1fe69ba1efc9f6570b1b3482.jpg)

![55ebfbc975fd4ccdecd80609f03ef384748b9e006c96b550313d6a1b0b881a30.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/55ebfbc975fd4ccdecd80609f03ef384748b9e006c96b550313d6a1b0b881a30.jpg)

![6a37bca9eb9ab7a818cb6ea32acde42d4b96d9485f23fe8b66a678881bb86129.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/6a37bca9eb9ab7a818cb6ea32acde42d4b96d9485f23fe8b66a678881bb86129.jpg)

![b19167ec9e48ad5d3f0b9753f8b17f3a499ee10f831a30e162d632eb91de3709.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/b19167ec9e48ad5d3f0b9753f8b17f3a499ee10f831a30e162d632eb91de3709.jpg)

![b71f59abf6e58caf579ecd0f57f16cb58e69f7080381637bc576a1a8c5291b44.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/b71f59abf6e58caf579ecd0f57f16cb58e69f7080381637bc576a1a8c5291b44.jpg)

![fa1371c935dfa2116a7fdcaea4ffff3265b3f458bec8de5c102a4b2071f47461.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/images/fa1371c935dfa2116a7fdcaea4ffff3265b3f458bec8de5c102a4b2071f47461.jpg)

### Tables

![07f7ee16b0e3065cc023dfa45eaa235151bbdf78a74d6bed035a929c266f036b.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/07f7ee16b0e3065cc023dfa45eaa235151bbdf78a74d6bed035a929c266f036b.jpg)

![3278c33a6e28ca061dc0e7fcd4e36959848756ff3cab1b796216ce33dd50b695.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/3278c33a6e28ca061dc0e7fcd4e36959848756ff3cab1b796216ce33dd50b695.jpg)

![4120fca43e1771ae3d22232f4c145ac43bc1136c62b311c0a8e4cda7b9231d4a.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/4120fca43e1771ae3d22232f4c145ac43bc1136c62b311c0a8e4cda7b9231d4a.jpg)

![422cee9cdb7e2a5ec59ab6eb2852ba636cf963d7a3ced7b36693eec0005a33cd.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/422cee9cdb7e2a5ec59ab6eb2852ba636cf963d7a3ced7b36693eec0005a33cd.jpg)

![4dbdedb80b837732dfc61170c2f37235e2bb41bb415e05747116edc949d44863.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/4dbdedb80b837732dfc61170c2f37235e2bb41bb415e05747116edc949d44863.jpg)

![5ecca3e6d869305bc5df083533f99e0bfbf94a1c8896d5a4270d2701e7178b94.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/5ecca3e6d869305bc5df083533f99e0bfbf94a1c8896d5a4270d2701e7178b94.jpg)

![b4fe8f08f2333f6060ff449195d230fa41ff3094ddd5923d58a01f51f49dd604.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/b4fe8f08f2333f6060ff449195d230fa41ff3094ddd5923d58a01f51f49dd604.jpg)

![f77a2d44a93fbb56630e114bdd62c7d232a12966a0e6704b757b86ed4db6253c.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/f77a2d44a93fbb56630e114bdd62c7d232a12966a0e6704b757b86ed4db6253c.jpg)

![fc16e320b501a7875eb2816a267a7d6ff4c3d5583674855eba53f7278c1ed248.jpg](../iclr_results/596_Designing Concise ConvNets with Columnar Stages/tables/fc16e320b501a7875eb2816a267a7d6ff4c3d5583674855eba53f7278c1ed248.jpg)

## MVTokenFlow: High-quality 4D Content Generation using Multiview Token Flow


### Images

![0eab6380018420da48639e89a52d464e3f48827882375750d26cdc496a9309b5.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/0eab6380018420da48639e89a52d464e3f48827882375750d26cdc496a9309b5.jpg)

![18e2597161227ccc0c60d2ff9321c46aef769de6189a2df691f75b6140071a89.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/18e2597161227ccc0c60d2ff9321c46aef769de6189a2df691f75b6140071a89.jpg)

![1b7779a74ebc4357c9d6efa1f46110732ec3c7b2a75ddd959aa759db0b1367ae.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/1b7779a74ebc4357c9d6efa1f46110732ec3c7b2a75ddd959aa759db0b1367ae.jpg)

![3d0de529568f78cf9fda6b94abe829a971a35a7d2ceaab12b60791969d8925f9.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/3d0de529568f78cf9fda6b94abe829a971a35a7d2ceaab12b60791969d8925f9.jpg)

![3f9fbe826313a625ddfdb9d6f5f6605b5a93516a794acdca54ccdc8aed9f08d3.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/3f9fbe826313a625ddfdb9d6f5f6605b5a93516a794acdca54ccdc8aed9f08d3.jpg)

![529685756546b7732e161827f117315a7a381515fcaaab1bc84eec96c2461eb6.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/529685756546b7732e161827f117315a7a381515fcaaab1bc84eec96c2461eb6.jpg)

![698f08eb793ab2e600c4d1f8a6b9c4e580946d7e3260e3ef11ed5a9113c76e14.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/698f08eb793ab2e600c4d1f8a6b9c4e580946d7e3260e3ef11ed5a9113c76e14.jpg)

![72c3b371067aae7bfa50621eb697620e78af2093cf985d05936b3dd7c19c4af8.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/72c3b371067aae7bfa50621eb697620e78af2093cf985d05936b3dd7c19c4af8.jpg)

![81acf1153c8cf0341c05a1d5dbe9af8e46461f90b0c5590fa84370b0848b3ced.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/81acf1153c8cf0341c05a1d5dbe9af8e46461f90b0c5590fa84370b0848b3ced.jpg)

![b1c9bf6778f826b435379684cfaabbbacda6a2ecd757b9e7236aa4e9b1fc26d4.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/b1c9bf6778f826b435379684cfaabbbacda6a2ecd757b9e7236aa4e9b1fc26d4.jpg)

![baa12edd11796396a2a2ca92f897358dd797c8c9a85471d37a1497975e941b36.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/images/baa12edd11796396a2a2ca92f897358dd797c8c9a85471d37a1497975e941b36.jpg)

### Tables

![5efe7c8506195e0a77ba344d707635a2e9e53982333efed8ef4dbd3bf9c781c1.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/tables/5efe7c8506195e0a77ba344d707635a2e9e53982333efed8ef4dbd3bf9c781c1.jpg)

![a24fc3e477989b70c04e99fccea4ca630024c56316708a58fd454dff1a9887e3.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/tables/a24fc3e477989b70c04e99fccea4ca630024c56316708a58fd454dff1a9887e3.jpg)

![ab1d1a0d37a204123df0ecf8163dc10f9137579252eee962d78e4f933e0521d4.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/tables/ab1d1a0d37a204123df0ecf8163dc10f9137579252eee962d78e4f933e0521d4.jpg)

![c3dfed7e779c47c61965a8ab940a0662564a8640177f8c21ec93882b0fdbe9e7.jpg](../iclr_results/597_MVTokenFlow_ High-quality 4D Content Generation using Multiview Token Flow/tables/c3dfed7e779c47c61965a8ab940a0662564a8640177f8c21ec93882b0fdbe9e7.jpg)

## Let the Code LLM Edit Itself When You Edit the Code


### Images

![344ee96ad989463b6a1b61013e4b50481ce8f9bf9050bb1d45f706ba62a98644.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/images/344ee96ad989463b6a1b61013e4b50481ce8f9bf9050bb1d45f706ba62a98644.jpg)

![726343bf0a5c61fa51aae8b77cb904a97adad5f2a6cd1fa089105b7c4d7a4661.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/images/726343bf0a5c61fa51aae8b77cb904a97adad5f2a6cd1fa089105b7c4d7a4661.jpg)

![7a4e450b151632265f3be53ea846bcada17bff2e0b8b0e8f77d052156bce0cbf.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/images/7a4e450b151632265f3be53ea846bcada17bff2e0b8b0e8f77d052156bce0cbf.jpg)

![7fbb33a721736ce3bb84f1d9ecc4638d37e23d5b0034c1d46970a3c7b000ca81.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/images/7fbb33a721736ce3bb84f1d9ecc4638d37e23d5b0034c1d46970a3c7b000ca81.jpg)

### Tables

![24859c4bbce8702f1dc1380e185b40542ac25e94495fe2ebafa7a150d321aed4.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/24859c4bbce8702f1dc1380e185b40542ac25e94495fe2ebafa7a150d321aed4.jpg)

![2725b444e5d573ca52c813b85aefad2a597c65a9c47fbe1535c49ac3b0f1dd36.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/2725b444e5d573ca52c813b85aefad2a597c65a9c47fbe1535c49ac3b0f1dd36.jpg)

![4330afbf6f87c5df72a827d69e61a965d779ee13536f4050879a532929be00a3.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/4330afbf6f87c5df72a827d69e61a965d779ee13536f4050879a532929be00a3.jpg)

![455991bc177ca2b8c957dc241b172a9f609c2812972657cd37ae9a166ebff546.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/455991bc177ca2b8c957dc241b172a9f609c2812972657cd37ae9a166ebff546.jpg)

![5ee6f46560c94b2e6d8e87663012f134874d3d9d590a354154cef68bcac8f3fb.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/5ee6f46560c94b2e6d8e87663012f134874d3d9d590a354154cef68bcac8f3fb.jpg)

![728aca67eec5eafff15c96b4203238a4e9352a8ee361ece98036a23f3d4ec3f9.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/728aca67eec5eafff15c96b4203238a4e9352a8ee361ece98036a23f3d4ec3f9.jpg)

![90683526e6c60046db9577c670da7a8def1a3a82faa1b419617cfde696e9538f.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/90683526e6c60046db9577c670da7a8def1a3a82faa1b419617cfde696e9538f.jpg)

![9cb17353ae8c78605fec2ca0d3c2c00a04556412be029b9f8ee81066fdfeabdd.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/9cb17353ae8c78605fec2ca0d3c2c00a04556412be029b9f8ee81066fdfeabdd.jpg)

![d05e382444e0fd3a46cde3901e8cc8392d4453a762be89647c79cc2edc86129b.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/d05e382444e0fd3a46cde3901e8cc8392d4453a762be89647c79cc2edc86129b.jpg)

![dc5328d2c69afc719f70716c3ac759a8ff3946cc6b2f7798594b10b2883640d7.jpg](../iclr_results/598_Let the Code LLM Edit Itself When You Edit the Code/tables/dc5328d2c69afc719f70716c3ac759a8ff3946cc6b2f7798594b10b2883640d7.jpg)

## Fewer May Be Better: Enhancing Offline Reinforcement Learning with Reduced Dataset


### Images

![15bb8426f9e95195fbd79c6bcbd8feab2d79c43b0499d58665702f3eabf2d54f.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/15bb8426f9e95195fbd79c6bcbd8feab2d79c43b0499d58665702f3eabf2d54f.jpg)

![2aa22f4d75cca9eb241758ad4ab76c9f386504ea366205b02ff23aeb915e1864.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/2aa22f4d75cca9eb241758ad4ab76c9f386504ea366205b02ff23aeb915e1864.jpg)

![3880c346b9cc265e81dd0672b6402dce2c44bbf382f948953f86ef209971888b.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/3880c346b9cc265e81dd0672b6402dce2c44bbf382f948953f86ef209971888b.jpg)

![45f9b589c2c507348e72e85186f69c33a760513462c085574fe8010504f526f6.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/45f9b589c2c507348e72e85186f69c33a760513462c085574fe8010504f526f6.jpg)

![7fab3ebbaf77ba5bf3d55cdbf213ffb408882d808c2c14c687101875178fb7e1.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/7fab3ebbaf77ba5bf3d55cdbf213ffb408882d808c2c14c687101875178fb7e1.jpg)

![8fbea53c8a679cc6acdc5eae0dff94092cee1d86e82ccc9328e401c8dcf5c4be.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/8fbea53c8a679cc6acdc5eae0dff94092cee1d86e82ccc9328e401c8dcf5c4be.jpg)

![a605a32212f95b0a46d8005bd79fa605c5446e82283ca40e46bd627b2d2243b9.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/a605a32212f95b0a46d8005bd79fa605c5446e82283ca40e46bd627b2d2243b9.jpg)

![b88f752ceb2ec8f881ae3580ba3c92f7a4e0fb6ae97f7b9b3a5c3c8f2da46900.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/b88f752ceb2ec8f881ae3580ba3c92f7a4e0fb6ae97f7b9b3a5c3c8f2da46900.jpg)

![dc583fd6f042fc66883e8a8b7f18c1a820daa885270f78da0cdeeebccbac8f98.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/dc583fd6f042fc66883e8a8b7f18c1a820daa885270f78da0cdeeebccbac8f98.jpg)

![dfdffd824b93dda6bcf5e7b7eea098505a0beed23d7fced5496688e613814ec6.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/dfdffd824b93dda6bcf5e7b7eea098505a0beed23d7fced5496688e613814ec6.jpg)

![dfe9b73cd7a4ed61221cf06548a72a1459d09453fdd95cbb209793da980bda3e.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/dfe9b73cd7a4ed61221cf06548a72a1459d09453fdd95cbb209793da980bda3e.jpg)

![ef6a1e90f5cc880727bceb4581a0e5b409088b421fad3d0ed7a26ccc152a9207.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/images/ef6a1e90f5cc880727bceb4581a0e5b409088b421fad3d0ed7a26ccc152a9207.jpg)

### Tables

![4973d31c0c68b357ad048e0d7f330cc2b6ca8ac9f782c49fe29edf793e2768d5.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/tables/4973d31c0c68b357ad048e0d7f330cc2b6ca8ac9f782c49fe29edf793e2768d5.jpg)

![4a13520ff21135c870ba1878117cb66745c83d91e3e742945e19ffa37c6e2b53.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/tables/4a13520ff21135c870ba1878117cb66745c83d91e3e742945e19ffa37c6e2b53.jpg)

![59f14569da84a82be758e84a6ecd264ce6067b56c8dcbd4d30beb71d49aaee3b.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/tables/59f14569da84a82be758e84a6ecd264ce6067b56c8dcbd4d30beb71d49aaee3b.jpg)

![60edcb25ddfddfa6ee73955eeb1717358d0e774726e70a281754ce20fb8fa1e0.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/tables/60edcb25ddfddfa6ee73955eeb1717358d0e774726e70a281754ce20fb8fa1e0.jpg)

![7abd4601e1862788f41ea7350281de2d3fca23afce88d14ced71c314655e2af0.jpg](../iclr_results/599_Fewer May Be Better_ Enhancing Offline Reinforcement Learning with Reduced Dataset/tables/7abd4601e1862788f41ea7350281de2d3fca23afce88d14ced71c314655e2af0.jpg)

## Generalizing Reasoning Problems to Longer Lengths


### Images

![a9434ff5a5f7ed5e15559d29bd563e31485a2714545159739e3e26e0b2efa5ac.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/images/a9434ff5a5f7ed5e15559d29bd563e31485a2714545159739e3e26e0b2efa5ac.jpg)

![fc4e1878c455ec48b6588486c18d3376676f1d1739bcbef4d30cc2cfbcbf0993.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/images/fc4e1878c455ec48b6588486c18d3376676f1d1739bcbef4d30cc2cfbcbf0993.jpg)

### Tables

![0142796ff4582b662696bba2ef63332b914a14895169b264f218fe03f57dc874.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/0142796ff4582b662696bba2ef63332b914a14895169b264f218fe03f57dc874.jpg)

![0c6041197a924ce8745b774bce9ce88fc52de0bc57b788902526da2edb857283.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/0c6041197a924ce8745b774bce9ce88fc52de0bc57b788902526da2edb857283.jpg)

![5acd6c94f5ecffab946ecfb3474d8b47f333852b946e323a82f74b9b5d1e0442.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/5acd6c94f5ecffab946ecfb3474d8b47f333852b946e323a82f74b9b5d1e0442.jpg)

![b454e54a857f8811a7bcd65094ad953f94f580e77f506463a188cebb37e8604e.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/b454e54a857f8811a7bcd65094ad953f94f580e77f506463a188cebb37e8604e.jpg)

![c3066319ad0ad89e258888c313a4ac6a679095f20f59fe233a0066b779dc1bbf.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/c3066319ad0ad89e258888c313a4ac6a679095f20f59fe233a0066b779dc1bbf.jpg)

![cdd63000cd7c160a8719eb70f1c6596a3f5211d4ef55e56e8195a01f498202ca.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/cdd63000cd7c160a8719eb70f1c6596a3f5211d4ef55e56e8195a01f498202ca.jpg)

![de975d05ffb2a743c0a38c180e4ae294cb7669b90ebbf51d5bcd3e6537eeb6c7.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/de975d05ffb2a743c0a38c180e4ae294cb7669b90ebbf51d5bcd3e6537eeb6c7.jpg)

![e66f5d7d2932b07262899959bc39dd403df5c0d751bb01201db1106e47055ddc.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/e66f5d7d2932b07262899959bc39dd403df5c0d751bb01201db1106e47055ddc.jpg)

![e6ab45ea0a9a9f3ae837e6bca5e8d14a1ffe992485e54cad5b691d34c9cd1b5a.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/e6ab45ea0a9a9f3ae837e6bca5e8d14a1ffe992485e54cad5b691d34c9cd1b5a.jpg)

![f004679db829b5ff2adc1863b27b2dcfdaa87b98737e713b92e5f6de8255b38e.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/f004679db829b5ff2adc1863b27b2dcfdaa87b98737e713b92e5f6de8255b38e.jpg)

![fc4b228ffed97d47daa75eb78ebdf419c5e917e27b5d796b624bfa5d157156e3.jpg](../iclr_results/600_Generalizing Reasoning Problems to Longer Lengths/tables/fc4b228ffed97d47daa75eb78ebdf419c5e917e27b5d796b624bfa5d157156e3.jpg)

## Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems


### Images

![0b518e583628024795c88e7ae5b5dba4920b7d3ed6ea1d6f2946780eb5efbadf.jpg](../iclr_results/601_Physics of Language Models_ Part 2.2, How to Learn From Mistakes on Grade-School Math Problems/images/0b518e583628024795c88e7ae5b5dba4920b7d3ed6ea1d6f2946780eb5efbadf.jpg)

![10e3f679160fe75b45e8a9dc50205d6f087498ba423aa91cd32e5b3a7910e483.jpg](../iclr_results/601_Physics of Language Models_ Part 2.2, How to Learn From Mistakes on Grade-School Math Problems/images/10e3f679160fe75b45e8a9dc50205d6f087498ba423aa91cd32e5b3a7910e483.jpg)

![54ad59ac8747d4e7d85d277da4d5bf6c5657c0141024813acb5a08ad0e792724.jpg](../iclr_results/601_Physics of Language Models_ Part 2.2, How to Learn From Mistakes on Grade-School Math Problems/images/54ad59ac8747d4e7d85d277da4d5bf6c5657c0141024813acb5a08ad0e792724.jpg)

![82bdf2701548b2efea2ac55ea4a7ca24a5b1cf68ec7861c9f492025614e3f064.jpg](../iclr_results/601_Physics of Language Models_ Part 2.2, How to Learn From Mistakes on Grade-School Math Problems/images/82bdf2701548b2efea2ac55ea4a7ca24a5b1cf68ec7861c9f492025614e3f064.jpg)

![c87b845c15e3e1566d773dc820666d3912b837693ad6c379eb33d5cd2b223978.jpg](../iclr_results/601_Physics of Language Models_ Part 2.2, How to Learn From Mistakes on Grade-School Math Problems/images/c87b845c15e3e1566d773dc820666d3912b837693ad6c379eb33d5cd2b223978.jpg)

## Understanding the Stability-based Generalization of Personalized Federated Learning


### Images

![33148e5def67ee1a78f8c861acb629884a3d7f428e9ca30d8cdb3b3a30f03f27.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/33148e5def67ee1a78f8c861acb629884a3d7f428e9ca30d8cdb3b3a30f03f27.jpg)

![3c60ff6dd27e0e55f12a7f4601ace66f442e7836af472c051df88a6db6ddd949.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/3c60ff6dd27e0e55f12a7f4601ace66f442e7836af472c051df88a6db6ddd949.jpg)

![5c33d6d40690af432c11c1d7c6512c10a7e05ecf07c7317b1054972e5c9db362.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/5c33d6d40690af432c11c1d7c6512c10a7e05ecf07c7317b1054972e5c9db362.jpg)

![630d47b286999d087050489b6886a11d69c941817bdffd016b3b8a0d6979c217.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/630d47b286999d087050489b6886a11d69c941817bdffd016b3b8a0d6979c217.jpg)

![6849eef4d42fdd594ef7b76d5a861760651f5ffc8fbd571d77b70543fe72f16f.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/6849eef4d42fdd594ef7b76d5a861760651f5ffc8fbd571d77b70543fe72f16f.jpg)

![79f294b5095ad91193bdce1737f755f1169e5f68508ac24db5805470d27abf14.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/79f294b5095ad91193bdce1737f755f1169e5f68508ac24db5805470d27abf14.jpg)

![87d3607052d60f8b9d2ac346bc2a4fb30c3d31aab6741874e36dafd90d9eccc7.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/87d3607052d60f8b9d2ac346bc2a4fb30c3d31aab6741874e36dafd90d9eccc7.jpg)

![ca1e89fcdac276f0197d6315d4f412c8724714181286a59dc5f54e79a110721b.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/images/ca1e89fcdac276f0197d6315d4f412c8724714181286a59dc5f54e79a110721b.jpg)

### Tables

![43095246d43a0243198a8a995f38c874785854546e410334c8d3f12b2f416a7e.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/tables/43095246d43a0243198a8a995f38c874785854546e410334c8d3f12b2f416a7e.jpg)

![4ff311d7dcd5868d88493d560f6cbcdcaf34a256f538e65f05f5d437bc374d98.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/tables/4ff311d7dcd5868d88493d560f6cbcdcaf34a256f538e65f05f5d437bc374d98.jpg)

![5bfd3277988cdc9e27c2acc55a577082964fd103e2c71becdb37268c259edab0.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/tables/5bfd3277988cdc9e27c2acc55a577082964fd103e2c71becdb37268c259edab0.jpg)

![de4f23229ac5ad30cee92b12d2cfc6b5625879758175df632b1ca3e396816082.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/tables/de4f23229ac5ad30cee92b12d2cfc6b5625879758175df632b1ca3e396816082.jpg)

![df5fc5aebb1ec70cdaf4a6a23d6b68f80c19c64b503382c0086b2a68bdf9e670.jpg](../iclr_results/602_Understanding the Stability-based Generalization of Personalized Federated Learning/tables/df5fc5aebb1ec70cdaf4a6a23d6b68f80c19c64b503382c0086b2a68bdf9e670.jpg)

## IgGM: A Generative Model for Functional Antibody and Nanobody Design


### Images

![353c8c77cfa36934dcb4aa42ac335db7b13320779a09ea4a26bd6987b0a3151c.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/353c8c77cfa36934dcb4aa42ac335db7b13320779a09ea4a26bd6987b0a3151c.jpg)

![62b94a453925c5045679ccd5d991eceed86069f013cb06c6da7312348d974862.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/62b94a453925c5045679ccd5d991eceed86069f013cb06c6da7312348d974862.jpg)

![9a098d9162ce5f3a46735ef7b0dff597157f619014c92ceb23844fed9fb1fbfe.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/9a098d9162ce5f3a46735ef7b0dff597157f619014c92ceb23844fed9fb1fbfe.jpg)

![9a932be1491e7050a91ef88b96e66377a80e949d79ab025db49ab178d7a14e97.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/9a932be1491e7050a91ef88b96e66377a80e949d79ab025db49ab178d7a14e97.jpg)

![9b1ff09b236e83617f6ad81363c32be0e95c443808d95c0172b3c040a9634a86.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/9b1ff09b236e83617f6ad81363c32be0e95c443808d95c0172b3c040a9634a86.jpg)

![a1cac53e18a2a1937db40049fa56db0d6ad5d971b45b38ccb92a4ad9f513431e.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/a1cac53e18a2a1937db40049fa56db0d6ad5d971b45b38ccb92a4ad9f513431e.jpg)

![a87a4a31c0c538199de70ad2a73dd1adb6494c945aeb9aeabb127cea71fed302.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/a87a4a31c0c538199de70ad2a73dd1adb6494c945aeb9aeabb127cea71fed302.jpg)

![bbd60d079ba4e9aff5acad08307f65e5b7f2cf1d59d1b8ac968857a852272320.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/bbd60d079ba4e9aff5acad08307f65e5b7f2cf1d59d1b8ac968857a852272320.jpg)

![f80a4ba85fd718082743dc66cefb57515dea668a08ce266faf9074011613ea30.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/images/f80a4ba85fd718082743dc66cefb57515dea668a08ce266faf9074011613ea30.jpg)

### Tables

![1782c021370b72cf1282c7ee37512bc05e7cde10d6b5c253c7b41c365829c8a1.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/tables/1782c021370b72cf1282c7ee37512bc05e7cde10d6b5c253c7b41c365829c8a1.jpg)

![2a2be23f476c286a30ba5da48416499c5443eb791f7d47ca86bb3b7abbf3986c.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/tables/2a2be23f476c286a30ba5da48416499c5443eb791f7d47ca86bb3b7abbf3986c.jpg)

![93f6269277c21e105df4a58f330590ac4eabf6b35f49decbe05b0d1857bb66a3.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/tables/93f6269277c21e105df4a58f330590ac4eabf6b35f49decbe05b0d1857bb66a3.jpg)

![a9e5f9002edb50c82b138acf96050216148b6b98d6b247af5550a9f91e591353.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/tables/a9e5f9002edb50c82b138acf96050216148b6b98d6b247af5550a9f91e591353.jpg)

![c3536cb168d396d2fca2c108f7eeffe1b6b1e0f25a098e0eb3524a309019537d.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/tables/c3536cb168d396d2fca2c108f7eeffe1b6b1e0f25a098e0eb3524a309019537d.jpg)

![d384edb0f08713e2723d408ea9df0b1b9b563f6bce9e9daa16439ce80bd288c0.jpg](../iclr_results/603_IgGM_ A Generative Model for Functional Antibody and Nanobody Design/tables/d384edb0f08713e2723d408ea9df0b1b9b563f6bce9e9daa16439ce80bd288c0.jpg)

## MMTEB: Massive Multilingual Text Embedding Benchmark


### Images

![061d811f871f7d88f733d643b4bc8d18930d0b5d14bc5b08bce2dee98f5f0b58.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/061d811f871f7d88f733d643b4bc8d18930d0b5d14bc5b08bce2dee98f5f0b58.jpg)

![0d7ca2e8f4a3814dd0604c9dcd0867de57ef612b6cf546c5a8e5b2fe096f180a.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/0d7ca2e8f4a3814dd0604c9dcd0867de57ef612b6cf546c5a8e5b2fe096f180a.jpg)

![1e1f0df2f12bf1d92f716c1f968fdd9e4611ccf7fd95f557da8d83eb9ded2618.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/1e1f0df2f12bf1d92f716c1f968fdd9e4611ccf7fd95f557da8d83eb9ded2618.jpg)

![78cd8cd7cc4bfa47d4239bcd23bb0749748233c6d47da4727cf8648ad92f048e.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/78cd8cd7cc4bfa47d4239bcd23bb0749748233c6d47da4727cf8648ad92f048e.jpg)

![81df21673844a3625246656bbc288ad99e77d23440ca97b247782cbbd293f19a.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/81df21673844a3625246656bbc288ad99e77d23440ca97b247782cbbd293f19a.jpg)

![85926f0295cffe2625390d23e95cdd7b1515963a29b683c0b3ae2ea158ce35d4.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/85926f0295cffe2625390d23e95cdd7b1515963a29b683c0b3ae2ea158ce35d4.jpg)

![8b11940bbe86f8ff9867eaf6a1afbdfdf9ca1c6c48f0541058ce2863a95bee0c.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/8b11940bbe86f8ff9867eaf6a1afbdfdf9ca1c6c48f0541058ce2863a95bee0c.jpg)

![9ac07b4e88e475306e112c8f07a68da6e0ef700189a4217dc619a744dbd67d8c.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/9ac07b4e88e475306e112c8f07a68da6e0ef700189a4217dc619a744dbd67d8c.jpg)

![d378df2a21b6f0209cafacaa1617372eb136f2dbde64e0846934288e64df2b49.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/d378df2a21b6f0209cafacaa1617372eb136f2dbde64e0846934288e64df2b49.jpg)

![dea929656696feb5c711791f74ef3b7a368ecec65505ea776b9e67bd8685559a.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/dea929656696feb5c711791f74ef3b7a368ecec65505ea776b9e67bd8685559a.jpg)

![e90b6bfd3af412709f3a87ba884655825938ac103a1b073fabaddfd04a0f8d92.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/images/e90b6bfd3af412709f3a87ba884655825938ac103a1b073fabaddfd04a0f8d92.jpg)

### Tables

![117b9ec4866db9080f05fa3bafa1b8faefff4ece1e5da01f23d143e89e575a5f.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/117b9ec4866db9080f05fa3bafa1b8faefff4ece1e5da01f23d143e89e575a5f.jpg)

![1a7ee2602f880296d5be19e48bf71db408cd21ddae5da15b562fb61d1ee9948f.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/1a7ee2602f880296d5be19e48bf71db408cd21ddae5da15b562fb61d1ee9948f.jpg)

![29420c87fe0f9fc0b16884040eae87b2e10a65c3021aa93618db4e4574608cc0.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/29420c87fe0f9fc0b16884040eae87b2e10a65c3021aa93618db4e4574608cc0.jpg)

![31e2c51103859b5e293b1dc1b2f853fc5916794286983bc3f91d877fdcaa502b.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/31e2c51103859b5e293b1dc1b2f853fc5916794286983bc3f91d877fdcaa502b.jpg)

![42230eceab7516604b12d56a0e9c67f58b556d6eb3caa6049f992c990f9b0438.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/42230eceab7516604b12d56a0e9c67f58b556d6eb3caa6049f992c990f9b0438.jpg)

![67d1bd5ff0cb4bac6fc3ae26c9586ab6ef0fbcbdbacd2a90915c154eea98a627.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/67d1bd5ff0cb4bac6fc3ae26c9586ab6ef0fbcbdbacd2a90915c154eea98a627.jpg)

![747a4504d33a398c56d7b2e8fee2aacbd180af9ae9c3392fef57c670c8cabc0b.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/747a4504d33a398c56d7b2e8fee2aacbd180af9ae9c3392fef57c670c8cabc0b.jpg)

![88285c7a6f93271e4122ee9a1b23338d1eaa694f56e66b4a4c24b23341ba0451.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/88285c7a6f93271e4122ee9a1b23338d1eaa694f56e66b4a4c24b23341ba0451.jpg)

![8d5352d9c447b9a6c9d2d31a751d53f7420bc098df09710564e1934698a2dbb7.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/8d5352d9c447b9a6c9d2d31a751d53f7420bc098df09710564e1934698a2dbb7.jpg)

![961fe4ee5faa7f02ad09bea9231848fca2746109465d634b79ac4b1601ac7a93.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/961fe4ee5faa7f02ad09bea9231848fca2746109465d634b79ac4b1601ac7a93.jpg)

![adc17b37d937693e8fbd2a23225bd581928f7804204dbc7611e393a0efd19c53.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/adc17b37d937693e8fbd2a23225bd581928f7804204dbc7611e393a0efd19c53.jpg)

![bf1e239fd0ebe371d4bd88a6ec0db3a88d47dd89faac7bb9387a652adcd417b4.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/bf1e239fd0ebe371d4bd88a6ec0db3a88d47dd89faac7bb9387a652adcd417b4.jpg)

![cb39cdc43a7b026c0c55aca166819486ee76fcf0b24f3890f9dc2ef61f5ca352.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/cb39cdc43a7b026c0c55aca166819486ee76fcf0b24f3890f9dc2ef61f5ca352.jpg)

![cbe52dc9a178f3327b44a7fa59a4bf21fa8a2dfed0caf03b76ca5216891b7acb.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/cbe52dc9a178f3327b44a7fa59a4bf21fa8a2dfed0caf03b76ca5216891b7acb.jpg)

![cdf07766ea751d079f8fc0053d16a1f4f02a40d0a8220da5a373a97a59fce784.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/cdf07766ea751d079f8fc0053d16a1f4f02a40d0a8220da5a373a97a59fce784.jpg)

![d3fb6510f4fc7277c49cafed312c285ab09eba27769266c1902db16d41345699.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/d3fb6510f4fc7277c49cafed312c285ab09eba27769266c1902db16d41345699.jpg)

![e1b1c6c322f2343d9f6319e6e00ef598c87bcaaac79d4a834ca51a009874ad2d.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/e1b1c6c322f2343d9f6319e6e00ef598c87bcaaac79d4a834ca51a009874ad2d.jpg)

![e662f1d60bc5f492ab6e0426d4279ec1ebe673910a352a1ab53e73ef255fb620.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/e662f1d60bc5f492ab6e0426d4279ec1ebe673910a352a1ab53e73ef255fb620.jpg)

![e7e771f097875d5acebb3e02aeafa3370127fbbd8dd28d0ffb04acfde24ab08b.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/e7e771f097875d5acebb3e02aeafa3370127fbbd8dd28d0ffb04acfde24ab08b.jpg)

![e9e2dae08f5bec9203e206d937d8175fd88f2a00580a57b09103de47039b7ac8.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/e9e2dae08f5bec9203e206d937d8175fd88f2a00580a57b09103de47039b7ac8.jpg)

![f7e92d8ecf123d12e751588ae5edb44d36fee83fabaae33a2f7b856e23a773b1.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/f7e92d8ecf123d12e751588ae5edb44d36fee83fabaae33a2f7b856e23a773b1.jpg)

![fcff33b558a07337e38ab929af7175a611c2150c1f02511215c7f904774cb11d.jpg](../iclr_results/604_MMTEB_ Massive Multilingual Text Embedding Benchmark/tables/fcff33b558a07337e38ab929af7175a611c2150c1f02511215c7f904774cb11d.jpg)

## Ultra-Sparse Memory Network


### Images

![0c48b0da614d9ff08bc8cf52a5dca81a999cfdf3025d102ef769f0d2da4c2e8f.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/0c48b0da614d9ff08bc8cf52a5dca81a999cfdf3025d102ef769f0d2da4c2e8f.jpg)

![1537411308f45cce9c57f5ee3a268a4eb6f450539a41be85008cb51ed04d9b17.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/1537411308f45cce9c57f5ee3a268a4eb6f450539a41be85008cb51ed04d9b17.jpg)

![28dedfa41aa6ae49bea10a8f15a85392ac88f8c8db89da7d75b7d307e63cb618.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/28dedfa41aa6ae49bea10a8f15a85392ac88f8c8db89da7d75b7d307e63cb618.jpg)

![7c996b077068f7ff841a3cc66819c9db7858845142fd6b26df2258eabbcf2d55.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/7c996b077068f7ff841a3cc66819c9db7858845142fd6b26df2258eabbcf2d55.jpg)

![a81ab62291496aa9b0993c9d0be18422e4f633656e9c2d5db3aeb32cbcf89ed8.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/a81ab62291496aa9b0993c9d0be18422e4f633656e9c2d5db3aeb32cbcf89ed8.jpg)

![b0d73297c356d0793522fc3d2b368e8751a17f97e6471e62bd812214c56f9917.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/b0d73297c356d0793522fc3d2b368e8751a17f97e6471e62bd812214c56f9917.jpg)

![da4268046d36d917ca9a7e20b432104987caf9035c9d0fe3f5ca475c11756def.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/da4268046d36d917ca9a7e20b432104987caf9035c9d0fe3f5ca475c11756def.jpg)

![dc1009512a082e0f56658784d38da6b9f302a1729a76cd284c79f0ab5ab8c973.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/dc1009512a082e0f56658784d38da6b9f302a1729a76cd284c79f0ab5ab8c973.jpg)

![dce2fde938b7a1e3b0a09e08b77d7cfcf7b635e539ead106df91d7ba5ae553aa.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/dce2fde938b7a1e3b0a09e08b77d7cfcf7b635e539ead106df91d7ba5ae553aa.jpg)

![dec5589c71608765e01ea70392f831c359887df87c5339d57e31f594b8339a25.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/dec5589c71608765e01ea70392f831c359887df87c5339d57e31f594b8339a25.jpg)

![fcec52640055214c3f11ec632ff01290c66751141c049b72848dc5889122fb6c.jpg](../iclr_results/605_Ultra-Sparse Memory Network/images/fcec52640055214c3f11ec632ff01290c66751141c049b72848dc5889122fb6c.jpg)

### Tables

![079d54f824b27944f6937fb4bc6e1ee7b1f38e3381d935f1f5bcf418db6afe1e.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/079d54f824b27944f6937fb4bc6e1ee7b1f38e3381d935f1f5bcf418db6afe1e.jpg)

![0b434cde0aa4bad7d6d061a9f293e209643ff1d8b087bdc69c22d94f5e4b723b.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/0b434cde0aa4bad7d6d061a9f293e209643ff1d8b087bdc69c22d94f5e4b723b.jpg)

![4e6b741042177076c60c857347d9a287fbea6db535899f04e5009556d8f1f0aa.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/4e6b741042177076c60c857347d9a287fbea6db535899f04e5009556d8f1f0aa.jpg)

![686120ed3ef96af1794c739f56ca70f3a19ed1ad5c3036d2d8fbce21681b5622.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/686120ed3ef96af1794c739f56ca70f3a19ed1ad5c3036d2d8fbce21681b5622.jpg)

![750b43b62c942c25de55a28b7bf9dac40833d7fef967f3e1cf3ffe7024da3002.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/750b43b62c942c25de55a28b7bf9dac40833d7fef967f3e1cf3ffe7024da3002.jpg)

![7eeb805bb4fe746996d699921fe8e098d5077260dde03f9909f386aa1eb4be08.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/7eeb805bb4fe746996d699921fe8e098d5077260dde03f9909f386aa1eb4be08.jpg)

![a5b0167be67c32e461995f43096b192967f16aa66c0e9f1cc7794daa8f24838e.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/a5b0167be67c32e461995f43096b192967f16aa66c0e9f1cc7794daa8f24838e.jpg)

![cbb763b6087fe84c000bbf21ebc9218ba6b3928fc757c092e5efa96594a84304.jpg](../iclr_results/605_Ultra-Sparse Memory Network/tables/cbb763b6087fe84c000bbf21ebc9218ba6b3928fc757c092e5efa96594a84304.jpg)

## Lines of Thought in Large Language Models


### Images

![0353013af41ae3fee7abadb8d3bf3dabc12c0dfb0b479304bc75973c43cee8ef.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/0353013af41ae3fee7abadb8d3bf3dabc12c0dfb0b479304bc75973c43cee8ef.jpg)

![09b3c32b29079438509d5962b4b64c0edf29edea984b1f05aaf148d355a75a72.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/09b3c32b29079438509d5962b4b64c0edf29edea984b1f05aaf148d355a75a72.jpg)

![0acfc5c7dfa5de93e901ceacc88ae071cd6f17e012215e8fe4959f5367871d38.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/0acfc5c7dfa5de93e901ceacc88ae071cd6f17e012215e8fe4959f5367871d38.jpg)

![0caa253eafcc7cbc4a302f853fd15a25b2c9776438cd1a32844f3bb04b053294.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/0caa253eafcc7cbc4a302f853fd15a25b2c9776438cd1a32844f3bb04b053294.jpg)

![186c94a340f0e48d5e40f92c2691e35a9addfb173f6f91453470dde01db89fac.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/186c94a340f0e48d5e40f92c2691e35a9addfb173f6f91453470dde01db89fac.jpg)

![56bacec9b04dfad5a7e76e2bbd6e04b4f4161592701ca0ca9c305e18074bb87a.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/56bacec9b04dfad5a7e76e2bbd6e04b4f4161592701ca0ca9c305e18074bb87a.jpg)

![6742a7ed0cc690c4811159f0ae11e6908e4db0defc165e34d1ac7f3620aa0cde.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/6742a7ed0cc690c4811159f0ae11e6908e4db0defc165e34d1ac7f3620aa0cde.jpg)

![768b6bb13f458e3bd62be0aabdf4298380d58260a70c2cd2225d813ed7be65e3.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/768b6bb13f458e3bd62be0aabdf4298380d58260a70c2cd2225d813ed7be65e3.jpg)

![7f08fde601141db541548ad8beb33952bbf1a9077fd4ab20a8c0301564c86356.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/7f08fde601141db541548ad8beb33952bbf1a9077fd4ab20a8c0301564c86356.jpg)

![804debe13cf765761959b5efd918476292ed378147fd9c73997f62b16f3fdf67.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/804debe13cf765761959b5efd918476292ed378147fd9c73997f62b16f3fdf67.jpg)

![81e99c513fa9f542abbd4a2e3a4d397ec1c63b8d54e9a46eabd7800ee901c314.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/81e99c513fa9f542abbd4a2e3a4d397ec1c63b8d54e9a46eabd7800ee901c314.jpg)

![9e9597f3cfe9dcef1326113893ff57c061482fee6ee1ff045fcb6235c3865ca8.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/9e9597f3cfe9dcef1326113893ff57c061482fee6ee1ff045fcb6235c3865ca8.jpg)

![b37dbffd7dd949825d0102661080b7cd7966090d938f35ccbd01d21c9231f5f3.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/b37dbffd7dd949825d0102661080b7cd7966090d938f35ccbd01d21c9231f5f3.jpg)

![c26c6545f5c7854445977ce8d5e901f14d121e1acff305c578adb8032c71ecd0.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/c26c6545f5c7854445977ce8d5e901f14d121e1acff305c578adb8032c71ecd0.jpg)

![e8cd1e6e711a05259cc7e8c7e42c1efbd91526528616c172a837757fb171a7a0.jpg](../iclr_results/606_Lines of Thought in Large Language Models/images/e8cd1e6e711a05259cc7e8c7e42c1efbd91526528616c172a837757fb171a7a0.jpg)

### Tables

![64d0f87f9a646defdb822227a885fedcc26ecd5394cbc2925f60457522a3e2ee.jpg](../iclr_results/606_Lines of Thought in Large Language Models/tables/64d0f87f9a646defdb822227a885fedcc26ecd5394cbc2925f60457522a3e2ee.jpg)

## Do You Keep an Eye on What I Ask? Mitigating Multimodal Hallucination via Attention-Guided Ensemble Decoding


### Images

![42e580605bd304f3104f70eeb443ccbf00ae34b5c9b79845bbc133765b6c569a.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/42e580605bd304f3104f70eeb443ccbf00ae34b5c9b79845bbc133765b6c569a.jpg)

![562f7da6b8e7efe034e585aee4fa0c9176228df3a1d6e16af4a791286d42fe8a.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/562f7da6b8e7efe034e585aee4fa0c9176228df3a1d6e16af4a791286d42fe8a.jpg)

![572fcaa67f5fcffeebc2c27fc90ddb71bff6dd99bd4e32ec55cc8e1f8d9a5a76.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/572fcaa67f5fcffeebc2c27fc90ddb71bff6dd99bd4e32ec55cc8e1f8d9a5a76.jpg)

![58ee5bb020e52e86d4b7ab060630c6aab928952d540d158deefa500fa3da7128.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/58ee5bb020e52e86d4b7ab060630c6aab928952d540d158deefa500fa3da7128.jpg)

![59264fb47431ea40351a0d8204b888138552600d3e2e9177ca91b9f5db3ad133.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/59264fb47431ea40351a0d8204b888138552600d3e2e9177ca91b9f5db3ad133.jpg)

![676ba74369e8463914139432e5aa421d34e262f093301d8404fcb6e4eb2afa27.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/676ba74369e8463914139432e5aa421d34e262f093301d8404fcb6e4eb2afa27.jpg)

![7db0022405a8ebaec01b874f277beac6455d8be54a141b77fc93537eed35f503.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/7db0022405a8ebaec01b874f277beac6455d8be54a141b77fc93537eed35f503.jpg)

![84f08aea0dcd70d2c303df2c53c9139a04d519fb14c68d7a8af2faaf34e6fa12.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/84f08aea0dcd70d2c303df2c53c9139a04d519fb14c68d7a8af2faaf34e6fa12.jpg)

![bf3fbeff07a6b45c25b1c57b0b382eafb3ea3199d6ed690a604faa1cd0ee74a9.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/bf3fbeff07a6b45c25b1c57b0b382eafb3ea3199d6ed690a604faa1cd0ee74a9.jpg)

![ce19472bc61a480bdcb597ba315c68185f6a550d403133dff9c748d12fc332a7.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/ce19472bc61a480bdcb597ba315c68185f6a550d403133dff9c748d12fc332a7.jpg)

![ef275610b300e072dfc3ff76fde04f4904c4252296ffbea6ca361115f3a1c2b6.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/ef275610b300e072dfc3ff76fde04f4904c4252296ffbea6ca361115f3a1c2b6.jpg)

![f28c84f349cb6f9ef674ee26966825180832b71c50f55fa82cd0a240ccb546e6.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /images/f28c84f349cb6f9ef674ee26966825180832b71c50f55fa82cd0a240ccb546e6.jpg)

### Tables

![0c0d97f49c8814fd7d2ef2da621435066edf9a4ce33ac000c5efd1cba3883fd4.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/0c0d97f49c8814fd7d2ef2da621435066edf9a4ce33ac000c5efd1cba3883fd4.jpg)

![421bfc237b2be4ec8b4d96f0d59eb3b2fda92ce44921d37f96bcd04fa15c5ccd.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/421bfc237b2be4ec8b4d96f0d59eb3b2fda92ce44921d37f96bcd04fa15c5ccd.jpg)

![5559b669caa81202cc00c284332c58e2bef0cb274d32a96ddd44cea3af2d9b8a.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/5559b669caa81202cc00c284332c58e2bef0cb274d32a96ddd44cea3af2d9b8a.jpg)

![591382f7868baf9f57007897f68df2a85e2c26c3892dd9d37ef27aea787d05b3.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/591382f7868baf9f57007897f68df2a85e2c26c3892dd9d37ef27aea787d05b3.jpg)

![70c66f1e03c84364906353f374cf1ba326075910b4ed04d7673c22de7d3c260e.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/70c66f1e03c84364906353f374cf1ba326075910b4ed04d7673c22de7d3c260e.jpg)

![7bf0760b2f8bf432b7fbbba772e6a885e73b1b859814fed43f749731fc78c783.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/7bf0760b2f8bf432b7fbbba772e6a885e73b1b859814fed43f749731fc78c783.jpg)

![88dc9b883cce58db8cc58ba5a5c690555e4c11320fc12b7dd2f4a8e640c42050.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/88dc9b883cce58db8cc58ba5a5c690555e4c11320fc12b7dd2f4a8e640c42050.jpg)

![d1023cac04155eb3cce0ca6e5940d91de34b2e5b6c2a9a5ad3b02b16594350c9.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/d1023cac04155eb3cce0ca6e5940d91de34b2e5b6c2a9a5ad3b02b16594350c9.jpg)

![f404f56e28ca5b12a3d986ef23500a0a06f8e3c87e49647b693e590b6e475121.jpg](../iclr_results/607_Do You Keep an Eye on What I Ask_ Mitigating Multimodal Hallucination via Attention-Guided Ensemble /tables/f404f56e28ca5b12a3d986ef23500a0a06f8e3c87e49647b693e590b6e475121.jpg)

## Exact Community Recovery under Side Information: Optimality of Spectral Algorithms


### Images

![73510be4e1b280d4c32926800328a3f706e72f9070ecd28fb592de4f5959bdc1.jpg](../iclr_results/608_Exact Community Recovery under Side Information_ Optimality of Spectral Algorithms/images/73510be4e1b280d4c32926800328a3f706e72f9070ecd28fb592de4f5959bdc1.jpg)

![7a107a5d120ade41a89b5359d3d90498a8ea7e4e6256d4c2bbd3808878f8cdf7.jpg](../iclr_results/608_Exact Community Recovery under Side Information_ Optimality of Spectral Algorithms/images/7a107a5d120ade41a89b5359d3d90498a8ea7e4e6256d4c2bbd3808878f8cdf7.jpg)

![8274961b67ea6dc477e92a6b6b1d2b300b32a08211bf06729514015151e0e306.jpg](../iclr_results/608_Exact Community Recovery under Side Information_ Optimality of Spectral Algorithms/images/8274961b67ea6dc477e92a6b6b1d2b300b32a08211bf06729514015151e0e306.jpg)

![8ba88ab86b30b252965208de33be7956864c9e64b95ee320eb068e55786f68b9.jpg](../iclr_results/608_Exact Community Recovery under Side Information_ Optimality of Spectral Algorithms/images/8ba88ab86b30b252965208de33be7956864c9e64b95ee320eb068e55786f68b9.jpg)

![a9d5d51e09033ffec7649deb718a02dde5793c19f0445845bc3b5d15bf8e0e27.jpg](../iclr_results/608_Exact Community Recovery under Side Information_ Optimality of Spectral Algorithms/images/a9d5d51e09033ffec7649deb718a02dde5793c19f0445845bc3b5d15bf8e0e27.jpg)

![e33b8f8f870d9b7005f69023a22bfeae23f7e774e34c6ff0c78d2e6104f4c634.jpg](../iclr_results/608_Exact Community Recovery under Side Information_ Optimality of Spectral Algorithms/images/e33b8f8f870d9b7005f69023a22bfeae23f7e774e34c6ff0c78d2e6104f4c634.jpg)

![e39fe6a50a881d3244fcd99cdc7c66ed0cc5e095cc59793ad752d5ac7cae1b1b.jpg](../iclr_results/608_Exact Community Recovery under Side Information_ Optimality of Spectral Algorithms/images/e39fe6a50a881d3244fcd99cdc7c66ed0cc5e095cc59793ad752d5ac7cae1b1b.jpg)

## Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data


### Images

![03d04c55f3323585d79134808dce1dda78dca1f65a251ef61a1065c4fe1bd3b8.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/03d04c55f3323585d79134808dce1dda78dca1f65a251ef61a1065c4fe1bd3b8.jpg)

![136bef0ae0d3f6a6a72b8fca2be73c0eb3d4d833710c34cc7e70d34022b1f789.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/136bef0ae0d3f6a6a72b8fca2be73c0eb3d4d833710c34cc7e70d34022b1f789.jpg)

![1c9fbde8e863f8da62da718817ddcd32519353b0a0e9d2596f0294efd82b2eab.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/1c9fbde8e863f8da62da718817ddcd32519353b0a0e9d2596f0294efd82b2eab.jpg)

![241541d7d6cab957e07571eccc12063bef519a76375b772c7d680f22ba2a757b.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/241541d7d6cab957e07571eccc12063bef519a76375b772c7d680f22ba2a757b.jpg)

![294dc6f112cfd585b14ebfa476867c7dc335759dcae6c6fe506d5d9d764fe177.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/294dc6f112cfd585b14ebfa476867c7dc335759dcae6c6fe506d5d9d764fe177.jpg)

![6640ab1bbc1ca942eef90c1a418b42f2212f35b46abbfb470b5d1b688b5fd1f7.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/6640ab1bbc1ca942eef90c1a418b42f2212f35b46abbfb470b5d1b688b5fd1f7.jpg)

![8f66345e37c290eeed4ec7ede0765034922f28d4d2d87b7dbf990db3043be141.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/8f66345e37c290eeed4ec7ede0765034922f28d4d2d87b7dbf990db3043be141.jpg)

![b214ae95d5d214e3ef30a7e5b78b81f48fc016291b80908fed2e4acd96348803.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/b214ae95d5d214e3ef30a7e5b78b81f48fc016291b80908fed2e4acd96348803.jpg)

![bcbe269e95332e5a7d0912576bfe2638debf487c490441a0ec88a01c4c70e851.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/bcbe269e95332e5a7d0912576bfe2638debf487c490441a0ec88a01c4c70e851.jpg)

![cd478ce99a750ca4fdb76cb7f4dbc2ba6b9b796bd696b8aa60776ac4496c8fba.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/cd478ce99a750ca4fdb76cb7f4dbc2ba6b9b796bd696b8aa60776ac4496c8fba.jpg)

![e9255f31a78ab06782645ff7f1930434b616442005f1e2bf326806fb9c620a4e.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/e9255f31a78ab06782645ff7f1930434b616442005f1e2bf326806fb9c620a4e.jpg)

![f246f4736cb07ae713fcd0632eda88216210f3a278b7e84f4fac4ab743bdcf18.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/images/f246f4736cb07ae713fcd0632eda88216210f3a278b7e84f4fac4ab743bdcf18.jpg)

### Tables

![08306b3d4e7ab31791b3ba70de8e460eaf97c81c9aa9d299e7277e822340063d.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/08306b3d4e7ab31791b3ba70de8e460eaf97c81c9aa9d299e7277e822340063d.jpg)

![12e6e379c108bb87fa979f7817bbf1f2b9a593155c2ef3f1b400415ae300fd44.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/12e6e379c108bb87fa979f7817bbf1f2b9a593155c2ef3f1b400415ae300fd44.jpg)

![194145d72813e430002f5bce11f8190eaac8a47f22a34db6e0cfdb3ccb4f469f.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/194145d72813e430002f5bce11f8190eaac8a47f22a34db6e0cfdb3ccb4f469f.jpg)

![2c4b1e542aae6463dda9074bf7d3771526ff2727c06686326f9ec88453afa827.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/2c4b1e542aae6463dda9074bf7d3771526ff2727c06686326f9ec88453afa827.jpg)

![3cbeb162b43ca1500bf83bfd1047ed2690dfc918a32f74b70b7962edfc0c0743.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/3cbeb162b43ca1500bf83bfd1047ed2690dfc918a32f74b70b7962edfc0c0743.jpg)

![441b07e1ad20850e4f8b4bef22ecbc40d9c26e99335b0a8e8f004ae392d6ef46.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/441b07e1ad20850e4f8b4bef22ecbc40d9c26e99335b0a8e8f004ae392d6ef46.jpg)

![49b657c505b528f6d37b3c7106f87dbe2e0ad2a0ebec9230c10162058f47948b.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/49b657c505b528f6d37b3c7106f87dbe2e0ad2a0ebec9230c10162058f47948b.jpg)

![501c51abbac9800cc26d070294d3d0df10ea6746a19ca6c1775a50d9e5604d7f.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/501c51abbac9800cc26d070294d3d0df10ea6746a19ca6c1775a50d9e5604d7f.jpg)

![6c3b17229bb96bc6d1c0bce9fb044c5b16269210a9327d5f6345124398a1c5d1.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/6c3b17229bb96bc6d1c0bce9fb044c5b16269210a9327d5f6345124398a1c5d1.jpg)

![771a00cab638360e40da337512e5000a29f5ab1a6f0cda8490545fcddd49e5cb.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/771a00cab638360e40da337512e5000a29f5ab1a6f0cda8490545fcddd49e5cb.jpg)

![84645589b0bf6a094db94d3524f6e7ca7020b46e9b85c25afa381d2d1c136c51.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/84645589b0bf6a094db94d3524f6e7ca7020b46e9b85c25afa381d2d1c136c51.jpg)

![a0d99d0498311a8feba0c4a27f8ebdaa59f2e74219cddb59d78adb87f7aa4c94.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/a0d99d0498311a8feba0c4a27f8ebdaa59f2e74219cddb59d78adb87f7aa4c94.jpg)

![af51d715604c1587d9b3675f6ddb259b0a485efdcc07097a6fd5668b809d5b33.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/af51d715604c1587d9b3675f6ddb259b0a485efdcc07097a6fd5668b809d5b33.jpg)

![bd97c3b8df5241926b1f61ae4bf0c8eef96b3d8ada46af8e0f01abc07d8e43f8.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/bd97c3b8df5241926b1f61ae4bf0c8eef96b3d8ada46af8e0f01abc07d8e43f8.jpg)

![cc3cf772ed43a5c1fd1e8daf44c0c9782af9d0e985d036008be97a5d446da14a.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/cc3cf772ed43a5c1fd1e8daf44c0c9782af9d0e985d036008be97a5d446da14a.jpg)

![eb05d76a4641a191b29eea1beed036f987d269e901b6a1a1890fd182c4e650e2.jpg](../iclr_results/609_Context Clues_ Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data/tables/eb05d76a4641a191b29eea1beed036f987d269e901b6a1a1890fd182c4e650e2.jpg)

## Deconstructing What Makes a Good Optimizer for Autoregressive Language Models


### Images

![13b236c0641c5996218fc6d9380f7e54a78055e3cf4cd76cba338ee70f328cd8.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/13b236c0641c5996218fc6d9380f7e54a78055e3cf4cd76cba338ee70f328cd8.jpg)

![15eb2863687dac5d571f03abd6c9b2d8dc40c40e192fc10d7a148274484ca067.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/15eb2863687dac5d571f03abd6c9b2d8dc40c40e192fc10d7a148274484ca067.jpg)

![24dbb1fbbd9cad0d002a0f28b92a221120b709d4be80a84a6a1286fb7acfb9ea.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/24dbb1fbbd9cad0d002a0f28b92a221120b709d4be80a84a6a1286fb7acfb9ea.jpg)

![2831d01655ffe2df6f28075bb90e1c062aab9de2fbf1eebcbfff3d4ebcfd59ac.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/2831d01655ffe2df6f28075bb90e1c062aab9de2fbf1eebcbfff3d4ebcfd59ac.jpg)

![2855769081eb134217bfb63fc478e2a1f3c373c44996dce426877b65a39358a6.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/2855769081eb134217bfb63fc478e2a1f3c373c44996dce426877b65a39358a6.jpg)

![2c04223512102a6acc1aa066382740666689aefc9486061e238b2421dac53bb5.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/2c04223512102a6acc1aa066382740666689aefc9486061e238b2421dac53bb5.jpg)

![2e9c9286d2233635337de4cbb26e47979acc25af833ef06a879be0c9c75b4886.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/2e9c9286d2233635337de4cbb26e47979acc25af833ef06a879be0c9c75b4886.jpg)

![2fee847262631c43364758c30123bbf3ed6059502f0071a4b9a1619f40def553.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/2fee847262631c43364758c30123bbf3ed6059502f0071a4b9a1619f40def553.jpg)

![369b56fc9b43528bb5d96d3c7bda6e48e3abd98d1ac9cc7463f802e34324272a.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/369b56fc9b43528bb5d96d3c7bda6e48e3abd98d1ac9cc7463f802e34324272a.jpg)

![3b00e786e663417365b00349c060f5f0c85ec69983522b874a0e14be2607fbd6.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/3b00e786e663417365b00349c060f5f0c85ec69983522b874a0e14be2607fbd6.jpg)

![49a1d50572f2b32d3f006bbe670fdbe137cf7d9630549130700b788f4b5cc6ac.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/49a1d50572f2b32d3f006bbe670fdbe137cf7d9630549130700b788f4b5cc6ac.jpg)

![575f2826817bd7f655ee9f66805fe66b4a30faee7f9f03464c7b33e5099f0e6f.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/575f2826817bd7f655ee9f66805fe66b4a30faee7f9f03464c7b33e5099f0e6f.jpg)

![64d9b44d6085bc65925cd3cf24561b1dcb2dae9c9a30852ec56ddd879f7ea69d.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/64d9b44d6085bc65925cd3cf24561b1dcb2dae9c9a30852ec56ddd879f7ea69d.jpg)

![70a15ef952aa6735968252878ff551baa7e1ce09ed823c6501b9fcea739b3cc3.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/70a15ef952aa6735968252878ff551baa7e1ce09ed823c6501b9fcea739b3cc3.jpg)

![a4087351ef61836cf7de1a2dda913705a497f791cce0b14f953031d65d4bd604.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/a4087351ef61836cf7de1a2dda913705a497f791cce0b14f953031d65d4bd604.jpg)

![a57f9d0c21382f17ae2bb0fa70adc40bc9414e82c827915ef1d13318e0bba551.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/a57f9d0c21382f17ae2bb0fa70adc40bc9414e82c827915ef1d13318e0bba551.jpg)

![ac4cf06dc6e7b773a7ff5fa67456803ed02dea9d6cff093b792a12b1c7db30c7.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/ac4cf06dc6e7b773a7ff5fa67456803ed02dea9d6cff093b792a12b1c7db30c7.jpg)

![b8d3e06c387ad00eff237817c4068041364086a9bca5eac2e3247cab19b94e7b.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/b8d3e06c387ad00eff237817c4068041364086a9bca5eac2e3247cab19b94e7b.jpg)

![db38647f928fcacfa7f1b77741174c015a2499d39679d1d91320cebcfa8d73bf.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/db38647f928fcacfa7f1b77741174c015a2499d39679d1d91320cebcfa8d73bf.jpg)

![fac5ee39de2f1df8b43e803ce62e179acd25d16ac6f88dbf65872b01b9e28443.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/images/fac5ee39de2f1df8b43e803ce62e179acd25d16ac6f88dbf65872b01b9e28443.jpg)

### Tables

![618cd19281063d98e5ef9034f86606d49db22547ee8c1db965909528eb086dbb.jpg](../iclr_results/610_Deconstructing What Makes a Good Optimizer for Autoregressive Language Models/tables/618cd19281063d98e5ef9034f86606d49db22547ee8c1db965909528eb086dbb.jpg)

## Dataset Ownership Verification in Contrastive Pre-trained Models


### Images

![08240f14b9da55909b77917d816e0132d7d79bdcac488f085fe23d3291657c8d.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/08240f14b9da55909b77917d816e0132d7d79bdcac488f085fe23d3291657c8d.jpg)

![3282e9d0d27c24b1421728a78777389f095da81122a86f74492b73bf063d8ae1.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/3282e9d0d27c24b1421728a78777389f095da81122a86f74492b73bf063d8ae1.jpg)

![3cdb6b8fefb3bf31a34939916587f503634ea1f77b10380ec6ab9cd7bd99ef99.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/3cdb6b8fefb3bf31a34939916587f503634ea1f77b10380ec6ab9cd7bd99ef99.jpg)

![6024bde87f4664c775c3fecc446bd37fa595a0dd11915c1705ef94a85022f62a.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/6024bde87f4664c775c3fecc446bd37fa595a0dd11915c1705ef94a85022f62a.jpg)

![b03dc1e51687eb6583683daea811863de851f5e1fea797b1e72f93756945cf9e.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/b03dc1e51687eb6583683daea811863de851f5e1fea797b1e72f93756945cf9e.jpg)

![b9fca240e6d47a1aba16b587c3c063e3fc8adb38da054058edf2aacb7afcb051.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/b9fca240e6d47a1aba16b587c3c063e3fc8adb38da054058edf2aacb7afcb051.jpg)

![ea3cf05daeb309deb4e9fed700ef47edad413e792c0c9b71f1537758ecca58f4.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/ea3cf05daeb309deb4e9fed700ef47edad413e792c0c9b71f1537758ecca58f4.jpg)

![f240f0d6a7e576c0641b5ddbd14f6646fb84a8cbc9039ac2f7b3a2813cdc6c84.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/images/f240f0d6a7e576c0641b5ddbd14f6646fb84a8cbc9039ac2f7b3a2813cdc6c84.jpg)

### Tables

![01eb5c95525df49e02541175c0d63b5c1ace7d81de84b0a79e30a0424523e2bd.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/01eb5c95525df49e02541175c0d63b5c1ace7d81de84b0a79e30a0424523e2bd.jpg)

![03559b804d58883072f423dc9c07474822e605fa36ea412d85bd02dbd33db48f.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/03559b804d58883072f423dc9c07474822e605fa36ea412d85bd02dbd33db48f.jpg)

![13ede16430322fc3c2567154be5ba459e32fce0041c6b49ed74935e660bd171f.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/13ede16430322fc3c2567154be5ba459e32fce0041c6b49ed74935e660bd171f.jpg)

![14911003e1313b37665a1f461dc2cdee6cd5f4abcbd6233cd17f186427ad1c16.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/14911003e1313b37665a1f461dc2cdee6cd5f4abcbd6233cd17f186427ad1c16.jpg)

![2e8f6dfd77a4e3d4effb2459f8aafe1f5881368690f453e286de9837e42ba4e6.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/2e8f6dfd77a4e3d4effb2459f8aafe1f5881368690f453e286de9837e42ba4e6.jpg)

![4bd7231bca9f5b501ad451bd33f82f006a11941dddd0034cea8d837e1b777982.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/4bd7231bca9f5b501ad451bd33f82f006a11941dddd0034cea8d837e1b777982.jpg)

![68409eb5fb58d24af5a9ca12f125ad9c46fbaada98187b714a89ebe504981a1e.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/68409eb5fb58d24af5a9ca12f125ad9c46fbaada98187b714a89ebe504981a1e.jpg)

![6c1dbdb1d7ee0d2a362486652a931963f8351d36cf894540fa546bc67389c6d8.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/6c1dbdb1d7ee0d2a362486652a931963f8351d36cf894540fa546bc67389c6d8.jpg)

![8b3dcfb1a22143f6ac3a7ba0d06ef04e93f25714291068aa3b3686b909802ca6.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/8b3dcfb1a22143f6ac3a7ba0d06ef04e93f25714291068aa3b3686b909802ca6.jpg)

![9e9edb33703fc599ada8e139f98695bdc6edaa4b42e205f1f02b2ec2e27cd184.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/9e9edb33703fc599ada8e139f98695bdc6edaa4b42e205f1f02b2ec2e27cd184.jpg)

![a08e782863090e870b19bda41e1af721a31ff56082994da46844e151ed255cc2.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/a08e782863090e870b19bda41e1af721a31ff56082994da46844e151ed255cc2.jpg)

![a5907be290538513a83cb08c1228d090e67eb6d22583848ac95b3182c1b152c4.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/a5907be290538513a83cb08c1228d090e67eb6d22583848ac95b3182c1b152c4.jpg)

![a6f5c9b7505e0f62484f4172952413088c120cb10e6f6c381eb4fcb8e299f60c.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/a6f5c9b7505e0f62484f4172952413088c120cb10e6f6c381eb4fcb8e299f60c.jpg)

![aafa53db4a8d4cd9961c302c2dfdeaa0c4025b2b63c2c37f4f1f471b6eb3d749.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/aafa53db4a8d4cd9961c302c2dfdeaa0c4025b2b63c2c37f4f1f471b6eb3d749.jpg)

![b8095c5a1b060afdfd54becd69bfc2dac436ea637e90f16a9461d7920f93efea.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/b8095c5a1b060afdfd54becd69bfc2dac436ea637e90f16a9461d7920f93efea.jpg)

![c2f991ad6d8fe283f6824a136db3ae1a7069aa8433975061dfa07d74a82d1a46.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/c2f991ad6d8fe283f6824a136db3ae1a7069aa8433975061dfa07d74a82d1a46.jpg)

![dd7d3f80e0f16bcd8850f9e8b2d1fcb6662f5916d52af33cb7932fa3c752b5c9.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/dd7d3f80e0f16bcd8850f9e8b2d1fcb6662f5916d52af33cb7932fa3c752b5c9.jpg)

![f21a2b236679845a05a7132aa3a21f8ae2fa47ee1372a608d6016aaca4ecc9f5.jpg](../iclr_results/611_Dataset Ownership Verification in Contrastive Pre-trained Models/tables/f21a2b236679845a05a7132aa3a21f8ae2fa47ee1372a608d6016aaca4ecc9f5.jpg)

## System 1.x: Learning to Balance Fast and Slow Planning with Language Models


### Images

![040b23194ef3741ec82ca153699e1e7bc677affd96b94055f80eb00292cb9e6c.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/040b23194ef3741ec82ca153699e1e7bc677affd96b94055f80eb00292cb9e6c.jpg)

![51257540eae69740466bc3981beed0bfc2cbabe7968fa6b5cd11015554fe6f7f.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/51257540eae69740466bc3981beed0bfc2cbabe7968fa6b5cd11015554fe6f7f.jpg)

![5ec790d240bd675cd8a2f896934e77bd161f1f67ccbc77bd753d8a862fe551f8.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/5ec790d240bd675cd8a2f896934e77bd161f1f67ccbc77bd753d8a862fe551f8.jpg)

![702025141ec0086f48fe2aa3650c1d0f5fbb939c65befbbab00ffa28059c26a9.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/702025141ec0086f48fe2aa3650c1d0f5fbb939c65befbbab00ffa28059c26a9.jpg)

![ae5f51c3e5ef24c99bd37f833e78791e16b227acb2b27431317f7297977e9128.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/ae5f51c3e5ef24c99bd37f833e78791e16b227acb2b27431317f7297977e9128.jpg)

![c3a3cbbb3d0098d9ad11446c1b3a7f327298b40987cfd1e489a3b751f0aa61ef.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/c3a3cbbb3d0098d9ad11446c1b3a7f327298b40987cfd1e489a3b751f0aa61ef.jpg)

![d0df083d77b637c984350b9ac0dc9eb69caa6c9a12c2c606ae9822f320a31388.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/d0df083d77b637c984350b9ac0dc9eb69caa6c9a12c2c606ae9822f320a31388.jpg)

![ec33c23328a3b261c20c31fa947705f9267d557db66063aa6bb853f3b2256119.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/ec33c23328a3b261c20c31fa947705f9267d557db66063aa6bb853f3b2256119.jpg)

![f35c399bde00c1ff682701882fdc730e85970c14b67f660f37d13eb5a57aec3b.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/f35c399bde00c1ff682701882fdc730e85970c14b67f660f37d13eb5a57aec3b.jpg)

![fe60474df8005df88dcd07d4ae3518b81e9ab3e2968e6ede2dab87a0cfb3bb28.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/images/fe60474df8005df88dcd07d4ae3518b81e9ab3e2968e6ede2dab87a0cfb3bb28.jpg)

### Tables

![0f2774b1df4c56e91bf9dfbdbc2f4d647323aeab044251c845732fcbfdc4f0fc.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/0f2774b1df4c56e91bf9dfbdbc2f4d647323aeab044251c845732fcbfdc4f0fc.jpg)

![16b521d5ba1e02fae4bcf3143a396f72de4bbbacbe219ea26f2face4d8a23a32.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/16b521d5ba1e02fae4bcf3143a396f72de4bbbacbe219ea26f2face4d8a23a32.jpg)

![21e131dd7ded4ea27146a16cf46597e3bac919a1c7febd61c05bee6829416adf.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/21e131dd7ded4ea27146a16cf46597e3bac919a1c7febd61c05bee6829416adf.jpg)

![67fe0bbb1cf45219099b2e287017893d09f33bbbe8443c002a302e48e655f420.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/67fe0bbb1cf45219099b2e287017893d09f33bbbe8443c002a302e48e655f420.jpg)

![87b1312c8e7c42f5647fb09ebd7c513e1b0ac9c2e3c1f2fd75cf2661e85aa387.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/87b1312c8e7c42f5647fb09ebd7c513e1b0ac9c2e3c1f2fd75cf2661e85aa387.jpg)

![b29aa27d26c45a75cb91693753345f80facc88ec8ffa92de5a3f061b76836624.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/b29aa27d26c45a75cb91693753345f80facc88ec8ffa92de5a3f061b76836624.jpg)

![baefa5175b339fabacbf29eef87479c4651b65dd631c403e573194f063802542.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/baefa5175b339fabacbf29eef87479c4651b65dd631c403e573194f063802542.jpg)

![c15dce1da9496714a472cc4cf938536f042a7547456ad2148737a734bffddf7b.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/c15dce1da9496714a472cc4cf938536f042a7547456ad2148737a734bffddf7b.jpg)

![dcf03ed17a574e27708bfcc24d397a2f359e28ec2317833239e99bfc3301bd28.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/dcf03ed17a574e27708bfcc24d397a2f359e28ec2317833239e99bfc3301bd28.jpg)

![e7663e48ae9c89a8476440a7252908a2d8173badbc66c65a732aaa32d101e155.jpg](../iclr_results/612_System 1.x_ Learning to Balance Fast and Slow Planning with Language Models/tables/e7663e48ae9c89a8476440a7252908a2d8173badbc66c65a732aaa32d101e155.jpg)

## Time-to-Event Pretraining for 3D Medical Imaging


### Images

![126e0246ae8e2b1ef78a8d058e8f0f3f0b8ba1c9e45ad7e044537e068bcefe26.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/126e0246ae8e2b1ef78a8d058e8f0f3f0b8ba1c9e45ad7e044537e068bcefe26.jpg)

![1727049c45fbd7e9521654510cedf78630752172d4ee9a81e9ae4b759cc0d396.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/1727049c45fbd7e9521654510cedf78630752172d4ee9a81e9ae4b759cc0d396.jpg)

![2ca8722a1222bc1632f1974483fa1ef4b18751ad221961a0791518e7749d2407.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/2ca8722a1222bc1632f1974483fa1ef4b18751ad221961a0791518e7749d2407.jpg)

![48dfc036aca127d93006b7b594ec92c5223d9d74fa0512793a35b7b257fff546.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/48dfc036aca127d93006b7b594ec92c5223d9d74fa0512793a35b7b257fff546.jpg)

![4e38fc1acf171e113352833214b89ace6f229cf7e10c9352f570331149da08a6.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/4e38fc1acf171e113352833214b89ace6f229cf7e10c9352f570331149da08a6.jpg)

![5989ddf574c2edb449ebe7a1850bb4249bb296bcfe288a5144df229c283c1d47.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/5989ddf574c2edb449ebe7a1850bb4249bb296bcfe288a5144df229c283c1d47.jpg)

![7842f7caa98a76c78beb0150ab7d6b52292807a29c1e6f43e9fcb143d215ef6a.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/7842f7caa98a76c78beb0150ab7d6b52292807a29c1e6f43e9fcb143d215ef6a.jpg)

![7d7e2989f5086b5c589c2965a5ba85bffe518db9de5aec241b47dc7eb0b93c48.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/7d7e2989f5086b5c589c2965a5ba85bffe518db9de5aec241b47dc7eb0b93c48.jpg)

![87b03677323b4faa124240cd67c84c5dfc12da2e831d2900684366b9ce482d61.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/87b03677323b4faa124240cd67c84c5dfc12da2e831d2900684366b9ce482d61.jpg)

![8888be2ebc01d7a4031df18a73348091b69ca5b8b4fa1cc6e42f5fff8b3b5837.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/8888be2ebc01d7a4031df18a73348091b69ca5b8b4fa1cc6e42f5fff8b3b5837.jpg)

![89e9ff550baefff753fecb4d453d7dd4b7d843dbb447569e0a91c73440e382e6.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/89e9ff550baefff753fecb4d453d7dd4b7d843dbb447569e0a91c73440e382e6.jpg)

![8bb4e57d2218018bb80166d115af76a6706fed0be9e98ff8e4338372336c3354.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/8bb4e57d2218018bb80166d115af76a6706fed0be9e98ff8e4338372336c3354.jpg)

![8c6261381f6ff9ea95d1541b275df79efa75a5741d5f17e99fb2639e06850553.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/8c6261381f6ff9ea95d1541b275df79efa75a5741d5f17e99fb2639e06850553.jpg)

![a92f5891c66ed48cac71555985251b79ca06d3d9e2ee6dbd64f4c4f3aa70e7b5.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/a92f5891c66ed48cac71555985251b79ca06d3d9e2ee6dbd64f4c4f3aa70e7b5.jpg)

![ab471894614dbf19824805f912f1aa91c8c147772aabecaeecfa7de1e855d7a9.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/ab471894614dbf19824805f912f1aa91c8c147772aabecaeecfa7de1e855d7a9.jpg)

![baee1865ce68347ab0b73945d392a3326fb0950a15f818f6fbdf13c43dcb433a.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/baee1865ce68347ab0b73945d392a3326fb0950a15f818f6fbdf13c43dcb433a.jpg)

![cdd9aa057fbf04bdde734b13d3c48a2d437f29902c20cc83c80edf47f2a79305.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/cdd9aa057fbf04bdde734b13d3c48a2d437f29902c20cc83c80edf47f2a79305.jpg)

![e2cb0bf24a9e4c3993446ef66cd16981b39d9c72963aaf14a0f09aaac6512df0.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/e2cb0bf24a9e4c3993446ef66cd16981b39d9c72963aaf14a0f09aaac6512df0.jpg)

![f50ffdd152401b33e5e3a44f755c7747be7ba3dc5548edf72dbf3b4e35388365.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/images/f50ffdd152401b33e5e3a44f755c7747be7ba3dc5548edf72dbf3b4e35388365.jpg)

### Tables

![0220c3fca9c7d08121b660f9638a473d95c2b45af88916e53d1e9c154949230b.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/0220c3fca9c7d08121b660f9638a473d95c2b45af88916e53d1e9c154949230b.jpg)

![113d72a161ddbae0689660ffe7128f2788e7b7fe0a94141ada5837b1a8d8b7cf.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/113d72a161ddbae0689660ffe7128f2788e7b7fe0a94141ada5837b1a8d8b7cf.jpg)

![14b9582f019931f44e17ad6e4cd79a5f8334bcb0696e79b9d879b675b0198a1c.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/14b9582f019931f44e17ad6e4cd79a5f8334bcb0696e79b9d879b675b0198a1c.jpg)

![238edf16ce834321c958745e0945f25169f39da6bc9f11faa26b486a5fed62ed.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/238edf16ce834321c958745e0945f25169f39da6bc9f11faa26b486a5fed62ed.jpg)

![32c29bb1006ec0c3e578da6f607db6f44ee8ca2a9a342d71bb2b420daf772d90.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/32c29bb1006ec0c3e578da6f607db6f44ee8ca2a9a342d71bb2b420daf772d90.jpg)

![356695dcfac9e61f5db85b9f616b331e9605f9462c6f8e653887c80dbf67db1b.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/356695dcfac9e61f5db85b9f616b331e9605f9462c6f8e653887c80dbf67db1b.jpg)

![3c192cd0761b48a0d8d9c6d737bf709ebf930fc4250277274758f17b6767face.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/3c192cd0761b48a0d8d9c6d737bf709ebf930fc4250277274758f17b6767face.jpg)

![421f2ced587934d9265d15a7fde09d32f60aae3827675a21ed53e0d8cf62410c.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/421f2ced587934d9265d15a7fde09d32f60aae3827675a21ed53e0d8cf62410c.jpg)

![4ef162867dd5e5625b66c70b8a1a3cf96410a9a33f310a37592c7c050ea2e57b.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/4ef162867dd5e5625b66c70b8a1a3cf96410a9a33f310a37592c7c050ea2e57b.jpg)

![5014120d37f3050c163fc591594d18226467c119acaa962fbd83c4ebd6e73e57.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/5014120d37f3050c163fc591594d18226467c119acaa962fbd83c4ebd6e73e57.jpg)

![54d71bbb9410e94cbac46498ab7022305f93598aaeaaa70c48c4679625975837.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/54d71bbb9410e94cbac46498ab7022305f93598aaeaaa70c48c4679625975837.jpg)

![5e567c80af874d2d191d020499d7bc00d20a728511298d77335168507e83c5d1.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/5e567c80af874d2d191d020499d7bc00d20a728511298d77335168507e83c5d1.jpg)

![70ee77cb987a4c536fca0d61f4fdd1cb0d9a345921bffaad8aafe017926011f3.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/70ee77cb987a4c536fca0d61f4fdd1cb0d9a345921bffaad8aafe017926011f3.jpg)

![77895132605be21c54afbad4b3921b48f91490c468dc7e35c3be7ba407e018a9.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/77895132605be21c54afbad4b3921b48f91490c468dc7e35c3be7ba407e018a9.jpg)

![7fe2b561192c00e3c9df3abc626c5e0746f68dcacf71c6daa0cb7d7d6fbcbcc9.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/7fe2b561192c00e3c9df3abc626c5e0746f68dcacf71c6daa0cb7d7d6fbcbcc9.jpg)

![80f8c6ac8006d14a8b3b03427793615a870c1ac59f14f9262a99f8b2a4790a4b.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/80f8c6ac8006d14a8b3b03427793615a870c1ac59f14f9262a99f8b2a4790a4b.jpg)

![86c6f69e52ca3668d612c64bd001129c6fd1fd1c5e11386d5d0be4ec6af9dc9a.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/86c6f69e52ca3668d612c64bd001129c6fd1fd1c5e11386d5d0be4ec6af9dc9a.jpg)

![8fd4b0d79a40dfb6b3b2142ee0b8b417f44808f97fd18e1a066aaf39d068358b.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/8fd4b0d79a40dfb6b3b2142ee0b8b417f44808f97fd18e1a066aaf39d068358b.jpg)

![9d8fb32390be7e4bec88baed48dc8f651276fc4519729ead0ea95e60f464f779.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/9d8fb32390be7e4bec88baed48dc8f651276fc4519729ead0ea95e60f464f779.jpg)

![a02d47d438096f6cd49ef987b5715fda5d25757fc7975fa0653a78a35b977703.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/a02d47d438096f6cd49ef987b5715fda5d25757fc7975fa0653a78a35b977703.jpg)

![a27b44a5aefa47cbb24c4f1f9d5ac8773c8c7025befea5fddbc0636430f71652.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/a27b44a5aefa47cbb24c4f1f9d5ac8773c8c7025befea5fddbc0636430f71652.jpg)

![a5bc0a1a7b2241c8b586defb450e251717717c20e370a7b7a80d908cd42a0dc4.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/a5bc0a1a7b2241c8b586defb450e251717717c20e370a7b7a80d908cd42a0dc4.jpg)

![a67b86342ea599db98aa72b1777e6e47858742482e2182cd0bc62779a6d23841.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/a67b86342ea599db98aa72b1777e6e47858742482e2182cd0bc62779a6d23841.jpg)

![c67059bf80c03d355fac7e07b00670d2ada21fac5e74957c8511c26d92b63464.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/c67059bf80c03d355fac7e07b00670d2ada21fac5e74957c8511c26d92b63464.jpg)

![d01f1ce5bfa85559446b98cd5adef751dcae6bc59304893f931a84f08e5ba176.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/d01f1ce5bfa85559446b98cd5adef751dcae6bc59304893f931a84f08e5ba176.jpg)

![ed68bef9899932e6eae276a0ac1d1e4728e358a2dd31137ddb1eb0386b0b0fc8.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/ed68bef9899932e6eae276a0ac1d1e4728e358a2dd31137ddb1eb0386b0b0fc8.jpg)

![f07d1f97c44e862344b3e865db426b51f4a65236041890c2b27c5b668a2692e0.jpg](../iclr_results/613_Time-to-Event Pretraining for 3D Medical Imaging/tables/f07d1f97c44e862344b3e865db426b51f4a65236041890c2b27c5b668a2692e0.jpg)

## Semialgebraic Neural Networks: From roots to representations


### Images

![12416e3bfb1b8e8cb81f443822dc564fcc38c04dc2b2faedc59e0ef315bed7b0.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/images/12416e3bfb1b8e8cb81f443822dc564fcc38c04dc2b2faedc59e0ef315bed7b0.jpg)

![72306748befaf22f06d9c414daf0a7f38846985d00cb138887ff8f4a6f5be91a.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/images/72306748befaf22f06d9c414daf0a7f38846985d00cb138887ff8f4a6f5be91a.jpg)

![8a6dd3fef6597086ae74a6b77045435c9a9f27a6a76a454f50b273f0eec6b0bc.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/images/8a6dd3fef6597086ae74a6b77045435c9a9f27a6a76a454f50b273f0eec6b0bc.jpg)

![93ecb69ebd171871821b3832a1846476e16362a421522b73ae6eae83984f4957.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/images/93ecb69ebd171871821b3832a1846476e16362a421522b73ae6eae83984f4957.jpg)

![a760b1853aaf62a54951e8299e67e2c53eb7bf5abdc04854a91e03f45b2414be.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/images/a760b1853aaf62a54951e8299e67e2c53eb7bf5abdc04854a91e03f45b2414be.jpg)

![b04a45ec3abf04ba9a61e3c3d8acdfdaf5cba7728fc118c8a8233c8c801a58a2.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/images/b04a45ec3abf04ba9a61e3c3d8acdfdaf5cba7728fc118c8a8233c8c801a58a2.jpg)

![eb8ab8c991d7b2ccf47815b9cdb6dfefe9e990d949b3f8d4d53a6c9c7eb2bba8.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/images/eb8ab8c991d7b2ccf47815b9cdb6dfefe9e990d949b3f8d4d53a6c9c7eb2bba8.jpg)

### Tables

![3615a934033847f07a518ac2d9f88361ad30ad828eec2fd65b8197df67de30f4.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/tables/3615a934033847f07a518ac2d9f88361ad30ad828eec2fd65b8197df67de30f4.jpg)

![51c6bc2f160e2e98089f305caea9bd8500b3de6c818f4307b695edf5dba647a2.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/tables/51c6bc2f160e2e98089f305caea9bd8500b3de6c818f4307b695edf5dba647a2.jpg)

![facdb1315e54d17151ec2e482a78d4c2aaae9fbe591b3b00d232b1dfcf48aa8f.jpg](../iclr_results/614_Semialgebraic Neural Networks_ From roots to representations/tables/facdb1315e54d17151ec2e482a78d4c2aaae9fbe591b3b00d232b1dfcf48aa8f.jpg)

## h4rm3l: A Language for Composable Jailbreak Attack Synthesis


### Images

![0bb2ac19003cf56785fe3e52e7866345adbfa986bdc365acdaec2a9253337952.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/0bb2ac19003cf56785fe3e52e7866345adbfa986bdc365acdaec2a9253337952.jpg)

![2f1ef2674c6345c51b018d24aee501a68955b9157e733ed62121c1685b63c799.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/2f1ef2674c6345c51b018d24aee501a68955b9157e733ed62121c1685b63c799.jpg)

![2f6ca30698d24520c501c1ff060a81d6da2c5a1f59ff0de8ae024fe51b36a966.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/2f6ca30698d24520c501c1ff060a81d6da2c5a1f59ff0de8ae024fe51b36a966.jpg)

![3bdb8922968ca4ff2db4a1eaa20e2a3c813345df138a282f334116e423f11456.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/3bdb8922968ca4ff2db4a1eaa20e2a3c813345df138a282f334116e423f11456.jpg)

![3dbdd1b37017783f044a98f6bb01f70dc0b1768a60982fb2433db7df3cf6f042.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/3dbdd1b37017783f044a98f6bb01f70dc0b1768a60982fb2433db7df3cf6f042.jpg)

![4b4984c8b2421a826023fdc8839291f4bd2ba0ae9251e2ae34ecec854411454f.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/4b4984c8b2421a826023fdc8839291f4bd2ba0ae9251e2ae34ecec854411454f.jpg)

![959e88ccab01de41a49deeb2042854ab30bed8f1ddcbf7c472d4982cfa856bdd.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/959e88ccab01de41a49deeb2042854ab30bed8f1ddcbf7c472d4982cfa856bdd.jpg)

![a139203c5cb6eed3dea641285bc1f544c1de2b225421befe5d18328ffd387a1a.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/a139203c5cb6eed3dea641285bc1f544c1de2b225421befe5d18328ffd387a1a.jpg)

![a892f93e6e859d9d7c040513ecc93eacfdeed791613a9b624b8cf71dcba44079.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/a892f93e6e859d9d7c040513ecc93eacfdeed791613a9b624b8cf71dcba44079.jpg)

![b6cb7cd912dac84597d236152dd5a8bbef0788a4adfbba8c379d93db64bad4be.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/b6cb7cd912dac84597d236152dd5a8bbef0788a4adfbba8c379d93db64bad4be.jpg)

![bb5c3178b6e6d544eeb998ab27cff983e18a2112ea85c35031c8e45646c7bfaa.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/bb5c3178b6e6d544eeb998ab27cff983e18a2112ea85c35031c8e45646c7bfaa.jpg)

![c30233e31b2b595217ff514902d6830b21818b6ddb9944844c12a55e918828e4.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/c30233e31b2b595217ff514902d6830b21818b6ddb9944844c12a55e918828e4.jpg)

![cab482ffa0a2937cdcc20b6b46f25367dc4cd1a9be0fb1cc90d0b63eced33fac.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/cab482ffa0a2937cdcc20b6b46f25367dc4cd1a9be0fb1cc90d0b63eced33fac.jpg)

![cfb11def53f321a3667b46db174cc260c073373878ec2a34fd14f98604e3f820.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/cfb11def53f321a3667b46db174cc260c073373878ec2a34fd14f98604e3f820.jpg)

![d6e92c9dc78bcca1db21f98493df827ab012f7b22b01af7009da5796c487b1fa.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/images/d6e92c9dc78bcca1db21f98493df827ab012f7b22b01af7009da5796c487b1fa.jpg)

### Tables

![22cd5dd74d22cf9152a0d8a08f559758237883f84c178ac2a5b563f6afb56278.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/tables/22cd5dd74d22cf9152a0d8a08f559758237883f84c178ac2a5b563f6afb56278.jpg)

![3c1eb6e9db57e3b572dea996b121ef863893aa1c7e29a7deb867558b88c09e87.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/tables/3c1eb6e9db57e3b572dea996b121ef863893aa1c7e29a7deb867558b88c09e87.jpg)

![41bdda1b73dfb5ecb648e2d106b77c8f45196925a179badd60f2ad6256515d24.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/tables/41bdda1b73dfb5ecb648e2d106b77c8f45196925a179badd60f2ad6256515d24.jpg)

![552f662c3c7c32fef0774a1160aacf6240b33f8f95052440225152d683512e3a.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/tables/552f662c3c7c32fef0774a1160aacf6240b33f8f95052440225152d683512e3a.jpg)

![b881e619b7979f1b1b5a65d38205a8ec04afe2ecd7d4b6bbea018eed59153163.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/tables/b881e619b7979f1b1b5a65d38205a8ec04afe2ecd7d4b6bbea018eed59153163.jpg)

![c7d4410375b779fe1275d1ce267603117cb8f1022bbac27a4ca8bbf6995eabee.jpg](../iclr_results/615_h4rm3l_ A Language for Composable Jailbreak Attack Synthesis/tables/c7d4410375b779fe1275d1ce267603117cb8f1022bbac27a4ca8bbf6995eabee.jpg)

## Synthesizing Realistic fMRI: A Physiological Dynamics-Driven Hierarchical Diffusion Model for Efficient fMRI Acquisition

### Images

![184af935549bd108661d41b3654f32c20f6bb37d5441c82be8c13f6857cc65d0.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/images/184af935549bd108661d41b3654f32c20f6bb37d5441c82be8c13f6857cc65d0.jpg)

![5890d129992b30f0a6aa19899ea89b8fbe0b3f2da67eb8ef729b8f898c3bbff8.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/images/5890d129992b30f0a6aa19899ea89b8fbe0b3f2da67eb8ef729b8f898c3bbff8.jpg)

![a58cb17cf1dc819c19938858f4790e5b54ce5323cf6201303458394f3c521551.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/images/a58cb17cf1dc819c19938858f4790e5b54ce5323cf6201303458394f3c521551.jpg)

![c1273b889bd4f8d5d25929219b5505d760b110d46762354173b2365fb1116375.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/images/c1273b889bd4f8d5d25929219b5505d760b110d46762354173b2365fb1116375.jpg)

![c9ec2293cacaf1e188bfb32c1533824f3c1dcf4d1714f747a1a4c25abe31b901.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/images/c9ec2293cacaf1e188bfb32c1533824f3c1dcf4d1714f747a1a4c25abe31b901.jpg)

### Tables

![3d446374ed0c988d438c35dc220ffcbbfe109ffbd32d0faf1d4f0bc112346371.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/tables/3d446374ed0c988d438c35dc220ffcbbfe109ffbd32d0faf1d4f0bc112346371.jpg)

![4a26e492875c207188adb14e1e09270d2b11ae94359087817b3f1bfbd905391a.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/tables/4a26e492875c207188adb14e1e09270d2b11ae94359087817b3f1bfbd905391a.jpg)

![9009b1c86b09e1bc38eea1ec057e69dbcb7bc0e574386a6896eaf22f6a878799.jpg](../iclr_results/616_Synthesizing Realistic fMRI_ A Physiological Dynamics-Driven Hierarchical Diffusion Model for Effici/tables/9009b1c86b09e1bc38eea1ec057e69dbcb7bc0e574386a6896eaf22f6a878799.jpg)
