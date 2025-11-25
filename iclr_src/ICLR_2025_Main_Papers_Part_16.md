# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 16 of 100

## 目录 (Table of Contents)

1. [Planning in Natural Language Improves LLM Search for Code Generation](#Planning-in-Natural-Language-Improves-LLM-Search-for-Code-Generation)
2. [On Quantizing Neural Representation for Variable-Rate Video Coding](#On-Quantizing-Neural-Representation-for-Variable-Rate-Video-Coding)
3. [DiffPuter: Empowering Diffusion Models for Missing Data Imputation](#DiffPuter-Empowering-Diffusion-Models-for-Missing-Data-Imputation)
4. [What Does It Mean to Be a Transformer? Insights from a Theoretical Hessian Analysis](#What-Does-It-Mean-to-Be-a-Transformer-Insights-from-a-Theoretical-Hessian-Analysis)
5. [Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs with Semantic Space](#Broaden-your-SCOPE-Efficient-Multi-turn-Conversation-Planning-for-LLMs-with-Semantic-Space)
6. [LeFusion: Controllable Pathology Synthesis via Lesion-Focused Diffusion Models](#LeFusion-Controllable-Pathology-Synthesis-via-Lesion-Focused-Diffusion-Models)
7. [Multi-Field Adaptive Retrieval](#Multi-Field-Adaptive-Retrieval)
8. [RelitLRM: Generative Relightable Radiance for Large Reconstruction Models](#RelitLRM-Generative-Relightable-Radiance-for-Large-Reconstruction-Models)
9. [Learning Spatiotemporal Dynamical Systems from Point Process Observations](#Learning-Spatiotemporal-Dynamical-Systems-from-Point-Process-Observations)
10. [uniINF: Best-of-Both-Worlds Algorithm for Parameter-Free Heavy-Tailed MABs](#uniINF-Best-of-Both-Worlds-Algorithm-for-Parameter-Free-Heavy-Tailed-MABs)
11. [The Superposition of Diffusion Models Using the Itô Density Estimator](#The-Superposition-of-Diffusion-Models-Using-the-Itô-Density-Estimator)
12. [Towards hyperparameter-free optimization with differential privacy](#Towards-hyperparameter-free-optimization-with-differential-privacy)
13. [Discovering Temporally Compositional Neural Manifolds with Switching Infinite GPFA](#Discovering-Temporally-Compositional-Neural-Manifolds-with-Switching-Infinite-GPFA)
14. [DeepRTL: Bridging Verilog Understanding and Generation with a Unified Representation Model](#DeepRTL-Bridging-Verilog-Understanding-and-Generation-with-a-Unified-Representation-Model)
15. [DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference](#DeFT-Decoding-with-Flash-Tree-attention-for-Efficient-Tree-structured-LLM-Inference)
16. [CREIMBO: Cross-Regional Ensemble Interactions in Multi-view Brain Observations](#CREIMBO-Cross-Regional-Ensemble-Interactions-in-Multi-view-Brain-Observations)
17. [High-dimensional Analysis of Knowledge Distillation: Weak-to-Strong Generalization and Scaling Laws](#High-dimensional-Analysis-of-Knowledge-Distillation-Weak-to-Strong-Generalization-and-Scaling-Laws)
18. [NetMoE: Accelerating MoE Training through Dynamic Sample Placement](#NetMoE-Accelerating-MoE-Training-through-Dynamic-Sample-Placement)
19. [Bayesian Optimization via Continual Variational Last Layer Training](#Bayesian-Optimization-via-Continual-Variational-Last-Layer-Training)
20. [Fast Uncovering of Protein Sequence Diversity from Structure](#Fast-Uncovering-of-Protein-Sequence-Diversity-from-Structure)
21. [Century: A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images](#Century-A-Framework-and-Dataset-for-Evaluating-Historical-Contextualisation-of-Sensitive-Images)
22. [MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code](#MathCoder2-Better-Math-Reasoning-from-Continued-Pretraining-on-Model-translated-Mathematical-Code)
23. [OmniRe: Omni Urban Scene Reconstruction](#OmniRe-Omni-Urban-Scene-Reconstruction)
24. [In vivo cell-type and brain region classification via multimodal contrastive learning](#In-vivo-cell-type-and-brain-region-classification-via-multimodal-contrastive-learning)
25. [Monitoring Latent World States in Language Models with Propositional Probes](#Monitoring-Latent-World-States-in-Language-Models-with-Propositional-Probes)
26. [Universal generalization guarantees for Wasserstein distributionally robust models](#Universal-generalization-guarantees-for-Wasserstein-distributionally-robust-models)
27. [PETRA: Parallel End-to-end Training with Reversible Architectures](#PETRA-Parallel-End-to-end-Training-with-Reversible-Architectures)
28. [ThunderKittens: Simple, Fast, and $\textit{Adorable}$ Kernels](#ThunderKittens-Simple-Fast-and-textitAdorable-Kernels)
29. [Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage](#Multi-modal-Agent-Tuning-Building-a-VLM-Driven-Agent-for-Efficient-Tool-Usage)
30. [Holistically Evaluating the Environmental Impact of Creating Language Models](#Holistically-Evaluating-the-Environmental-Impact-of-Creating-Language-Models)
31. [Adaptive Gradient Clipping for Robust Federated Learning](#Adaptive-Gradient-Clipping-for-Robust-Federated-Learning)
32. [DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback](#DataEnvGym-Data-Generation-Agents-in-Teacher-Environments-with-Student-Feedback)
33. [Re-Imagining Multimodal Instruction Tuning: A Representation View](#Re-Imagining-Multimodal-Instruction-Tuning-A-Representation-View)
34. [Inverse decision-making using neural amortized Bayesian actors](#Inverse-decision-making-using-neural-amortized-Bayesian-actors)
35. [Designing Concise ConvNets with Columnar Stages](#Designing-Concise-ConvNets-with-Columnar-Stages)
36. [MVTokenFlow: High-quality 4D Content Generation using Multiview Token Flow](#MVTokenFlow-High-quality-4D-Content-Generation-using-Multiview-Token-Flow)
37. [Let the Code LLM Edit Itself When You Edit the Code](#Let-the-Code-LLM-Edit-Itself-When-You-Edit-the-Code)

---


## Planning in Natural Language Improves LLM Search for Code Generation

### Images

![431651a088c6f0146524b366d63dbe0b3344bcfb670c59a04b4482c1dcfc8a18.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/431651a088c6f0146524b366d63dbe0b3344bcfb670c59a04b4482c1dcfc8a18.jpg)

![5ddca3bd6dac1f81434deab8856d469a181afdd1d992df9d586b2a79095b09c2.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/5ddca3bd6dac1f81434deab8856d469a181afdd1d992df9d586b2a79095b09c2.jpg)

![5fcb550343f3308ebc0dbed366c1d110f49c8611eacf49d2c95194aa98a472b4.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/5fcb550343f3308ebc0dbed366c1d110f49c8611eacf49d2c95194aa98a472b4.jpg)

![84f2606063ea80098ad74a750387ff0961cb7041fc0fe3ce4f360685be8e4eeb.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/84f2606063ea80098ad74a750387ff0961cb7041fc0fe3ce4f360685be8e4eeb.jpg)

![8fbf340c34b191697cab8f24f3a3497539c51fbaef64cf755b8caabeba51a7f6.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/8fbf340c34b191697cab8f24f3a3497539c51fbaef64cf755b8caabeba51a7f6.jpg)

![94b204d8890367600cf49a8d2975d85459ca75065ef9e54a1272e9458dbf0934.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/94b204d8890367600cf49a8d2975d85459ca75065ef9e54a1272e9458dbf0934.jpg)

![a27cfb5c9fa36b627f70929817c39106ad2d759c9eb2b3a2e5308ce05aa37353.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/a27cfb5c9fa36b627f70929817c39106ad2d759c9eb2b3a2e5308ce05aa37353.jpg)

![c5fff4f0203c908aff5dd69034410acc7f77a81110b5c0a37c9f5453bb547366.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/images/c5fff4f0203c908aff5dd69034410acc7f77a81110b5c0a37c9f5453bb547366.jpg)

### Tables

![8ea1aaa56282a55c60389f87f7097622d767afa476522c389c5eea1b57934f35.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/tables/8ea1aaa56282a55c60389f87f7097622d767afa476522c389c5eea1b57934f35.jpg)

![a00043b8ef76249e1d02f5fe82db63f689f7a301d014101d6e8bec4a9ebe78da.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/tables/a00043b8ef76249e1d02f5fe82db63f689f7a301d014101d6e8bec4a9ebe78da.jpg)

![f8cb39a1544dfc61153d1d9591dcffc6cb9498faae87a9894944ceae0b891f8b.jpg](../iclr_results/561_Learning%20from%20negative%20feedback%2C%20or%20positive%20feedback%20or%20both/tables/f8cb39a1544dfc61153d1d9591dcffc6cb9498faae87a9894944ceae0b891f8b.jpg)

## Planning in Natural Language Improves LLM Search for Code Generation


### Images

![0b9ec46e22ea33a1fa58f0d5ace9486dc067d8320cab73850cfd0466e8a47cc7.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/0b9ec46e22ea33a1fa58f0d5ace9486dc067d8320cab73850cfd0466e8a47cc7.jpg)

![0c801849fc5566df2fb68fecc1cb6db82157352ebd8c5dc2f7ecb0e6a0cf3efa.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/0c801849fc5566df2fb68fecc1cb6db82157352ebd8c5dc2f7ecb0e6a0cf3efa.jpg)

![0e45f2e7e6cbcd45bb0f355344f02fada4bfc848be1aa1c13661ad2a7ce744b9.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/0e45f2e7e6cbcd45bb0f355344f02fada4bfc848be1aa1c13661ad2a7ce744b9.jpg)

![1a3bec5c8b5591e95e38cdec7be84bbe69ab20dafdd69f9070276a6ae813cdaf.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/1a3bec5c8b5591e95e38cdec7be84bbe69ab20dafdd69f9070276a6ae813cdaf.jpg)

![28bab11289568f02db326529b3a4626406c33e3cea01a5df2dbd9f2402dd2154.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/28bab11289568f02db326529b3a4626406c33e3cea01a5df2dbd9f2402dd2154.jpg)

![2a526cf43c82647eb25e9e3103a6f9618a9eea7939cd1b61dab5548d5cda2068.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/2a526cf43c82647eb25e9e3103a6f9618a9eea7939cd1b61dab5548d5cda2068.jpg)

![2fe801390c27a21cb67f2c83fd31f6ad3c69a1e8fcd0e339a4230a20277234a7.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/2fe801390c27a21cb67f2c83fd31f6ad3c69a1e8fcd0e339a4230a20277234a7.jpg)

![336bf5940c1d46b3aaf7c09e3306fff4aec44a2785b8865da3331a71738545ed.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/336bf5940c1d46b3aaf7c09e3306fff4aec44a2785b8865da3331a71738545ed.jpg)

![39dd6ac48989a9831147608c1b05de6a33d5cd96e62ac98fad9b48fa50606bd4.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/39dd6ac48989a9831147608c1b05de6a33d5cd96e62ac98fad9b48fa50606bd4.jpg)

![3d1a59469cf31c3f209f5a0325735d892a8911013e1d86e80c5b656dac61080c.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/3d1a59469cf31c3f209f5a0325735d892a8911013e1d86e80c5b656dac61080c.jpg)

![3f46a9e19512944c5b2db93e69d08eec123ebc676a9ed381bbf87e10ca54fba5.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/3f46a9e19512944c5b2db93e69d08eec123ebc676a9ed381bbf87e10ca54fba5.jpg)

![44e4524775fc3ed5051bbbc3a4f274df55fa0a1753e9f71ae1ecea0437845aee.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/44e4524775fc3ed5051bbbc3a4f274df55fa0a1753e9f71ae1ecea0437845aee.jpg)

![4720eb14f1f377671dd5054d23adf3d5eb482bba0f3d3ade1474771ede500d63.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/4720eb14f1f377671dd5054d23adf3d5eb482bba0f3d3ade1474771ede500d63.jpg)

![51426f17ef21dd7d29ad93665453755443caee4cbdcc04bd4b2887d6eac70988.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/51426f17ef21dd7d29ad93665453755443caee4cbdcc04bd4b2887d6eac70988.jpg)

![53a43167b2f931dfd16c723edb9d557cb6086ec668dc43677dc320640302e4ee.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/53a43167b2f931dfd16c723edb9d557cb6086ec668dc43677dc320640302e4ee.jpg)

![55ef89e4d5ddfbed4ea3768cf90a5c5f0cb3ea5502d4fbe35bc7d7d600baa3e3.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/55ef89e4d5ddfbed4ea3768cf90a5c5f0cb3ea5502d4fbe35bc7d7d600baa3e3.jpg)

![59ee2f63c46f5a485e8929deb01bddcb260831f4ffc3a9d2c9649d3625e62610.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/59ee2f63c46f5a485e8929deb01bddcb260831f4ffc3a9d2c9649d3625e62610.jpg)

![5b0c9b4a89749b5b53143fe78e4d9efaa4b2ba5939b0baff735551ac553e35b5.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/5b0c9b4a89749b5b53143fe78e4d9efaa4b2ba5939b0baff735551ac553e35b5.jpg)

![6a5e9738ca31317468e48a2777deb97286cc9e484395458873029d01c51ed1c6.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/6a5e9738ca31317468e48a2777deb97286cc9e484395458873029d01c51ed1c6.jpg)

![6c40cb65d56f8ec6e16074a185326a12ec4dd3c851bfff334acacd8879359db8.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/6c40cb65d56f8ec6e16074a185326a12ec4dd3c851bfff334acacd8879359db8.jpg)

![6c70d796157f66baa05145f4c19d0014ab262207a5cfd3f8dc363cc1ec1054f8.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/6c70d796157f66baa05145f4c19d0014ab262207a5cfd3f8dc363cc1ec1054f8.jpg)

![6f4c3cae41467877b811897b8e6f801e5dada5f6a765b118d5b1a6b759d9ab2d.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/6f4c3cae41467877b811897b8e6f801e5dada5f6a765b118d5b1a6b759d9ab2d.jpg)

![8edcf1da59c30a12d15a82bae873318086b1c72a1f7a91d2f72c9723bb1b10b7.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/8edcf1da59c30a12d15a82bae873318086b1c72a1f7a91d2f72c9723bb1b10b7.jpg)

![a636bcda002e515fcd6e093d0969b546ebaedfa52fb8e1b9d4ea8fed8afb35bd.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/a636bcda002e515fcd6e093d0969b546ebaedfa52fb8e1b9d4ea8fed8afb35bd.jpg)

![a72fd82147b532b82ce51303a2919df68539f9039c22c0232b4e57c50504103e.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/a72fd82147b532b82ce51303a2919df68539f9039c22c0232b4e57c50504103e.jpg)

![a828c84325ac2b64c1d622e12b4f4adba1082fd95cfda659fc62e677d7a55386.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/a828c84325ac2b64c1d622e12b4f4adba1082fd95cfda659fc62e677d7a55386.jpg)

![a879dbdefa6f0971a77f161cfd51d2825c88403dd4346f847caeaf53cdae2838.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/a879dbdefa6f0971a77f161cfd51d2825c88403dd4346f847caeaf53cdae2838.jpg)

![b1ca71b7699f3b2a5e93653be0ca9fefd68f7360b0fc0fe48befa0957adda734.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/b1ca71b7699f3b2a5e93653be0ca9fefd68f7360b0fc0fe48befa0957adda734.jpg)

![b5cd5e53582ecc487edf4ef107e20ee4dd55a13dd9830e4ad25b6741a0944782.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/b5cd5e53582ecc487edf4ef107e20ee4dd55a13dd9830e4ad25b6741a0944782.jpg)

![b73c1ec62ee831bad7b16b12378df8a046998a58c0088a363ad8bf44ae289551.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/b73c1ec62ee831bad7b16b12378df8a046998a58c0088a363ad8bf44ae289551.jpg)

![b85da28c1bc96f3492b094cd93b3e79c4ce66da94fc98b815289d9226afa19d2.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/b85da28c1bc96f3492b094cd93b3e79c4ce66da94fc98b815289d9226afa19d2.jpg)

![ba45762f89ab28e769f912327dcdd6dcaf5c5a44042e70672fa68d23b184b30d.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/ba45762f89ab28e769f912327dcdd6dcaf5c5a44042e70672fa68d23b184b30d.jpg)

![bc14421027ac44f44e6391a60f4cfc8740e636884077114d25ad1ae578ec8b35.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/bc14421027ac44f44e6391a60f4cfc8740e636884077114d25ad1ae578ec8b35.jpg)

![ca33005452f8737bac768386395443484d3a85899c074030f0dc3587a218ddb7.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/ca33005452f8737bac768386395443484d3a85899c074030f0dc3587a218ddb7.jpg)

![d2198a6c0e887f69b74d488b5431162abb04f97b1c466f9b12ba3c41b6b4a8ea.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/d2198a6c0e887f69b74d488b5431162abb04f97b1c466f9b12ba3c41b6b4a8ea.jpg)

![d26bfd4abe502e4f8c2fd5b15c8e41fb2eb59ca5d7a67e176afd4438f74d96d7.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/d26bfd4abe502e4f8c2fd5b15c8e41fb2eb59ca5d7a67e176afd4438f74d96d7.jpg)

![d95ad59305b2d95936e1d52302dbd9b9f367e1ef45a0206e9761a8d491e43269.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/d95ad59305b2d95936e1d52302dbd9b9f367e1ef45a0206e9761a8d491e43269.jpg)

![e04c893755c60954de231d1ac9f645c47a314d0db52a35fdf042f98d9ba3f2eb.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/e04c893755c60954de231d1ac9f645c47a314d0db52a35fdf042f98d9ba3f2eb.jpg)

![e35008e5a1f92d9fc5babc042226a4106c2b3e7751f141dc243963eb2052be14.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/e35008e5a1f92d9fc5babc042226a4106c2b3e7751f141dc243963eb2052be14.jpg)

![e4368b0eaa9cfcbfc7f69e4aab96312dcaedc677d78786749b5ae663856f0002.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/e4368b0eaa9cfcbfc7f69e4aab96312dcaedc677d78786749b5ae663856f0002.jpg)

![e9a867195a1c748ea17e3f3ba0cd670320315d72318c9bfc91371db9e029cd45.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/e9a867195a1c748ea17e3f3ba0cd670320315d72318c9bfc91371db9e029cd45.jpg)

![eb7bdf239009320277a6fc680dc08f3ea7c26df07d6b08870574810545b66a3d.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/eb7bdf239009320277a6fc680dc08f3ea7c26df07d6b08870574810545b66a3d.jpg)

![ee5a7136e0663f3267ce1ef5f2a8cd69c07516cb811a8b96a8448a2a4d575b17.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/ee5a7136e0663f3267ce1ef5f2a8cd69c07516cb811a8b96a8448a2a4d575b17.jpg)

![fcb873aadf586d2438fe8afe1ca7a3871b92d6ce65cb55539a38697a35b0e912.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/images/fcb873aadf586d2438fe8afe1ca7a3871b92d6ce65cb55539a38697a35b0e912.jpg)

### Tables

![2a51973131e968ee41bf16bd0eab523263e3d59dcaee2c373e1c7d7036180a2c.jpg](../iclr_results/562_Planning%20in%20Natural%20Language%20Improves%20LLM%20Search%20for%20Code%20Generation/tables/2a51973131e968ee41bf16bd0eab523263e3d59dcaee2c373e1c7d7036180a2c.jpg)

## On Quantizing Neural Representation for Variable-Rate Video Coding


### Images

![14cfc227907ffc3445d188887e4eb3dda1cf76dcb9df19e0b5027e49d90dd266.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/14cfc227907ffc3445d188887e4eb3dda1cf76dcb9df19e0b5027e49d90dd266.jpg)

![235c55d74235c25841022988a55ef2c63ea61667037703d86523069e9e1bec72.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/235c55d74235c25841022988a55ef2c63ea61667037703d86523069e9e1bec72.jpg)

![2c7379a657ee2579b7fd4bd39e8e2df6262a689289ddb52fc44b59f3d30618fb.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/2c7379a657ee2579b7fd4bd39e8e2df6262a689289ddb52fc44b59f3d30618fb.jpg)

![2cf92bd0e49ce59f8380cb51c722a0dc2ad81588e4076b05b6667eb8c31e0396.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/2cf92bd0e49ce59f8380cb51c722a0dc2ad81588e4076b05b6667eb8c31e0396.jpg)

![2fb4d129e33a6228c1ba699d14238363eed5431d32e323bee4f5be8ff483f0b1.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/2fb4d129e33a6228c1ba699d14238363eed5431d32e323bee4f5be8ff483f0b1.jpg)

![3478593e14cf4b0c584021d68027aa1cfa12094637def167e5e0b2b7b4b9ad26.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/3478593e14cf4b0c584021d68027aa1cfa12094637def167e5e0b2b7b4b9ad26.jpg)

![3ccdfaee04e1473ab2b80f30cbd9caad643537a2abcf05965ac7f024ea0f7939.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/3ccdfaee04e1473ab2b80f30cbd9caad643537a2abcf05965ac7f024ea0f7939.jpg)

![73f1efb95964b29fe62567845dcc87b92a2d6e9384e1c7032ce6bedcbc4b0022.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/73f1efb95964b29fe62567845dcc87b92a2d6e9384e1c7032ce6bedcbc4b0022.jpg)

![7e71722757320d699d4a8e78d02d40bc7b051981ea8e488007eaef203367a44f.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/7e71722757320d699d4a8e78d02d40bc7b051981ea8e488007eaef203367a44f.jpg)

![c5d1340834975f45bb4622b76df3480528846bd351ddf1b25c64d7831cb1e1c2.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/c5d1340834975f45bb4622b76df3480528846bd351ddf1b25c64d7831cb1e1c2.jpg)

![cd93b70779c42de6c6703520c857f7cae1e9220e28b1047cdf2ae2e7a3af325d.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/cd93b70779c42de6c6703520c857f7cae1e9220e28b1047cdf2ae2e7a3af325d.jpg)

![dae26874b61d351e70ad9a7814efd368d8eef0dcea5285688dfd351b49a48119.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/dae26874b61d351e70ad9a7814efd368d8eef0dcea5285688dfd351b49a48119.jpg)

![fa4d232512086fdf9041f22bbdd6743106bb20a268f53505738c302dd7664474.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/images/fa4d232512086fdf9041f22bbdd6743106bb20a268f53505738c302dd7664474.jpg)

### Tables

![26b5b68c5da1eed90a3553d0aaf05b485279149fd4d7a4893ce48c1520317eaa.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/26b5b68c5da1eed90a3553d0aaf05b485279149fd4d7a4893ce48c1520317eaa.jpg)

![35615d47226250d5fd8cad92edf539a98dae234db30990970d2cae7eb15647c6.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/35615d47226250d5fd8cad92edf539a98dae234db30990970d2cae7eb15647c6.jpg)

![62d71c9cdfbb990843d212c07683219977cc7cc54b8ffd3ca33136fc2130b99d.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/62d71c9cdfbb990843d212c07683219977cc7cc54b8ffd3ca33136fc2130b99d.jpg)

![77f8dcf79c11cad8cc8f6a593323c1b6a5b84c0863acdb94d8672f585ba02ce8.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/77f8dcf79c11cad8cc8f6a593323c1b6a5b84c0863acdb94d8672f585ba02ce8.jpg)

![83da88fa9dc916f660ee97e7f5fc3149769290ade742b45d9afd8e1d4215c94e.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/83da88fa9dc916f660ee97e7f5fc3149769290ade742b45d9afd8e1d4215c94e.jpg)

![9c97ce979c5a85722da06ba386b89ac7f6f14374e948f53bb29ac87939e7f4d6.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/9c97ce979c5a85722da06ba386b89ac7f6f14374e948f53bb29ac87939e7f4d6.jpg)

![e73481234b7986877338efb3c318932aa4f681639215062f12930afda3ed7328.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/e73481234b7986877338efb3c318932aa4f681639215062f12930afda3ed7328.jpg)

![eb553c956b24974f8532569b51c4a7a6c63aa9f398470f016c4cf173b6d0e299.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/eb553c956b24974f8532569b51c4a7a6c63aa9f398470f016c4cf173b6d0e299.jpg)

![eb9f1cd3d39a6ae349ef11b06bb8565ba4d61635338d609268e1436aabe983f8.jpg](../iclr_results/563_On%20Quantizing%20Neural%20Representation%20for%20Variable-Rate%20Video%20Coding/tables/eb9f1cd3d39a6ae349ef11b06bb8565ba4d61635338d609268e1436aabe983f8.jpg)

## DiffPuter: Empowering Diffusion Models for Missing Data Imputation


### Images

![49627742778f2d30583b09098eab62f4bf6c8167908f7f5750ebd2bbbf3f8fbb.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/images/49627742778f2d30583b09098eab62f4bf6c8167908f7f5750ebd2bbbf3f8fbb.jpg)

![56df6d4ef63353a1df67c2ffbf25322f95837d2ba945666e126e0f2224f7eec8.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/images/56df6d4ef63353a1df67c2ffbf25322f95837d2ba945666e126e0f2224f7eec8.jpg)

![bacc3427e5836443c808937da494ac87bcd68e4902fafa3a09e1f197e23024e8.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/images/bacc3427e5836443c808937da494ac87bcd68e4902fafa3a09e1f197e23024e8.jpg)

![c968d719e3c6ceb8f072cbe2185a1d705ca80940251f74879a6f8fe68cc25266.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/images/c968d719e3c6ceb8f072cbe2185a1d705ca80940251f74879a6f8fe68cc25266.jpg)

![f811ccf7600ecc6bf39445477f5d2d8100629172a162b5680c4691af471c0212.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/images/f811ccf7600ecc6bf39445477f5d2d8100629172a162b5680c4691af471c0212.jpg)

### Tables

![01ca0ec70d20e15e801977a500b3c3c9cba2455463125acf7e250dc6b471ed19.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/01ca0ec70d20e15e801977a500b3c3c9cba2455463125acf7e250dc6b471ed19.jpg)

![0d3dff440d401f6e000cafa6529baaf18f5cdafb11dd5542c4c67fd777d5f0f0.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/0d3dff440d401f6e000cafa6529baaf18f5cdafb11dd5542c4c67fd777d5f0f0.jpg)

![18b1bf6fae7da1eb9626e89e1595812c6b2447ee60818612e8c6fac3b9de2b62.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/18b1bf6fae7da1eb9626e89e1595812c6b2447ee60818612e8c6fac3b9de2b62.jpg)

![667c23e20f51db41b1e6225e720fed108bce36781f732ce17ff28733da397159.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/667c23e20f51db41b1e6225e720fed108bce36781f732ce17ff28733da397159.jpg)

![ad0da8511516e5e49cc6561fbb8ad0640b2f662c5b87064a3216f540e3fef253.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/ad0da8511516e5e49cc6561fbb8ad0640b2f662c5b87064a3216f540e3fef253.jpg)

![c348dd99ffeef2873ac51677678b6a667805b6741bf945f1538acd224036187b.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/c348dd99ffeef2873ac51677678b6a667805b6741bf945f1538acd224036187b.jpg)

![cf7010551193c6175c33a3ace2b115c10820ec6dcd715f1ffe8c53b0496001e8.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/cf7010551193c6175c33a3ace2b115c10820ec6dcd715f1ffe8c53b0496001e8.jpg)

![ddf07b0e88a9a9a0bfdc5967220d4735e497768464f92001d1e43a3b5da08f96.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/ddf07b0e88a9a9a0bfdc5967220d4735e497768464f92001d1e43a3b5da08f96.jpg)

![edb46d104df4a022026a73fae46651413ea7dd9f62c903b86e5b2c40abdb7abb.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/edb46d104df4a022026a73fae46651413ea7dd9f62c903b86e5b2c40abdb7abb.jpg)

![ff4a60ef78cf36f78e74d3c954c8553caae6824ca9d97741e6c1cd3838264d5f.jpg](../iclr_results/564_DiffPuter_%20Empowering%20Diffusion%20Models%20for%20Missing%20Data%20Imputation/tables/ff4a60ef78cf36f78e74d3c954c8553caae6824ca9d97741e6c1cd3838264d5f.jpg)

## What Does It Mean to Be a Transformer? Insights from a Theoretical Hessian Analysis


### Images

![07e47b78f01bd64717808a58530d6767c000ef12226b09257b0a0f6f29bc9d88.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/07e47b78f01bd64717808a58530d6767c000ef12226b09257b0a0f6f29bc9d88.jpg)

![1a9a1c44ed0320afb97123c2cde3862bf38649f1de465612e7e77441d757e3dc.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/1a9a1c44ed0320afb97123c2cde3862bf38649f1de465612e7e77441d757e3dc.jpg)

![434b61f68c32ef9d900e9df1260366c95ca6d7c17ff868a3d05ff3a6760a1a91.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/434b61f68c32ef9d900e9df1260366c95ca6d7c17ff868a3d05ff3a6760a1a91.jpg)

![6dd31defbf8ab6cd26302554d1f547c7cd4870d6e46e5d6c2a389ddb962cb39d.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/6dd31defbf8ab6cd26302554d1f547c7cd4870d6e46e5d6c2a389ddb962cb39d.jpg)

![7411bc120cf2b440960edb857be4aa5a0eeeb367058d51133bf441409fccc0b6.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/7411bc120cf2b440960edb857be4aa5a0eeeb367058d51133bf441409fccc0b6.jpg)

![b38a4712d5d839068dc906447ac4860be4d13f5591683f308fba59bd10bec5a4.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/b38a4712d5d839068dc906447ac4860be4d13f5591683f308fba59bd10bec5a4.jpg)

![de6388e1d6a12723989c0775dc6330129ea3d1fc0d384158bdedbeeaf90fdeb4.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/de6388e1d6a12723989c0775dc6330129ea3d1fc0d384158bdedbeeaf90fdeb4.jpg)

![f78368cc1007be168140b1bc1f2fed816732c240656503bde37adb4391288114.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/images/f78368cc1007be168140b1bc1f2fed816732c240656503bde37adb4391288114.jpg)

### Tables

![5688fd68463fc0accf95c791d4b1603fb4f7f306bdb0151611519fabe9ead346.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/tables/5688fd68463fc0accf95c791d4b1603fb4f7f306bdb0151611519fabe9ead346.jpg)

![b7e02b87908a9e7d8e8ccf826f817e7ae75549b3d64275c94be440dea6e814f7.jpg](../iclr_results/565_What%20Does%20It%20Mean%20to%20Be%20a%20Transformer_%20Insights%20from%20a%20Theoretical%20Hessian%20Analysis/tables/b7e02b87908a9e7d8e8ccf826f817e7ae75549b3d64275c94be440dea6e814f7.jpg)

## Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs with Semantic Space


### Images

![0bd0832d5a31898cdf7eb4cb04a1bd5d1456800d21b749c7691328af8c2faf6f.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/0bd0832d5a31898cdf7eb4cb04a1bd5d1456800d21b749c7691328af8c2faf6f.jpg)

![2ca9f87cd7854a62fd5a187969d19a2e167b295fb7406c50d74a4826145314a1.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/2ca9f87cd7854a62fd5a187969d19a2e167b295fb7406c50d74a4826145314a1.jpg)

![5d743c82f264af3f164f4cbc02c7be8a18e268fddd94a7d0b09edb0d5049b121.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/5d743c82f264af3f164f4cbc02c7be8a18e268fddd94a7d0b09edb0d5049b121.jpg)

![81e27f7d494a349a0d94706f8b8c490aaa4201e7d64bc92bc1028f7add8a8f22.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/81e27f7d494a349a0d94706f8b8c490aaa4201e7d64bc92bc1028f7add8a8f22.jpg)

![82bdc08d076bc4c51c3a45f4de98a2aa49a9be9ebe7916e9618adb19c5c7be2a.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/82bdc08d076bc4c51c3a45f4de98a2aa49a9be9ebe7916e9618adb19c5c7be2a.jpg)

![9cd605aae53b3f0fb19895d9f6e4f70a317fd1cf9261f806faec8e3ca6d88b03.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/9cd605aae53b3f0fb19895d9f6e4f70a317fd1cf9261f806faec8e3ca6d88b03.jpg)

![ad0ebd8477f5e928b7fc67bb988f3da17aaead16db772687c074eee2a0bc2235.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/ad0ebd8477f5e928b7fc67bb988f3da17aaead16db772687c074eee2a0bc2235.jpg)

![b4a5b084f9dcf686fe356aa1d2df318914c5b8fee53ac9f276408396a4c63bb3.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/b4a5b084f9dcf686fe356aa1d2df318914c5b8fee53ac9f276408396a4c63bb3.jpg)

![bf4aff0f5456a25152f79d2f8099117402b9536ec20af3f3d01ad58c238d2df7.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/bf4aff0f5456a25152f79d2f8099117402b9536ec20af3f3d01ad58c238d2df7.jpg)

![de67fefee480468406d2dc3c83ca534d074dbd0795b66af4f0f9b743ad12e0c2.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/de67fefee480468406d2dc3c83ca534d074dbd0795b66af4f0f9b743ad12e0c2.jpg)

![ed979ddde49b56021bf326d8ef5b7fc6262958caf6a55a303bea9ef00411f003.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/images/ed979ddde49b56021bf326d8ef5b7fc6262958caf6a55a303bea9ef00411f003.jpg)

### Tables

![3d95b5b65afe3e991167bc26780b95bb641914f1166d2476806d3ac502b0ef66.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/tables/3d95b5b65afe3e991167bc26780b95bb641914f1166d2476806d3ac502b0ef66.jpg)

![693a18c2da3d4b65ec93c6c6871b1935c0d2878fdeec8309dd15fc223b407928.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/tables/693a18c2da3d4b65ec93c6c6871b1935c0d2878fdeec8309dd15fc223b407928.jpg)

![6d04bc450f28ba5b8df314a1785e34fc271283ec44af1b3ae940917582f820c3.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/tables/6d04bc450f28ba5b8df314a1785e34fc271283ec44af1b3ae940917582f820c3.jpg)

![776ff86c6c9d498ac689cad7333009b473e5075cc23c9a525a8520aa95b1ff21.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/tables/776ff86c6c9d498ac689cad7333009b473e5075cc23c9a525a8520aa95b1ff21.jpg)

![be15cc8d6297c79fd1964635c66f2f4acd5e4ce402e87f5f758eb9c4718d2719.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/tables/be15cc8d6297c79fd1964635c66f2f4acd5e4ce402e87f5f758eb9c4718d2719.jpg)

![bf1508814d1c5f8a53f6d127e16d14654bea645a2f9513815020d6537d9f9ad4.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/tables/bf1508814d1c5f8a53f6d127e16d14654bea645a2f9513815020d6537d9f9ad4.jpg)

![c5c68a8dda2f01a10321b7eaa01388e7d2da99147970e17eb4d16085141bec23.jpg](../iclr_results/566_Broaden%20your%20SCOPE%21%20Efficient%20Multi-turn%20Conversation%20Planning%20for%20LLMs%20with%20Semantic%20Space/tables/c5c68a8dda2f01a10321b7eaa01388e7d2da99147970e17eb4d16085141bec23.jpg)

## LeFusion: Controllable Pathology Synthesis via Lesion-Focused Diffusion Models


### Images

![02b0fe9d3c9a0b6fb834675b640ba38d2b8ecd0d5c0f2e9779960818158551a5.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/02b0fe9d3c9a0b6fb834675b640ba38d2b8ecd0d5c0f2e9779960818158551a5.jpg)

![054b7d6e993bd3e031a533bd0353e7be1250a13d1056011e22a226c6fe531362.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/054b7d6e993bd3e031a533bd0353e7be1250a13d1056011e22a226c6fe531362.jpg)

![2a676441f9b9cd7372f87ce5a32fab2bffae524ed0ef7363fd52679c95de6d52.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/2a676441f9b9cd7372f87ce5a32fab2bffae524ed0ef7363fd52679c95de6d52.jpg)

![37cf44a42fb1fa2deac2c06e4bf4665c4355eba86e37fcfbf6e71d73922ec7cb.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/37cf44a42fb1fa2deac2c06e4bf4665c4355eba86e37fcfbf6e71d73922ec7cb.jpg)

![4a4b5d3f4f8bee219f680022f64f5eaff574932101be6c3cefe7f84018cbe74d.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/4a4b5d3f4f8bee219f680022f64f5eaff574932101be6c3cefe7f84018cbe74d.jpg)

![6b3822924d63c9a66942bdc5261a614285aad0f714f6495a068c8db2ce74df3e.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/6b3822924d63c9a66942bdc5261a614285aad0f714f6495a068c8db2ce74df3e.jpg)

![6de0d99afae7b044ddacf5492464da807664c4dd71d919fac5ba5b1a764601b1.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/6de0d99afae7b044ddacf5492464da807664c4dd71d919fac5ba5b1a764601b1.jpg)

![73b4ee1068025eb2fbe0370273d91135c1f4d4b66aaa5278520972975f8ff121.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/73b4ee1068025eb2fbe0370273d91135c1f4d4b66aaa5278520972975f8ff121.jpg)

![79a67a867ca28a6c52d86934c71b7ca370fd9d0a5212ab73c885b56b84daa084.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/79a67a867ca28a6c52d86934c71b7ca370fd9d0a5212ab73c885b56b84daa084.jpg)

![9fbd4591dc91b3aded240c893e7455c09d384bb6acbcbe6295cb2468397aae59.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/9fbd4591dc91b3aded240c893e7455c09d384bb6acbcbe6295cb2468397aae59.jpg)

![b0de3f592283bd92163301e0ceef58595a170439176fe9284575729d504bc7c9.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/b0de3f592283bd92163301e0ceef58595a170439176fe9284575729d504bc7c9.jpg)

![ba92217f9e9a047affa25d201776e8f8813037ca6616d39a41f46f149c8285c6.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/ba92217f9e9a047affa25d201776e8f8813037ca6616d39a41f46f149c8285c6.jpg)

![c24d55004931392907dce9a83fefb84658e94bbfd6259d2105f5bbcb8feb41cd.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/c24d55004931392907dce9a83fefb84658e94bbfd6259d2105f5bbcb8feb41cd.jpg)

![c9b5e2f09c7c1f2ecb5dbe01421331629af8467cb04c121605958f743b22dc9a.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/c9b5e2f09c7c1f2ecb5dbe01421331629af8467cb04c121605958f743b22dc9a.jpg)

![cf2a994b7a71859f82d539342f9b7b5cc8b52a1a1e05424c823719bf56e3eb1c.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/images/cf2a994b7a71859f82d539342f9b7b5cc8b52a1a1e05424c823719bf56e3eb1c.jpg)

### Tables

![029bac3f8256084d5a3aec02fc1632bd9472f04ec480cab3a6162e33a50ec010.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/tables/029bac3f8256084d5a3aec02fc1632bd9472f04ec480cab3a6162e33a50ec010.jpg)

![0569ed4aa4e2791770ab4fd9b11b727b47aab41624b606e8afca61d5ad44846d.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/tables/0569ed4aa4e2791770ab4fd9b11b727b47aab41624b606e8afca61d5ad44846d.jpg)

![17bbf87726f1b4427a7bef7a358ae2f7c6a75b716b16aa44f37ab822cb601bbe.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/tables/17bbf87726f1b4427a7bef7a358ae2f7c6a75b716b16aa44f37ab822cb601bbe.jpg)

![60350df1380f74fc8f0ffe099214d91dc7653556e19fcebd997f7208afb26e3d.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/tables/60350df1380f74fc8f0ffe099214d91dc7653556e19fcebd997f7208afb26e3d.jpg)

![b197321f8dedba65623ce455f7281b9aa4f8c3c1a2a694a3550cfc9efd9b0e6a.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/tables/b197321f8dedba65623ce455f7281b9aa4f8c3c1a2a694a3550cfc9efd9b0e6a.jpg)

![b6d904821b3ae62bbe71a80618959441dbc3bdf9e7f3532a972e4a4965a4d45c.jpg](../iclr_results/567_LeFusion_%20Controllable%20Pathology%20Synthesis%20via%20Lesion-Focused%20Diffusion%20Models/tables/b6d904821b3ae62bbe71a80618959441dbc3bdf9e7f3532a972e4a4965a4d45c.jpg)

## Multi-Field Adaptive Retrieval


### Images

![31f746fecc81d50e74fdc54fda1c7f6bc042bc5290e8ab370b70814c4789c07c.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/images/31f746fecc81d50e74fdc54fda1c7f6bc042bc5290e8ab370b70814c4789c07c.jpg)

### Tables

![08118c574624af610d4ac87faa548804e275876d0ba6180ca0fe1405c1f5c62b.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/08118c574624af610d4ac87faa548804e275876d0ba6180ca0fe1405c1f5c62b.jpg)

![15728516cc73100d11cba712cf11083e0fdaa7a78875345e9ac3ac0de48cdcb6.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/15728516cc73100d11cba712cf11083e0fdaa7a78875345e9ac3ac0de48cdcb6.jpg)

![1726cb88d5a01a787c3d49f6d9bd13407eecb37d25b1d25fb2b2e41f7162ded3.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/1726cb88d5a01a787c3d49f6d9bd13407eecb37d25b1d25fb2b2e41f7162ded3.jpg)

![1d394795b3acbfd8bb2f76a2494ea8d78d72013fcfba89a7222ea55e135ecff4.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/1d394795b3acbfd8bb2f76a2494ea8d78d72013fcfba89a7222ea55e135ecff4.jpg)

![1d9e5faeded2913abb19e94159d9dec5132b972b46047bd0f9a0a0fb485d54bd.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/1d9e5faeded2913abb19e94159d9dec5132b972b46047bd0f9a0a0fb485d54bd.jpg)

![278f45eea6a84624749ebd92ec74902bd469e2b8c42c2691c48c528b4562c352.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/278f45eea6a84624749ebd92ec74902bd469e2b8c42c2691c48c528b4562c352.jpg)

![377e15d470f27c6c35fbeb3454cef47cb8e17754a0116713dcc3dadb108da58b.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/377e15d470f27c6c35fbeb3454cef47cb8e17754a0116713dcc3dadb108da58b.jpg)

![445be262f47dc30d14cc3a1b6cea778938b3cc15b283a41d57987e51c9341125.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/445be262f47dc30d14cc3a1b6cea778938b3cc15b283a41d57987e51c9341125.jpg)

![506f302a55993d68f2c14b29c53dffc779b3b6754d58ecda445704d7929afb72.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/506f302a55993d68f2c14b29c53dffc779b3b6754d58ecda445704d7929afb72.jpg)

![6b45330b3703739e5053252ceaf13aa67f00bf6b3e6e0a34a55d53f6cbdce11f.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/6b45330b3703739e5053252ceaf13aa67f00bf6b3e6e0a34a55d53f6cbdce11f.jpg)

![7a7a4e6f1026c439d256a036d742488d0bd1b07553eda4cf4b481d14e74aee4c.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/7a7a4e6f1026c439d256a036d742488d0bd1b07553eda4cf4b481d14e74aee4c.jpg)

![8757ce78da94bac3c26e050e27d50916b8a12c2a4031cc20c48ef68deb980da9.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/8757ce78da94bac3c26e050e27d50916b8a12c2a4031cc20c48ef68deb980da9.jpg)

![af611ef08e74406e0d010c80b2e4941fcd2bf2bcbee704b1cbc26c9e27d845bf.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/af611ef08e74406e0d010c80b2e4941fcd2bf2bcbee704b1cbc26c9e27d845bf.jpg)

![b154f6a5c0c05d78b0079bf6df312a9f571975ab64862ef70f7842af70e81970.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/b154f6a5c0c05d78b0079bf6df312a9f571975ab64862ef70f7842af70e81970.jpg)

![bc0807fe6ed349ac52946bb65d500301a7bca441940b1dea1d8994eefeb5197d.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/bc0807fe6ed349ac52946bb65d500301a7bca441940b1dea1d8994eefeb5197d.jpg)

![bfd58a28bf1eff44a183436baf83c21c81698a6968b52cb13244d0bf65d52466.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/bfd58a28bf1eff44a183436baf83c21c81698a6968b52cb13244d0bf65d52466.jpg)

![ddd90bd777b4f6c7c85e1086a81b0e724b02e5c14a3c8a2d1bfddea6a529acb1.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/ddd90bd777b4f6c7c85e1086a81b0e724b02e5c14a3c8a2d1bfddea6a529acb1.jpg)

![eeb50c484297463d62fc3ad232425cfee58850bf9c7d0f60db28eb769956b34d.jpg](../iclr_results/568_Multi-Field%20Adaptive%20Retrieval/tables/eeb50c484297463d62fc3ad232425cfee58850bf9c7d0f60db28eb769956b34d.jpg)

## RelitLRM: Generative Relightable Radiance for Large Reconstruction Models


### Images

![1ce20f5b3b555939cc209818a0f90392811c6a1fbadecda8c1d074ccdbbb909e.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/1ce20f5b3b555939cc209818a0f90392811c6a1fbadecda8c1d074ccdbbb909e.jpg)

![1d415d7b3b705a458371ddc62f7a59177fdb9b2691d1ecc503da81aeabf45b48.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/1d415d7b3b705a458371ddc62f7a59177fdb9b2691d1ecc503da81aeabf45b48.jpg)

![24296aee078beb4753169d30c22ea3d0d1f9fd8c679b32cf1d39e088cb68074b.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/24296aee078beb4753169d30c22ea3d0d1f9fd8c679b32cf1d39e088cb68074b.jpg)

![26a05925a9cd79b47a72bd0931f0d0c8b693f739395908a1492fdd8e95e0ba78.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/26a05925a9cd79b47a72bd0931f0d0c8b693f739395908a1492fdd8e95e0ba78.jpg)

![2bac7da85266a0b98696100adbc29ae76f4d3fe8389d1111471ab2663b08315c.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/2bac7da85266a0b98696100adbc29ae76f4d3fe8389d1111471ab2663b08315c.jpg)

![3a146d4af3a23a74c0c8cb698f9cefb5e9c160ef974b9be8caded388855715a3.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/3a146d4af3a23a74c0c8cb698f9cefb5e9c160ef974b9be8caded388855715a3.jpg)

![60be67a436d6fc0c6f8307ec270f2c1455b0d18c8b441fec24ea4683f44dd4aa.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/60be67a436d6fc0c6f8307ec270f2c1455b0d18c8b441fec24ea4683f44dd4aa.jpg)

![67026b5f136dab8e9d6a41f69261d8880375c1df9787256391ce1bfe60108a8d.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/67026b5f136dab8e9d6a41f69261d8880375c1df9787256391ce1bfe60108a8d.jpg)

![8dffdcf3cc8826185a3d4f7baa1a0070541c743d094d55806bfcb89fef35dbaf.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/8dffdcf3cc8826185a3d4f7baa1a0070541c743d094d55806bfcb89fef35dbaf.jpg)

![8f55ed1e035e5d2b3d158f6c28dab229f8ff70d54859e64c6d6d1e2448a65d60.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/8f55ed1e035e5d2b3d158f6c28dab229f8ff70d54859e64c6d6d1e2448a65d60.jpg)

![c01fa90832bea85dfda50eadb16296f2c9104369622bea09a599f9655a30f8ef.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/c01fa90832bea85dfda50eadb16296f2c9104369622bea09a599f9655a30f8ef.jpg)

![d547f54e619df7c779d2318f6a861a5905461a6a4a0999f872469a85bf6f9c2e.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/d547f54e619df7c779d2318f6a861a5905461a6a4a0999f872469a85bf6f9c2e.jpg)

![dd1fb4933cb90eb4b8001132d605b1ec949000a25fa4f75d464f43c7ebe24683.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/images/dd1fb4933cb90eb4b8001132d605b1ec949000a25fa4f75d464f43c7ebe24683.jpg)

### Tables

![16efba63eb60211161dc1986cfbfa2d797651f7f31b4b10b38b31a58881968a7.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/16efba63eb60211161dc1986cfbfa2d797651f7f31b4b10b38b31a58881968a7.jpg)

![1fe2472ae1a267495091c66a5de71bbcc324693ac7e0c263c39cbb3f9042e0fa.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/1fe2472ae1a267495091c66a5de71bbcc324693ac7e0c263c39cbb3f9042e0fa.jpg)

![2f05fff28a3bbdeab72f3e4efa6747db13327143af541b2f17279b7787911f6a.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/2f05fff28a3bbdeab72f3e4efa6747db13327143af541b2f17279b7787911f6a.jpg)

![431b15d062fb2315c1da1717d225ca6c0bf22842870f0e7043d4bc849356b41b.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/431b15d062fb2315c1da1717d225ca6c0bf22842870f0e7043d4bc849356b41b.jpg)

![7b4460ebb46769cd2ed4dbc6a48cf777702952cea0a6b31b6eb0fc5b19d7e386.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/7b4460ebb46769cd2ed4dbc6a48cf777702952cea0a6b31b6eb0fc5b19d7e386.jpg)

![89da187cbb84de796651aa168a8b54ed6867ea974a6a82c9a792b476c1a85b6d.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/89da187cbb84de796651aa168a8b54ed6867ea974a6a82c9a792b476c1a85b6d.jpg)

![9caf0f9427d516607de8cfdf0f69e09f411fad877debaf32d58551271d02c5db.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/9caf0f9427d516607de8cfdf0f69e09f411fad877debaf32d58551271d02c5db.jpg)

![e196bf9a1a435705bd7989d9ac040d5427bd752481b69e7098c5ef7dcc9be0af.jpg](../iclr_results/569_RelitLRM_%20Generative%20Relightable%20Radiance%20for%20Large%20Reconstruction%20Models/tables/e196bf9a1a435705bd7989d9ac040d5427bd752481b69e7098c5ef7dcc9be0af.jpg)

## Learning Spatiotemporal Dynamical Systems from Point Process Observations


### Images

![16f3036b552030768ab538aade50fa0e30fe2a2d7fe12b96ab616f3dda9c126e.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/16f3036b552030768ab538aade50fa0e30fe2a2d7fe12b96ab616f3dda9c126e.jpg)

![183627bb5de4dc60b34b49789d9d6eb4559cd4c1ba91c56a327a1eb56da7f1da.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/183627bb5de4dc60b34b49789d9d6eb4559cd4c1ba91c56a327a1eb56da7f1da.jpg)

![1f8d8b417dce3fa7da664326eca89de00c3fa6992a6fe13b4742551cce8cf906.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/1f8d8b417dce3fa7da664326eca89de00c3fa6992a6fe13b4742551cce8cf906.jpg)

![2607bbda8e4bf17223ad0494ecf65f531153aa563627aebb761527ca10549199.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/2607bbda8e4bf17223ad0494ecf65f531153aa563627aebb761527ca10549199.jpg)

![35a96d2d9b9a02cc9d822ee58d584baf02c6a838c77fca9e10ee81abd761ee17.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/35a96d2d9b9a02cc9d822ee58d584baf02c6a838c77fca9e10ee81abd761ee17.jpg)

![3adc712466a3705fb755af1419e235fd115ddeaa124e831bd15c027b80230743.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/3adc712466a3705fb755af1419e235fd115ddeaa124e831bd15c027b80230743.jpg)

![4d711d46d96908ec5462dc90a421ab1994f1b5a0a3d78b0fc372037aa59ba86e.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/4d711d46d96908ec5462dc90a421ab1994f1b5a0a3d78b0fc372037aa59ba86e.jpg)

![6626fd6d2d9d89fef35c21624c3adf39f6e97599643546cea429ab3078f910fc.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/6626fd6d2d9d89fef35c21624c3adf39f6e97599643546cea429ab3078f910fc.jpg)

![bf177b4ec17b8aade1aa9b5e6b4de638b617bdedf475bd83aff89ca9e7e994a8.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/images/bf177b4ec17b8aade1aa9b5e6b4de638b617bdedf475bd83aff89ca9e7e994a8.jpg)

### Tables

![03dcc5cabcfda051f886659ee774b2226a999c782845039d29228a1cca97fd1c.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/tables/03dcc5cabcfda051f886659ee774b2226a999c782845039d29228a1cca97fd1c.jpg)

![3b0f46c3681ee038e22e59699bf1f8557607c85162df5e36fef37e4f9424f070.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/tables/3b0f46c3681ee038e22e59699bf1f8557607c85162df5e36fef37e4f9424f070.jpg)

![88b938c9a83cb2a4eafb47ecaab0070e22b6869f2fc064d21e32629b5ad9ec90.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/tables/88b938c9a83cb2a4eafb47ecaab0070e22b6869f2fc064d21e32629b5ad9ec90.jpg)

![f34592183aeab6ca0f7597f18c2a40be487ec3eb5e37302aae85a9ca906faa75.jpg](../iclr_results/570_Learning%20Spatiotemporal%20Dynamical%20Systems%20from%20Point%20Process%20Observations/tables/f34592183aeab6ca0f7597f18c2a40be487ec3eb5e37302aae85a9ca906faa75.jpg)

## uniINF: Best-of-Both-Worlds Algorithm for Parameter-Free Heavy-Tailed MABs


### Tables

![524c1b078c62c947a089aa93bce62863ad6e3a328bb775722cb7203327de513d.jpg](../iclr_results/571_uniINF_%20Best-of-Both-Worlds%20Algorithm%20for%20Parameter-Free%20Heavy-Tailed%20MABs/tables/524c1b078c62c947a089aa93bce62863ad6e3a328bb775722cb7203327de513d.jpg)

![57d524052b51c91eb74d94a526e53ab2e570f6d9419a59858f84bca928807091.jpg](../iclr_results/571_uniINF_%20Best-of-Both-Worlds%20Algorithm%20for%20Parameter-Free%20Heavy-Tailed%20MABs/tables/57d524052b51c91eb74d94a526e53ab2e570f6d9419a59858f84bca928807091.jpg)

## The Superposition of Diffusion Models Using the Itô Density Estimator


### Images

![0bf5f11094df618610411ac25285934e1b65179dfaf88a44f8e2c5ed752aff57.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/0bf5f11094df618610411ac25285934e1b65179dfaf88a44f8e2c5ed752aff57.jpg)

![222f318a5955ee43838b11b2b0e1513f56f8cde3018186e3abd3c08785841b40.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/222f318a5955ee43838b11b2b0e1513f56f8cde3018186e3abd3c08785841b40.jpg)

![26585a2b1ae791505c27af18849d33ef88f93d838d56ecdb5be9d9e3f7692e19.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/26585a2b1ae791505c27af18849d33ef88f93d838d56ecdb5be9d9e3f7692e19.jpg)

![300ac63484ee503a3b695c6c1d17c68a7382cfe218c0c54891255a550d00b3b7.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/300ac63484ee503a3b695c6c1d17c68a7382cfe218c0c54891255a550d00b3b7.jpg)

![3a10502122890d696b43b7b7ae35dff3a607ec084ccbf8e93c879c1391754bb6.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/3a10502122890d696b43b7b7ae35dff3a607ec084ccbf8e93c879c1391754bb6.jpg)

![4c24225bbcddd707ae078e8a03c97a4bf04fdbc7444465f4b520b7f9b96ed025.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/4c24225bbcddd707ae078e8a03c97a4bf04fdbc7444465f4b520b7f9b96ed025.jpg)

![5386f94fef5937849f909c4d2487acb34a5fa68a37c549598c42a40fe4acc1ea.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/5386f94fef5937849f909c4d2487acb34a5fa68a37c549598c42a40fe4acc1ea.jpg)

![5452d8a47f1dd5e3af50a1f3b99df2acff7b686aa20c03cfbb1fbd7a21fe7e8e.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/5452d8a47f1dd5e3af50a1f3b99df2acff7b686aa20c03cfbb1fbd7a21fe7e8e.jpg)

![823c4ca029d2173cefe5d466d316f4a7e50a88c8ca312baabf65c7465259a37d.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/823c4ca029d2173cefe5d466d316f4a7e50a88c8ca312baabf65c7465259a37d.jpg)

![8870d73bcef86f1349a8c5541617b7a99da6701c9fab08d24834037104ce47ee.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/8870d73bcef86f1349a8c5541617b7a99da6701c9fab08d24834037104ce47ee.jpg)

![8893fab141ca645327e92fdf83322640caced896d3db0c34f4a027243822e80d.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/8893fab141ca645327e92fdf83322640caced896d3db0c34f4a027243822e80d.jpg)

![93e680d7e05979ff5d5a73d4e19694e71f728ac882a6489879045c5b52b7252d.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/93e680d7e05979ff5d5a73d4e19694e71f728ac882a6489879045c5b52b7252d.jpg)

![97e4c76ff4ac859da9373df0e47f88bdca8a6cfc4c73e766e51818f1e9dc2e37.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/97e4c76ff4ac859da9373df0e47f88bdca8a6cfc4c73e766e51818f1e9dc2e37.jpg)

![9d0608621893786f8e9ec80651c566719d58febe83fb90431387fb980a26c429.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/9d0608621893786f8e9ec80651c566719d58febe83fb90431387fb980a26c429.jpg)

![9f0fe317e704a8e695ea184d763bbffacc26dc68f25cc5ce268cae4ca0c8c5ea.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/9f0fe317e704a8e695ea184d763bbffacc26dc68f25cc5ce268cae4ca0c8c5ea.jpg)

![a0fa67b8c39221d89d4c342cd24df1f496436d3cc4afefa752d0b0af2d8df498.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/a0fa67b8c39221d89d4c342cd24df1f496436d3cc4afefa752d0b0af2d8df498.jpg)

![a42d147d2c21857c763a0bcb94a6b11230e1d1c93c32130af7eeeb5bc3e6625b.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/a42d147d2c21857c763a0bcb94a6b11230e1d1c93c32130af7eeeb5bc3e6625b.jpg)

![a52f062bfb255732a71d79abd843d57e9ebdda10924fb6c88350fd71692f71a5.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/a52f062bfb255732a71d79abd843d57e9ebdda10924fb6c88350fd71692f71a5.jpg)

![ab273206c40407bea8cc5295cba4a436c0abc3a345e56ce5e6b798c6577101f6.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/ab273206c40407bea8cc5295cba4a436c0abc3a345e56ce5e6b798c6577101f6.jpg)

![ac255363c45b2aaad5f4930938edf018df33ece8ba1e9b3fb43960809bc15d9c.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/ac255363c45b2aaad5f4930938edf018df33ece8ba1e9b3fb43960809bc15d9c.jpg)

![b72c7308e098aa02d13329f0db71ab0339c8ebf3c382c1a0d8db2a46da67f510.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/b72c7308e098aa02d13329f0db71ab0339c8ebf3c382c1a0d8db2a46da67f510.jpg)

![c18f105422835bec057e6e4aa9de750ac5540d77db5f710dfebf1e758abd742e.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/c18f105422835bec057e6e4aa9de750ac5540d77db5f710dfebf1e758abd742e.jpg)

![c3e21b99f63c507458336503801a26d4b41fccd327ada5786f539a49900fa679.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/c3e21b99f63c507458336503801a26d4b41fccd327ada5786f539a49900fa679.jpg)

![c9a14acc0e67ab661605ac3dbe22c089730c35ffa6b68ebec921c5eac9dd8b86.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/c9a14acc0e67ab661605ac3dbe22c089730c35ffa6b68ebec921c5eac9dd8b86.jpg)

![e4f3f449cfe278254cdc364a0f120d10bba55187084cf1d07be0a09d5a27450e.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/e4f3f449cfe278254cdc364a0f120d10bba55187084cf1d07be0a09d5a27450e.jpg)

![eee4fc9aa9d1bca5863399999e518ee6caed7db0e019f47d693a3e42487d45cb.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/eee4fc9aa9d1bca5863399999e518ee6caed7db0e019f47d693a3e42487d45cb.jpg)

![f35417d5d5a09be0502560ebaf18b9f66d3e4c267f1ff614dc2230d94d99c207.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/f35417d5d5a09be0502560ebaf18b9f66d3e4c267f1ff614dc2230d94d99c207.jpg)

![fa1a41be03b4f46092724bd6bf1b386cc6b97ad6a06d87e16d2d84cba7956da5.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/fa1a41be03b4f46092724bd6bf1b386cc6b97ad6a06d87e16d2d84cba7956da5.jpg)

![fa4e6eb445b3756b4eb58316d86fb3d917a7d38d3d61b3cc8840b1a911cada7d.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/fa4e6eb445b3756b4eb58316d86fb3d917a7d38d3d61b3cc8840b1a911cada7d.jpg)

![fc925347b6cea5765088b0ae6a82bde53cde58e0065ee5ad822a60ed8e8817b2.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/images/fc925347b6cea5765088b0ae6a82bde53cde58e0065ee5ad822a60ed8e8817b2.jpg)

### Tables

![4b4533a35997417b4240f5f259ee972e97c58117621a4ad52cb3ba00e225c11a.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/tables/4b4533a35997417b4240f5f259ee972e97c58117621a4ad52cb3ba00e225c11a.jpg)

![788934d1b341160426aa2bf242f77739d3a0d28aa6633935456dfbb3cd24c23c.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/tables/788934d1b341160426aa2bf242f77739d3a0d28aa6633935456dfbb3cd24c23c.jpg)

![9a4db69c590fa3221bcd2dfc95d1b05d6eb2ca8e911308049f895e8b70bc307d.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/tables/9a4db69c590fa3221bcd2dfc95d1b05d6eb2ca8e911308049f895e8b70bc307d.jpg)

![aa240b36f311652fb2b7f227b7194300a8d2d789350840ded083e79b29f17af4.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/tables/aa240b36f311652fb2b7f227b7194300a8d2d789350840ded083e79b29f17af4.jpg)

![f30789b0e18e036623ec2721d7bd10a465063b6e7c267f61d03f89d78ca48b42.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/tables/f30789b0e18e036623ec2721d7bd10a465063b6e7c267f61d03f89d78ca48b42.jpg)

![fac714230b9d7b15da5d90c22081915b48c4bfee0b00eacb77fa8c2aa1e8e417.jpg](../iclr_results/572_The%20Superposition%20of%20Diffusion%20Models%20Using%20the%20It%C3%B4%20Density%20Estimator/tables/fac714230b9d7b15da5d90c22081915b48c4bfee0b00eacb77fa8c2aa1e8e417.jpg)

## Towards hyperparameter-free optimization with differential privacy


### Images

![0d1d76172d3a7144098296d3ab9610c26160b231cc9baaf3330733430fa03973.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/0d1d76172d3a7144098296d3ab9610c26160b231cc9baaf3330733430fa03973.jpg)

![49fcdc0635801225020ed034545a97aa9aaf88e6682e0ca147205e6de03256f3.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/49fcdc0635801225020ed034545a97aa9aaf88e6682e0ca147205e6de03256f3.jpg)

![65bbc416590c523ef13300e5711a78c96a2fa6c92fd33946ff75c6ab336416d5.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/65bbc416590c523ef13300e5711a78c96a2fa6c92fd33946ff75c6ab336416d5.jpg)

![76deb7e196c06e6d66b932772330f8918a5aefc82cd8f33f588d207bf5eba8e2.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/76deb7e196c06e6d66b932772330f8918a5aefc82cd8f33f588d207bf5eba8e2.jpg)

![ac97fc1f6177347314222e16bcd32deef981b53d7e64aeb4863ce9ffe43b00e7.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/ac97fc1f6177347314222e16bcd32deef981b53d7e64aeb4863ce9ffe43b00e7.jpg)

![b04c26f64076f9142b29b362a4bfe2e372191b9f0b2f91e50a997961593cba24.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/b04c26f64076f9142b29b362a4bfe2e372191b9f0b2f91e50a997961593cba24.jpg)

![dcd987d1ac2b9e731f79866d5ac8138365d2714efa7e8f786e0adbf3cac7813d.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/dcd987d1ac2b9e731f79866d5ac8138365d2714efa7e8f786e0adbf3cac7813d.jpg)

![e4a7a734741ba84147f7153b548a0a379ae2f23f139063ed325d606ea1eb7f3c.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/images/e4a7a734741ba84147f7153b548a0a379ae2f23f139063ed325d606ea1eb7f3c.jpg)

### Tables

![06f309ec31ecd632098660c35fdcdcdd4d07a07a7626310d9b4265373dd811aa.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/06f309ec31ecd632098660c35fdcdcdd4d07a07a7626310d9b4265373dd811aa.jpg)

![24433241303468c1c75fbd580dc922a85889c151c646ab4db3a71817af3d5d22.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/24433241303468c1c75fbd580dc922a85889c151c646ab4db3a71817af3d5d22.jpg)

![2b660347c5b857626a4eb92c846cbcb44ddad83930ef7bf51eb20191c288ec52.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/2b660347c5b857626a4eb92c846cbcb44ddad83930ef7bf51eb20191c288ec52.jpg)

![4a1b074573810a46199e90756d67db5f051a3fa1560018b6b32517580646348b.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/4a1b074573810a46199e90756d67db5f051a3fa1560018b6b32517580646348b.jpg)

![59c879e2197e8f35d62f2ce452fece1e67a6e21a13b1b1cb76c67469ef92e46a.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/59c879e2197e8f35d62f2ce452fece1e67a6e21a13b1b1cb76c67469ef92e46a.jpg)

![6c69044fcc609ecbc0d2db0b4689f44bfd30d1862fd8dfb37a373faca0a9b089.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/6c69044fcc609ecbc0d2db0b4689f44bfd30d1862fd8dfb37a373faca0a9b089.jpg)

![881cd6617a53a0033a83e111bb57cdc62688ce67642fed33a51ddcace03bff8b.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/881cd6617a53a0033a83e111bb57cdc62688ce67642fed33a51ddcace03bff8b.jpg)

![db195b7d23df56b086b627d7990f4e921411c704cd701f351082e4c9b36f83ef.jpg](../iclr_results/573_Towards%20hyperparameter-free%20optimization%20with%20differential%20privacy/tables/db195b7d23df56b086b627d7990f4e921411c704cd701f351082e4c9b36f83ef.jpg)

## Discovering Temporally Compositional Neural Manifolds with Switching Infinite GPFA


### Images

![07b0a5c8432f745a2b66f58248c56ad7925d0877cec46cc2769e3af3ad326896.jpg](../iclr_results/574_Discovering%20Temporally%20Compositional%20Neural%20Manifolds%20with%20Switching%20Infinite%20GPFA/images/07b0a5c8432f745a2b66f58248c56ad7925d0877cec46cc2769e3af3ad326896.jpg)

![0b687cc5a44fb94cffbd1e1e55944dc2795f304cc7eaa043cea42e54db69e67a.jpg](../iclr_results/574_Discovering%20Temporally%20Compositional%20Neural%20Manifolds%20with%20Switching%20Infinite%20GPFA/images/0b687cc5a44fb94cffbd1e1e55944dc2795f304cc7eaa043cea42e54db69e67a.jpg)

![2a63edc5c86f82e1d22df7b3440b768b8be01a6b4f735958836b1c278fe1d4e4.jpg](../iclr_results/574_Discovering%20Temporally%20Compositional%20Neural%20Manifolds%20with%20Switching%20Infinite%20GPFA/images/2a63edc5c86f82e1d22df7b3440b768b8be01a6b4f735958836b1c278fe1d4e4.jpg)

![69deda04532e26e09346cc15d4f1877935da9c484516f17a106fcfcc2848d19c.jpg](../iclr_results/574_Discovering%20Temporally%20Compositional%20Neural%20Manifolds%20with%20Switching%20Infinite%20GPFA/images/69deda04532e26e09346cc15d4f1877935da9c484516f17a106fcfcc2848d19c.jpg)

![730eae95469100d672debdc9e31da36a86dcb6a5b7a15751eafb31c354729366.jpg](../iclr_results/574_Discovering%20Temporally%20Compositional%20Neural%20Manifolds%20with%20Switching%20Infinite%20GPFA/images/730eae95469100d672debdc9e31da36a86dcb6a5b7a15751eafb31c354729366.jpg)

![efc6e05071e2f79e6abe5cd9692814f2669748314bb5a04afedbeb8ca3f6905d.jpg](../iclr_results/574_Discovering%20Temporally%20Compositional%20Neural%20Manifolds%20with%20Switching%20Infinite%20GPFA/images/efc6e05071e2f79e6abe5cd9692814f2669748314bb5a04afedbeb8ca3f6905d.jpg)

## DeepRTL: Bridging Verilog Understanding and Generation with a Unified Representation Model


### Images

![24a5d887ac2c94c442ddc2c7b6dd2bedda5e5fcd2f981ad44cea71e6a4a520d3.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/24a5d887ac2c94c442ddc2c7b6dd2bedda5e5fcd2f981ad44cea71e6a4a520d3.jpg)

![698cdb455eac90da051e23356a66390b5785f0154e1dc99f0d4b570042608474.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/698cdb455eac90da051e23356a66390b5785f0154e1dc99f0d4b570042608474.jpg)

![8c1820635456fc5a26ce010105322d173e8a9e83ef950393541632724084208c.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/8c1820635456fc5a26ce010105322d173e8a9e83ef950393541632724084208c.jpg)

![8d8a6c0d9d50cafe94ee7a2d2d19bac7e2277f9ebfcf7eb864dab1bc741df9fb.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/8d8a6c0d9d50cafe94ee7a2d2d19bac7e2277f9ebfcf7eb864dab1bc741df9fb.jpg)

![a1b03f02ba948a767b2be3a75d9cdf2fcce8dc965397c7a1c6647f50e98a213b.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/a1b03f02ba948a767b2be3a75d9cdf2fcce8dc965397c7a1c6647f50e98a213b.jpg)

![a38242a07cff826fe163ef602d9f27678e946cd52ab81629a7285c658025af30.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/a38242a07cff826fe163ef602d9f27678e946cd52ab81629a7285c658025af30.jpg)

![a40639f8fc95a54636c40bebe84517afdb4739e01fbc1d528588da882f12fada.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/a40639f8fc95a54636c40bebe84517afdb4739e01fbc1d528588da882f12fada.jpg)

![ad559e4ae62d69dd38085cb630b942462537f1a62ebae3ed9df00c63697bcbb2.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/ad559e4ae62d69dd38085cb630b942462537f1a62ebae3ed9df00c63697bcbb2.jpg)

![eac7157c3b266c86d4989f4f7a9aa78c7850878247edba736a9d8fbc2394b844.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/images/eac7157c3b266c86d4989f4f7a9aa78c7850878247edba736a9d8fbc2394b844.jpg)

### Tables

![052cb824a844fe3c4b70fc3d7eacbd6fbe5a391d53fcb7ed84f32045f57d9035.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/tables/052cb824a844fe3c4b70fc3d7eacbd6fbe5a391d53fcb7ed84f32045f57d9035.jpg)

![1041661ce6d1742155596670a0badfb7b59c2ad88649564b5e7e1af60f678e2b.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/tables/1041661ce6d1742155596670a0badfb7b59c2ad88649564b5e7e1af60f678e2b.jpg)

![334f3ae1073961a89ac8b2c7b6d3c316756e12cfebdbc7ba1f3e96187728fe06.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/tables/334f3ae1073961a89ac8b2c7b6d3c316756e12cfebdbc7ba1f3e96187728fe06.jpg)

![6778aff1bdd75ed34e29a04d1204ef6446270be1729ebb84f82e7f31ac205099.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/tables/6778aff1bdd75ed34e29a04d1204ef6446270be1729ebb84f82e7f31ac205099.jpg)

![708ee751d76815fc7f3d243d8bc33bf7e63bbbc9e658ef74d509fc09da42545e.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/tables/708ee751d76815fc7f3d243d8bc33bf7e63bbbc9e658ef74d509fc09da42545e.jpg)

![9e7534778fbbf5370822dda368a0e79306149c237a6e9babf370f6eb7c534f0f.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/tables/9e7534778fbbf5370822dda368a0e79306149c237a6e9babf370f6eb7c534f0f.jpg)

![a1ab07e1e4224a0a444ca3f9d15dbbf871032c6ee300cb571eaa46ac26da6c68.jpg](../iclr_results/575_DeepRTL_%20Bridging%20Verilog%20Understanding%20and%20Generation%20with%20a%20Unified%20Representation%20Model/tables/a1ab07e1e4224a0a444ca3f9d15dbbf871032c6ee300cb571eaa46ac26da6c68.jpg)

## DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference


### Images

![0b8be97869b9a3cd7cb78eb98fb176e0fa25272d2d6e7c03525b72503d8e4f74.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/0b8be97869b9a3cd7cb78eb98fb176e0fa25272d2d6e7c03525b72503d8e4f74.jpg)

![0d5720efb2a364046f7be5f3f6033c615076fd39cecfdb27c6eeb503ba8594b7.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/0d5720efb2a364046f7be5f3f6033c615076fd39cecfdb27c6eeb503ba8594b7.jpg)

![12a8f79125f98eacb783613be27f1ec77eb76a08cefc0bdbd918a3b9565bcc93.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/12a8f79125f98eacb783613be27f1ec77eb76a08cefc0bdbd918a3b9565bcc93.jpg)

![1e4d2eebfdcdd0e35da6ebddff6d166d47d7d8f7da9d880f45da42a2edc8be78.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/1e4d2eebfdcdd0e35da6ebddff6d166d47d7d8f7da9d880f45da42a2edc8be78.jpg)

![55e1d9461f73e9c8e43b51f61501cd455e0a3d3ea3026b2c77bc9e5fdb96870e.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/55e1d9461f73e9c8e43b51f61501cd455e0a3d3ea3026b2c77bc9e5fdb96870e.jpg)

![62e1d71181d91af7020d26627f1d41247dd32f43077f76c3145a813d9be43d04.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/62e1d71181d91af7020d26627f1d41247dd32f43077f76c3145a813d9be43d04.jpg)

![80d5ca7a41b1431aa4ab3ab6d6af4004bff7297bf8a6be0c041426d95149b073.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/80d5ca7a41b1431aa4ab3ab6d6af4004bff7297bf8a6be0c041426d95149b073.jpg)

![859c313eb5e9b6930ef1f076ecb44daca0859c68296dba780498fc3f1077f446.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/859c313eb5e9b6930ef1f076ecb44daca0859c68296dba780498fc3f1077f446.jpg)

![9449c3ccedc252f7c3a0b1a28cbec26539c879f0d3457d1bcf58146cd15259ce.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/9449c3ccedc252f7c3a0b1a28cbec26539c879f0d3457d1bcf58146cd15259ce.jpg)

![9a1f360a26ba5696fbee731282f93fe314cb8c6d1e38cf655a18a92678019679.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/9a1f360a26ba5696fbee731282f93fe314cb8c6d1e38cf655a18a92678019679.jpg)

![9e2d69cf5f53b7e7b6e6419bb35be009d83c41f252d5857a64cc8c57b579f457.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/9e2d69cf5f53b7e7b6e6419bb35be009d83c41f252d5857a64cc8c57b579f457.jpg)

![a3c20077496a782ad36a5f8a78a0e9f0d1372be1486520b38546ee62376d39e4.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/a3c20077496a782ad36a5f8a78a0e9f0d1372be1486520b38546ee62376d39e4.jpg)

![ab081ab1864b59f052dc166a206459b785ff0e58926f4278836acc7ae16188a2.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/ab081ab1864b59f052dc166a206459b785ff0e58926f4278836acc7ae16188a2.jpg)

![b6ec92e53016ea97266bf1b1b3ae72dba32dcdf1ff07bbf5563f018b75becffc.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/b6ec92e53016ea97266bf1b1b3ae72dba32dcdf1ff07bbf5563f018b75becffc.jpg)

![e159f42dc27ab318c5198f3c95e246fc16813d419c416bf84f28306632336621.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/e159f42dc27ab318c5198f3c95e246fc16813d419c416bf84f28306632336621.jpg)

![e7f121a3475d5ff705f57878a38622cc5fdbb7a95b41ddc2c4dcebae1d14dc58.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/e7f121a3475d5ff705f57878a38622cc5fdbb7a95b41ddc2c4dcebae1d14dc58.jpg)

![f29716fb8d2a9e2c85d017869f1e23a148800cf1bc10fd8f824d2b76c11f9363.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/f29716fb8d2a9e2c85d017869f1e23a148800cf1bc10fd8f824d2b76c11f9363.jpg)

![f98f3dcb9f9413d790ddab53e486ea1155d5750c5253965529a56c572b472f00.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/f98f3dcb9f9413d790ddab53e486ea1155d5750c5253965529a56c572b472f00.jpg)

![fbae95e859d78ee7fe879913aebfd5fc0923b7359c52f0e3f9a6139ac0577f58.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/fbae95e859d78ee7fe879913aebfd5fc0923b7359c52f0e3f9a6139ac0577f58.jpg)

![fbfae240b0dc1818f9898addc5d61d734058085651dba1eba0b7c91193d7ce1e.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/images/fbfae240b0dc1818f9898addc5d61d734058085651dba1eba0b7c91193d7ce1e.jpg)

### Tables

![106e9716e731db8a8b9005407884ad7ea8e5b6cb1ece0910f476bffaa6af4b94.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/106e9716e731db8a8b9005407884ad7ea8e5b6cb1ece0910f476bffaa6af4b94.jpg)

![1528e8ed45cc9af896a8d7da0546c37c1f7c606fee057f43e0aaafe51bbed770.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/1528e8ed45cc9af896a8d7da0546c37c1f7c606fee057f43e0aaafe51bbed770.jpg)

![230f18a826aaca8ac46156e440a6c50acf3483088ec3ee24fa3711c5f22b0668.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/230f18a826aaca8ac46156e440a6c50acf3483088ec3ee24fa3711c5f22b0668.jpg)

![259309c678abe972ab792ebfa06ba96ed91421b850440ab3a5c5e9f54b34e655.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/259309c678abe972ab792ebfa06ba96ed91421b850440ab3a5c5e9f54b34e655.jpg)

![3642341cfbb3b103a57597d3c209d6add898e5f73545aecc69ed654ee961b029.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/3642341cfbb3b103a57597d3c209d6add898e5f73545aecc69ed654ee961b029.jpg)

![416ad870c65fd08733e6e5d101bd20269c54726a7f24e5ea18a5b7aea3359201.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/416ad870c65fd08733e6e5d101bd20269c54726a7f24e5ea18a5b7aea3359201.jpg)

![450ccc6aabe1b12691ebf66fd596f92110070dccad6811b571db882bef13accd.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/450ccc6aabe1b12691ebf66fd596f92110070dccad6811b571db882bef13accd.jpg)

![4b2d8251fa4a107ca9e72c1b43dc310b6ab06856179d21137e0ca49de694c041.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/4b2d8251fa4a107ca9e72c1b43dc310b6ab06856179d21137e0ca49de694c041.jpg)

![5116c1188557bf62bfad77352a805c8198faf1f94b1770a331549143c6ff5f17.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/5116c1188557bf62bfad77352a805c8198faf1f94b1770a331549143c6ff5f17.jpg)

![5bd5a49cd0b3cb8c4f6b182f8e83bbeb584b6038c0ed9fbc1dca9c767b71c7ad.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/5bd5a49cd0b3cb8c4f6b182f8e83bbeb584b6038c0ed9fbc1dca9c767b71c7ad.jpg)

![81fff70d44e03b88b3fbe3bb1ba9039045eb6a2d506039328e526ebdee61a502.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/81fff70d44e03b88b3fbe3bb1ba9039045eb6a2d506039328e526ebdee61a502.jpg)

![8251b03c7b7f827a51287012fa1b110c7e55dc8e1083d059491d86ff195e1663.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/8251b03c7b7f827a51287012fa1b110c7e55dc8e1083d059491d86ff195e1663.jpg)

![8ba45a9cc35015079be0e6e48ed25d773144b2900940ee11f08d8197c5aabcd4.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/8ba45a9cc35015079be0e6e48ed25d773144b2900940ee11f08d8197c5aabcd4.jpg)

![8cb8e8996828e14f09a55be80baaf1b052923075c6cf9b2d3db28c24eed1bab2.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/8cb8e8996828e14f09a55be80baaf1b052923075c6cf9b2d3db28c24eed1bab2.jpg)

![90eb032d189cb144bf6a96e15f01bf4d8a44c13e5e5454ce326d226615310527.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/90eb032d189cb144bf6a96e15f01bf4d8a44c13e5e5454ce326d226615310527.jpg)

![9b0a854864932841d5b0d1067c11255fb89c383fa6b244027d5df07766fdd356.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/9b0a854864932841d5b0d1067c11255fb89c383fa6b244027d5df07766fdd356.jpg)

![acf9443769f5ea6162e6ff5a7e08beed3ad334a621c2199286ef01bf5535fb80.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/acf9443769f5ea6162e6ff5a7e08beed3ad334a621c2199286ef01bf5535fb80.jpg)

![bd1fae36d345b7ee4894e92f35ade1b1bd082049df84e1dceab8526f3ea55df4.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/bd1fae36d345b7ee4894e92f35ade1b1bd082049df84e1dceab8526f3ea55df4.jpg)

![de987300c47de50e21bf4bf2b7628ff88094c418d012dc135d28b9c6af916d70.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/de987300c47de50e21bf4bf2b7628ff88094c418d012dc135d28b9c6af916d70.jpg)

![efea2dcb87ab5047005db01f8066e3761f013f51dc8bba61a7aaef63bdffdea9.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/efea2dcb87ab5047005db01f8066e3761f013f51dc8bba61a7aaef63bdffdea9.jpg)

![fb8b09e9a4699da760057dab2f39c032d65b6815effd19b225080e6564dcb3b3.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/fb8b09e9a4699da760057dab2f39c032d65b6815effd19b225080e6564dcb3b3.jpg)

![ff147f820f8a65a81d30ea8f1c1cf1b903fbd4f30a4167024f1041c644839fa4.jpg](../iclr_results/576_DeFT_%20Decoding%20with%20Flash%20Tree-attention%20for%20Efficient%20Tree-structured%20LLM%20Inference/tables/ff147f820f8a65a81d30ea8f1c1cf1b903fbd4f30a4167024f1041c644839fa4.jpg)

## CREIMBO: Cross-Regional Ensemble Interactions in Multi-view Brain Observations


### Images

![022bb39d816adf9f9fc572b2b3aa05642fc9fd7c44e384444694354fdf49d6bd.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/022bb39d816adf9f9fc572b2b3aa05642fc9fd7c44e384444694354fdf49d6bd.jpg)

![03492e652b75a2310df0a57699f16a43fee724140665405dfcb6bf751271fd63.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/03492e652b75a2310df0a57699f16a43fee724140665405dfcb6bf751271fd63.jpg)

![03bc4fbb8cf11cb3821b12b72fb8a05ce83ae44f1715252100e8552fefe957bc.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/03bc4fbb8cf11cb3821b12b72fb8a05ce83ae44f1715252100e8552fefe957bc.jpg)

![096e10dcfe308775872e3c1a6973e240c89edbe9adaf6be09c8d62fa2f6f2c47.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/096e10dcfe308775872e3c1a6973e240c89edbe9adaf6be09c8d62fa2f6f2c47.jpg)

![103b4c741803fee9774dda94c71ea5cb83c513eb49b2c4ad4e48e756729649cd.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/103b4c741803fee9774dda94c71ea5cb83c513eb49b2c4ad4e48e756729649cd.jpg)

![104c07b9c6fa82257f94e1252a9b3059169099d8fe0f7a7758c30644217c4018.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/104c07b9c6fa82257f94e1252a9b3059169099d8fe0f7a7758c30644217c4018.jpg)

![132525dcb08608ddf1871bf460c71bd57d0c95c47acebf96b9063f6624415703.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/132525dcb08608ddf1871bf460c71bd57d0c95c47acebf96b9063f6624415703.jpg)

![1fb48ee5d743f9e48e6a4d1f46e2e408af98aca7bd7d153ed2552d8f4fc285f4.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/1fb48ee5d743f9e48e6a4d1f46e2e408af98aca7bd7d153ed2552d8f4fc285f4.jpg)

![27a0006312666b4d84a98d07ba8fb352292a76b67d123b2d69b8fffaeb5dce72.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/27a0006312666b4d84a98d07ba8fb352292a76b67d123b2d69b8fffaeb5dce72.jpg)

![2bd861f55ec1db4d4b005b2b9ff4dedfa94552ff31c181108c9917f2bd73e323.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/2bd861f55ec1db4d4b005b2b9ff4dedfa94552ff31c181108c9917f2bd73e323.jpg)

![31487f495b59c5d629270596f5c3f168585cfac7a319a5a5aee36542789f6559.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/31487f495b59c5d629270596f5c3f168585cfac7a319a5a5aee36542789f6559.jpg)

![34ba758a8d953fd02db9360c818eb63023246f9a2d679842daf9693e1b130ba3.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/34ba758a8d953fd02db9360c818eb63023246f9a2d679842daf9693e1b130ba3.jpg)

![3d4fdd17d238b862287c0cef56d0921269881e5696d374d1b861f73ee988dea5.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/3d4fdd17d238b862287c0cef56d0921269881e5696d374d1b861f73ee988dea5.jpg)

![407a826719c7281c4cc6d1127def5605007a67590d5ff2b8007f78414f2b5de8.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/407a826719c7281c4cc6d1127def5605007a67590d5ff2b8007f78414f2b5de8.jpg)

![47c0b421698042923cc5b4819bf2298be588607331b13df26481eba2191c3c52.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/47c0b421698042923cc5b4819bf2298be588607331b13df26481eba2191c3c52.jpg)

![4a476196ec748e35986eb972b1454860a9d3713e740ccb70a0682b15611cbff4.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/4a476196ec748e35986eb972b1454860a9d3713e740ccb70a0682b15611cbff4.jpg)

![5290e4b2ca40779ab661318abca13e66a8e4aeed26cbc66a54ab8d40f2a1e038.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/5290e4b2ca40779ab661318abca13e66a8e4aeed26cbc66a54ab8d40f2a1e038.jpg)

![52faae0a5f6f4d88686f5bcd4219e71b5e4dc9ca5fb50dd062f1311ec14361ff.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/52faae0a5f6f4d88686f5bcd4219e71b5e4dc9ca5fb50dd062f1311ec14361ff.jpg)

![5d45441e546142463a5b82bb7bd0dffa2be72fbc9374a10523c448526b014023.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/5d45441e546142463a5b82bb7bd0dffa2be72fbc9374a10523c448526b014023.jpg)

![5d7c9f3d27c25e03e630dad331e2c11f742509b74a61ed9164f2ea27a0822198.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/5d7c9f3d27c25e03e630dad331e2c11f742509b74a61ed9164f2ea27a0822198.jpg)

![60930784c61b7312c240a240a1a2ae91823977716ac0bb6128e8a779c2f36ba8.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/60930784c61b7312c240a240a1a2ae91823977716ac0bb6128e8a779c2f36ba8.jpg)

![6a60cbf855c923717f544fe14e0c42bd336e9f58bec8fc84d5752571ff9c2725.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/6a60cbf855c923717f544fe14e0c42bd336e9f58bec8fc84d5752571ff9c2725.jpg)

![84435f5d921ac81a40690717258f9a02e06f66827a0bdffe589552c82c15b80f.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/84435f5d921ac81a40690717258f9a02e06f66827a0bdffe589552c82c15b80f.jpg)

![87079800344b4879b7b6c6f9b8b069aa90a2f9382c22a51eeb30447d1c24d965.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/87079800344b4879b7b6c6f9b8b069aa90a2f9382c22a51eeb30447d1c24d965.jpg)

![95d9a4073f17dd8a9a3e042faa382853033eec7903baa0d99af9ad8679b4e9d7.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/95d9a4073f17dd8a9a3e042faa382853033eec7903baa0d99af9ad8679b4e9d7.jpg)

![9678ad372fab4145351a41501d34a9b8ad1b1a26f9d69ebd98e1a0bbdc9de0da.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/9678ad372fab4145351a41501d34a9b8ad1b1a26f9d69ebd98e1a0bbdc9de0da.jpg)

![aa6c93c1af02afdf804dd0e8e5f7d2d4f35b649887e9964fc39b83a62f269b85.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/aa6c93c1af02afdf804dd0e8e5f7d2d4f35b649887e9964fc39b83a62f269b85.jpg)

![ab3b7727b012aeee983eef9c3fbdf18865b57a9b6430cc15f5e932d439e906bc.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/ab3b7727b012aeee983eef9c3fbdf18865b57a9b6430cc15f5e932d439e906bc.jpg)

![b0d47c612a7252f45af91c3002472cb4defb065028372635d80afe2c8126839e.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/b0d47c612a7252f45af91c3002472cb4defb065028372635d80afe2c8126839e.jpg)

![b32236fb22a31026c6bff16a6ad385c43fdcbba37c15ae9df1a5d98cb1c4600d.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/b32236fb22a31026c6bff16a6ad385c43fdcbba37c15ae9df1a5d98cb1c4600d.jpg)

![bbefb05ab1e3cca0a58d35c6f6abe8fee73bdca322ee5d7e51e1ed941cb2a66e.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/bbefb05ab1e3cca0a58d35c6f6abe8fee73bdca322ee5d7e51e1ed941cb2a66e.jpg)

![caf5b6c94430a3ce7d65b46b6fc0fd6ffdd21a92f39499f032880e4938bee5a6.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/caf5b6c94430a3ce7d65b46b6fc0fd6ffdd21a92f39499f032880e4938bee5a6.jpg)

![ea5db59d396d303ad62216342a5a6222c56a2ee01d1fb76f320f40ddb872d8d6.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/images/ea5db59d396d303ad62216342a5a6222c56a2ee01d1fb76f320f40ddb872d8d6.jpg)

### Tables

![2bc20a9bf9459522e74bcfcf034a417e0d32b423696cdaaeca7920f9d5f0d3dc.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/tables/2bc20a9bf9459522e74bcfcf034a417e0d32b423696cdaaeca7920f9d5f0d3dc.jpg)

![6b92a7dd0369ed62d3eae1a608da7d8f21c78545e598eeb4b5676fb945c2db77.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/tables/6b92a7dd0369ed62d3eae1a608da7d8f21c78545e598eeb4b5676fb945c2db77.jpg)

![c45632ab58cb8ca576716bd2bd77fb3a0a54c015316e1880f44099154963c8ea.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/tables/c45632ab58cb8ca576716bd2bd77fb3a0a54c015316e1880f44099154963c8ea.jpg)

![f123343b4000934ff2d8c0adf6279f046799e7e8f54584efc4f38a59c3202b6c.jpg](../iclr_results/577_CREIMBO_%20Cross-Regional%20Ensemble%20Interactions%20in%20Multi-view%20Brain%20Observations/tables/f123343b4000934ff2d8c0adf6279f046799e7e8f54584efc4f38a59c3202b6c.jpg)

## High-dimensional Analysis of Knowledge Distillation: Weak-to-Strong Generalization and Scaling Laws


### Images

![71432a5dd7cc9b2b5545f0d2d6283d70fb9d1e8466cd7f8b272a8bb368812b31.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/images/71432a5dd7cc9b2b5545f0d2d6283d70fb9d1e8466cd7f8b272a8bb368812b31.jpg)

![929bb0d1bcdbbb848c7d9889e24134e0b31984cc16c60db6ad94f18c97c81bdc.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/images/929bb0d1bcdbbb848c7d9889e24134e0b31984cc16c60db6ad94f18c97c81bdc.jpg)

![bafe6f9f4c53f143d590555ac3992561ab1c884a4482f493a8deec1eeb6c0414.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/images/bafe6f9f4c53f143d590555ac3992561ab1c884a4482f493a8deec1eeb6c0414.jpg)

![c731b26df1013372eb3a5249621d4fb19e96231b941204023b8db6c203c15eb9.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/images/c731b26df1013372eb3a5249621d4fb19e96231b941204023b8db6c203c15eb9.jpg)

![d321c86e30bd5331cb5b108ceed5dacbacc1443bfb23ea94dfa36ad1521eceba.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/images/d321c86e30bd5331cb5b108ceed5dacbacc1443bfb23ea94dfa36ad1521eceba.jpg)

![e9df1a100d86e117bfb8c7df9300ec41ecc320e3705aebffa27e1013638f8f06.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/images/e9df1a100d86e117bfb8c7df9300ec41ecc320e3705aebffa27e1013638f8f06.jpg)

### Tables

![23366f834c8e6a591a3ebae957bc2eec59028a0fb19c0a7f6066f3cde2fb8007.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/tables/23366f834c8e6a591a3ebae957bc2eec59028a0fb19c0a7f6066f3cde2fb8007.jpg)

![fca167094bae9583e7515c9fe79f693edd8790f162a9770fd199630b8a2f0d81.jpg](../iclr_results/578_High-dimensional%20Analysis%20of%20Knowledge%20Distillation_%20Weak-to-Strong%20Generalization%20and%20Scaling%20Laws/tables/fca167094bae9583e7515c9fe79f693edd8790f162a9770fd199630b8a2f0d81.jpg)

## NetMoE: Accelerating MoE Training through Dynamic Sample Placement


### Images

![5d60326e17d3e88dd9f9096685d52a4bc5eb178084a27c9900bb2d4680fb1929.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/5d60326e17d3e88dd9f9096685d52a4bc5eb178084a27c9900bb2d4680fb1929.jpg)

![7be2647dd724605a8d5ea7f3e578a6da4b06715fd170eb04d69965ed1adca537.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/7be2647dd724605a8d5ea7f3e578a6da4b06715fd170eb04d69965ed1adca537.jpg)

![8da88076786fe499979db7e32fd1c50ed21e7edfa516c3881ed9a09db4430d64.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/8da88076786fe499979db7e32fd1c50ed21e7edfa516c3881ed9a09db4430d64.jpg)

![920e6585c7f361d14a12fa68fe9746998352a6fa56fe541bbbe5c47b0ae65733.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/920e6585c7f361d14a12fa68fe9746998352a6fa56fe541bbbe5c47b0ae65733.jpg)

![97b2c4f60d0b9e095debf497a4b14da63500a88fc412aea4681c9a63d9337f53.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/97b2c4f60d0b9e095debf497a4b14da63500a88fc412aea4681c9a63d9337f53.jpg)

![99d0d7160e22ca7e2be0a38850c040802ab5d52d672d969280faf46cad9f8cb8.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/99d0d7160e22ca7e2be0a38850c040802ab5d52d672d969280faf46cad9f8cb8.jpg)

![a0a9a5c8dda47c0a1fede798be805581c9102a57d97b5b10f9784d3f6d49a9b1.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/a0a9a5c8dda47c0a1fede798be805581c9102a57d97b5b10f9784d3f6d49a9b1.jpg)

![a10c9ef346efd64f739f71659f0103dc4f411771bd3c3a0267105856189b6346.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/a10c9ef346efd64f739f71659f0103dc4f411771bd3c3a0267105856189b6346.jpg)

![a3e82271d82754905bb3443dd8c35126abcaf905865d3a676f7c9510ec9aad7c.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/a3e82271d82754905bb3443dd8c35126abcaf905865d3a676f7c9510ec9aad7c.jpg)

![af1ff744be15c959bf83b92fc291955dfba0eec698a378983145b5cb0507dcce.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/images/af1ff744be15c959bf83b92fc291955dfba0eec698a378983145b5cb0507dcce.jpg)

### Tables

![4b9f95eee0c21d18503d62d97c7da294748470cc709005138b11048399198788.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/tables/4b9f95eee0c21d18503d62d97c7da294748470cc709005138b11048399198788.jpg)

![74a04bf586742550d2716ad93c0e3e5fcf2076301b86414154750284dce38090.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/tables/74a04bf586742550d2716ad93c0e3e5fcf2076301b86414154750284dce38090.jpg)

![99f7b2a1d6466a3fb90ddff77cda8e1d34d5c7eebea6f7cf6ee175c354e8fbd4.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/tables/99f7b2a1d6466a3fb90ddff77cda8e1d34d5c7eebea6f7cf6ee175c354e8fbd4.jpg)

![cf7ba3f0279b39ef4b3aa125a96a5275523ebc87ee7106474650e344ffac8a1f.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/tables/cf7ba3f0279b39ef4b3aa125a96a5275523ebc87ee7106474650e344ffac8a1f.jpg)

![f4d95e8d325fe430d6412b000019f1a152b70f8e2c3d94c9eccda3c0d7e4871f.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/tables/f4d95e8d325fe430d6412b000019f1a152b70f8e2c3d94c9eccda3c0d7e4871f.jpg)

![ffc57d1db74613144cb73576b8c9651945950203b22cbb58247f9fe98174ebe6.jpg](../iclr_results/579_NetMoE_%20Accelerating%20MoE%20Training%20through%20Dynamic%20Sample%20Placement/tables/ffc57d1db74613144cb73576b8c9651945950203b22cbb58247f9fe98174ebe6.jpg)

## Bayesian Optimization via Continual Variational Last Layer Training


### Images

![10bd3930939948158e8e922e8112ca04f6498d931641b8f273d72fb74da4c069.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/10bd3930939948158e8e922e8112ca04f6498d931641b8f273d72fb74da4c069.jpg)

![1476f66b8315e959965a63e0a2a51126b7c2755e14714450f4308a057b5da76d.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/1476f66b8315e959965a63e0a2a51126b7c2755e14714450f4308a057b5da76d.jpg)

![303a5c8cbfcabf14ee47d83649a6fe89efdfaa3a323897223c7a295bbb101e6e.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/303a5c8cbfcabf14ee47d83649a6fe89efdfaa3a323897223c7a295bbb101e6e.jpg)

![30b0f5d6f291d62fdc91d80298d6c8ff27c27de3b082e1717991947706de9178.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/30b0f5d6f291d62fdc91d80298d6c8ff27c27de3b082e1717991947706de9178.jpg)

![35ccd72e631110f2f4df522a8610418cdca7ec3799ee53b2583232ec53068371.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/35ccd72e631110f2f4df522a8610418cdca7ec3799ee53b2583232ec53068371.jpg)

![59b79a2caf2a6dfde17ffaca078cb84435d424c1de7e0669ca0d1d5405a988f0.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/59b79a2caf2a6dfde17ffaca078cb84435d424c1de7e0669ca0d1d5405a988f0.jpg)

![5ad483bb4c1a7a665c6eb81ea7dd779f706f064976974154ca7c731a3e975950.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/5ad483bb4c1a7a665c6eb81ea7dd779f706f064976974154ca7c731a3e975950.jpg)

![5b388f5edafac3d569d3e78f98afcdde2af0132b28428f64f5d486e9b41c75b0.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/5b388f5edafac3d569d3e78f98afcdde2af0132b28428f64f5d486e9b41c75b0.jpg)

![5cf72d56f4b156aa9b44c483c1c1fdf2b9e1067f93f1b45d81c1bb3732934d7f.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/5cf72d56f4b156aa9b44c483c1c1fdf2b9e1067f93f1b45d81c1bb3732934d7f.jpg)

![5d19e5e504be73fb9a6e4770e8ddf4bbffb042477c3dc3eef46a3dd4440669cd.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/5d19e5e504be73fb9a6e4770e8ddf4bbffb042477c3dc3eef46a3dd4440669cd.jpg)

![5fae8e7c9f20c164ad1f780367ed22b40f8e072908248273c899891fe10dfb82.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/5fae8e7c9f20c164ad1f780367ed22b40f8e072908248273c899891fe10dfb82.jpg)

![6f38b603ab541b4c19424e01929a9468a78f0aa2dbfa7638d865ccbde2a3997a.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/6f38b603ab541b4c19424e01929a9468a78f0aa2dbfa7638d865ccbde2a3997a.jpg)

![71d26784430e1e54844ca009929f69d0023fcacce98bc817e609dd1a98c2738d.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/71d26784430e1e54844ca009929f69d0023fcacce98bc817e609dd1a98c2738d.jpg)

![7ffee7248b4eb250ecb688a7f8d88436b142783235fd4944e1c4084c56714ff5.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/7ffee7248b4eb250ecb688a7f8d88436b142783235fd4944e1c4084c56714ff5.jpg)

![9be1a3568822d5aee55e4296f10416d01f6dd4968eeb0b8952256bb68cc2172c.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/9be1a3568822d5aee55e4296f10416d01f6dd4968eeb0b8952256bb68cc2172c.jpg)

![9bedae638faae14ebb99c77e3465e4136f4345372f32a0f755760af83d21102a.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/9bedae638faae14ebb99c77e3465e4136f4345372f32a0f755760af83d21102a.jpg)

![af13270173294c1b890bd132bfe6385906f800be96a85f337440c080fc821308.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/af13270173294c1b890bd132bfe6385906f800be96a85f337440c080fc821308.jpg)

![b2648f448136ffb0baaede4725a7d62f91831bb98fd4f4410b1d9653f4c2e352.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/b2648f448136ffb0baaede4725a7d62f91831bb98fd4f4410b1d9653f4c2e352.jpg)

![b53ff65584d74dc8cf015462810e39ba47d0a910ddab01437cde0d17681f0448.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/b53ff65584d74dc8cf015462810e39ba47d0a910ddab01437cde0d17681f0448.jpg)

![d267f656b8cce95d71e3994aaae197e05a4747046df96b80dbc24491e2568a7a.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/d267f656b8cce95d71e3994aaae197e05a4747046df96b80dbc24491e2568a7a.jpg)

![d84af66192dace75a199839834fae6e95b1bb9184c66045910f64d8714525905.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/d84af66192dace75a199839834fae6e95b1bb9184c66045910f64d8714525905.jpg)

![e10581f5cfacd96bc10caff23f349ae11077ec39cc6d489625c4db074df3c51c.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/images/e10581f5cfacd96bc10caff23f349ae11077ec39cc6d489625c4db074df3c51c.jpg)

### Tables

![accb19019398d784fc239bddeff82600e930d6c31e2775b1f3401a231d969a81.jpg](../iclr_results/580_Bayesian%20Optimization%20via%20Continual%20Variational%20Last%20Layer%20Training/tables/accb19019398d784fc239bddeff82600e930d6c31e2775b1f3401a231d969a81.jpg)

## Fast Uncovering of Protein Sequence Diversity from Structure


### Images

![0d44cedf1d0cd3d1bd9e314f197810ed55e404dfc3a964c56f912be72afd1248.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/0d44cedf1d0cd3d1bd9e314f197810ed55e404dfc3a964c56f912be72afd1248.jpg)

![153b915f1a09f104682d7f05ce17e9d45ef4b1d5fc603e140f052034a38c903c.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/153b915f1a09f104682d7f05ce17e9d45ef4b1d5fc603e140f052034a38c903c.jpg)

![2499a4e0ec54efc598047d219838a4ffc294ba8bb2a729bb9580547ec07b9dc8.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/2499a4e0ec54efc598047d219838a4ffc294ba8bb2a729bb9580547ec07b9dc8.jpg)

![2ce64a322cfffb27ad3d5fefa1c4f4ae0112fd7d66451b623cc9a99e1a377293.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/2ce64a322cfffb27ad3d5fefa1c4f4ae0112fd7d66451b623cc9a99e1a377293.jpg)

![469c254f80be1555bff56c8d77387e62813890662807d5b18f9270be2ab927c7.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/469c254f80be1555bff56c8d77387e62813890662807d5b18f9270be2ab927c7.jpg)

![5500784d6383e95cd7668bc3dfccb00715fbe45536483e8c60d766afe93585f7.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/5500784d6383e95cd7668bc3dfccb00715fbe45536483e8c60d766afe93585f7.jpg)

![604be6c8f89b2070bf85752c5bad1c809f22677cf98cb0398dca1a06a82b3661.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/604be6c8f89b2070bf85752c5bad1c809f22677cf98cb0398dca1a06a82b3661.jpg)

![6221b6926a109438f03788e67b74b7cf2bdd400349f400a423c7e09dd719c974.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/6221b6926a109438f03788e67b74b7cf2bdd400349f400a423c7e09dd719c974.jpg)

![6acea3aec9c20068078a252ecefb4a90df81c2d186feb51a8d1c8c5e7660ad61.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/6acea3aec9c20068078a252ecefb4a90df81c2d186feb51a8d1c8c5e7660ad61.jpg)

![724e604fa167d77fb42cb451cdf1f984e92bb7e2a5d38c6f0b310551e0d83bae.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/724e604fa167d77fb42cb451cdf1f984e92bb7e2a5d38c6f0b310551e0d83bae.jpg)

![78a733dcc7f1d0e0ce3f5902ab837f972f4a2e5882556909fd8cd893a71be22b.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/78a733dcc7f1d0e0ce3f5902ab837f972f4a2e5882556909fd8cd893a71be22b.jpg)

![82de8a0a510c6b335eb5120efef26478bffeee2366a3d3fd811738a1c31642ba.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/82de8a0a510c6b335eb5120efef26478bffeee2366a3d3fd811738a1c31642ba.jpg)

![8c755aee690333081f41ce2140ac2b8a1e3514ac73a9ffa55843b9bce5292ace.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/8c755aee690333081f41ce2140ac2b8a1e3514ac73a9ffa55843b9bce5292ace.jpg)

![972729d903e49eb9d5836f27fa027a0ba44b3dfb448bc06226eae12514aa649f.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/972729d903e49eb9d5836f27fa027a0ba44b3dfb448bc06226eae12514aa649f.jpg)

![9c8632c904783a217fbf638ba4f50fe7928df9a459e1e9854af3d1f68e86877d.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/9c8632c904783a217fbf638ba4f50fe7928df9a459e1e9854af3d1f68e86877d.jpg)

![a90d41f275557638c9ba73b5ce49801118b01c868dd45dfaa1cc165d2f6e83eb.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/a90d41f275557638c9ba73b5ce49801118b01c868dd45dfaa1cc165d2f6e83eb.jpg)

![aa6ed9966e784aa9a73b832c6235451125a54f9fd8611ff91208f9460c9572fd.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/aa6ed9966e784aa9a73b832c6235451125a54f9fd8611ff91208f9460c9572fd.jpg)

![c16010b536db2256c9139e0746d1d81d912b87b9bcae826859ad0aa3dfec33b9.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/c16010b536db2256c9139e0746d1d81d912b87b9bcae826859ad0aa3dfec33b9.jpg)

![d3543da87958499477c44c8710f60590cf039fb84cd62a6c5aa245bf06e1d139.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/d3543da87958499477c44c8710f60590cf039fb84cd62a6c5aa245bf06e1d139.jpg)

![e1bdc93d010a0564f6e5d701866778df3867dcbcfba584a5701ecad16d9f9b5c.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/e1bdc93d010a0564f6e5d701866778df3867dcbcfba584a5701ecad16d9f9b5c.jpg)

![ed2a27315a0794a37b6374140803b4cef6b72d9f7c0b83119e38f0e7c8d686db.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/ed2a27315a0794a37b6374140803b4cef6b72d9f7c0b83119e38f0e7c8d686db.jpg)

![ee6f1db6b8f3d53d367be77c17cd36f9735945c811811b242c68a28bc3a96caa.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/ee6f1db6b8f3d53d367be77c17cd36f9735945c811811b242c68a28bc3a96caa.jpg)

![eea6585504b717110826486c25a6c79b0f8480396ced48807cd805f6dcf3aff9.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/eea6585504b717110826486c25a6c79b0f8480396ced48807cd805f6dcf3aff9.jpg)

![fc6cd54e44a6a768325961dbef75a44d8f15fe9a148040f1de8d2bc35a25b4bf.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/images/fc6cd54e44a6a768325961dbef75a44d8f15fe9a148040f1de8d2bc35a25b4bf.jpg)

### Tables

![6528c018f5ae26c53576fabf8cf1807d32c0cdf7ee9eace221e91e0cc784dc5a.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/tables/6528c018f5ae26c53576fabf8cf1807d32c0cdf7ee9eace221e91e0cc784dc5a.jpg)

![99fbca9d03d22a56e4c2acbcc2d444a9e0d90edc153fb93d42f960155aa64c15.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/tables/99fbca9d03d22a56e4c2acbcc2d444a9e0d90edc153fb93d42f960155aa64c15.jpg)

![c0f2cf9102cb9613e774574f8bd44e82587a66dfc9ac777781c3d3495708859c.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/tables/c0f2cf9102cb9613e774574f8bd44e82587a66dfc9ac777781c3d3495708859c.jpg)

![d79d2ea9dc885bc3c500140152f4a53ecc93f7d2b0b38d9f3f14aff4f16490b1.jpg](../iclr_results/581_Fast%20Uncovering%20of%20Protein%20Sequence%20Diversity%20from%20Structure/tables/d79d2ea9dc885bc3c500140152f4a53ecc93f7d2b0b38d9f3f14aff4f16490b1.jpg)

## Century: A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images


### Images

![0579968824855e5134c2293e30c2e6d15497c160eaf9f399ab28dffd59d2181d.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/0579968824855e5134c2293e30c2e6d15497c160eaf9f399ab28dffd59d2181d.jpg)

![0fb0fe3566c7b004ee9fe39063a4734c4213e86c7530661423ba35c9b4d00127.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/0fb0fe3566c7b004ee9fe39063a4734c4213e86c7530661423ba35c9b4d00127.jpg)

![175f56d28191cd25a5694cd0bfdeda9ecf4c52c5c698ca2a34d3335773335ec3.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/175f56d28191cd25a5694cd0bfdeda9ecf4c52c5c698ca2a34d3335773335ec3.jpg)

![1cc14af8ce83d383500adce4185ecda9a7af09a5af94ad7a8be0f7b44bf4673e.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/1cc14af8ce83d383500adce4185ecda9a7af09a5af94ad7a8be0f7b44bf4673e.jpg)

![3e81478c7d8cd477285afc1d1c445eacced90b4968db544db240bfec966c3ffd.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/3e81478c7d8cd477285afc1d1c445eacced90b4968db544db240bfec966c3ffd.jpg)

![3f0fb96fe7a007a6dffd9ba7c01d78de65c8efed8bf645e134be59fd25d74957.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/3f0fb96fe7a007a6dffd9ba7c01d78de65c8efed8bf645e134be59fd25d74957.jpg)

![4e2e6c377721f0d2aeb1b1f35594b35d2350f5622638b12d63285096feed81db.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/4e2e6c377721f0d2aeb1b1f35594b35d2350f5622638b12d63285096feed81db.jpg)

![671b9c64a1fec278fdd2df67381a692ccfce1264f0d47a4850d5775a4ee93a47.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/671b9c64a1fec278fdd2df67381a692ccfce1264f0d47a4850d5775a4ee93a47.jpg)

![6b57723618c0498d9de0b320f2625869b3a519915627818a40dffa4aff50047e.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/6b57723618c0498d9de0b320f2625869b3a519915627818a40dffa4aff50047e.jpg)

![8617c94da482d6bbdf36a5e3a32b06541fb749ced42dd62e8a9eb1f7df1b5694.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/8617c94da482d6bbdf36a5e3a32b06541fb749ced42dd62e8a9eb1f7df1b5694.jpg)

![91b848cf0ea46c2e507a6593398ca8248fbd4765888fdcc1c03a341044fb0a3e.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/91b848cf0ea46c2e507a6593398ca8248fbd4765888fdcc1c03a341044fb0a3e.jpg)

![91fe7bb5147e72e4c104463d4022cfbec3a66f2a72b744e7901e6b794f41d006.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/91fe7bb5147e72e4c104463d4022cfbec3a66f2a72b744e7901e6b794f41d006.jpg)

![9a10f714cae58bab57c732a6f0dbd7e125321fc99c0bc9403f92cde88f043c2d.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/9a10f714cae58bab57c732a6f0dbd7e125321fc99c0bc9403f92cde88f043c2d.jpg)

![a3349b3d6605d03da2fb331f14b2cddc1ac156c434ba7c3c7d5faa4a89b1f1ca.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/a3349b3d6605d03da2fb331f14b2cddc1ac156c434ba7c3c7d5faa4a89b1f1ca.jpg)

![a92cfc12274113af7a4f78520d82c6dac73b47653f53755dc766e66fdbf2eb59.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/a92cfc12274113af7a4f78520d82c6dac73b47653f53755dc766e66fdbf2eb59.jpg)

![b36de30bfb2d6c730059c49d8a97e0bbc0730c2c2ad8e083d471bb618f20f907.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/b36de30bfb2d6c730059c49d8a97e0bbc0730c2c2ad8e083d471bb618f20f907.jpg)

![bb91d730e70bdde3950d68afb39d7921efed7100f0e88e1df554fd218d50fd68.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/bb91d730e70bdde3950d68afb39d7921efed7100f0e88e1df554fd218d50fd68.jpg)

![bf0d404bc19ca22eef444638130f4fc66aa50404d15952c2f911a48f7eddd182.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/bf0d404bc19ca22eef444638130f4fc66aa50404d15952c2f911a48f7eddd182.jpg)

![c6a69ab01b7454c4e191ba86c041557f6644bb269b5c08ab7d163d4530c36bac.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/c6a69ab01b7454c4e191ba86c041557f6644bb269b5c08ab7d163d4530c36bac.jpg)

![c966eea222dad07db72029b3f0ebeb047c8440c78e516acda33b563af9330bf6.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/c966eea222dad07db72029b3f0ebeb047c8440c78e516acda33b563af9330bf6.jpg)

![d377d4481ea123ceb7628b55772f098b79cdcdd891ca15fb9b38dbb53f471a84.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/d377d4481ea123ceb7628b55772f098b79cdcdd891ca15fb9b38dbb53f471a84.jpg)

![dbd5b3342c76270cafe493f1e36637de1224be277130dd7f22bd5342ef463b7a.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/dbd5b3342c76270cafe493f1e36637de1224be277130dd7f22bd5342ef463b7a.jpg)

![de19fe47ba3e3cb12a1ffd3e91993d47c8f49c0b8dd5f7c56a432d790227788e.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/de19fe47ba3e3cb12a1ffd3e91993d47c8f49c0b8dd5f7c56a432d790227788e.jpg)

![e5987002c8a88d43e4f8ebae7fecccba2613d88fe64e87abe0838f8cee95f5ba.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/images/e5987002c8a88d43e4f8ebae7fecccba2613d88fe64e87abe0838f8cee95f5ba.jpg)

### Tables

![0426785f2d0c5465d816cda81c51da85f86cfa19b04f91991ebabb5476f40856.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/0426785f2d0c5465d816cda81c51da85f86cfa19b04f91991ebabb5476f40856.jpg)

![111bb6119129e24e99b07fd7b59ae8d197788716102c96a39e548209b1fc8389.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/111bb6119129e24e99b07fd7b59ae8d197788716102c96a39e548209b1fc8389.jpg)

![15f474ebd1652943ab3690a4b3be859de1ec64e01b468e1c6f4192dd9455f09b.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/15f474ebd1652943ab3690a4b3be859de1ec64e01b468e1c6f4192dd9455f09b.jpg)

![3054590d030b5103443f43bcf981fb711e283da70a732af4e236228d8a11471e.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/3054590d030b5103443f43bcf981fb711e283da70a732af4e236228d8a11471e.jpg)

![3e3bc708cbfe3b2066aedf31aba117ab4ef0580e1dcae13afed508a0c663c2dd.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/3e3bc708cbfe3b2066aedf31aba117ab4ef0580e1dcae13afed508a0c663c2dd.jpg)

![52898fa8c2690653588cec6498e9b6edef5d1fccae9a372d8988174efd5f4851.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/52898fa8c2690653588cec6498e9b6edef5d1fccae9a372d8988174efd5f4851.jpg)

![5dd1e753cb6581fd21b2819e0565366e16cbbfd9f2f7debee3aaf820dd80b7bd.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/5dd1e753cb6581fd21b2819e0565366e16cbbfd9f2f7debee3aaf820dd80b7bd.jpg)

![6a4548508e45a16813f63b55065538ec1747fab92678c754c675dd0bf8322e0a.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/6a4548508e45a16813f63b55065538ec1747fab92678c754c675dd0bf8322e0a.jpg)

![8298f90bca465adfbba0abd705f76d6c5f640e59ea3d7526513d73a2f55470e1.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/8298f90bca465adfbba0abd705f76d6c5f640e59ea3d7526513d73a2f55470e1.jpg)

![99ce50b5a9e20651f50e94ae62639bf79151763ce0b9f2c850cd918342ad78aa.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/99ce50b5a9e20651f50e94ae62639bf79151763ce0b9f2c850cd918342ad78aa.jpg)

![a7117883a883c678eaddddb6455e162702bfa4e136045122cef646183c4ee79b.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/a7117883a883c678eaddddb6455e162702bfa4e136045122cef646183c4ee79b.jpg)

![aa023c191dc9b1d3f06f27e49e6c001895894d1fdaf64bd9f695daa09dd26f36.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/aa023c191dc9b1d3f06f27e49e6c001895894d1fdaf64bd9f695daa09dd26f36.jpg)

![d03ef20ef7d00a77782b6dbd745e58db0a8dff87b227af01e69e7cf572cd562d.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/d03ef20ef7d00a77782b6dbd745e58db0a8dff87b227af01e69e7cf572cd562d.jpg)

![d8ccdb9889bc6f577ea7f919cbb095c8c954d21c3cbcc30b381418c915097919.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/d8ccdb9889bc6f577ea7f919cbb095c8c954d21c3cbcc30b381418c915097919.jpg)

![ef95f8467e28e52caea023b9b1eda93d13bfd8c87b8b843301401a82e57eac8e.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/ef95f8467e28e52caea023b9b1eda93d13bfd8c87b8b843301401a82e57eac8e.jpg)

![efe071dea6729b471e4fd0424caf2078e2bca2cc03340a596003d210c3ee991b.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/efe071dea6729b471e4fd0424caf2078e2bca2cc03340a596003d210c3ee991b.jpg)

![f04790ac6c400f36dfbce28227727a917468a49cbf8bd1ed8e007ec46326f152.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/f04790ac6c400f36dfbce28227727a917468a49cbf8bd1ed8e007ec46326f152.jpg)

![f7ecac16323564c5647920e051699faacc71cfd08f96e67d15dcd6ea6a31668d.jpg](../iclr_results/582_Century_%20A%20Framework%20and%20Dataset%20for%20Evaluating%20Historical%20Contextualisation%20of%20Sensitive%20Images/tables/f7ecac16323564c5647920e051699faacc71cfd08f96e67d15dcd6ea6a31668d.jpg)

## MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code


### Images

![4dce28d6b5b5d5644bfcccec89d1dd84fcdff05f84e392ce4d9c507b852eddcc.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/images/4dce28d6b5b5d5644bfcccec89d1dd84fcdff05f84e392ce4d9c507b852eddcc.jpg)

![600a90139e7c1b90af05455bc8debbeab665ea0d0d3af409bc5470f481ad3b17.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/images/600a90139e7c1b90af05455bc8debbeab665ea0d0d3af409bc5470f481ad3b17.jpg)

![8987c706e7d0e0355432f667ecea994932ca62024f688f070abee01a54cc4bf5.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/images/8987c706e7d0e0355432f667ecea994932ca62024f688f070abee01a54cc4bf5.jpg)

### Tables

![2795a3e10022453fce360e82798a5e88b06bf621a990e3efd76346b07659adfb.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/2795a3e10022453fce360e82798a5e88b06bf621a990e3efd76346b07659adfb.jpg)

![3f944b8055548dc0e801c4052c4753d9a25fbbd1cef74b2c77098c3fa31ac40d.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/3f944b8055548dc0e801c4052c4753d9a25fbbd1cef74b2c77098c3fa31ac40d.jpg)

![5ff72dc663a1af2a7c8c88bbf46db6f2fdef05b66dfecf5bac1c6bbbb171d758.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/5ff72dc663a1af2a7c8c88bbf46db6f2fdef05b66dfecf5bac1c6bbbb171d758.jpg)

![745cf3f0304346849b95e8c3a9eede9cfcc566ce8df99af82c009a3d2a5bc1d5.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/745cf3f0304346849b95e8c3a9eede9cfcc566ce8df99af82c009a3d2a5bc1d5.jpg)

![89ca232b14a301ac2c3aa126c46404d418be274d5433076e1746ae4a1e67f4ba.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/89ca232b14a301ac2c3aa126c46404d418be274d5433076e1746ae4a1e67f4ba.jpg)

![9561b97296e7208e5c3ca78f9e78f7bfb2470dd1b578d536a67cfb7fc42a65ba.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/9561b97296e7208e5c3ca78f9e78f7bfb2470dd1b578d536a67cfb7fc42a65ba.jpg)

![addebd45dee0a591014183bf9cf4ca7b1f10bafab1c1ee0be31021a7e9a7e4fe.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/addebd45dee0a591014183bf9cf4ca7b1f10bafab1c1ee0be31021a7e9a7e4fe.jpg)

![afd975e82b0eaae3b7e947f71c243a36ca6ae95d0b524aa13044bc61e71e780a.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/afd975e82b0eaae3b7e947f71c243a36ca6ae95d0b524aa13044bc61e71e780a.jpg)

![dbf3c2b93992c7503f8f47197cca98ef4555564359801a5f64125b50ce81be16.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/dbf3c2b93992c7503f8f47197cca98ef4555564359801a5f64125b50ce81be16.jpg)

![ea11766bff8c88706f2ca7baec0b2f6425442f9a38d7246bacc59c526f9f7de0.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/ea11766bff8c88706f2ca7baec0b2f6425442f9a38d7246bacc59c526f9f7de0.jpg)

![ea8d4ad7164f3287e75d3077152a17e3599d6e254e688df9b9674a7e0d55ec04.jpg](../iclr_results/583_MathCoder2_%20Better%20Math%20Reasoning%20from%20Continued%20Pretraining%20on%20Model-translated%20Mathematical%20Code/tables/ea8d4ad7164f3287e75d3077152a17e3599d6e254e688df9b9674a7e0d55ec04.jpg)

## OmniRe: Omni Urban Scene Reconstruction


### Images

![00d60c3dc5ebc77e5b8888f013dbb351b620c573a7dfd319ee57837557fbfb40.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/00d60c3dc5ebc77e5b8888f013dbb351b620c573a7dfd319ee57837557fbfb40.jpg)

![16b1bb647bc02fd1da7b66693c91606df47ba95507b1da21f2ff096b3ff3b155.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/16b1bb647bc02fd1da7b66693c91606df47ba95507b1da21f2ff096b3ff3b155.jpg)

![27d824fb740e4ef07cee6a1681053d7b25f5c46dd4e2bf5f9cbbf92c7fb905c9.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/27d824fb740e4ef07cee6a1681053d7b25f5c46dd4e2bf5f9cbbf92c7fb905c9.jpg)

![6a098a64f4c663632678961672e2830e567b0e4521e21b32f455560485cb88c8.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/6a098a64f4c663632678961672e2830e567b0e4521e21b32f455560485cb88c8.jpg)

![7748f2d0c5a1bad059ebb4ab0ef476c47c48a0ebf96c11a9d63ad173f93cce1e.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/7748f2d0c5a1bad059ebb4ab0ef476c47c48a0ebf96c11a9d63ad173f93cce1e.jpg)

![995eda197c5ee5627b2573048db2653697494aff3d9daf4a0d7965456562ed30.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/995eda197c5ee5627b2573048db2653697494aff3d9daf4a0d7965456562ed30.jpg)

![a58aa2e292347d1a096f6012e513ccc09db4a445f4f992c8e230fa770f9dbc5d.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/a58aa2e292347d1a096f6012e513ccc09db4a445f4f992c8e230fa770f9dbc5d.jpg)

![b77f68836e86e2acdddc4b3bdab20c4c13a220a141bb1b2681eec509a24a780a.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/b77f68836e86e2acdddc4b3bdab20c4c13a220a141bb1b2681eec509a24a780a.jpg)

![d60b3bc9071f4407fb753a96dd6b373bd8b8e14e6b254f7604a73fe4706bb0b8.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/d60b3bc9071f4407fb753a96dd6b373bd8b8e14e6b254f7604a73fe4706bb0b8.jpg)

![d838400dce39d20aba4b334d48c3b94e0ac5771d56bcef47fa83bb83a076741b.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/d838400dce39d20aba4b334d48c3b94e0ac5771d56bcef47fa83bb83a076741b.jpg)

![f20535b63eb5de8b4d1b7ad9477274cc613cb0fc7025f9e2902052b3a80ad079.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/f20535b63eb5de8b4d1b7ad9477274cc613cb0fc7025f9e2902052b3a80ad079.jpg)

![fccd93a4169eaa0a74c189db06cad77c2df0e449b87624c1db218a851244b0cc.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/images/fccd93a4169eaa0a74c189db06cad77c2df0e449b87624c1db218a851244b0cc.jpg)

### Tables

![1b6eb03ed87a0b698a6856fd9c4d1cecdea7c80e60b0799701c34906e3d5c8c3.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/1b6eb03ed87a0b698a6856fd9c4d1cecdea7c80e60b0799701c34906e3d5c8c3.jpg)

![474fc82f271c1b9c25ceec3a549c80d2448c86ee665165ceae0a238ec7d67d14.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/474fc82f271c1b9c25ceec3a549c80d2448c86ee665165ceae0a238ec7d67d14.jpg)

![553ee98bc489f96bba4b3ae2e970341ba08f6652dd694783825fb2306c8ab2c3.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/553ee98bc489f96bba4b3ae2e970341ba08f6652dd694783825fb2306c8ab2c3.jpg)

![5eafaee1861299439fbaf097d2d1922dd58e28df374aa2a1be1edff3bfa9ab35.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/5eafaee1861299439fbaf097d2d1922dd58e28df374aa2a1be1edff3bfa9ab35.jpg)

![6fbb55322f442f1adead0ef00358c1c2396f9385f085164596fe40da35ce366a.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/6fbb55322f442f1adead0ef00358c1c2396f9385f085164596fe40da35ce366a.jpg)

![7d09e70ca63e8315ae753b4e9a5811ea5609f92e552e611e9b7963ca88ddf1a0.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/7d09e70ca63e8315ae753b4e9a5811ea5609f92e552e611e9b7963ca88ddf1a0.jpg)

![875286e9a929e81c66adae587c21326a92ee3509c12740d2ad0b302c59e8910c.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/875286e9a929e81c66adae587c21326a92ee3509c12740d2ad0b302c59e8910c.jpg)

![91c2ac53ba6ae45df585b46d3a3385929ce4cea49580059ff14e8be8ef19e0bc.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/91c2ac53ba6ae45df585b46d3a3385929ce4cea49580059ff14e8be8ef19e0bc.jpg)

![9729a77bd36d5152f256cad5352c6208c5a923010ccab3ee9728d8db41c66cd7.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/9729a77bd36d5152f256cad5352c6208c5a923010ccab3ee9728d8db41c66cd7.jpg)

![a2994cf93060df46395ba75f8a1fbf641606b0f032aa1bfd50f15d197b2cd4ab.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/a2994cf93060df46395ba75f8a1fbf641606b0f032aa1bfd50f15d197b2cd4ab.jpg)

![d1fa74c1e07401b8070878a7c738a4f2912e4f2ceec25bd289c7129775059028.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/d1fa74c1e07401b8070878a7c738a4f2912e4f2ceec25bd289c7129775059028.jpg)

![d5bde4781f72a61acf7eb30cafc0a66f77f97c0e372983bd8dea16b6d363b86e.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/d5bde4781f72a61acf7eb30cafc0a66f77f97c0e372983bd8dea16b6d363b86e.jpg)

![df52936636aa6ecd07e24a8db8570885e53d67b8e5688db9305afb45ac65e239.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/df52936636aa6ecd07e24a8db8570885e53d67b8e5688db9305afb45ac65e239.jpg)

![e23c54b461a67a017734aaac59c8a7b6f9de6707312d239fce35be3ab826110c.jpg](../iclr_results/584_OmniRe_%20Omni%20Urban%20Scene%20Reconstruction/tables/e23c54b461a67a017734aaac59c8a7b6f9de6707312d239fce35be3ab826110c.jpg)

## In vivo cell-type and brain region classification via multimodal contrastive learning


### Images

![0d1c1577676dc115c007c27dea46a926483d0781adce71bd7c22071adce2e001.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/0d1c1577676dc115c007c27dea46a926483d0781adce71bd7c22071adce2e001.jpg)

![10cc402de1fb08a2d6019ca3b139c2ea997f339c5884c3fd1477afa001a4644f.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/10cc402de1fb08a2d6019ca3b139c2ea997f339c5884c3fd1477afa001a4644f.jpg)

![2138ad2358b857dd2fcf83483f80ae743e4ea44b2b66490a9699cc3d30b5b7c0.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/2138ad2358b857dd2fcf83483f80ae743e4ea44b2b66490a9699cc3d30b5b7c0.jpg)

![2305746fd23638832fb1ff375ac7b9e767a6d23632bf46691d05f00e529ff9f1.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/2305746fd23638832fb1ff375ac7b9e767a6d23632bf46691d05f00e529ff9f1.jpg)

![26cdf36da878ef51940e21ef825255fada48b664bfbe41390857498e560c1232.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/26cdf36da878ef51940e21ef825255fada48b664bfbe41390857498e560c1232.jpg)

![2ab8c8b9f2dfce54af27b8201c86f276a371bce0604c72f2191bb16a7092bf26.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/2ab8c8b9f2dfce54af27b8201c86f276a371bce0604c72f2191bb16a7092bf26.jpg)

![30afbe624ce2f15e52e487e010831af251b0e2d1e6d28bbcf73edbf7f58537a7.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/30afbe624ce2f15e52e487e010831af251b0e2d1e6d28bbcf73edbf7f58537a7.jpg)

![4e9b62d468d714ed32e6b66eed463b2bb56c9e6e5fcaa01c9775e33027799d97.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/4e9b62d468d714ed32e6b66eed463b2bb56c9e6e5fcaa01c9775e33027799d97.jpg)

![51cbb763922ba922a06ea8843478e57738676448448110d5e94b5566e4a530ca.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/51cbb763922ba922a06ea8843478e57738676448448110d5e94b5566e4a530ca.jpg)

![533a939bad4e3c11a1d4dea30edc77940f4ac478dc390f7ccddbfca7152f3895.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/533a939bad4e3c11a1d4dea30edc77940f4ac478dc390f7ccddbfca7152f3895.jpg)

![5da339daf2f025f7c27562e226accb6b76c600ecf7845484e40ee464f7557e5f.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/5da339daf2f025f7c27562e226accb6b76c600ecf7845484e40ee464f7557e5f.jpg)

![5dec3efa166c9961c56f48df084c3c83195148eb492f9577ad8a0f3b3776cd3d.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/5dec3efa166c9961c56f48df084c3c83195148eb492f9577ad8a0f3b3776cd3d.jpg)

![6e15a6632265ea314ec2921fa6ce2f37e999f159c758ecc7c28800cbcc7903ee.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/6e15a6632265ea314ec2921fa6ce2f37e999f159c758ecc7c28800cbcc7903ee.jpg)

![7284cd169826e64fe9d24767cf97131872d92926ef4b8a52c6fa770b18701c4a.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/7284cd169826e64fe9d24767cf97131872d92926ef4b8a52c6fa770b18701c4a.jpg)

![758e7154f748237c73fc892521233f02245cc46fbbac6337021cd9fec6964bd2.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/758e7154f748237c73fc892521233f02245cc46fbbac6337021cd9fec6964bd2.jpg)

![785d4e8fdb2a6b62ece0027e2ad581ab13d2688711727f7283ca0f565bf17408.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/785d4e8fdb2a6b62ece0027e2ad581ab13d2688711727f7283ca0f565bf17408.jpg)

![864c8467d20cd5e416c7f31af5a70849be013c892ec163eb165815b40d9eeaf9.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/864c8467d20cd5e416c7f31af5a70849be013c892ec163eb165815b40d9eeaf9.jpg)

![94b198fe1ee626aee4aa568463cf2ca8cecda68ef77bd12fb558513804eac034.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/94b198fe1ee626aee4aa568463cf2ca8cecda68ef77bd12fb558513804eac034.jpg)

![95919b8f586f54b0d5780af1ba23f779124c4971142854f945aae3d32f2e41bb.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/95919b8f586f54b0d5780af1ba23f779124c4971142854f945aae3d32f2e41bb.jpg)

![a83d2b7fec1aabbdaceaecfbe2eb6a29e0dd9f0f93aaffa3d14a86fd70454c63.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/a83d2b7fec1aabbdaceaecfbe2eb6a29e0dd9f0f93aaffa3d14a86fd70454c63.jpg)

![a850813a1d65265d19222a8637f13eede8a466bd06abce41c8add4acec6e7f43.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/a850813a1d65265d19222a8637f13eede8a466bd06abce41c8add4acec6e7f43.jpg)

![a9451decec5022a8e68f8998f757b2562b6cfda09346f1d5924aa1299755d793.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/a9451decec5022a8e68f8998f757b2562b6cfda09346f1d5924aa1299755d793.jpg)

![a9a63aee596f948044599d0d798c0ebc0aa8968d5d3dd6888b9383ab4082377b.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/a9a63aee596f948044599d0d798c0ebc0aa8968d5d3dd6888b9383ab4082377b.jpg)

![b241b304c3a8dfc99488e687d1caa3db29b6530cfffda5d75e620ae6a51325f6.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/b241b304c3a8dfc99488e687d1caa3db29b6530cfffda5d75e620ae6a51325f6.jpg)

![c71906ea5558badbd5e3f926c319aeebfac30e48b3141835980aac5d2ca04565.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/c71906ea5558badbd5e3f926c319aeebfac30e48b3141835980aac5d2ca04565.jpg)

![c9d1c34683bd03d044ab8bbe40ff76c97f6062112046f9b2317abb7363ba9329.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/c9d1c34683bd03d044ab8bbe40ff76c97f6062112046f9b2317abb7363ba9329.jpg)

![d0a6ed8aca638643a165d1f06183482ae3b4755e269865bdc2e26f314027f722.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/d0a6ed8aca638643a165d1f06183482ae3b4755e269865bdc2e26f314027f722.jpg)

![e8662221e29e55e1c3c00313ea6181aa501932368735700f738e03d4c89101aa.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/e8662221e29e55e1c3c00313ea6181aa501932368735700f738e03d4c89101aa.jpg)

![e99f15a3c0cfbeb6da0e56a4607313b916e2dd56dafd1a0d86996bae49882c14.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/e99f15a3c0cfbeb6da0e56a4607313b916e2dd56dafd1a0d86996bae49882c14.jpg)

![f645672cce802e55116d93ab987d3ac763f36785ff612f998c682ee7b54f8068.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/images/f645672cce802e55116d93ab987d3ac763f36785ff612f998c682ee7b54f8068.jpg)

### Tables

![0bbe92b1f6363abe7b7119afa01b5f341517c1ff3c708d27db2f0748167b0af3.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/0bbe92b1f6363abe7b7119afa01b5f341517c1ff3c708d27db2f0748167b0af3.jpg)

![25b7e5afd5c6384ecf595935261ea518542c41076c3e41c4789c0e86c9544f30.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/25b7e5afd5c6384ecf595935261ea518542c41076c3e41c4789c0e86c9544f30.jpg)

![284ca0ad137ec1c04e2c5d58286660696077b62632770c66d6014ae62c087887.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/284ca0ad137ec1c04e2c5d58286660696077b62632770c66d6014ae62c087887.jpg)

![5a631a12321300e6c841fb77b1a8638a3b94d24f8d9056043ad8526c45a6310c.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/5a631a12321300e6c841fb77b1a8638a3b94d24f8d9056043ad8526c45a6310c.jpg)

![5b2419e185afaf5123bdec9d99cd793634025e9fd351d871d49a1198d750e4c2.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/5b2419e185afaf5123bdec9d99cd793634025e9fd351d871d49a1198d750e4c2.jpg)

![6810be95cc19817d9f421348768fed7c0a00cd465679251b0b898cadfbca7256.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/6810be95cc19817d9f421348768fed7c0a00cd465679251b0b898cadfbca7256.jpg)

![68a63082b6ce51df45ed9118f7761b42bea362e22cd4606142265efcbdfb483b.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/68a63082b6ce51df45ed9118f7761b42bea362e22cd4606142265efcbdfb483b.jpg)

![9769741bafcf81431fc7a9a7c66268c13a3355b856aeaca826d03e8bcf106f58.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/9769741bafcf81431fc7a9a7c66268c13a3355b856aeaca826d03e8bcf106f58.jpg)

![9fb07a8834e1330cd51557e173a6b22349ec9996089ce99ae48c03cdc672affc.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/9fb07a8834e1330cd51557e173a6b22349ec9996089ce99ae48c03cdc672affc.jpg)

![b8f04a859aabfc897a89bdcfc304e64a4e2117ac5b26b94320bbad4f17fe8f59.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/b8f04a859aabfc897a89bdcfc304e64a4e2117ac5b26b94320bbad4f17fe8f59.jpg)

![cde484c6daaa603b51e1367970b52fcec4b34eae3f696806df4ef49006e9e8ab.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/cde484c6daaa603b51e1367970b52fcec4b34eae3f696806df4ef49006e9e8ab.jpg)

![d50d36568ddf4e65533ba861695032c9307254bdef779fa50f6014d265fa1dc8.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/d50d36568ddf4e65533ba861695032c9307254bdef779fa50f6014d265fa1dc8.jpg)

![d9f2a461a828ec3c3f1ced1e9c7fe18c0cf2c2ce1ba26e331fd2df313ec2feed.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/d9f2a461a828ec3c3f1ced1e9c7fe18c0cf2c2ce1ba26e331fd2df313ec2feed.jpg)

![e5d2e4514c532c424b1d913b244ac142924f7f6f1ca2dd6323f4137d106ef727.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/e5d2e4514c532c424b1d913b244ac142924f7f6f1ca2dd6323f4137d106ef727.jpg)

![f264f42acd2131f3f04cb9f7e7d4411668ddd260db1102e5003aeeb8ea23b9d0.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/f264f42acd2131f3f04cb9f7e7d4411668ddd260db1102e5003aeeb8ea23b9d0.jpg)

![fe8d12254a3a5623dc20a10de583a281390f94c311a729d481c22d58055996e1.jpg](../iclr_results/585_In%20vivo%20cell-type%20and%20brain%20region%20classification%20via%20multimodal%20contrastive%20learning/tables/fe8d12254a3a5623dc20a10de583a281390f94c311a729d481c22d58055996e1.jpg)

## Monitoring Latent World States in Language Models with Propositional Probes


### Images

![00382adac5d5c0001c3c2c920e83db94971e33bbd76ec9181f8212ccfe8a04c8.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/00382adac5d5c0001c3c2c920e83db94971e33bbd76ec9181f8212ccfe8a04c8.jpg)

![0120a3f5512725c3f55e18aef760cb5f36e698e2ce46b752e41d5d3e31ca623f.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/0120a3f5512725c3f55e18aef760cb5f36e698e2ce46b752e41d5d3e31ca623f.jpg)

![12faac9ba564e82f83767bcef8c6c8747b64b2b5b57df288da4bc6d42a815232.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/12faac9ba564e82f83767bcef8c6c8747b64b2b5b57df288da4bc6d42a815232.jpg)

![2bbae1a7e2a4211f227cc3e9e0aa61a7dff43c8a2eb463051d51b8e4c1838a05.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/2bbae1a7e2a4211f227cc3e9e0aa61a7dff43c8a2eb463051d51b8e4c1838a05.jpg)

![34b76a415704c1e57a67fa73d2a4a0517af8752d9bc275a95d829e64d317ed44.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/34b76a415704c1e57a67fa73d2a4a0517af8752d9bc275a95d829e64d317ed44.jpg)

![6c2ce3a0fa982721f2f8e986d57baa7398412a211b56b9738d8a785c82dd69d1.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/6c2ce3a0fa982721f2f8e986d57baa7398412a211b56b9738d8a785c82dd69d1.jpg)

![81637e96ca2e3311c1394f1cc1ae06e06f76672a5844e2700f457df32bd24c7a.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/81637e96ca2e3311c1394f1cc1ae06e06f76672a5844e2700f457df32bd24c7a.jpg)

![864c981072dad7c010f9bbb643f14c0f2eaec398eb7dee03b48d84fbff0b2097.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/864c981072dad7c010f9bbb643f14c0f2eaec398eb7dee03b48d84fbff0b2097.jpg)

![8aa0615e0adbaefc385ae80c2bee609648490b920eb6f9775085a48adc34bfea.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/8aa0615e0adbaefc385ae80c2bee609648490b920eb6f9775085a48adc34bfea.jpg)

![9b5ae3d0196c7969be156cb8eb709aea3f5312b81e014164dac5be9695d0af81.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/9b5ae3d0196c7969be156cb8eb709aea3f5312b81e014164dac5be9695d0af81.jpg)

![9bd16e7ec1bc7a6325134c8c9b3a73f3e8a7b7df5123c81f10e360448fec554f.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/9bd16e7ec1bc7a6325134c8c9b3a73f3e8a7b7df5123c81f10e360448fec554f.jpg)

![a309d6713da88266b630dbaa299d9fb8c538d92a7d44ae8616abea1c8549983a.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/a309d6713da88266b630dbaa299d9fb8c538d92a7d44ae8616abea1c8549983a.jpg)

![a83a4217d46c0c42b695bed00cad110e5a1ea64fa0b9d1c4c8d87fb41fae3c9b.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/a83a4217d46c0c42b695bed00cad110e5a1ea64fa0b9d1c4c8d87fb41fae3c9b.jpg)

![b0b9084e8f40a8feae82f226148668810e51e724befa6bc77b094421630ca43e.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/b0b9084e8f40a8feae82f226148668810e51e724befa6bc77b094421630ca43e.jpg)

![b81b8917f2fa9c1323b4c45985295bd1ef9fc5cfa88ef09438222a62853dfe4e.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/b81b8917f2fa9c1323b4c45985295bd1ef9fc5cfa88ef09438222a62853dfe4e.jpg)

![bb6bda360c5ba97ea10f28e553eb36b826e6ef0ce98498db19202921f2019d0a.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/bb6bda360c5ba97ea10f28e553eb36b826e6ef0ce98498db19202921f2019d0a.jpg)

![c74197363fc1a5c314b73632ad217d644c9eeff743e0da0621b69c2124c560cf.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/c74197363fc1a5c314b73632ad217d644c9eeff743e0da0621b69c2124c560cf.jpg)

![e59c8868b20493c6c60fa4f8c897629de2f4e76081d6185fdfa6def61f6ea14c.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/e59c8868b20493c6c60fa4f8c897629de2f4e76081d6185fdfa6def61f6ea14c.jpg)

![fba8c0263f6c11a5385eaac4d1cfbd25094a5164b45f49fa19e9e43a912336a1.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/images/fba8c0263f6c11a5385eaac4d1cfbd25094a5164b45f49fa19e9e43a912336a1.jpg)

### Tables

![0dcefde4edc97424d7960dca80fde29d1c6ce3483ac2f9137c03314a63c185f7.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/tables/0dcefde4edc97424d7960dca80fde29d1c6ce3483ac2f9137c03314a63c185f7.jpg)

![af911a3d4ef6371e52b438592af89444343504ee52369a1e4664a351325c2b3e.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/tables/af911a3d4ef6371e52b438592af89444343504ee52369a1e4664a351325c2b3e.jpg)

![d96c5b476e60d83565c5ea462f04dbaa73ef10304e6e74fef998fffab55726d9.jpg](../iclr_results/586_Monitoring%20Latent%20World%20States%20in%20Language%20Models%20with%20Propositional%20Probes/tables/d96c5b476e60d83565c5ea462f04dbaa73ef10304e6e74fef998fffab55726d9.jpg)

## Universal generalization guarantees for Wasserstein distributionally robust models


### Images

![f24f2c52be1f531899f8e2de3ac44f449c14c10ab7f5b80112297a076c0a500a.jpg](../iclr_results/587_Universal%20generalization%20guarantees%20for%20Wasserstein%20distributionally%20robust%20models/images/f24f2c52be1f531899f8e2de3ac44f449c14c10ab7f5b80112297a076c0a500a.jpg)

![fd8abf05e43e0f0b6f48adbed42006db22810dd1b1e0a97598be21ca555757b9.jpg](../iclr_results/587_Universal%20generalization%20guarantees%20for%20Wasserstein%20distributionally%20robust%20models/images/fd8abf05e43e0f0b6f48adbed42006db22810dd1b1e0a97598be21ca555757b9.jpg)

## PETRA: Parallel End-to-end Training with Reversible Architectures


### Images

![31e656ba173da5d3d875e91a8403e141906200065b92a4d51fb6e83b20053c83.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/31e656ba173da5d3d875e91a8403e141906200065b92a4d51fb6e83b20053c83.jpg)

![42fd084357fd86e54f88c18b122c0cad206af92193b036b48559840c023a3097.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/42fd084357fd86e54f88c18b122c0cad206af92193b036b48559840c023a3097.jpg)

![47da7b00b0383609a36146f0be9adc122ed83b5aed5cb307943b4306f2ff829a.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/47da7b00b0383609a36146f0be9adc122ed83b5aed5cb307943b4306f2ff829a.jpg)

![55e3d1b5a227a3c879eb9a948ebf2eefa455991d1bdf27947a3879ef4ad25151.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/55e3d1b5a227a3c879eb9a948ebf2eefa455991d1bdf27947a3879ef4ad25151.jpg)

![581ba2aa59166430e30c41e2954511ead9dcf5e236337a4dd91c4eb675565d3b.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/581ba2aa59166430e30c41e2954511ead9dcf5e236337a4dd91c4eb675565d3b.jpg)

![62f54c7b46192a9e59e96cd786898d0fb6248d8b4fae7ab09c57cf33367f12ab.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/62f54c7b46192a9e59e96cd786898d0fb6248d8b4fae7ab09c57cf33367f12ab.jpg)

![67844e7ebee1ec14e2a425e36db8b5e83e708014c1eef7de5f2030bd5a6956e8.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/67844e7ebee1ec14e2a425e36db8b5e83e708014c1eef7de5f2030bd5a6956e8.jpg)

![a2897a99f70a24a2b18d88a877087d116fe875f5e6ac6e0d2972eac13ae4eb33.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/a2897a99f70a24a2b18d88a877087d116fe875f5e6ac6e0d2972eac13ae4eb33.jpg)

![c92536ce0e5da799a3ae180ef6086cac3b2329c57ad502b701a7bc6753c50c60.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/c92536ce0e5da799a3ae180ef6086cac3b2329c57ad502b701a7bc6753c50c60.jpg)

![d5a9be8a8bce87801c9faa812e73d99c1530b68cd7a0acab917534fa68f67457.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/d5a9be8a8bce87801c9faa812e73d99c1530b68cd7a0acab917534fa68f67457.jpg)

![ee4d2a5b9a10e1505f9d3e0e8c333a22036747820b9d1ac5a0b64d52dfd59271.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/ee4d2a5b9a10e1505f9d3e0e8c333a22036747820b9d1ac5a0b64d52dfd59271.jpg)

![f31ae603d3ccf689de76e21672ede12d0336a470eb9dd585b74dc46ac997703e.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/images/f31ae603d3ccf689de76e21672ede12d0336a470eb9dd585b74dc46ac997703e.jpg)

### Tables

![0ee0e1cb4a411644b6de76a1228968e058584558c44bfd25775fd103432294a3.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/tables/0ee0e1cb4a411644b6de76a1228968e058584558c44bfd25775fd103432294a3.jpg)

![45c5f7e864f8167f084d299d86678383769804f92d8a4e435db976b5deedf4b0.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/tables/45c5f7e864f8167f084d299d86678383769804f92d8a4e435db976b5deedf4b0.jpg)

![5186dc0448ee9286ccaa6e7dea9c3c2fe9a02f0481aa8ba0d612d5f417543069.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/tables/5186dc0448ee9286ccaa6e7dea9c3c2fe9a02f0481aa8ba0d612d5f417543069.jpg)

![81819f015aa25e7f81988fb26152c0198b97d99ba84681b73714320538d8177f.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/tables/81819f015aa25e7f81988fb26152c0198b97d99ba84681b73714320538d8177f.jpg)

![83b8f6e24c7276259dc856ab354fc46a4dede2e2ed42de4eb2e68911e06386ef.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/tables/83b8f6e24c7276259dc856ab354fc46a4dede2e2ed42de4eb2e68911e06386ef.jpg)

![da4c3ef962a2f8172c4fb6902f188632c746d70d1000f759ad5d3c3526318878.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/tables/da4c3ef962a2f8172c4fb6902f188632c746d70d1000f759ad5d3c3526318878.jpg)

![e91c6947090667931cf4089447f637d3d30107a90d18f150e4e5691fbdc56407.jpg](../iclr_results/588_PETRA_%20Parallel%20End-to-end%20Training%20with%20Reversible%20Architectures/tables/e91c6947090667931cf4089447f637d3d30107a90d18f150e4e5691fbdc56407.jpg)

## ThunderKittens: Simple, Fast, and $\textit{Adorable}$ Kernels


### Images

![0816f9e91d6643d89944760799d4339e45943360166927fcbf1eb146dd62a0fc.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/0816f9e91d6643d89944760799d4339e45943360166927fcbf1eb146dd62a0fc.jpg)

![0d5a4814368182ae9366697d45858e6d720dd70223b05100f50b84993e3dd405.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/0d5a4814368182ae9366697d45858e6d720dd70223b05100f50b84993e3dd405.jpg)

![182ccd40da49841e0f43e64159ec2e2707351e512ee9bb3a0bf482c9c7462a48.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/182ccd40da49841e0f43e64159ec2e2707351e512ee9bb3a0bf482c9c7462a48.jpg)

![1c2b478fd8fb6d03c99000e6c445c0cf3240595d55e460c2d98fca2abe39c7f0.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/1c2b478fd8fb6d03c99000e6c445c0cf3240595d55e460c2d98fca2abe39c7f0.jpg)

![206764faef81eab14a42535947ed7d58571d36940f12ae560fc530027820f7ad.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/206764faef81eab14a42535947ed7d58571d36940f12ae560fc530027820f7ad.jpg)

![214cec9d240731e57584f26c9575ca8f39d93bb3e328949fb4b2c236c2a2c551.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/214cec9d240731e57584f26c9575ca8f39d93bb3e328949fb4b2c236c2a2c551.jpg)

![2566c62d3410af44bef1f7d0317a28fd42e3d7d7328f22f1ceeaf6b5f3658770.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/2566c62d3410af44bef1f7d0317a28fd42e3d7d7328f22f1ceeaf6b5f3658770.jpg)

![3492f0783e008f10efe430dbe03be95de51935c7956777f88a11fc6b4dd05bae.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/3492f0783e008f10efe430dbe03be95de51935c7956777f88a11fc6b4dd05bae.jpg)

![3992f7dc332839b3610dc6a4debaf479cf89288cafa195c1807c1ae4c78d96c3.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/3992f7dc332839b3610dc6a4debaf479cf89288cafa195c1807c1ae4c78d96c3.jpg)

![3b866064ccc90859b141b0e7c457ffe5da4448bdd27a25c7f2bc0302050dc4b4.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/3b866064ccc90859b141b0e7c457ffe5da4448bdd27a25c7f2bc0302050dc4b4.jpg)

![3ce816018b6044da7b868d2c11f2b207459fc5bfa5992c9921c8b91571d596ef.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/3ce816018b6044da7b868d2c11f2b207459fc5bfa5992c9921c8b91571d596ef.jpg)

![4a7c8e74cf3250babf00c915c3d32ec87329d19c88e4a51765ef0549ec1bf61b.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/4a7c8e74cf3250babf00c915c3d32ec87329d19c88e4a51765ef0549ec1bf61b.jpg)

![4b29925f21c0b35377c2c3bda6b8e47c5f680f31a92cfaaf93d78a24f76bc0c8.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/4b29925f21c0b35377c2c3bda6b8e47c5f680f31a92cfaaf93d78a24f76bc0c8.jpg)

![4de53237705e438b2da0563d3bee04ce22cb26799fd414362f19e0f489b06642.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/4de53237705e438b2da0563d3bee04ce22cb26799fd414362f19e0f489b06642.jpg)

![4e77dfce84717a6e19bb956f3422483e393f98bc9fcee49209222d89d5a98ae3.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/4e77dfce84717a6e19bb956f3422483e393f98bc9fcee49209222d89d5a98ae3.jpg)

![4f7b3eefe12c37ee1331858b8969b25576dfeab93e8809b2da0f1213e1d5f28e.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/4f7b3eefe12c37ee1331858b8969b25576dfeab93e8809b2da0f1213e1d5f28e.jpg)

![534806484e638aad6b3149d11cfc43f05ca4cfd7f291f5eb26ce3f3bcac420bc.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/534806484e638aad6b3149d11cfc43f05ca4cfd7f291f5eb26ce3f3bcac420bc.jpg)

![64c41292aa2ed32b77f9bb1e25fa8cbe2cab5b76d283fe41fd2dc15c8d5414ca.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/64c41292aa2ed32b77f9bb1e25fa8cbe2cab5b76d283fe41fd2dc15c8d5414ca.jpg)

![693cf69a7d60008c9d24ed8baeb0d74835fabea58c16258ce2434bd63aa6337a.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/693cf69a7d60008c9d24ed8baeb0d74835fabea58c16258ce2434bd63aa6337a.jpg)

![693e315070164810765de88c37fcdc2463056e8fbb18d1ad41ab339a2229b1ea.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/693e315070164810765de88c37fcdc2463056e8fbb18d1ad41ab339a2229b1ea.jpg)

![797e7107f17ec97bc467c5f2c2a9cdf93b0b6ebb6d0085433cf3d8d22100d6df.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/797e7107f17ec97bc467c5f2c2a9cdf93b0b6ebb6d0085433cf3d8d22100d6df.jpg)

![b589171b52672f3c4afc2fe1cc26415fd75080d038207e2296693bafb1ac85d5.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/b589171b52672f3c4afc2fe1cc26415fd75080d038207e2296693bafb1ac85d5.jpg)

![cab4c9c4e796c7c06fa322bc7e0ce9e6e4fe1f6a6ba89b8fe627bad1b4aebc43.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/cab4c9c4e796c7c06fa322bc7e0ce9e6e4fe1f6a6ba89b8fe627bad1b4aebc43.jpg)

![d620c11a9dc1ad4942912188b2d2fd6d845642b124d37966a300a25d11a78702.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/d620c11a9dc1ad4942912188b2d2fd6d845642b124d37966a300a25d11a78702.jpg)

![d78b7f96303952e166f565b802ff309271469e556e275ee164e2ebf9c3004e95.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/d78b7f96303952e166f565b802ff309271469e556e275ee164e2ebf9c3004e95.jpg)

![e247109d95ca3f8bfc568994fad5a1798a915d12f8147b3c3509f093a9866217.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/e247109d95ca3f8bfc568994fad5a1798a915d12f8147b3c3509f093a9866217.jpg)

![e8cb4e293b5ee2f743a23d5cab19b605214b0e8c3a0e27713acfba684fbca415.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/e8cb4e293b5ee2f743a23d5cab19b605214b0e8c3a0e27713acfba684fbca415.jpg)

![eab4fac321dbae299e849d783899e9060457d184fa1cf7346d4aa4f4094415cc.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/eab4fac321dbae299e849d783899e9060457d184fa1cf7346d4aa4f4094415cc.jpg)

![ebc556c627977339e02a60dcbeb3607365d51aac3a71ddda07520cb84e99af88.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/ebc556c627977339e02a60dcbeb3607365d51aac3a71ddda07520cb84e99af88.jpg)

![f621e6814e4e225e78d7f736b7962aa7fcf5b55686213fc702314d79c52fa9e2.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/f621e6814e4e225e78d7f736b7962aa7fcf5b55686213fc702314d79c52fa9e2.jpg)

![f6bd7190538ab967ef68ceeaa7807a912b34a70419946ee52a5ed4bdea1bc830.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/f6bd7190538ab967ef68ceeaa7807a912b34a70419946ee52a5ed4bdea1bc830.jpg)

![fb10f3544ce83752e1377fd76fbad0f38be13bdb5597fc7f7665647b08c56d7f.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/fb10f3544ce83752e1377fd76fbad0f38be13bdb5597fc7f7665647b08c56d7f.jpg)

![fba81ba6c42e52e73ecaf1ce686bcfdb436876bdcb42c3a1f6df363a91f96143.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/images/fba81ba6c42e52e73ecaf1ce686bcfdb436876bdcb42c3a1f6df363a91f96143.jpg)

### Tables

![04481e3b73e0b785f55bc50f9295d96d85b968bd6637b467fec5b892c1ebb66b.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/04481e3b73e0b785f55bc50f9295d96d85b968bd6637b467fec5b892c1ebb66b.jpg)

![05406ad4df0fa33d9ba75b03d31a55cce76f36ebbcca8f08458ca6b6a422eb5e.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/05406ad4df0fa33d9ba75b03d31a55cce76f36ebbcca8f08458ca6b6a422eb5e.jpg)

![4235b116d3f7fb8fe620083f8f0a6dc07067cd29d1f2c596fcd78463d9728f7d.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/4235b116d3f7fb8fe620083f8f0a6dc07067cd29d1f2c596fcd78463d9728f7d.jpg)

![599b33b319b5a075caed360df4303186204222649c645e82c77364a438f505bc.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/599b33b319b5a075caed360df4303186204222649c645e82c77364a438f505bc.jpg)

![5d2a28fa0549c7c087de27012dbd65da9d64b7dd663a29fae63b5e42b61fe266.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/5d2a28fa0549c7c087de27012dbd65da9d64b7dd663a29fae63b5e42b61fe266.jpg)

![7a68f52fca7a4ef3e0c2c5502e53961ad3ffb7fb51ca7eae3108ce3b737c19dd.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/7a68f52fca7a4ef3e0c2c5502e53961ad3ffb7fb51ca7eae3108ce3b737c19dd.jpg)

![9a5bae1e5de479dc3fde7dce3a019c2d52b0a920a0805ff1e851346dc4729e7b.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/9a5bae1e5de479dc3fde7dce3a019c2d52b0a920a0805ff1e851346dc4729e7b.jpg)

![9be39c6908f83dbcfbe68c4139e65758c57bc41855a53c81a5ef6fc71db7cf4c.jpg](../iclr_results/589_ThunderKittens_%20Simple%2C%20Fast%2C%20and%20%24_textit%7BAdorable%7D%24%20Kernels/tables/9be39c6908f83dbcfbe68c4139e65758c57bc41855a53c81a5ef6fc71db7cf4c.jpg)

## Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage


### Images

![03677b183cc10c2aaf1339b41417f8b1238c4bf88e970fc7f15f59187221f856.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/03677b183cc10c2aaf1339b41417f8b1238c4bf88e970fc7f15f59187221f856.jpg)

![1763f10a809fc078627eb257ccdb38931083bae02cf706ae6b0e477e618a6baa.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/1763f10a809fc078627eb257ccdb38931083bae02cf706ae6b0e477e618a6baa.jpg)

![20285bd2ea075962366dac5db54c5c839c9d4becb75dbf5537f15f1cbb5c1382.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/20285bd2ea075962366dac5db54c5c839c9d4becb75dbf5537f15f1cbb5c1382.jpg)

![37d80d564938a1d090a2e08d7f992e7d96bdb087ddf8c81c0fa087389efb59fb.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/37d80d564938a1d090a2e08d7f992e7d96bdb087ddf8c81c0fa087389efb59fb.jpg)

![626452f7afe09c70bddcc8c7639acaa2e5fe9b03e2b613f51b40b9ca26d49de2.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/626452f7afe09c70bddcc8c7639acaa2e5fe9b03e2b613f51b40b9ca26d49de2.jpg)

![7705c1fe23f86ebb9deaf348e854d93ed2eed0f8750851a5de534704579736ca.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/7705c1fe23f86ebb9deaf348e854d93ed2eed0f8750851a5de534704579736ca.jpg)

![b41fcc8ee84907bbe1e56fab6582fcc425899391183673085d415fd5b3b9b874.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/b41fcc8ee84907bbe1e56fab6582fcc425899391183673085d415fd5b3b9b874.jpg)

![be912a680ab4288f59e8e9b4e5d23510cf0c63bbe98689b0a656b4efa58140b4.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/be912a680ab4288f59e8e9b4e5d23510cf0c63bbe98689b0a656b4efa58140b4.jpg)

![c7b2ecb58c2bf52c5b5ceed99f72b9649e726625dd9669e156c88775ed8a63e6.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/c7b2ecb58c2bf52c5b5ceed99f72b9649e726625dd9669e156c88775ed8a63e6.jpg)

![ca7fdc99fd57aca60f4eed2a272a5a9dda6755dde6a950d9e66c4dae3eeb15ec.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/ca7fdc99fd57aca60f4eed2a272a5a9dda6755dde6a950d9e66c4dae3eeb15ec.jpg)

![cbd38634171a6c7c75f9bbd8f644b3ec73789c729bf96714d381ce6a3f63b3d4.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/cbd38634171a6c7c75f9bbd8f644b3ec73789c729bf96714d381ce6a3f63b3d4.jpg)

![d2fe94b6480e2207726a399bd94760911dbd54d3938b473f16db23e0e7bd02c7.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/d2fe94b6480e2207726a399bd94760911dbd54d3938b473f16db23e0e7bd02c7.jpg)

![e7c7b78532ba4ddd7048605399b0bcc9b83180ae1e00777e32f0b4341ddaa280.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/images/e7c7b78532ba4ddd7048605399b0bcc9b83180ae1e00777e32f0b4341ddaa280.jpg)

### Tables

![052e81db36f43fa567c91c15b556f5313f2ceed8113d3ff4341f55f9d2eb2441.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/052e81db36f43fa567c91c15b556f5313f2ceed8113d3ff4341f55f9d2eb2441.jpg)

![0fb39d41a72b5265fcb9f0c9004d0cbc89be4dfd20a6419cc09c6fbd95dba324.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/0fb39d41a72b5265fcb9f0c9004d0cbc89be4dfd20a6419cc09c6fbd95dba324.jpg)

![19b4e227ee1efebe3c32fcf266567c8dc2d19eb061af38dc82e06ddac1d7788a.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/19b4e227ee1efebe3c32fcf266567c8dc2d19eb061af38dc82e06ddac1d7788a.jpg)

![1acd5325e7607bae89a8727bd6480514c566b6c6226e137f95379ef0238b7332.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/1acd5325e7607bae89a8727bd6480514c566b6c6226e137f95379ef0238b7332.jpg)

![2485410e7ccc1cf933e91bba8f2a14d84f0cf4fd31a5918b9a63373edb4b6d2b.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/2485410e7ccc1cf933e91bba8f2a14d84f0cf4fd31a5918b9a63373edb4b6d2b.jpg)

![3933e2c8ce311574ce6ca5699441362219091219037ae09f97f799bd540819bb.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/3933e2c8ce311574ce6ca5699441362219091219037ae09f97f799bd540819bb.jpg)

![494beed986af3d377e58d96ad58b6c72c296b2bdbe844ee559450957ad757204.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/494beed986af3d377e58d96ad58b6c72c296b2bdbe844ee559450957ad757204.jpg)

![57159596ef101f11765a0e2ea79aeea136a63ab73db7ad0b9d87e3fef84fdfb4.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/57159596ef101f11765a0e2ea79aeea136a63ab73db7ad0b9d87e3fef84fdfb4.jpg)

![95fd81ec8785012655c7b5031cc76f84d99612981439fb7ce453ee26c404ab19.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/95fd81ec8785012655c7b5031cc76f84d99612981439fb7ce453ee26c404ab19.jpg)

![d1e91d7b04db11ab42578ab2c962eebf2974467b1518ac5aebaadefd8c834c99.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/d1e91d7b04db11ab42578ab2c962eebf2974467b1518ac5aebaadefd8c834c99.jpg)

![dfab3f9f1877bf0cf9a4bdf53d6865476e29940e96a54682a54ce8e20ebbc2e7.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/dfab3f9f1877bf0cf9a4bdf53d6865476e29940e96a54682a54ce8e20ebbc2e7.jpg)

![e557fa7a1ee2f1e5e11f09a5ca1c7ceb36a48a6d3fbfa01204bc9198391cbd8c.jpg](../iclr_results/590_Multi-modal%20Agent%20Tuning_%20Building%20a%20VLM-Driven%20Agent%20for%20Efficient%20Tool%20Usage/tables/e557fa7a1ee2f1e5e11f09a5ca1c7ceb36a48a6d3fbfa01204bc9198391cbd8c.jpg)

## Holistically Evaluating the Environmental Impact of Creating Language Models


### Images

![af6afca81e80bae10ba51fd251bd13f7f4af9a5c8559d724c22da22aea0171e4.jpg](../iclr_results/591_Holistically%20Evaluating%20the%20Environmental%20Impact%20of%20Creating%20Language%20Models/images/af6afca81e80bae10ba51fd251bd13f7f4af9a5c8559d724c22da22aea0171e4.jpg)

![ca32abcace37e22c9a2bc945c8a3a17764d952cea0b7df2a3a2fd5e76ba3bf7e.jpg](../iclr_results/591_Holistically%20Evaluating%20the%20Environmental%20Impact%20of%20Creating%20Language%20Models/images/ca32abcace37e22c9a2bc945c8a3a17764d952cea0b7df2a3a2fd5e76ba3bf7e.jpg)

### Tables

![2cb0910f98f9418ba9dd1c8e714e6acb675fe6f27a1787378f532210d5ce0a61.jpg](../iclr_results/591_Holistically%20Evaluating%20the%20Environmental%20Impact%20of%20Creating%20Language%20Models/tables/2cb0910f98f9418ba9dd1c8e714e6acb675fe6f27a1787378f532210d5ce0a61.jpg)

![a81dec0258a5a711ddc34606d95ebea24f76ffc239984d16033a2ba5f06b5677.jpg](../iclr_results/591_Holistically%20Evaluating%20the%20Environmental%20Impact%20of%20Creating%20Language%20Models/tables/a81dec0258a5a711ddc34606d95ebea24f76ffc239984d16033a2ba5f06b5677.jpg)

![afd92b6343d74004b399c34a4fdffb41cb7fcfe9153f68579cf9e142415a0cec.jpg](../iclr_results/591_Holistically%20Evaluating%20the%20Environmental%20Impact%20of%20Creating%20Language%20Models/tables/afd92b6343d74004b399c34a4fdffb41cb7fcfe9153f68579cf9e142415a0cec.jpg)

![b04ddcfea3b61e83d8338475d39f1b0faf0fe7701a2c4e0cf8d565a001e5acd0.jpg](../iclr_results/591_Holistically%20Evaluating%20the%20Environmental%20Impact%20of%20Creating%20Language%20Models/tables/b04ddcfea3b61e83d8338475d39f1b0faf0fe7701a2c4e0cf8d565a001e5acd0.jpg)

## Adaptive Gradient Clipping for Robust Federated Learning


### Images

![02ee545b4788cdc667316cac111fc2e2f196c576032e8021ab79e52526a14123.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/02ee545b4788cdc667316cac111fc2e2f196c576032e8021ab79e52526a14123.jpg)

![03d7436e2e2c2d7f32bc872b026e444ceef2140a32a39a867727a1d270b77786.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/03d7436e2e2c2d7f32bc872b026e444ceef2140a32a39a867727a1d270b77786.jpg)

![083265e5f9940a5053a071efa76717e09eac219605a4000389528a4040ba0637.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/083265e5f9940a5053a071efa76717e09eac219605a4000389528a4040ba0637.jpg)

![104ad214d47f9520b7f2adcd11e5439c049083b189f66adef443d6b122cdf535.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/104ad214d47f9520b7f2adcd11e5439c049083b189f66adef443d6b122cdf535.jpg)

![1feaff50cb2fb5d0890c56198c062502587db0452accede3283a8dbf2eaa3670.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/1feaff50cb2fb5d0890c56198c062502587db0452accede3283a8dbf2eaa3670.jpg)

![24efb479c192aa455f3743a4af7db58d7d8daf00a6242d0aba236b6c29d47ec4.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/24efb479c192aa455f3743a4af7db58d7d8daf00a6242d0aba236b6c29d47ec4.jpg)

![473c05f61a17856f164c903b3a57ea4cc41532c885e5b47271978116b3b87272.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/473c05f61a17856f164c903b3a57ea4cc41532c885e5b47271978116b3b87272.jpg)

![49a28c94df3067a102fc6d8ed507f29939347a21006cc9b2c4a672342f042229.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/49a28c94df3067a102fc6d8ed507f29939347a21006cc9b2c4a672342f042229.jpg)

![59effbb3a4db4ca3520b0fc8f037937a05effc111d815703acb5ed0e36c5cc8e.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/59effbb3a4db4ca3520b0fc8f037937a05effc111d815703acb5ed0e36c5cc8e.jpg)

![5d068a408513848c9231ea4e14440afcd778e6852aca9d649a7385ecf68213e8.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/5d068a408513848c9231ea4e14440afcd778e6852aca9d649a7385ecf68213e8.jpg)

![627f66e98d73a736f9085538c0921ad458fbc0543dd19637f5e7b8dd6b5ae9a9.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/627f66e98d73a736f9085538c0921ad458fbc0543dd19637f5e7b8dd6b5ae9a9.jpg)

![673f2fe2939cb1541b395a76331ae86846da9aa07ba17417ecab3b524fcd20b7.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/673f2fe2939cb1541b395a76331ae86846da9aa07ba17417ecab3b524fcd20b7.jpg)

![740f5ad090ae11e9f2093effc18eaf2e6030c760b39d94434015145c56f2b269.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/740f5ad090ae11e9f2093effc18eaf2e6030c760b39d94434015145c56f2b269.jpg)

![7e4bfdd428462674f84bca8005c4bb9a6f728728db7ca73c581b580072792aee.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/7e4bfdd428462674f84bca8005c4bb9a6f728728db7ca73c581b580072792aee.jpg)

![7fb1d24b26421cd98304b0d5aa484866fefb67e1ef0d37c2504b3e2af4649607.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/7fb1d24b26421cd98304b0d5aa484866fefb67e1ef0d37c2504b3e2af4649607.jpg)

![8a1d69eae3d7f431e9de5f19e3dce203813ea381337ab05fe452a90db54709a3.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/8a1d69eae3d7f431e9de5f19e3dce203813ea381337ab05fe452a90db54709a3.jpg)

![a31237df371f8e299023dc619292b9b1fc0625f13c08083e709f237c9a5a37cb.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/a31237df371f8e299023dc619292b9b1fc0625f13c08083e709f237c9a5a37cb.jpg)

![a3faa64ec6c8cee1fb954eac422ea00692b119f6b0ea979305b06b6831b8d254.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/a3faa64ec6c8cee1fb954eac422ea00692b119f6b0ea979305b06b6831b8d254.jpg)

![a812975e144cb744013d91edbd6b3db8325516b8755939f4b29c8e08322f5b85.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/a812975e144cb744013d91edbd6b3db8325516b8755939f4b29c8e08322f5b85.jpg)

![b05b209775ecfdff352dedfc9e6f118965bd5f098a0fbd6aa9e0f4e053cfc829.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/b05b209775ecfdff352dedfc9e6f118965bd5f098a0fbd6aa9e0f4e053cfc829.jpg)

![bc95ad0a6dc89ad4532f7b33dc14935145dadc8b1e112e22586ab5e7384befce.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/bc95ad0a6dc89ad4532f7b33dc14935145dadc8b1e112e22586ab5e7384befce.jpg)

![c125c7827817d9389ed5d8b8606b253f8f0e5815bc77c0d83a09fe6c88730fd5.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/c125c7827817d9389ed5d8b8606b253f8f0e5815bc77c0d83a09fe6c88730fd5.jpg)

![d08a9096374c4bc07343e8e224b00de06d056acdd1c615d2c2c536e23ab8c7c2.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/d08a9096374c4bc07343e8e224b00de06d056acdd1c615d2c2c536e23ab8c7c2.jpg)

![d153d07567a771ae733264683627b8a995320a26b0b2034de099bc552894c120.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/d153d07567a771ae733264683627b8a995320a26b0b2034de099bc552894c120.jpg)

![d1708d84c549766b4f8a9858c9293f21bf43bb431bf39848c8e8ef51bda56679.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/d1708d84c549766b4f8a9858c9293f21bf43bb431bf39848c8e8ef51bda56679.jpg)

![d2cb8936e802e253cd16ad628864cdea8441fba2d02131f524d81ab8bf0c7f1e.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/d2cb8936e802e253cd16ad628864cdea8441fba2d02131f524d81ab8bf0c7f1e.jpg)

![f94daff88f45dcf227503b346eec9bc2c09f93d33998fb5c2df2d71367b582d4.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/images/f94daff88f45dcf227503b346eec9bc2c09f93d33998fb5c2df2d71367b582d4.jpg)

### Tables

![04cc6ccebdbf22f291d7e2fa9357c1e427a3234c6551a697a11f19214aa63481.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/tables/04cc6ccebdbf22f291d7e2fa9357c1e427a3234c6551a697a11f19214aa63481.jpg)

![2cc6d43b00a474ba86857947fd792429fcaf082399fd7b3a9c71b316ab985971.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/tables/2cc6d43b00a474ba86857947fd792429fcaf082399fd7b3a9c71b316ab985971.jpg)

![2d8f6135792b130722d4ae7994a54b161d97ee4493998d91766daa8855a99fff.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/tables/2d8f6135792b130722d4ae7994a54b161d97ee4493998d91766daa8855a99fff.jpg)

![67434d19f6d89ea4ab18ec00576aa122d91007b91959942ac1a7bd65319a6ad2.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/tables/67434d19f6d89ea4ab18ec00576aa122d91007b91959942ac1a7bd65319a6ad2.jpg)

![776d783358a1113f5b30ffd9311d4afce25b18820cd67f6f1fa9d5387550a90a.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/tables/776d783358a1113f5b30ffd9311d4afce25b18820cd67f6f1fa9d5387550a90a.jpg)

![c7604aea19b8d81c0043e1f879f0bd89b1c8cbac75a66f0404a966bb68b49263.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/tables/c7604aea19b8d81c0043e1f879f0bd89b1c8cbac75a66f0404a966bb68b49263.jpg)

![cf147e538a077615c070859835c5d68c95eca5e090f845a3863be21e726b43f2.jpg](../iclr_results/592_Adaptive%20Gradient%20Clipping%20for%20Robust%20Federated%20Learning/tables/cf147e538a077615c070859835c5d68c95eca5e090f845a3863be21e726b43f2.jpg)

## DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback


### Images

![06528ff676e8b2c64388f64bfd2df33a112094677d1496c280ab3ff3dde91ccd.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/06528ff676e8b2c64388f64bfd2df33a112094677d1496c280ab3ff3dde91ccd.jpg)

![1195ea31820eb7d3b8e6beb343d51fe16162a43f0ae5c931a23926a457623e41.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/1195ea31820eb7d3b8e6beb343d51fe16162a43f0ae5c931a23926a457623e41.jpg)

![1df44d3cf85ab2c511c3929e2862c8b4b86972693212fe0f7041e959a26d3e55.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/1df44d3cf85ab2c511c3929e2862c8b4b86972693212fe0f7041e959a26d3e55.jpg)

![55a920a9c8b7f08f84b89bbab0a45a3f407a42c9a3906e334fc62afdd9094925.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/55a920a9c8b7f08f84b89bbab0a45a3f407a42c9a3906e334fc62afdd9094925.jpg)

![5f70d2b93df2aafce2d38b1b89e5dc5d86779907c398bb5dab05b9162fe6035b.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/5f70d2b93df2aafce2d38b1b89e5dc5d86779907c398bb5dab05b9162fe6035b.jpg)

![6c79d7433742e99b0f6dc58916e1f594860b9976e53c2e7cd8c877fd9d9337b8.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/6c79d7433742e99b0f6dc58916e1f594860b9976e53c2e7cd8c877fd9d9337b8.jpg)

![6feb293a886d516212d5e387bcc1bfd4acc87bcbbacf1e172bae67367e8856fb.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/6feb293a886d516212d5e387bcc1bfd4acc87bcbbacf1e172bae67367e8856fb.jpg)

![7cac564817c8e235943bd42fc3f18625fb5b62ae17582ee63b22c9f49fbf8445.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/7cac564817c8e235943bd42fc3f18625fb5b62ae17582ee63b22c9f49fbf8445.jpg)

![88da51740df79737a0d3707e258420578db4f07d1d6df6f21cc7aed744210ca2.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/88da51740df79737a0d3707e258420578db4f07d1d6df6f21cc7aed744210ca2.jpg)

![ae71a9db51efaef10b41ba145296635f42a264743edc65ce60c15b59973eca66.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/ae71a9db51efaef10b41ba145296635f42a264743edc65ce60c15b59973eca66.jpg)

![b21b3332be4018bbfb36f18c6fb54a55cc9239c2d08c3ed2d6735e6203d6731b.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/b21b3332be4018bbfb36f18c6fb54a55cc9239c2d08c3ed2d6735e6203d6731b.jpg)

![bae9fb45d2701def5e1a47945e40dec84b6b1d2004832b168f8627f8ef7c2e7b.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/bae9fb45d2701def5e1a47945e40dec84b6b1d2004832b168f8627f8ef7c2e7b.jpg)

![bcb31e6643f29632f1c20aa1edd2761dfa8c5ccf6fad2b169cc21589147a6fe7.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/bcb31e6643f29632f1c20aa1edd2761dfa8c5ccf6fad2b169cc21589147a6fe7.jpg)

![bcba92f087629d9ab1314c218bf40c93704b98f6ea8f5f8d804f03f6501d393e.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/bcba92f087629d9ab1314c218bf40c93704b98f6ea8f5f8d804f03f6501d393e.jpg)

![c00bba759cccb27bc627039193d22e49f7d8849bca9bef3d8dc2d7df3e7f4ba4.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/c00bba759cccb27bc627039193d22e49f7d8849bca9bef3d8dc2d7df3e7f4ba4.jpg)

![c219ceda944b995a4fb8ecfc0d40532c16a0f74da68de6d10bb7c65b70ca00e3.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/c219ceda944b995a4fb8ecfc0d40532c16a0f74da68de6d10bb7c65b70ca00e3.jpg)

![de26d6f7ae1a9c4430d7865bf39c148f0c0645bc13a4f10c875637b68d8c1b39.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/de26d6f7ae1a9c4430d7865bf39c148f0c0645bc13a4f10c875637b68d8c1b39.jpg)

![fbd5decf3554205f5060a1ec78ed59b6f2284b8ae2bee8673a2ff30cb4ce00af.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/images/fbd5decf3554205f5060a1ec78ed59b6f2284b8ae2bee8673a2ff30cb4ce00af.jpg)

### Tables

![057d4b07c6fc54df8a20a2d649624c6ad55e7a262a49b232aa857bc490323614.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/057d4b07c6fc54df8a20a2d649624c6ad55e7a262a49b232aa857bc490323614.jpg)

![23b497a4c714199904d47b99a5fc621da4a6d89ee3f5a0717323e5817be17eee.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/23b497a4c714199904d47b99a5fc621da4a6d89ee3f5a0717323e5817be17eee.jpg)

![38ce09570f03e48f28eace741f0b48a6b186efb3adc0f2c89fa72e33ee73592d.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/38ce09570f03e48f28eace741f0b48a6b186efb3adc0f2c89fa72e33ee73592d.jpg)

![47e72441de3ba87aa5e1b9834948e42d78d65dfea66bd12a29d6709b55e94a9c.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/47e72441de3ba87aa5e1b9834948e42d78d65dfea66bd12a29d6709b55e94a9c.jpg)

![b613f07b378deb69e1b8afd33c5a854f5581983d6fbef5f86c86812d7a2bc6a8.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/b613f07b378deb69e1b8afd33c5a854f5581983d6fbef5f86c86812d7a2bc6a8.jpg)

![d178c7c7b29f7f8d46585897c59f7b1a9fc962aae60cd81a0b2c931db44899d0.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/d178c7c7b29f7f8d46585897c59f7b1a9fc962aae60cd81a0b2c931db44899d0.jpg)

![e4d3aeb68d9a716cecee729c21aee4bdaf6dfe2a5d9370600874f39d42d56076.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/e4d3aeb68d9a716cecee729c21aee4bdaf6dfe2a5d9370600874f39d42d56076.jpg)

![f8448f613691be3b03c7144264bc782b73c1a5e89772675afe821eb4bf446ef2.jpg](../iclr_results/593_DataEnvGym_%20Data%20Generation%20Agents%20in%20Teacher%20Environments%20with%20Student%20Feedback/tables/f8448f613691be3b03c7144264bc782b73c1a5e89772675afe821eb4bf446ef2.jpg)

## Re-Imagining Multimodal Instruction Tuning: A Representation View


### Images

![2205ef7cdbe34e371df96527dedec16e550288405da47b04ab39c0ac4bd56d0b.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/2205ef7cdbe34e371df96527dedec16e550288405da47b04ab39c0ac4bd56d0b.jpg)

![270ea1dcc6b5b522177ff550426eb297a48c3f1e48055d5bfd387926e7b7df18.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/270ea1dcc6b5b522177ff550426eb297a48c3f1e48055d5bfd387926e7b7df18.jpg)

![58bf63304385dd9eb423eeb20f45c80b11fec280cfe54907f2a9aca2d986a3e1.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/58bf63304385dd9eb423eeb20f45c80b11fec280cfe54907f2a9aca2d986a3e1.jpg)

![75715e430155f6ecf998cab27ffd1a50f14b7767c4fc8ba52dfbb5ecb9501027.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/75715e430155f6ecf998cab27ffd1a50f14b7767c4fc8ba52dfbb5ecb9501027.jpg)

![868f4ac4750396ac521a4621c215284abe24dc10587cd74ff98c29abc805b7b2.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/868f4ac4750396ac521a4621c215284abe24dc10587cd74ff98c29abc805b7b2.jpg)

![92bf76e1864c8eb2111e9c864b257a8c0d506abd07d9871484b4ef543221d014.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/92bf76e1864c8eb2111e9c864b257a8c0d506abd07d9871484b4ef543221d014.jpg)

![c1d11429791f5f9fb07c7c5150f0b91e00ce10282a5e81872a1f3d837a6cb52b.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/c1d11429791f5f9fb07c7c5150f0b91e00ce10282a5e81872a1f3d837a6cb52b.jpg)

![c23d77015f52c853caba92d3019b20cd7383d99d2f3e4d7cca06d41a3f5b6ada.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/c23d77015f52c853caba92d3019b20cd7383d99d2f3e4d7cca06d41a3f5b6ada.jpg)

![ddf70cf18af71bcde99a88da5de67f365340e848f0c6b3e95c31337d24491524.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/images/ddf70cf18af71bcde99a88da5de67f365340e848f0c6b3e95c31337d24491524.jpg)

### Tables

![057e17c7b8ff0000f9a156a030bde843c586138ba35eea259d86ecf6f775cf7d.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/057e17c7b8ff0000f9a156a030bde843c586138ba35eea259d86ecf6f775cf7d.jpg)

![223718410e54364f205a3ef7256aada4e333c67ddc0ce8041e5b475fecf0bd4a.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/223718410e54364f205a3ef7256aada4e333c67ddc0ce8041e5b475fecf0bd4a.jpg)

![24fb599d0c90153b4f9de2517d30521f807f5616768245e6eda520c80f5c29ea.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/24fb599d0c90153b4f9de2517d30521f807f5616768245e6eda520c80f5c29ea.jpg)

![26e84c3545939f44bc3afda770384878d5e498e0ed5a63497e374a5bf405bd64.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/26e84c3545939f44bc3afda770384878d5e498e0ed5a63497e374a5bf405bd64.jpg)

![42f1b5a95babe6653ac8c62d4ca336dff2792146c9983d0ca3d9606a30e51c14.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/42f1b5a95babe6653ac8c62d4ca336dff2792146c9983d0ca3d9606a30e51c14.jpg)

![5f4aa1ab57fffd7bcb24ba9fdc9462c1b9c778f5b3855919131bc2f9e5fb4ec1.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/5f4aa1ab57fffd7bcb24ba9fdc9462c1b9c778f5b3855919131bc2f9e5fb4ec1.jpg)

![67607dfd07476fd1e17df5947568f8a4bcb455ca280e819cfb3f75d561671dbd.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/67607dfd07476fd1e17df5947568f8a4bcb455ca280e819cfb3f75d561671dbd.jpg)

![7310ba98cfa99737388fb1e4c3d0b40e8b6b0159fb01ceb002122a17cd0bd12f.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/7310ba98cfa99737388fb1e4c3d0b40e8b6b0159fb01ceb002122a17cd0bd12f.jpg)

![90635c7ef2c60ea4aeda7db3b1fce74b6be432c88774458f45c6c4642cf11c1c.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/90635c7ef2c60ea4aeda7db3b1fce74b6be432c88774458f45c6c4642cf11c1c.jpg)

![c1a1ac79980cb35673a4284eb821be32ef96ff1f1639ef9c41baf0ff54accae2.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/c1a1ac79980cb35673a4284eb821be32ef96ff1f1639ef9c41baf0ff54accae2.jpg)

![c5d8ddd616a1dbe2a72815a5e22b003683c2772c1d4e170ceb60583ed1558b31.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/c5d8ddd616a1dbe2a72815a5e22b003683c2772c1d4e170ceb60583ed1558b31.jpg)

![ca7d8b6bcb33b3f6f9f3a5267f9939d73cfb3ae6889f6d4e08334cc9d3df515d.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/ca7d8b6bcb33b3f6f9f3a5267f9939d73cfb3ae6889f6d4e08334cc9d3df515d.jpg)

![eeb02927d25a885ebf734a49ee1cbdcaffa40a310abd7f60b7b85fabf4b9bd5d.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/eeb02927d25a885ebf734a49ee1cbdcaffa40a310abd7f60b7b85fabf4b9bd5d.jpg)

![f4e973d0536ae0e8191ad895e77074f55d8959aa35944ee0cd95797bf3e27dfa.jpg](../iclr_results/594_Re-Imagining%20Multimodal%20Instruction%20Tuning_%20A%20Representation%20View/tables/f4e973d0536ae0e8191ad895e77074f55d8959aa35944ee0cd95797bf3e27dfa.jpg)

## Inverse decision-making using neural amortized Bayesian actors


### Images

![5df12ef7061dc629c473f147288eecb8f8acc5a291869d958d72c6d602e789ef.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/5df12ef7061dc629c473f147288eecb8f8acc5a291869d958d72c6d602e789ef.jpg)

![7a5b13ff5116f30a43d421d7e914b703c34e2de210d108476aae87de93596afe.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/7a5b13ff5116f30a43d421d7e914b703c34e2de210d108476aae87de93596afe.jpg)

![91980472e386ff8a1f30c76a5e6c6bc94a0a4bbeadeed94f7f8c2cc1c2ea9e1a.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/91980472e386ff8a1f30c76a5e6c6bc94a0a4bbeadeed94f7f8c2cc1c2ea9e1a.jpg)

![bc577177a2a28ac980fb9f18e442d8c783b3f9a84088a81bda22d97846902d1a.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/bc577177a2a28ac980fb9f18e442d8c783b3f9a84088a81bda22d97846902d1a.jpg)

![c18ac4e2f68f5d5d767cc61092939521624e1606bf821f370898a39eff8cd31d.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/c18ac4e2f68f5d5d767cc61092939521624e1606bf821f370898a39eff8cd31d.jpg)

![c63a754e68b88c1e2fd0bc41d554ca51b34a90878eefa713161865706133dd1a.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/c63a754e68b88c1e2fd0bc41d554ca51b34a90878eefa713161865706133dd1a.jpg)

![e685894059de70b6839266bab30a9dd197ba93d6f1a3184f788dc7dbe10b4468.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/e685894059de70b6839266bab30a9dd197ba93d6f1a3184f788dc7dbe10b4468.jpg)

![ec941f366807065b2e3ad5b19a4706f93d1220d3e437b9a54502d74355502fe4.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/ec941f366807065b2e3ad5b19a4706f93d1220d3e437b9a54502d74355502fe4.jpg)

![f03ba9b76cd6800ad9e4c69446058ad3641b4ccad95090c3540bdb345077ab11.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/images/f03ba9b76cd6800ad9e4c69446058ad3641b4ccad95090c3540bdb345077ab11.jpg)

### Tables

![6a5f533f1d3c237083fb3be1ac783a8f2fd51aada6441ba11a43dadf9ad094a7.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/tables/6a5f533f1d3c237083fb3be1ac783a8f2fd51aada6441ba11a43dadf9ad094a7.jpg)

![926fe0da0a22655df6215a27d567c5e3ab3903481fe4013d37a33b9a35645efc.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/tables/926fe0da0a22655df6215a27d567c5e3ab3903481fe4013d37a33b9a35645efc.jpg)

![e0ef52165d876068f6d49d549dc6cd1b72ab79fa49c1a3f8bd249eaa78927cc7.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/tables/e0ef52165d876068f6d49d549dc6cd1b72ab79fa49c1a3f8bd249eaa78927cc7.jpg)

![fe4d3b56d323f21dbb5da88560423613af496a70cd9e9d3a6e5ae3a9b275cf97.jpg](../iclr_results/595_Inverse%20decision-making%20using%20neural%20amortized%20Bayesian%20actors/tables/fe4d3b56d323f21dbb5da88560423613af496a70cd9e9d3a6e5ae3a9b275cf97.jpg)

## Designing Concise ConvNets with Columnar Stages


### Images

![0652de361ebe33ce87e1c17575b8f6f5677de6a5316e9b0b2be2796b807c9e51.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/0652de361ebe33ce87e1c17575b8f6f5677de6a5316e9b0b2be2796b807c9e51.jpg)

![086d05b921cb31b8244a8b3dc1010ae60c0ce16d245e712351efae6430c74265.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/086d05b921cb31b8244a8b3dc1010ae60c0ce16d245e712351efae6430c74265.jpg)

![3dddc0d42625f2be2945c42deb2915fbff1642aa1fe69ba1efc9f6570b1b3482.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/3dddc0d42625f2be2945c42deb2915fbff1642aa1fe69ba1efc9f6570b1b3482.jpg)

![55ebfbc975fd4ccdecd80609f03ef384748b9e006c96b550313d6a1b0b881a30.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/55ebfbc975fd4ccdecd80609f03ef384748b9e006c96b550313d6a1b0b881a30.jpg)

![6a37bca9eb9ab7a818cb6ea32acde42d4b96d9485f23fe8b66a678881bb86129.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/6a37bca9eb9ab7a818cb6ea32acde42d4b96d9485f23fe8b66a678881bb86129.jpg)

![b19167ec9e48ad5d3f0b9753f8b17f3a499ee10f831a30e162d632eb91de3709.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/b19167ec9e48ad5d3f0b9753f8b17f3a499ee10f831a30e162d632eb91de3709.jpg)

![b71f59abf6e58caf579ecd0f57f16cb58e69f7080381637bc576a1a8c5291b44.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/b71f59abf6e58caf579ecd0f57f16cb58e69f7080381637bc576a1a8c5291b44.jpg)

![fa1371c935dfa2116a7fdcaea4ffff3265b3f458bec8de5c102a4b2071f47461.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/images/fa1371c935dfa2116a7fdcaea4ffff3265b3f458bec8de5c102a4b2071f47461.jpg)

### Tables

![07f7ee16b0e3065cc023dfa45eaa235151bbdf78a74d6bed035a929c266f036b.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/07f7ee16b0e3065cc023dfa45eaa235151bbdf78a74d6bed035a929c266f036b.jpg)

![3278c33a6e28ca061dc0e7fcd4e36959848756ff3cab1b796216ce33dd50b695.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/3278c33a6e28ca061dc0e7fcd4e36959848756ff3cab1b796216ce33dd50b695.jpg)

![4120fca43e1771ae3d22232f4c145ac43bc1136c62b311c0a8e4cda7b9231d4a.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/4120fca43e1771ae3d22232f4c145ac43bc1136c62b311c0a8e4cda7b9231d4a.jpg)

![422cee9cdb7e2a5ec59ab6eb2852ba636cf963d7a3ced7b36693eec0005a33cd.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/422cee9cdb7e2a5ec59ab6eb2852ba636cf963d7a3ced7b36693eec0005a33cd.jpg)

![4dbdedb80b837732dfc61170c2f37235e2bb41bb415e05747116edc949d44863.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/4dbdedb80b837732dfc61170c2f37235e2bb41bb415e05747116edc949d44863.jpg)

![5ecca3e6d869305bc5df083533f99e0bfbf94a1c8896d5a4270d2701e7178b94.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/5ecca3e6d869305bc5df083533f99e0bfbf94a1c8896d5a4270d2701e7178b94.jpg)

![b4fe8f08f2333f6060ff449195d230fa41ff3094ddd5923d58a01f51f49dd604.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/b4fe8f08f2333f6060ff449195d230fa41ff3094ddd5923d58a01f51f49dd604.jpg)

![f77a2d44a93fbb56630e114bdd62c7d232a12966a0e6704b757b86ed4db6253c.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/f77a2d44a93fbb56630e114bdd62c7d232a12966a0e6704b757b86ed4db6253c.jpg)

![fc16e320b501a7875eb2816a267a7d6ff4c3d5583674855eba53f7278c1ed248.jpg](../iclr_results/596_Designing%20Concise%20ConvNets%20with%20Columnar%20Stages/tables/fc16e320b501a7875eb2816a267a7d6ff4c3d5583674855eba53f7278c1ed248.jpg)

## MVTokenFlow: High-quality 4D Content Generation using Multiview Token Flow


### Images

![0eab6380018420da48639e89a52d464e3f48827882375750d26cdc496a9309b5.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/0eab6380018420da48639e89a52d464e3f48827882375750d26cdc496a9309b5.jpg)

![18e2597161227ccc0c60d2ff9321c46aef769de6189a2df691f75b6140071a89.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/18e2597161227ccc0c60d2ff9321c46aef769de6189a2df691f75b6140071a89.jpg)

![1b7779a74ebc4357c9d6efa1f46110732ec3c7b2a75ddd959aa759db0b1367ae.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/1b7779a74ebc4357c9d6efa1f46110732ec3c7b2a75ddd959aa759db0b1367ae.jpg)

![3d0de529568f78cf9fda6b94abe829a971a35a7d2ceaab12b60791969d8925f9.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/3d0de529568f78cf9fda6b94abe829a971a35a7d2ceaab12b60791969d8925f9.jpg)

![3f9fbe826313a625ddfdb9d6f5f6605b5a93516a794acdca54ccdc8aed9f08d3.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/3f9fbe826313a625ddfdb9d6f5f6605b5a93516a794acdca54ccdc8aed9f08d3.jpg)

![529685756546b7732e161827f117315a7a381515fcaaab1bc84eec96c2461eb6.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/529685756546b7732e161827f117315a7a381515fcaaab1bc84eec96c2461eb6.jpg)

![698f08eb793ab2e600c4d1f8a6b9c4e580946d7e3260e3ef11ed5a9113c76e14.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/698f08eb793ab2e600c4d1f8a6b9c4e580946d7e3260e3ef11ed5a9113c76e14.jpg)

![72c3b371067aae7bfa50621eb697620e78af2093cf985d05936b3dd7c19c4af8.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/72c3b371067aae7bfa50621eb697620e78af2093cf985d05936b3dd7c19c4af8.jpg)

![81acf1153c8cf0341c05a1d5dbe9af8e46461f90b0c5590fa84370b0848b3ced.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/81acf1153c8cf0341c05a1d5dbe9af8e46461f90b0c5590fa84370b0848b3ced.jpg)

![b1c9bf6778f826b435379684cfaabbbacda6a2ecd757b9e7236aa4e9b1fc26d4.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/b1c9bf6778f826b435379684cfaabbbacda6a2ecd757b9e7236aa4e9b1fc26d4.jpg)

![baa12edd11796396a2a2ca92f897358dd797c8c9a85471d37a1497975e941b36.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/images/baa12edd11796396a2a2ca92f897358dd797c8c9a85471d37a1497975e941b36.jpg)

### Tables

![5efe7c8506195e0a77ba344d707635a2e9e53982333efed8ef4dbd3bf9c781c1.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/tables/5efe7c8506195e0a77ba344d707635a2e9e53982333efed8ef4dbd3bf9c781c1.jpg)

![a24fc3e477989b70c04e99fccea4ca630024c56316708a58fd454dff1a9887e3.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/tables/a24fc3e477989b70c04e99fccea4ca630024c56316708a58fd454dff1a9887e3.jpg)

![ab1d1a0d37a204123df0ecf8163dc10f9137579252eee962d78e4f933e0521d4.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/tables/ab1d1a0d37a204123df0ecf8163dc10f9137579252eee962d78e4f933e0521d4.jpg)

![c3dfed7e779c47c61965a8ab940a0662564a8640177f8c21ec93882b0fdbe9e7.jpg](../iclr_results/597_MVTokenFlow_%20High-quality%204D%20Content%20Generation%20using%20Multiview%20Token%20Flow/tables/c3dfed7e779c47c61965a8ab940a0662564a8640177f8c21ec93882b0fdbe9e7.jpg)

## Let the Code LLM Edit Itself When You Edit the Code

### Images

![344ee96ad989463b6a1b61013e4b50481ce8f9bf9050bb1d45f706ba62a98644.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/images/344ee96ad989463b6a1b61013e4b50481ce8f9bf9050bb1d45f706ba62a98644.jpg)

![726343bf0a5c61fa51aae8b77cb904a97adad5f2a6cd1fa089105b7c4d7a4661.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/images/726343bf0a5c61fa51aae8b77cb904a97adad5f2a6cd1fa089105b7c4d7a4661.jpg)

![7a4e450b151632265f3be53ea846bcada17bff2e0b8b0e8f77d052156bce0cbf.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/images/7a4e450b151632265f3be53ea846bcada17bff2e0b8b0e8f77d052156bce0cbf.jpg)

![7fbb33a721736ce3bb84f1d9ecc4638d37e23d5b0034c1d46970a3c7b000ca81.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/images/7fbb33a721736ce3bb84f1d9ecc4638d37e23d5b0034c1d46970a3c7b000ca81.jpg)

### Tables

![24859c4bbce8702f1dc1380e185b40542ac25e94495fe2ebafa7a150d321aed4.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/24859c4bbce8702f1dc1380e185b40542ac25e94495fe2ebafa7a150d321aed4.jpg)

![2725b444e5d573ca52c813b85aefad2a597c65a9c47fbe1535c49ac3b0f1dd36.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/2725b444e5d573ca52c813b85aefad2a597c65a9c47fbe1535c49ac3b0f1dd36.jpg)

![4330afbf6f87c5df72a827d69e61a965d779ee13536f4050879a532929be00a3.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/4330afbf6f87c5df72a827d69e61a965d779ee13536f4050879a532929be00a3.jpg)

![455991bc177ca2b8c957dc241b172a9f609c2812972657cd37ae9a166ebff546.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/455991bc177ca2b8c957dc241b172a9f609c2812972657cd37ae9a166ebff546.jpg)

![5ee6f46560c94b2e6d8e87663012f134874d3d9d590a354154cef68bcac8f3fb.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/5ee6f46560c94b2e6d8e87663012f134874d3d9d590a354154cef68bcac8f3fb.jpg)

![728aca67eec5eafff15c96b4203238a4e9352a8ee361ece98036a23f3d4ec3f9.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/728aca67eec5eafff15c96b4203238a4e9352a8ee361ece98036a23f3d4ec3f9.jpg)

![90683526e6c60046db9577c670da7a8def1a3a82faa1b419617cfde696e9538f.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/90683526e6c60046db9577c670da7a8def1a3a82faa1b419617cfde696e9538f.jpg)

![9cb17353ae8c78605fec2ca0d3c2c00a04556412be029b9f8ee81066fdfeabdd.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/9cb17353ae8c78605fec2ca0d3c2c00a04556412be029b9f8ee81066fdfeabdd.jpg)

![d05e382444e0fd3a46cde3901e8cc8392d4453a762be89647c79cc2edc86129b.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/d05e382444e0fd3a46cde3901e8cc8392d4453a762be89647c79cc2edc86129b.jpg)

![dc5328d2c69afc719f70716c3ac759a8ff3946cc6b2f7798594b10b2883640d7.jpg](../iclr_results/598_Let%20the%20Code%20LLM%20Edit%20Itself%20When%20You%20Edit%20the%20Code/tables/dc5328d2c69afc719f70716c3ac759a8ff3946cc6b2f7798594b10b2883640d7.jpg)
