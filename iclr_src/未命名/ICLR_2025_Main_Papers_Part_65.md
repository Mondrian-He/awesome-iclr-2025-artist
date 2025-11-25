# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 65 of 100

## 目录 (Table of Contents)

1. [LLM-wrapper: Black-Box Semantic-Aware Adaptation of Vision-Language Models for Referring Expression Comprehension](#LLM-wrapper-Black-Box-Semantic-Aware-Adaptation-of-Vision-Language-Models-for-Referring-Expression-Comprehension)
2. [Deep MMD Gradient Flow without adversarial training](#Deep-MMD-Gradient-Flow-without-adversarial-training)
3. [A transfer learning framework for weak to strong generalization](#A-transfer-learning-framework-for-weak-to-strong-generalization)
4. [What to align in multimodal contrastive learning?](#What-to-align-in-multimodal-contrastive-learning)
5. [GenVP: Generating Visual Puzzles with Contrastive Hierarchical VAEs](#GenVP-Generating-Visual-Puzzles-with-Contrastive-Hierarchical-VAEs)
6. [Towards Continuous Reuse of Graph Models via Holistic Memory Diversification](#Towards-Continuous-Reuse-of-Graph-Models-via-Holistic-Memory-Diversification)
7. [Lightweight Predictive 3D Gaussian Splats](#Lightweight-Predictive-3D-Gaussian-Splats)
8. [Universal Image Restoration Pre-training via Degradation Classification](#Universal-Image-Restoration-Pre-training-via-Degradation-Classification)
9. [Connectome Mapping: Shape-Memory Network via Interpretation of Contextual Semantic Information](#Connectome-Mapping-Shape-Memory-Network-via-Interpretation-of-Contextual-Semantic-Information)
10. [LLaMA-Omni: Seamless Speech Interaction with Large Language Models](#LLaMA-Omni-Seamless-Speech-Interaction-with-Large-Language-Models)
11. [SONICS: Synthetic Or Not - Identifying Counterfeit Songs](#SONICS-Synthetic-Or-Not-Identifying-Counterfeit-Songs)
12. [Execution-guided within-prompt search for programming-by-example](#Execution-guided-within-prompt-search-for-programming-by-example)
13. [Interpretable Unsupervised Joint Denoising and Enhancement for Real-World low-light Scenarios](#Interpretable-Unsupervised-Joint-Denoising-and-Enhancement-for-Real-World-low-light-Scenarios)
14. [A Training-Free Sub-quadratic Cost Transformer Model Serving Framework with Hierarchically Pruned Attention](#A-Training-Free-Sub-quadratic-Cost-Transformer-Model-Serving-Framework-with-Hierarchically-Pruned-Attention)
15. [ReSi: A Comprehensive Benchmark for Representational Similarity Measures](#ReSi-A-Comprehensive-Benchmark-for-Representational-Similarity-Measures)
16. [Autoregressive Pretraining with Mamba in Vision](#Autoregressive-Pretraining-with-Mamba-in-Vision)
17. [Improved Training Technique for Latent Consistency Models](#Improved-Training-Technique-for-Latent-Consistency-Models)
18. [Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization](#Mitigating-Reward-Over-Optimization-in-RLHF-via-Behavior-Supported-Regularization)
19. [Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional, Black-box Systems](#Gaussian-Ensemble-Belief-Propagation-for-Efficient-Inference-in-High-Dimensional-Black-box-Systems)
20. [MS-Diffusion: Multi-subject Zero-shot Image Personalization with Layout Guidance](#MS-Diffusion-Multi-subject-Zero-shot-Image-Personalization-with-Layout-Guidance)
21. [Generalization and Distributed Learning of GFlowNets](#Generalization-and-Distributed-Learning-of-GFlowNets)
22. [Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors](#Humanizing-the-Machine-Proxy-Attacks-to-Mislead-LLM-Detectors)
23. [Unlocking Efficient, Scalable, and Continual Knowledge Editing with Basis-Level Representation Fine-Tuning](#Unlocking-Efficient-Scalable-and-Continual-Knowledge-Editing-with-Basis-Level-Representation-Fine-Tuning)
24. [RTop-K: Ultra-Fast Row-Wise Top-K Selection for Neural Network Acceleration on GPUs](#RTop-K-Ultra-Fast-Row-Wise-Top-K-Selection-for-Neural-Network-Acceleration-on-GPUs)
25. [Learning Robust Representations with Long-Term Information for Generalization in Visual Reinforcement Learning](#Learning-Robust-Representations-with-Long-Term-Information-for-Generalization-in-Visual-Reinforcement-Learning)
26. [Magnetic Preference Optimization: Achieving Last-iterate Convergence for Language Model Alignment](#Magnetic-Preference-Optimization-Achieving-Last-iterate-Convergence-for-Language-Model-Alignment)
27. [Select before Act: Spatially Decoupled Action Repetition for Continuous Control](#Select-before-Act-Spatially-Decoupled-Action-Repetition-for-Continuous-Control)
28. [Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models](#Can-Video-LLMs-Refuse-to-Answer-Alignment-for-Answerability-in-Video-Large-Language-Models)
29. [Diffusion Models Are Real-Time Game Engines](#Diffusion-Models-Are-Real-Time-Game-Engines)
30. [Swift Hydra:  Self-Reinforcing Generative Framework for Anomaly Detection with Multiple Mamba Models](#Swift-Hydra-Self-Reinforcing-Generative-Framework-for-Anomaly-Detection-with-Multiple-Mamba-Models)
31. [QuaDiM: A Conditional Diffusion Model For Quantum State Property Estimation](#QuaDiM-A-Conditional-Diffusion-Model-For-Quantum-State-Property-Estimation)
32. [Expected Sliced Transport Plans](#Expected-Sliced-Transport-Plans)
33. [B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners](#B-STaR-Monitoring-and-Balancing-Exploration-and-Exploitation-in-Self-Taught-Reasoners)
34. [Provable Benefit of Annealed Langevin Monte Carlo for Non-log-concave Sampling](#Provable-Benefit-of-Annealed-Langevin-Monte-Carlo-for-Non-log-concave-Sampling)
35. [Modeling Fine-Grained Hand-Object Dynamics for Egocentric Video Representation Learning](#Modeling-Fine-Grained-Hand-Object-Dynamics-for-Egocentric-Video-Representation-Learning)
36. [ACES: Automatic Cohort Extraction System for Event-Stream Datasets](#ACES-Automatic-Cohort-Extraction-System-for-Event-Stream-Datasets)
37. [GaussianAnything: Interactive Point Cloud Flow Matching for 3D Generation](#GaussianAnything-Interactive-Point-Cloud-Flow-Matching-for-3D-Generation)

---


## LLM-wrapper: Black-Box Semantic-Aware Adaptation of Vision-Language Models for Referring Expression Comprehension

### Images

![1369fe7a5e74a2ccacfcd93e20e4a5470f1201d06373c810b92818f00cdddf5b.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/1369fe7a5e74a2ccacfcd93e20e4a5470f1201d06373c810b92818f00cdddf5b.jpg)

![38ea515101c3ac195ca3aa55b675f3681f2ec7688cc8ee2bd85d2f55974bc4f9.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/38ea515101c3ac195ca3aa55b675f3681f2ec7688cc8ee2bd85d2f55974bc4f9.jpg)

![4c04e832efc575c2e130a0459dcbf82216efb4a2a9a3b457ccdff8ec5a3c399f.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/4c04e832efc575c2e130a0459dcbf82216efb4a2a9a3b457ccdff8ec5a3c399f.jpg)

![5950557a0675af3e454842a00778c6a4f987e57b246db4d2842d374eff48283e.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/5950557a0675af3e454842a00778c6a4f987e57b246db4d2842d374eff48283e.jpg)

![662e5bbd9242bfdedd37e4834267ac6a55ab660e8e63af76850e407b0a61edaf.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/662e5bbd9242bfdedd37e4834267ac6a55ab660e8e63af76850e407b0a61edaf.jpg)

![695ce17a9a9c2bf98cef5a0f21654c0ba25c809749aba4d56ac0ed09c3787c86.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/695ce17a9a9c2bf98cef5a0f21654c0ba25c809749aba4d56ac0ed09c3787c86.jpg)

![8f78cc46e597d73f8b771918f3785e636842a3c12ba7660548d1325f6abc418f.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/8f78cc46e597d73f8b771918f3785e636842a3c12ba7660548d1325f6abc418f.jpg)

![a429e093837b16a91f5a731485ea9d566322e069c101f78bf71d709fdc61e656.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/a429e093837b16a91f5a731485ea9d566322e069c101f78bf71d709fdc61e656.jpg)

![a4dc3a022885894ad76740c959f996ad3291998ef7561e35a405803b3c704762.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/a4dc3a022885894ad76740c959f996ad3291998ef7561e35a405803b3c704762.jpg)

![ad432bbdb36205ded4dc6075a1f6881cf613e9b3ecc91f6912c91ac465fb1443.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/ad432bbdb36205ded4dc6075a1f6881cf613e9b3ecc91f6912c91ac465fb1443.jpg)

![af6d86672ccde700952b3d6316a50571e236553a5558fd527251ab6d55b43368.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/af6d86672ccde700952b3d6316a50571e236553a5558fd527251ab6d55b43368.jpg)

![c635d0f89cddae3596a38cf1c1f757a99605c5736e46b5df7a591ade117bb816.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/c635d0f89cddae3596a38cf1c1f757a99605c5736e46b5df7a591ade117bb816.jpg)

![cbc865e4166b3e09fc7995c8f830182b35389d61d42d746b0230a27f925a419e.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/cbc865e4166b3e09fc7995c8f830182b35389d61d42d746b0230a27f925a419e.jpg)

![d09a79a1edc85715b02fc1b5ad9697b535ad2f7e99db9c94af3e503f58b3b6c4.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/d09a79a1edc85715b02fc1b5ad9697b535ad2f7e99db9c94af3e503f58b3b6c4.jpg)

![e080a85438ff553f877ca789a4f048bcff3e5f7ed8056a9fb8b748cbbc31d160.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/e080a85438ff553f877ca789a4f048bcff3e5f7ed8056a9fb8b748cbbc31d160.jpg)

![f29f317e81f2ad5f88690144c59a36def4b1abf7603bc73e433b752b69681d60.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/f29f317e81f2ad5f88690144c59a36def4b1abf7603bc73e433b752b69681d60.jpg)

### Tables

![17c7511bd75f206749e03d431145598d1772ef5d84a426c973f0875eaf679a53.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/17c7511bd75f206749e03d431145598d1772ef5d84a426c973f0875eaf679a53.jpg)

![5ccde043f6240d4b443bfcef6b525f7b05c3756ca542b3170118b57834b8ef99.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/5ccde043f6240d4b443bfcef6b525f7b05c3756ca542b3170118b57834b8ef99.jpg)

![9604c9ff37e127239fd17823e2e110887b358a3de5c547d3a90daecc87a8a10a.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/9604c9ff37e127239fd17823e2e110887b358a3de5c547d3a90daecc87a8a10a.jpg)

![c32cc55bb4abde3e217f02064af8d796df8183f3a943c6cf5e661712df9d1fba.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/c32cc55bb4abde3e217f02064af8d796df8183f3a943c6cf5e661712df9d1fba.jpg)

![c986e326bcde2ace7d97f1471509551d2778f803120c9e55263e998dec566dc7.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/c986e326bcde2ace7d97f1471509551d2778f803120c9e55263e998dec566dc7.jpg)

![dc3e826a7fb91dd5cceeb73f2fa94fa3efde2df5e4e51a4614ca3ee6f16d9430.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/dc3e826a7fb91dd5cceeb73f2fa94fa3efde2df5e4e51a4614ca3ee6f16d9430.jpg)

## LLM-wrapper: Black-Box Semantic-Aware Adaptation of Vision-Language Models for Referring Expression Comprehension


### Images

![0d02c01b457da3914a06f28a014a6895cf591103541d7f257a3462f4c76c3bbd.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/0d02c01b457da3914a06f28a014a6895cf591103541d7f257a3462f4c76c3bbd.jpg)

![0fc3899ba0375c50149c0a4484e7a7d3b5c42b74e4ba5e89d9a709a053746733.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/0fc3899ba0375c50149c0a4484e7a7d3b5c42b74e4ba5e89d9a709a053746733.jpg)

![129dbc0e13a161b17cea750d13745e691ca09f12519bdb6c48af2d0c54d68fc4.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/129dbc0e13a161b17cea750d13745e691ca09f12519bdb6c48af2d0c54d68fc4.jpg)

![31c4f806946b2ec869564093c308796702e19a4234fe055a8fcfe732e3f2ece4.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/31c4f806946b2ec869564093c308796702e19a4234fe055a8fcfe732e3f2ece4.jpg)

![3239300bcf9eab38be3bc0b0dd44563c2b3450b69e49293627d4c7f8b1ceab84.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/3239300bcf9eab38be3bc0b0dd44563c2b3450b69e49293627d4c7f8b1ceab84.jpg)

![33f57eb8853b339fbb944e6d06cdaa2f10ecb37f0550e7fbafb2e471608c57d2.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/33f57eb8853b339fbb944e6d06cdaa2f10ecb37f0550e7fbafb2e471608c57d2.jpg)

![b8413ce7f2367263a3708b02cd1c29bdba5beb905123fc2da9643464d04ff66f.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/b8413ce7f2367263a3708b02cd1c29bdba5beb905123fc2da9643464d04ff66f.jpg)

![bb6a1b8d6b6faf07ea1f62ae8869057e5a5a6e7d7b523965155b32cbb4679f5a.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/bb6a1b8d6b6faf07ea1f62ae8869057e5a5a6e7d7b523965155b32cbb4679f5a.jpg)

![bc224754380723afb702fa282d7e2bb89fab13bd141870017a3d5d55f2253446.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/bc224754380723afb702fa282d7e2bb89fab13bd141870017a3d5d55f2253446.jpg)

![beb1e0626116abff0b89597939978571b132cb080d006d52be10f88e824e246f.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/beb1e0626116abff0b89597939978571b132cb080d006d52be10f88e824e246f.jpg)

![c21302e1b9c1a856167daf7c66e87ae02ec4cf6b88cd8219f1c8633b43a14623.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/c21302e1b9c1a856167daf7c66e87ae02ec4cf6b88cd8219f1c8633b43a14623.jpg)

![c2a8e246a0f415d58c38a443752164e862100e65de5cc3b529ac94b8396b7bc1.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/c2a8e246a0f415d58c38a443752164e862100e65de5cc3b529ac94b8396b7bc1.jpg)

![d2e9ec59ac253e425d6196364d821f4818af3382d6ecdfb54e2beefe6b3550c5.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/d2e9ec59ac253e425d6196364d821f4818af3382d6ecdfb54e2beefe6b3550c5.jpg)

![ecc8593c70ced573e22f130ffca8e05f34b83b4ef37b0fdc39fdc9244142014e.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/images/ecc8593c70ced573e22f130ffca8e05f34b83b4ef37b0fdc39fdc9244142014e.jpg)

### Tables

![0523dcf4c52647dd5f6725d9d79f8710de6fa73a1d7bcc4b9c4e7adc7993b02a.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/0523dcf4c52647dd5f6725d9d79f8710de6fa73a1d7bcc4b9c4e7adc7993b02a.jpg)

![05b418a39953796533c98b371b3e2e1befabd6180726e7aa3d55d4317c886815.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/05b418a39953796533c98b371b3e2e1befabd6180726e7aa3d55d4317c886815.jpg)

![0ead56e1ec6d18ee4beaa03423e718080eceb407b6891d8d95ccc70932cc35da.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/0ead56e1ec6d18ee4beaa03423e718080eceb407b6891d8d95ccc70932cc35da.jpg)

![1c56067c37aacf2f96875d237144b69954c46c85342cc390e85da1d5e31716eb.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/1c56067c37aacf2f96875d237144b69954c46c85342cc390e85da1d5e31716eb.jpg)

![5ad6b120e75ffeb0d6ec1c0d65e923971a0faefd5c4c16bc9d6e67042cf61c4a.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/5ad6b120e75ffeb0d6ec1c0d65e923971a0faefd5c4c16bc9d6e67042cf61c4a.jpg)

![7d6f8615da2bc41a78ad57ba3b09b323a938f1890691669d639e929a6ad7d925.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/7d6f8615da2bc41a78ad57ba3b09b323a938f1890691669d639e929a6ad7d925.jpg)

![9b420f918a390e0e76637c4c0f5e34c8076fb76220f0f8ca8adb45f5863cfe65.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/9b420f918a390e0e76637c4c0f5e34c8076fb76220f0f8ca8adb45f5863cfe65.jpg)

![a8f6bcbbaaa5dc1dfb137baa702ef7f378ca1015cd7ea894e0ccaa8d693d009c.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/a8f6bcbbaaa5dc1dfb137baa702ef7f378ca1015cd7ea894e0ccaa8d693d009c.jpg)

![bff9bb20ba4c4267f376d1bb301d38f3f72f5e31a3c385f29d6bf0738b0c5c17.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/bff9bb20ba4c4267f376d1bb301d38f3f72f5e31a3c385f29d6bf0738b0c5c17.jpg)

![cb0adf07beb2bf29dab1854ab680573a29e481ab87006c2a7ab9151df3bd8155.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/cb0adf07beb2bf29dab1854ab680573a29e481ab87006c2a7ab9151df3bd8155.jpg)

![cccfa7f272024b0bd8b7422f8507f49607057bb09358666ae91715652a7e3c01.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/cccfa7f272024b0bd8b7422f8507f49607057bb09358666ae91715652a7e3c01.jpg)

![e8f31e6a9a7fee39df790edc874fe9081b3feb353953323c26c28de469775b17.jpg](../iclr_results/2381_LLM-wrapper_%20Black-Box%20Semantic-Aware%20Adaptation%20of%20Vision-Language%20Models%20for%20Referring%20Expression%20/tables/e8f31e6a9a7fee39df790edc874fe9081b3feb353953323c26c28de469775b17.jpg)

## Deep MMD Gradient Flow without adversarial training


### Images

![6cbc72bde6db4ffaf0be66fa4d6e93af3c1809571e48c8a1e8ce584fc5ad011b.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/6cbc72bde6db4ffaf0be66fa4d6e93af3c1809571e48c8a1e8ce584fc5ad011b.jpg)

![747dc0f6d1663cc1a3011fae8753ad72f26099a33219d6a17cb8e44ce33135ae.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/747dc0f6d1663cc1a3011fae8753ad72f26099a33219d6a17cb8e44ce33135ae.jpg)

![8f1b61f63648870e66cb9b79ec8fe7e8527b20e986360700d696eb1ab1f7d82f.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/8f1b61f63648870e66cb9b79ec8fe7e8527b20e986360700d696eb1ab1f7d82f.jpg)

![914dad22b40990b88f1753d15bfa437a6cd27707f727555315f84c2484af473a.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/914dad22b40990b88f1753d15bfa437a6cd27707f727555315f84c2484af473a.jpg)

![93c728cd1542ed1c148f3d04b4ec30bced1c5d6cb8d1427830c40dc1dca918e6.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/93c728cd1542ed1c148f3d04b4ec30bced1c5d6cb8d1427830c40dc1dca918e6.jpg)

![9ec62b73a72c47886fdc3f7e0f39d73356be29943ae5647fa45c26b772b70fdb.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/9ec62b73a72c47886fdc3f7e0f39d73356be29943ae5647fa45c26b772b70fdb.jpg)

![d46536edd8435e40a36a1bc7c3707953f261a4d8072d0d9d5f12c2ec96b6f844.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/d46536edd8435e40a36a1bc7c3707953f261a4d8072d0d9d5f12c2ec96b6f844.jpg)

![e43180b20273a5fa2df66bb5ba1b6e33c5db056ce623b97889912048edb73d56.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/e43180b20273a5fa2df66bb5ba1b6e33c5db056ce623b97889912048edb73d56.jpg)

![e6b10477c3c571b7937d3ee19bf5807d20aed7908bfeaef41e2d53bdd6117b5b.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/e6b10477c3c571b7937d3ee19bf5807d20aed7908bfeaef41e2d53bdd6117b5b.jpg)

![f15c9e04de2112decaf6ce572d4494f5c5618b4bbd97be0e46c56d2810a0a89a.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/f15c9e04de2112decaf6ce572d4494f5c5618b4bbd97be0e46c56d2810a0a89a.jpg)

![f63f762536f868ee36177b108eba5df22938d5c43e3bff502de6b7fecb8e0f12.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/f63f762536f868ee36177b108eba5df22938d5c43e3bff502de6b7fecb8e0f12.jpg)

![f66ea42670cd768c038785ca0c78e0bb6db923b9bcc3962b4334ec0fa6ebb0ff.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/images/f66ea42670cd768c038785ca0c78e0bb6db923b9bcc3962b4334ec0fa6ebb0ff.jpg)

### Tables

![03cabde9c5dd73a148598c4a58281e104b5e6ec816420560fbfed5d3a81f764e.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/03cabde9c5dd73a148598c4a58281e104b5e6ec816420560fbfed5d3a81f764e.jpg)

![1226d998cd4049fef5029d8e4de3a2ed90e3f5db10dcdd9fc8dfa5f88a45f668.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/1226d998cd4049fef5029d8e4de3a2ed90e3f5db10dcdd9fc8dfa5f88a45f668.jpg)

![1891526dc100022f5874f054b844f2ec41c3ab70dff4d2c9c9f09611a4a1c954.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/1891526dc100022f5874f054b844f2ec41c3ab70dff4d2c9c9f09611a4a1c954.jpg)

![28a7a7a9c4826937d21b3c9fa9416777fdf9ff756ec9dc23f0d0645d9a560373.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/28a7a7a9c4826937d21b3c9fa9416777fdf9ff756ec9dc23f0d0645d9a560373.jpg)

![50b34a57e42260c38562f2db1778a76c5682d13896581e3f8d5f9ac997b938c5.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/50b34a57e42260c38562f2db1778a76c5682d13896581e3f8d5f9ac997b938c5.jpg)

![6143d1b69a8ff3ada611c0b76cee6c2649300165535ede16def9e7d08bb1a504.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/6143d1b69a8ff3ada611c0b76cee6c2649300165535ede16def9e7d08bb1a504.jpg)

![6b7a9a8d49bbfaf909fc880ff73d2eb7ee7d50cf30ffe9dee7b9121aea1b55c9.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/6b7a9a8d49bbfaf909fc880ff73d2eb7ee7d50cf30ffe9dee7b9121aea1b55c9.jpg)

![6ffa8d3868c875094e7a67ba15e7c4adea4e9d21ebf2acf8664c4dca2dc6dcff.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/6ffa8d3868c875094e7a67ba15e7c4adea4e9d21ebf2acf8664c4dca2dc6dcff.jpg)

![70ecb2fcc9d8e358ae0a0b2f85d2ab4bf4a0968b9bf57156ae7c96ba7b9a3ec0.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/70ecb2fcc9d8e358ae0a0b2f85d2ab4bf4a0968b9bf57156ae7c96ba7b9a3ec0.jpg)

![9177fedb823b7ee1c7ca75091d9961939b875f171e22608c417ad38a6bc6877d.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/9177fedb823b7ee1c7ca75091d9961939b875f171e22608c417ad38a6bc6877d.jpg)

![c674852b6244dd7893432d6938c875968de17deb8d132b2be156c114109a1027.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/c674852b6244dd7893432d6938c875968de17deb8d132b2be156c114109a1027.jpg)

![fff9a1c665a60e9b5c5cc33db4fe113489e1725039aba4996ba902927bf08010.jpg](../iclr_results/2382_Deep%20MMD%20Gradient%20Flow%20without%20adversarial%20training/tables/fff9a1c665a60e9b5c5cc33db4fe113489e1725039aba4996ba902927bf08010.jpg)

## A transfer learning framework for weak to strong generalization


### Images

![0826c12e523d76d8197ba8cc968d2695662715d98cb31fb56f5315b453159540.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/images/0826c12e523d76d8197ba8cc968d2695662715d98cb31fb56f5315b453159540.jpg)

![0df72309e2c694e2960253eedf135bb3f7a15d86a998b17a1281b3125dc410ea.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/images/0df72309e2c694e2960253eedf135bb3f7a15d86a998b17a1281b3125dc410ea.jpg)

![19bdb56d53e4d5be7ed0ecbaebbe17e23a7adb8cbc9b0cd001ea0812ec2d00f2.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/images/19bdb56d53e4d5be7ed0ecbaebbe17e23a7adb8cbc9b0cd001ea0812ec2d00f2.jpg)

![3091ca1a301e9cb764e21b34f4b13f3f3cfde05ebafdcdde9577a40868d8ba0e.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/images/3091ca1a301e9cb764e21b34f4b13f3f3cfde05ebafdcdde9577a40868d8ba0e.jpg)

![4fd87fc95a67e1a11bdf61117d8e078c4c31ec8a0e10cd6b3f8d4cfbe0c19b0f.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/images/4fd87fc95a67e1a11bdf61117d8e078c4c31ec8a0e10cd6b3f8d4cfbe0c19b0f.jpg)

![562f98281cc30c8d758fda24c39f3e351f29de5f86b15ad0303cda454038db3d.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/images/562f98281cc30c8d758fda24c39f3e351f29de5f86b15ad0303cda454038db3d.jpg)

![6b248a8c326d4856342e161f8fc897882cd03156a8cd38c041ab4c9bffce56b7.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/images/6b248a8c326d4856342e161f8fc897882cd03156a8cd38c041ab4c9bffce56b7.jpg)

### Tables

![83311116b4714a51d250c61acd7c4d77b141f72179ce774039b43664fbf0b2d8.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/tables/83311116b4714a51d250c61acd7c4d77b141f72179ce774039b43664fbf0b2d8.jpg)

![f2677c1d4c6bf7dddac8ee4dbb03ee0edebc8b47bc5972607ef92bf49f49db61.jpg](../iclr_results/2383_A%20transfer%20learning%20framework%20for%20weak%20to%20strong%20generalization/tables/f2677c1d4c6bf7dddac8ee4dbb03ee0edebc8b47bc5972607ef92bf49f49db61.jpg)

## What to align in multimodal contrastive learning?


### Images

![13076b0b131a11e770278f70eab38fa7d3323317240f239e7c6860ab0bc2f3c2.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/images/13076b0b131a11e770278f70eab38fa7d3323317240f239e7c6860ab0bc2f3c2.jpg)

![244a2ff5926de1cf4f3c497a725e58641172c1d1e721d3d3564c06ab23393f31.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/images/244a2ff5926de1cf4f3c497a725e58641172c1d1e721d3d3564c06ab23393f31.jpg)

![449f255aa1510bee686ac699ada5f7a254d7e9f083c9f40dedb1e54bd958901b.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/images/449f255aa1510bee686ac699ada5f7a254d7e9f083c9f40dedb1e54bd958901b.jpg)

![75469b900e449d10adf36d564fc305590694162ae367cf91ed64575ab513929e.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/images/75469b900e449d10adf36d564fc305590694162ae367cf91ed64575ab513929e.jpg)

![7ea9bf320323ac8029850d866beca4a2615470021495ee5e6afd27a2a661546d.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/images/7ea9bf320323ac8029850d866beca4a2615470021495ee5e6afd27a2a661546d.jpg)

![95611bbc55aa31d17b2da77fb2b09f1c3cfa400a5e86b9e860db3c55482d4f87.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/images/95611bbc55aa31d17b2da77fb2b09f1c3cfa400a5e86b9e860db3c55482d4f87.jpg)

![dc45d8f02f96a311cbe88214fb74f5ec2fb02a5006fb43d28da0e70942f4695c.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/images/dc45d8f02f96a311cbe88214fb74f5ec2fb02a5006fb43d28da0e70942f4695c.jpg)

### Tables

![1d1617f9698c3c6d8598cd509f1473278e804c92beed7676ab439a00048b2ddf.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/1d1617f9698c3c6d8598cd509f1473278e804c92beed7676ab439a00048b2ddf.jpg)

![2728f37c0d497f80ae8f58c9830f58935aabdc166fe67222144ede7657033139.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/2728f37c0d497f80ae8f58c9830f58935aabdc166fe67222144ede7657033139.jpg)

![2913a6e6fbbcabb43a7aff9fb94bef605cfe40e2dbd9f28f47631229ebbc740a.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/2913a6e6fbbcabb43a7aff9fb94bef605cfe40e2dbd9f28f47631229ebbc740a.jpg)

![364711682de4f6194fc2070f928b30864dd805c6b5f20da1273f52839055176a.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/364711682de4f6194fc2070f928b30864dd805c6b5f20da1273f52839055176a.jpg)

![48c53095a85a9b157637f761fe709b89eb4e8dfe1a1a9a2e1120d1f186e6ce77.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/48c53095a85a9b157637f761fe709b89eb4e8dfe1a1a9a2e1120d1f186e6ce77.jpg)

![dce5132cd6d3cd6fe702abb0849d0eaf670f219fe7bcdd5e11936a0c5d9914d7.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/dce5132cd6d3cd6fe702abb0849d0eaf670f219fe7bcdd5e11936a0c5d9914d7.jpg)

![e0d1d0d2729232b95a264c5d0545a96433d3e8d2db4568a0e68526c6a10bbb0c.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/e0d1d0d2729232b95a264c5d0545a96433d3e8d2db4568a0e68526c6a10bbb0c.jpg)

![e7fcdfef790a03a0ebd7fd6fcb5f2a301281f079995d41cb7f2af1adf26d510c.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/e7fcdfef790a03a0ebd7fd6fcb5f2a301281f079995d41cb7f2af1adf26d510c.jpg)

![fc533cd5b6dfd64ccdd08722b12adbe9085abba2e27ed46e461adadf3e3de490.jpg](../iclr_results/2384_What%20to%20align%20in%20multimodal%20contrastive%20learning_/tables/fc533cd5b6dfd64ccdd08722b12adbe9085abba2e27ed46e461adadf3e3de490.jpg)

## GenVP: Generating Visual Puzzles with Contrastive Hierarchical VAEs


### Images

![13c78b21dff293dae70a574ae6a2a036900c49322cd2843f9ba3b358e618c02e.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/13c78b21dff293dae70a574ae6a2a036900c49322cd2843f9ba3b358e618c02e.jpg)

![1e8e08718689cff4f868b2438ba612efcbe2d699cf75b99c415e492cafd93180.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/1e8e08718689cff4f868b2438ba612efcbe2d699cf75b99c415e492cafd93180.jpg)

![264529709282488ad7544c056199bcd95ab518369a1757b8b7f4b3e8c1fa958d.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/264529709282488ad7544c056199bcd95ab518369a1757b8b7f4b3e8c1fa958d.jpg)

![7d2efc523113850e45d0ddc264a54bd881bd9e35b3c0484037f23a61344290dc.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/7d2efc523113850e45d0ddc264a54bd881bd9e35b3c0484037f23a61344290dc.jpg)

![91bafe2b00c94a54175efa40a667ca7b8921dd2d5698661aadeeac561f194bef.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/91bafe2b00c94a54175efa40a667ca7b8921dd2d5698661aadeeac561f194bef.jpg)

![d563cb692eb9e14eaaaf8cfc976830c864abb6af7a7d9554a49d67f4e2f7c3eb.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/d563cb692eb9e14eaaaf8cfc976830c864abb6af7a7d9554a49d67f4e2f7c3eb.jpg)

![df1c2262ef3d3587a00494241cc9aeeb5326795bb43abb93922c899395177886.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/df1c2262ef3d3587a00494241cc9aeeb5326795bb43abb93922c899395177886.jpg)

![ecf5485c9ef7c4dca073d7a3b4fc39093dbe2cfa0811501389438198cec2a30d.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/images/ecf5485c9ef7c4dca073d7a3b4fc39093dbe2cfa0811501389438198cec2a30d.jpg)

### Tables

![052f119236202b38431c389b084265a95b60c6af825ddb08fc3f8434332b6ccf.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/052f119236202b38431c389b084265a95b60c6af825ddb08fc3f8434332b6ccf.jpg)

![05aa0e643722abc56eebe6deda264bc8b7a1aa65746f14483e268e8983ccb6c4.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/05aa0e643722abc56eebe6deda264bc8b7a1aa65746f14483e268e8983ccb6c4.jpg)

![14ae97fedbfe353a5994dcb99872d14a752f7e636adcf736122cd1b27455a051.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/14ae97fedbfe353a5994dcb99872d14a752f7e636adcf736122cd1b27455a051.jpg)

![150d3b0384e59e45c1680e97f6786e855d97d0abe0c34d43646b1499f7a0e251.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/150d3b0384e59e45c1680e97f6786e855d97d0abe0c34d43646b1499f7a0e251.jpg)

![38dc6f421ef6f11b7ee444aba6c6ceb3e1671bae5da3d16f092153e612c44126.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/38dc6f421ef6f11b7ee444aba6c6ceb3e1671bae5da3d16f092153e612c44126.jpg)

![4112163d4aac79d510d8ba020826ca0dff1979e12ec2c3612b1f6f41b275abc6.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/4112163d4aac79d510d8ba020826ca0dff1979e12ec2c3612b1f6f41b275abc6.jpg)

![44cb2d58f5d234f0e9656de7b5087f5dd5d9f09a86c869cb77351efeec83e203.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/44cb2d58f5d234f0e9656de7b5087f5dd5d9f09a86c869cb77351efeec83e203.jpg)

![5c6a1c40e9eeb3e8ea559877e6d171dc74261486043dc0d1ced92386e673b0d5.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/5c6a1c40e9eeb3e8ea559877e6d171dc74261486043dc0d1ced92386e673b0d5.jpg)

![5f0295e7b0c7f38bf197f7f8df2e31cb46e98312c8825a45e84e298694f6de0a.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/5f0295e7b0c7f38bf197f7f8df2e31cb46e98312c8825a45e84e298694f6de0a.jpg)

![6e4424a2a4a905c70d46a19bb606573eb04153b1ae3f0b340486b2d16f88c46f.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/6e4424a2a4a905c70d46a19bb606573eb04153b1ae3f0b340486b2d16f88c46f.jpg)

![7418b97d4d9b92f37d924d11126b3a11bcb2a0f87a2cca0e043d9cd487d8190d.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/7418b97d4d9b92f37d924d11126b3a11bcb2a0f87a2cca0e043d9cd487d8190d.jpg)

![7789ebcb91a20412ef5feb26c2050732953cbd9dd07909f50e806a3680d35df8.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/7789ebcb91a20412ef5feb26c2050732953cbd9dd07909f50e806a3680d35df8.jpg)

![8160f617a57581094b12a996375db0120609df03024a7fdf57bdad177408e3b5.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/8160f617a57581094b12a996375db0120609df03024a7fdf57bdad177408e3b5.jpg)

![81fc8aac3964c5160f2f2ccdc10681fa8d5ab686ce8055a45d9d0e194b76eec7.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/81fc8aac3964c5160f2f2ccdc10681fa8d5ab686ce8055a45d9d0e194b76eec7.jpg)

![c28a7ed26d67c622306f64eea9051bf7ce6e980c94f392a17a4e9ac75b9dfba1.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/c28a7ed26d67c622306f64eea9051bf7ce6e980c94f392a17a4e9ac75b9dfba1.jpg)

![d37e5ab91eae2bea872439f6b60e412baf5a4b7c03ecd24f1bfb8d56dc835545.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/d37e5ab91eae2bea872439f6b60e412baf5a4b7c03ecd24f1bfb8d56dc835545.jpg)

![e0d1c5188c6e3682f4f47aa6ef18ea8e7232f561aff0349cc565aef59f8535b7.jpg](../iclr_results/2385_GenVP_%20Generating%20Visual%20Puzzles%20with%20Contrastive%20Hierarchical%20VAEs/tables/e0d1c5188c6e3682f4f47aa6ef18ea8e7232f561aff0349cc565aef59f8535b7.jpg)

## Towards Continuous Reuse of Graph Models via Holistic Memory Diversification


### Images

![12245a422bed7b6104f9c60588e87ca49d7391c1c83bcc3116de1514da9c7011.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/images/12245a422bed7b6104f9c60588e87ca49d7391c1c83bcc3116de1514da9c7011.jpg)

![60990422b784b8b4b1a397ce8b348ab8fdc9464acbb52cf981036daaef3133bc.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/images/60990422b784b8b4b1a397ce8b348ab8fdc9464acbb52cf981036daaef3133bc.jpg)

![7bb1630dc4aaa9fb77cdf75edfc72fe60200fbfaae3d745e0c0dacb6c540f80a.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/images/7bb1630dc4aaa9fb77cdf75edfc72fe60200fbfaae3d745e0c0dacb6c540f80a.jpg)

![809ded4e3325c6a0a2d11b8bbd4d771b4ed7f0f32de162d4bd19941bff14db71.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/images/809ded4e3325c6a0a2d11b8bbd4d771b4ed7f0f32de162d4bd19941bff14db71.jpg)

![8f81a13322297f1c9f440302b90d015e18e8f40431cf07338dd06306946f7794.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/images/8f81a13322297f1c9f440302b90d015e18e8f40431cf07338dd06306946f7794.jpg)

![a24bbaec7658b2150d595e6c29ed63973cd574b7953043524a9b3252f1fa4cf5.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/images/a24bbaec7658b2150d595e6c29ed63973cd574b7953043524a9b3252f1fa4cf5.jpg)

![f9f36f2167c5c903dad709630434ce459ffac05772a7b7a5dbf2610ca7bd644f.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/images/f9f36f2167c5c903dad709630434ce459ffac05772a7b7a5dbf2610ca7bd644f.jpg)

### Tables

![279d502075ab5fc6fc73dd6f1de2f0d3561db53337219904db16c890f565922b.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/tables/279d502075ab5fc6fc73dd6f1de2f0d3561db53337219904db16c890f565922b.jpg)

![73ccbef4b6f5ff33eebe7d3f981f23353f48fa28f0a64b013c86469f4b41c23f.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/tables/73ccbef4b6f5ff33eebe7d3f981f23353f48fa28f0a64b013c86469f4b41c23f.jpg)

![b165f46db1b1da4055b8ff674c284032fab2bd4a1b443b4456048a7c3c8494bf.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/tables/b165f46db1b1da4055b8ff674c284032fab2bd4a1b443b4456048a7c3c8494bf.jpg)

![b44ea6c82d37867a849f75d083ea7bb4f78372c98dc6e8ab736356a6a103b08a.jpg](../iclr_results/2386_Towards%20Continuous%20Reuse%20of%20Graph%20Models%20via%20Holistic%20Memory%20Diversification/tables/b44ea6c82d37867a849f75d083ea7bb4f78372c98dc6e8ab736356a6a103b08a.jpg)

## Lightweight Predictive 3D Gaussian Splats


### Images

![2152290f4f7737f510420ca9ab83065b410df2a8176c1d790028973dee2eec86.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/2152290f4f7737f510420ca9ab83065b410df2a8176c1d790028973dee2eec86.jpg)

![3035bdbcbc3b593f294e8f11368b7476a33413f459d68d60bb702098dfb16b8f.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/3035bdbcbc3b593f294e8f11368b7476a33413f459d68d60bb702098dfb16b8f.jpg)

![4d236c118c17f83ae4e68c6c43aaf5ef05e2b5b7ba81b462bf4aade351bda7cc.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/4d236c118c17f83ae4e68c6c43aaf5ef05e2b5b7ba81b462bf4aade351bda7cc.jpg)

![50937a4b4294e3c778840f84a9eb60237b7ece06bf77e6b670135d7051da9b1d.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/50937a4b4294e3c778840f84a9eb60237b7ece06bf77e6b670135d7051da9b1d.jpg)

![a73aaee0277557d4384a4c1b8a5ac982d1326f589aa004855df6f00fb3d70a9d.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/a73aaee0277557d4384a4c1b8a5ac982d1326f589aa004855df6f00fb3d70a9d.jpg)

![ba6eb685bd16d39a6b78ec8016a5e7f1489e5cd21d341fb8b3f0585d304a87a8.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/ba6eb685bd16d39a6b78ec8016a5e7f1489e5cd21d341fb8b3f0585d304a87a8.jpg)

![d5ef702d36227d59aba81c8441b95af1d6acda48cee7d1a5d537b922ab95995d.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/d5ef702d36227d59aba81c8441b95af1d6acda48cee7d1a5d537b922ab95995d.jpg)

![dba66d56c1c567e61ed5450d5bdb6ce5dbe7e71c0c79c4c349e1df08c05c8c19.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/dba66d56c1c567e61ed5450d5bdb6ce5dbe7e71c0c79c4c349e1df08c05c8c19.jpg)

![e7150c8c1f0e9ef5f22bf31ec9dc736865cd3032ad166f57fd01ed9a564ec7c7.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/e7150c8c1f0e9ef5f22bf31ec9dc736865cd3032ad166f57fd01ed9a564ec7c7.jpg)

![f7b38e6c3bfba1923e09e0f2845918460f2f6fe884d0e11469633665fba0929d.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/images/f7b38e6c3bfba1923e09e0f2845918460f2f6fe884d0e11469633665fba0929d.jpg)

### Tables

![73c2c6e93ab0afea0750541b1aa5db7544b65637967abb4e84fca2aa56f3ca98.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/73c2c6e93ab0afea0750541b1aa5db7544b65637967abb4e84fca2aa56f3ca98.jpg)

![898f1d15de5898c7d88407f6483c5f32d61f29d8411ba263777021ff4f0a2f09.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/898f1d15de5898c7d88407f6483c5f32d61f29d8411ba263777021ff4f0a2f09.jpg)

![94d1568ce8e41c3310acb49b6724abb5dc5ab913ee3bb11132c1f7d7fee3ffaa.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/94d1568ce8e41c3310acb49b6724abb5dc5ab913ee3bb11132c1f7d7fee3ffaa.jpg)

![9f263c7bc4417066d7c1ac9671e850db3ae7a086a00fd59b265c7e2447b01a9e.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/9f263c7bc4417066d7c1ac9671e850db3ae7a086a00fd59b265c7e2447b01a9e.jpg)

![a0c4a51078fa6ccbf4a045e6e1849d57a15ff242a6bb276a01029dd5103947ef.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/a0c4a51078fa6ccbf4a045e6e1849d57a15ff242a6bb276a01029dd5103947ef.jpg)

![f30a7e4b8911777d4f96b0dc0a31c00f343a24e7cc7c595528c2512b33949312.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/f30a7e4b8911777d4f96b0dc0a31c00f343a24e7cc7c595528c2512b33949312.jpg)

![f367819d752654592aae84ea47581b91befa3b25271167dd578a72b6c235c93e.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/f367819d752654592aae84ea47581b91befa3b25271167dd578a72b6c235c93e.jpg)

![fdf27a728042da1791364165b365bb50311b7b263e4e7e90d759ad75be3a6029.jpg](../iclr_results/2387_Lightweight%20Predictive%203D%20Gaussian%20Splats/tables/fdf27a728042da1791364165b365bb50311b7b263e4e7e90d759ad75be3a6029.jpg)

## Universal Image Restoration Pre-training via Degradation Classification


### Images

![05a2f8c8f188a169d68a031d026df3b163f88cff13be726dd3c129902c040711.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/images/05a2f8c8f188a169d68a031d026df3b163f88cff13be726dd3c129902c040711.jpg)

![3ed172bcec791d16580c2566d908e58b3dd734039eb3ae73a5671d630b9e1360.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/images/3ed172bcec791d16580c2566d908e58b3dd734039eb3ae73a5671d630b9e1360.jpg)

![4ff51f7000bd41fb9ba18e2241feacc5d4cc2239a944efadff5ce8e2e048d64d.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/images/4ff51f7000bd41fb9ba18e2241feacc5d4cc2239a944efadff5ce8e2e048d64d.jpg)

![533c965a0caad4690420456d643293c69086b67c0a34a55e781f754cd45d4bbc.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/images/533c965a0caad4690420456d643293c69086b67c0a34a55e781f754cd45d4bbc.jpg)

![aeb1efdc73b0e88b3cd9642973c3427b61ac3cf06a2f3ae4151cb3f16e93fec9.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/images/aeb1efdc73b0e88b3cd9642973c3427b61ac3cf06a2f3ae4151cb3f16e93fec9.jpg)

![e4d3f224ebb2b10327651d19cdcc54af9f64c0570b4717bea1b53cff21290bfd.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/images/e4d3f224ebb2b10327651d19cdcc54af9f64c0570b4717bea1b53cff21290bfd.jpg)

![fffcf26240a17a3cf71270839b758126e8916526f794e5a1c59793c370779ca6.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/images/fffcf26240a17a3cf71270839b758126e8916526f794e5a1c59793c370779ca6.jpg)

### Tables

![07436603a09876374b8ac63b0dab49fc29a1bf1add3d4e642424bd9172e3db1c.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/07436603a09876374b8ac63b0dab49fc29a1bf1add3d4e642424bd9172e3db1c.jpg)

![0794f94823ec5fc77299619623b6e149bfc5a5b3662f385360fbefc09cb1518c.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/0794f94823ec5fc77299619623b6e149bfc5a5b3662f385360fbefc09cb1518c.jpg)

![082dcada6de8348c30651aad264ebc00a56037f4d028a24500764238ace60990.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/082dcada6de8348c30651aad264ebc00a56037f4d028a24500764238ace60990.jpg)

![083fdf4fd7ae3c89f92806b791fde11d794b90178531ee40d97340aac1ce1e48.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/083fdf4fd7ae3c89f92806b791fde11d794b90178531ee40d97340aac1ce1e48.jpg)

![53c8cb8e58061f7d2a5f50dfc5c72c232787a5768e732dd9b4fcbba23cf289c1.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/53c8cb8e58061f7d2a5f50dfc5c72c232787a5768e732dd9b4fcbba23cf289c1.jpg)

![6114a91801636e7f7fbaa7724a3de1a57889a9cdf2e71fd044ad8f3e530baf09.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/6114a91801636e7f7fbaa7724a3de1a57889a9cdf2e71fd044ad8f3e530baf09.jpg)

![63398a487a08972c58c687a6d894fb17898fe8914206495fd4d949891b2dedcc.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/63398a487a08972c58c687a6d894fb17898fe8914206495fd4d949891b2dedcc.jpg)

![66bec3c97b1317b79afc01b08e0fc70c73dd7f9e951a40fa0c48e399e1653504.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/66bec3c97b1317b79afc01b08e0fc70c73dd7f9e951a40fa0c48e399e1653504.jpg)

![7941013c87a793e1829733060a5f906fb494e443a38cc9adbc9e46d182c52211.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/7941013c87a793e1829733060a5f906fb494e443a38cc9adbc9e46d182c52211.jpg)

![79bdbc0388b6276d166c130c455cd627612a4532acfd2dfcf898d838b3d46e9c.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/79bdbc0388b6276d166c130c455cd627612a4532acfd2dfcf898d838b3d46e9c.jpg)

![7d955848efb3bbd6c9af79d218bcc0b959fd06e161ed537a1dbadb1f4cfbed84.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/7d955848efb3bbd6c9af79d218bcc0b959fd06e161ed537a1dbadb1f4cfbed84.jpg)

![8d71d7b5a90a2e502e6abb794b84a98ee90fbd729a026e71319d76a0d2e90572.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/8d71d7b5a90a2e502e6abb794b84a98ee90fbd729a026e71319d76a0d2e90572.jpg)

![c341206228de1428d8199928a05014885f33e2868fcc049337e3ae3acf25d291.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/c341206228de1428d8199928a05014885f33e2868fcc049337e3ae3acf25d291.jpg)

![c7e0444ef1f46641c837fb10739996582e15219bd665705c69745dda76a5f12f.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/c7e0444ef1f46641c837fb10739996582e15219bd665705c69745dda76a5f12f.jpg)

![cb5813d2765a02a4fccb8599b466d61864e2a31f10788a407e2b637663529272.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/cb5813d2765a02a4fccb8599b466d61864e2a31f10788a407e2b637663529272.jpg)

![cfe2684a176978a12e565e05f89f116857178c26ca275c0e13e300394f1a90b2.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/cfe2684a176978a12e565e05f89f116857178c26ca275c0e13e300394f1a90b2.jpg)

![e17a68729720e3987619f27a7d575d5ab503ef417cc927bded3bd1ee06bcec0f.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/e17a68729720e3987619f27a7d575d5ab503ef417cc927bded3bd1ee06bcec0f.jpg)

![e89cf4d63f9c7bc498927568b0e42d6215db3482b7cc5530919e89f45668a6a0.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/e89cf4d63f9c7bc498927568b0e42d6215db3482b7cc5530919e89f45668a6a0.jpg)

![ffc4e797b64e2a8294db549e50837920cecab0b32697c55db9dea9d76e20d92c.jpg](../iclr_results/2388_Universal%20Image%20Restoration%20Pre-training%20via%20Degradation%20Classification/tables/ffc4e797b64e2a8294db549e50837920cecab0b32697c55db9dea9d76e20d92c.jpg)

## Connectome Mapping: Shape-Memory Network via Interpretation of Contextual Semantic Information


### Images

![03bdf665f25b80d0bd345d11149a91d36179e36219144a727e18277699f7d0ce.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/03bdf665f25b80d0bd345d11149a91d36179e36219144a727e18277699f7d0ce.jpg)

![128f3b8bf0e51dc4d8f8ee087c145093b60e024b18e7afa39df643a498a11251.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/128f3b8bf0e51dc4d8f8ee087c145093b60e024b18e7afa39df643a498a11251.jpg)

![1a6a70547dc8c070c61b55f5c7ec9449dccabf510ed9b394e35f9b24bd324217.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/1a6a70547dc8c070c61b55f5c7ec9449dccabf510ed9b394e35f9b24bd324217.jpg)

![3ec64b8cb5e9a7a6df83f8223010aec4a5cbbfae93284933a0845ac699a3ce58.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/3ec64b8cb5e9a7a6df83f8223010aec4a5cbbfae93284933a0845ac699a3ce58.jpg)

![62173bc4d547c475441717284a4335b715974e02063f0c052c6b6482e445a32d.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/62173bc4d547c475441717284a4335b715974e02063f0c052c6b6482e445a32d.jpg)

![637a5d87c99f09fc4b3066232387a3b966f0507857bffd8af0a0885cdc9d279a.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/637a5d87c99f09fc4b3066232387a3b966f0507857bffd8af0a0885cdc9d279a.jpg)

![7b9ab16a390feb3ea7acd4e98db1fba764724944c384a724c8892a2a9cdd7fea.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/7b9ab16a390feb3ea7acd4e98db1fba764724944c384a724c8892a2a9cdd7fea.jpg)

![7ce88e8e467b32506088a7dd230b35a1b5c55546e0c3a350cfac468c23a7b8de.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/7ce88e8e467b32506088a7dd230b35a1b5c55546e0c3a350cfac468c23a7b8de.jpg)

![91c5d82fad0d96c6078e0a04760cb21c12f714825a33615e4a5e03ffcc9510d8.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/91c5d82fad0d96c6078e0a04760cb21c12f714825a33615e4a5e03ffcc9510d8.jpg)

![9d2d39883b059899ab94b092df7939a785ad06108191b2c9d78ef9925ffa2db6.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/9d2d39883b059899ab94b092df7939a785ad06108191b2c9d78ef9925ffa2db6.jpg)

![a5426ec1e50b6811abb23641f2e220939225e85e0c03bb5f8dbd2b9670077018.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/a5426ec1e50b6811abb23641f2e220939225e85e0c03bb5f8dbd2b9670077018.jpg)

![a5591836d6b94dfd9e23e7a806fe90b0cc7dafd644e857de1c67c06f6f25a2e9.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/a5591836d6b94dfd9e23e7a806fe90b0cc7dafd644e857de1c67c06f6f25a2e9.jpg)

![b042cfbb39adc7af393f47dcb4a9b7788749656616dd38e0a5f6e77cd89cb617.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/b042cfbb39adc7af393f47dcb4a9b7788749656616dd38e0a5f6e77cd89cb617.jpg)

![ba0b9ed00495732771ee4ef33f4baeb9a3efb18394759a9e0d0c98aa49559ad7.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/ba0b9ed00495732771ee4ef33f4baeb9a3efb18394759a9e0d0c98aa49559ad7.jpg)

![bd09329d3d00b6ee02c597dd90a294defbfe85c919aac080c399379343f1a0dc.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/bd09329d3d00b6ee02c597dd90a294defbfe85c919aac080c399379343f1a0dc.jpg)

![c3c43bd48493256b8788c92c797c7adae038a5ad47242fe0f15520dbecff04fe.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/images/c3c43bd48493256b8788c92c797c7adae038a5ad47242fe0f15520dbecff04fe.jpg)

### Tables

![0b8bd162162525fd0ca1e1fb61561ac92e564a9c6a74ed441f0bdd05d28d68e7.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/tables/0b8bd162162525fd0ca1e1fb61561ac92e564a9c6a74ed441f0bdd05d28d68e7.jpg)

![3226f3792183927e2c52cbded6533109e53ec4c88310c3f8b48e2b05c12416c1.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/tables/3226f3792183927e2c52cbded6533109e53ec4c88310c3f8b48e2b05c12416c1.jpg)

![46ae67b53b7b00f3db6559b908d217a60e9841713eea7882dfbb9b19235fa217.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/tables/46ae67b53b7b00f3db6559b908d217a60e9841713eea7882dfbb9b19235fa217.jpg)

![9d54f5d79e9465ca6511d0ebe335e6f56bb03d4ffe7731fc1eb50e4fd4a9d847.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/tables/9d54f5d79e9465ca6511d0ebe335e6f56bb03d4ffe7731fc1eb50e4fd4a9d847.jpg)

![a54206de60338d6a5baa8942ac807a2fbefa655872efdf960fe556a398c15ab7.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/tables/a54206de60338d6a5baa8942ac807a2fbefa655872efdf960fe556a398c15ab7.jpg)

![b521bd92d6d28447e6299efca8f85de006ec016c4dd09c4e46a53ef517ea1f2f.jpg](../iclr_results/2389_Connectome%20Mapping_%20Shape-Memory%20Network%20via%20Interpretation%20of%20Contextual%20Semantic%20Information/tables/b521bd92d6d28447e6299efca8f85de006ec016c4dd09c4e46a53ef517ea1f2f.jpg)

## LLaMA-Omni: Seamless Speech Interaction with Large Language Models


### Images

![5ea2d130f108f8ff1149193dcd35e0b091fe19e18c11aa7f3d9c10fe0bca4948.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/images/5ea2d130f108f8ff1149193dcd35e0b091fe19e18c11aa7f3d9c10fe0bca4948.jpg)

![6a54e2db59ee933b180e89592b3dbdcd77a2b7614d08b1117fc75763c9da2e3c.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/images/6a54e2db59ee933b180e89592b3dbdcd77a2b7614d08b1117fc75763c9da2e3c.jpg)

![deb5d75bcdc41d5e81b98bc92de2b51ab33ebbd1e6e65798c076868455250115.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/images/deb5d75bcdc41d5e81b98bc92de2b51ab33ebbd1e6e65798c076868455250115.jpg)

![ed8c510866e34ee3d13ca5c2f88c643be32c535b5fee18602766e961e31255da.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/images/ed8c510866e34ee3d13ca5c2f88c643be32c535b5fee18602766e961e31255da.jpg)

### Tables

![2445f47cfab2d97c6a4f35c01b2941e7f5550c4fa043a0d5146452a01a5e898b.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/tables/2445f47cfab2d97c6a4f35c01b2941e7f5550c4fa043a0d5146452a01a5e898b.jpg)

![34e48333f3d0a9c2b54e1cc9192a08717c3f93871f32e2cfcb6747ef46d764a3.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/tables/34e48333f3d0a9c2b54e1cc9192a08717c3f93871f32e2cfcb6747ef46d764a3.jpg)

![4a53170a5c8f8e06fef2b785d9f1519653800f0c349d93cca75bcb2ca617113f.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/tables/4a53170a5c8f8e06fef2b785d9f1519653800f0c349d93cca75bcb2ca617113f.jpg)

![66943a94bd815be31c71abaa16e4c32b105371cbe2d85ac09d9ba2c08be1480d.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/tables/66943a94bd815be31c71abaa16e4c32b105371cbe2d85ac09d9ba2c08be1480d.jpg)

![b1407b037d1582f6c57c7d6d3f820eb33bf4485876aedb76c513c85289f6544f.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/tables/b1407b037d1582f6c57c7d6d3f820eb33bf4485876aedb76c513c85289f6544f.jpg)

![ba270daee46d36522411824aed278e23be892796a12d40050d5cff610e1e4e3b.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/tables/ba270daee46d36522411824aed278e23be892796a12d40050d5cff610e1e4e3b.jpg)

![d43713d1b6aae2a7348ef14c69309b38ebc7f6512a4cde1d39999f54251dc7ff.jpg](../iclr_results/2390_LLaMA-Omni_%20Seamless%20Speech%20Interaction%20with%20Large%20Language%20Models/tables/d43713d1b6aae2a7348ef14c69309b38ebc7f6512a4cde1d39999f54251dc7ff.jpg)

## SONICS: Synthetic Or Not - Identifying Counterfeit Songs


### Images

![2ee9a1cc1074ea60e8ecafd6eeb1bfd51bab2783219675c1f50604af4148bbcf.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/2ee9a1cc1074ea60e8ecafd6eeb1bfd51bab2783219675c1f50604af4148bbcf.jpg)

![37610a72e8eb21246f0a320f40a33532585646a632d6e039fa3b6e3f4460e21c.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/37610a72e8eb21246f0a320f40a33532585646a632d6e039fa3b6e3f4460e21c.jpg)

![5ba5f5ec371fc15b2cd20185edcc164127f365d903d244ac4e5c2e097a168ec8.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/5ba5f5ec371fc15b2cd20185edcc164127f365d903d244ac4e5c2e097a168ec8.jpg)

![66ed2646cb05b6b76f9962e6a76557a0beb972408b92108c3d3134f7c3adebd2.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/66ed2646cb05b6b76f9962e6a76557a0beb972408b92108c3d3134f7c3adebd2.jpg)

![73d4f1b380c84edcf8cb9bdb97d711f8e4e1e1f3ef1a3b63aca43a056ffbabab.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/73d4f1b380c84edcf8cb9bdb97d711f8e4e1e1f3ef1a3b63aca43a056ffbabab.jpg)

![76883305e070086c8dad4714433d90a3a33803568711bea2945b3623611d0639.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/76883305e070086c8dad4714433d90a3a33803568711bea2945b3623611d0639.jpg)

![785915fc6d4f0d02c243b64d0c0bcaf367701cd8cbbdda3abc92f88595a6d5cd.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/785915fc6d4f0d02c243b64d0c0bcaf367701cd8cbbdda3abc92f88595a6d5cd.jpg)

![a66843ffe6400b4c5df2b9f47befac513d89f7e99ecf37bc6500599c9fc7839c.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/a66843ffe6400b4c5df2b9f47befac513d89f7e99ecf37bc6500599c9fc7839c.jpg)

![d02841d4c3c57b80573c65aaee725df379e19c99da69a2103b2bf1090569d5e1.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/d02841d4c3c57b80573c65aaee725df379e19c99da69a2103b2bf1090569d5e1.jpg)

![f0878034aabdf31c989aaaeb44f426e0ac662c8e5f890076351cc6f53e68cdd7.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/f0878034aabdf31c989aaaeb44f426e0ac662c8e5f890076351cc6f53e68cdd7.jpg)

![f60cc6ddd6d252922adec44e49160dd33beb5251441e0f7bb3d3fa3a7cfd76a4.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/images/f60cc6ddd6d252922adec44e49160dd33beb5251441e0f7bb3d3fa3a7cfd76a4.jpg)

### Tables

![4067ad10fa54a2db255b49764d014c33ff1d6bfbbe4b9ca5e08c02b91b6c4784.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/4067ad10fa54a2db255b49764d014c33ff1d6bfbbe4b9ca5e08c02b91b6c4784.jpg)

![57a30fd9cc073bc983d2a548729035c9e38631f73b79d20de1a20961ea3a3d66.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/57a30fd9cc073bc983d2a548729035c9e38631f73b79d20de1a20961ea3a3d66.jpg)

![8b67b287b05926844db9d5608805f382d5ed8bc638b9af52fdfa41d9bb7b477f.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/8b67b287b05926844db9d5608805f382d5ed8bc638b9af52fdfa41d9bb7b477f.jpg)

![97ea0617823cba3c555d8e1c392b83d85fccde7ea60004470cb7e7f7d7ffa342.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/97ea0617823cba3c555d8e1c392b83d85fccde7ea60004470cb7e7f7d7ffa342.jpg)

![a8ee66f2ef8bb2832f762d7006990a7ee12d19c5ce9a65738cba6e5f8471e89a.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/a8ee66f2ef8bb2832f762d7006990a7ee12d19c5ce9a65738cba6e5f8471e89a.jpg)

![affeaec953e9410e1977590b71a26d8b784a5377ebd4e944b9ea989e4436971c.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/affeaec953e9410e1977590b71a26d8b784a5377ebd4e944b9ea989e4436971c.jpg)

![c951978d46b85a9c7a0654666094c9780c21d3da1f42b95d2c23edab4d2daa3d.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/c951978d46b85a9c7a0654666094c9780c21d3da1f42b95d2c23edab4d2daa3d.jpg)

![ec8c54ec2ce54c1cfbf5664ce65861985f15ec2765c1578be6b4fc05ec7184be.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/ec8c54ec2ce54c1cfbf5664ce65861985f15ec2765c1578be6b4fc05ec7184be.jpg)

![eedae75ac60ca0be08289bf5e096eaf47df2a44e454d91643ab31db95ab76025.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/eedae75ac60ca0be08289bf5e096eaf47df2a44e454d91643ab31db95ab76025.jpg)

![f209e494b8e5a6da1e8216f183053a7af09ddb9354d1ff65ca12d4b86e69567f.jpg](../iclr_results/2391_SONICS_%20Synthetic%20Or%20Not%20-%20Identifying%20Counterfeit%20Songs/tables/f209e494b8e5a6da1e8216f183053a7af09ddb9354d1ff65ca12d4b86e69567f.jpg)

## Execution-guided within-prompt search for programming-by-example


### Images

![0e7cd6e753eec571e709b1918915538c428172e9d245bb54c401172234f142b1.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/0e7cd6e753eec571e709b1918915538c428172e9d245bb54c401172234f142b1.jpg)

![2cfc1899d1869643ad68e40fa878206c24c6255d079eb91e3e2193dc791d2cd9.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/2cfc1899d1869643ad68e40fa878206c24c6255d079eb91e3e2193dc791d2cd9.jpg)

![5301e8fc5c99af82adbb18d5855499fa8684be56622889c90d56bd963a4397bb.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/5301e8fc5c99af82adbb18d5855499fa8684be56622889c90d56bd963a4397bb.jpg)

![544d07aa328668a7af12c63105c7a559afdd6b527e25a28c4a39c2a9ab6b1260.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/544d07aa328668a7af12c63105c7a559afdd6b527e25a28c4a39c2a9ab6b1260.jpg)

![66c7e70b9ce1dd464b50842329585d96c7f6c5f7002dff6b0ca2510f2aba874f.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/66c7e70b9ce1dd464b50842329585d96c7f6c5f7002dff6b0ca2510f2aba874f.jpg)

![7634d73449559dda5c73b49792c24f4e3376ccb97d1b01f7affa9fb6c989b84b.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/7634d73449559dda5c73b49792c24f4e3376ccb97d1b01f7affa9fb6c989b84b.jpg)

![81a0bbb241d7eaddbecb8370ee8b89b96108e8e5d484c5d794525e76e738d4de.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/81a0bbb241d7eaddbecb8370ee8b89b96108e8e5d484c5d794525e76e738d4de.jpg)

![8c14866b2b906b345157316ba9cc5f56dc05f8f819910cfd91dba79759ae1a76.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/8c14866b2b906b345157316ba9cc5f56dc05f8f819910cfd91dba79759ae1a76.jpg)

![c4c0a5fdb6298523522be733e4aed2f723e138afd2d65e37cbe47555bfeb0fb0.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/c4c0a5fdb6298523522be733e4aed2f723e138afd2d65e37cbe47555bfeb0fb0.jpg)

![d7b69c3e4cd3e3ca804e17082b58487de0dec0fb481316e6d0c6a34d840f6286.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/d7b69c3e4cd3e3ca804e17082b58487de0dec0fb481316e6d0c6a34d840f6286.jpg)

![dfb39b5eb829d4cc6e5ec9c277d7c8479cca95940119c9ac62c9d05ad531ac4f.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/dfb39b5eb829d4cc6e5ec9c277d7c8479cca95940119c9ac62c9d05ad531ac4f.jpg)

![f57f5020b2cbc8aa1bcd8c0754b475a6a1e3dd1a2398abeedfae3ca4c974c7a6.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/f57f5020b2cbc8aa1bcd8c0754b475a6a1e3dd1a2398abeedfae3ca4c974c7a6.jpg)

![f99490f967e1944aa79b3d6a2ff2b76d0b7cd7047ecbaa20f43845bcfdcaecf5.jpg](../iclr_results/2392_Execution-guided%20within-prompt%20search%20for%20programming-by-example/images/f99490f967e1944aa79b3d6a2ff2b76d0b7cd7047ecbaa20f43845bcfdcaecf5.jpg)

## Interpretable Unsupervised Joint Denoising and Enhancement for Real-World low-light Scenarios


### Images

![183b5e7967b111f57fde05c80eee0c087adc3e91d12cf140aea15f5de1e654c1.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/183b5e7967b111f57fde05c80eee0c087adc3e91d12cf140aea15f5de1e654c1.jpg)

![479198dbcafeafdacabde3b19cb26c9bcde2701ddee717e6a2686ef0d6c8e3ac.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/479198dbcafeafdacabde3b19cb26c9bcde2701ddee717e6a2686ef0d6c8e3ac.jpg)

![5273f7cb0734e3adddbf9485223476a309ea8ea98f32a520e49ccbea52ee7652.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/5273f7cb0734e3adddbf9485223476a309ea8ea98f32a520e49ccbea52ee7652.jpg)

![82dbc5d89c8774f86066ed01dd6628c4be8f1668b1c83e20625f0e2fadd57c95.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/82dbc5d89c8774f86066ed01dd6628c4be8f1668b1c83e20625f0e2fadd57c95.jpg)

![8e36830ce75d6dd21a87a286575bf83c7586f002d9fdfa09b698edb22eeaf066.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/8e36830ce75d6dd21a87a286575bf83c7586f002d9fdfa09b698edb22eeaf066.jpg)

![ab65b01f13ee33bba9efa36939d67523af2e69d4916f09a7f5d72002f84a3c16.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/ab65b01f13ee33bba9efa36939d67523af2e69d4916f09a7f5d72002f84a3c16.jpg)

![e3215085143086b4436fda138a18ee28976310cfd07a22477c36910df01d40e4.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/e3215085143086b4436fda138a18ee28976310cfd07a22477c36910df01d40e4.jpg)

![e9bd98c48e8b30d36b82aeeaa90b7be8e4adfd9664611992a5ae5e44ce5da18a.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/e9bd98c48e8b30d36b82aeeaa90b7be8e4adfd9664611992a5ae5e44ce5da18a.jpg)

![f431a899dd72107f3bbabdec90f0c8840d143c26f749ea92e85f66f4a3fbdea2.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/images/f431a899dd72107f3bbabdec90f0c8840d143c26f749ea92e85f66f4a3fbdea2.jpg)

### Tables

![07c692916d7076f602921041079c951c39ee1cc78857cafb549c07301e8320be.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/tables/07c692916d7076f602921041079c951c39ee1cc78857cafb549c07301e8320be.jpg)

![2e253181835ddd8a1c26e7ac5607219f0b407bcdbcaa6aacb86331a7eed5e713.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/tables/2e253181835ddd8a1c26e7ac5607219f0b407bcdbcaa6aacb86331a7eed5e713.jpg)

![7fd0162e52744bb51814a23cffae8969dcf6ae1f8bdea233bde2e55b8bda0165.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/tables/7fd0162e52744bb51814a23cffae8969dcf6ae1f8bdea233bde2e55b8bda0165.jpg)

![e9e92a34ab5b10154ecdc3ba17b8b16be75e65e25c33da02cabab5c1305287ad.jpg](../iclr_results/2393_Interpretable%20Unsupervised%20Joint%20Denoising%20and%20Enhancement%20for%20Real-World%20low-light%20Scenarios/tables/e9e92a34ab5b10154ecdc3ba17b8b16be75e65e25c33da02cabab5c1305287ad.jpg)

## A Training-Free Sub-quadratic Cost Transformer Model Serving Framework with Hierarchically Pruned Attention


### Images

![095b87e4dc06bca7803694364e911f95738420a0ff4c3bcc67cdc2eef074400c.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/095b87e4dc06bca7803694364e911f95738420a0ff4c3bcc67cdc2eef074400c.jpg)

![0b1421968240058ff495272c9d9bb79e8622572d2428299d34f1de9fc18b7eed.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/0b1421968240058ff495272c9d9bb79e8622572d2428299d34f1de9fc18b7eed.jpg)

![0deff5c6582eb5a3d92c947e5a093120ef5bc5917c82fe81ecc85ef75558c74b.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/0deff5c6582eb5a3d92c947e5a093120ef5bc5917c82fe81ecc85ef75558c74b.jpg)

![16115094e82bb948187d675f997e6e2bb68d8cd6a6fb320610ca0f3028f5f34f.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/16115094e82bb948187d675f997e6e2bb68d8cd6a6fb320610ca0f3028f5f34f.jpg)

![1e8696a3540ee5be3674663fd674c4f2653710fd020675989e235b1272e2967d.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/1e8696a3540ee5be3674663fd674c4f2653710fd020675989e235b1272e2967d.jpg)

![2e87b16b53ee4493c88e7383f0702b4266c145efc6df8b25d7042d538b673530.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/2e87b16b53ee4493c88e7383f0702b4266c145efc6df8b25d7042d538b673530.jpg)

![4a80382ac161e786f932ab34851463a5b9a29b96e96136a7a9fa5cf57ebab913.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/4a80382ac161e786f932ab34851463a5b9a29b96e96136a7a9fa5cf57ebab913.jpg)

![4aae3725aa4fdec024a24ec9f484b588b2d43dd102550f629513b027495784bd.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/4aae3725aa4fdec024a24ec9f484b588b2d43dd102550f629513b027495784bd.jpg)

![4b8b4d5ca7892a671c1471901fcf355da4d4bdcd3f560cbf9dacb9fbd3181eef.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/4b8b4d5ca7892a671c1471901fcf355da4d4bdcd3f560cbf9dacb9fbd3181eef.jpg)

![4c3397c8037bacd1901e3cc55838f14dc586f73c9c8abea26f273d6a2fffdeab.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/4c3397c8037bacd1901e3cc55838f14dc586f73c9c8abea26f273d6a2fffdeab.jpg)

![4d239afde2139b004ad67c2afd6a47089e6bd92119afed4d6b32dd9fc3293278.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/4d239afde2139b004ad67c2afd6a47089e6bd92119afed4d6b32dd9fc3293278.jpg)

![65229322529819ad748b92333189907b514742756d05e2712f797b1c584209c2.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/65229322529819ad748b92333189907b514742756d05e2712f797b1c584209c2.jpg)

![782f46bde7b4078c643a5d2cee9343f32b4a6a3ae2d054bbcabb0393821f25d4.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/782f46bde7b4078c643a5d2cee9343f32b4a6a3ae2d054bbcabb0393821f25d4.jpg)

![893798d6a2ba300a456e5d832868a6fb96c20beddb7b473df060781f189d64dc.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/893798d6a2ba300a456e5d832868a6fb96c20beddb7b473df060781f189d64dc.jpg)

![976fb718ee983413eb865ecefbcff6f187ae89720a6d3998f1debbd76104fbed.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/976fb718ee983413eb865ecefbcff6f187ae89720a6d3998f1debbd76104fbed.jpg)

![9f35214412ec2c35c93b44760d8eda7fa401789403743bd607244d31f18f9ba7.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/9f35214412ec2c35c93b44760d8eda7fa401789403743bd607244d31f18f9ba7.jpg)

![ae5ff60198cfd67f3fdf1b8cb761eb0a36ef59e287744f29d6a18bb78232b8bf.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/ae5ff60198cfd67f3fdf1b8cb761eb0a36ef59e287744f29d6a18bb78232b8bf.jpg)

![b854c1b553336a10beb5cd5c4d91b215311c565cf7f72784cf32c51f2cdc6713.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/b854c1b553336a10beb5cd5c4d91b215311c565cf7f72784cf32c51f2cdc6713.jpg)

![bff7a0d72e33704957067f404823204ac00a1ad932dec2f6b6d7d48415010987.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/bff7a0d72e33704957067f404823204ac00a1ad932dec2f6b6d7d48415010987.jpg)

![c3f7f7f59c7cc6fd3f6ff57a584ea049246ec523a32449296f17631637c74fcf.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/c3f7f7f59c7cc6fd3f6ff57a584ea049246ec523a32449296f17631637c74fcf.jpg)

![ce63b0c8f65ec40c31c5b498fc8ed85a82943f060acb3a921eae699df186e731.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/ce63b0c8f65ec40c31c5b498fc8ed85a82943f060acb3a921eae699df186e731.jpg)

![d84f400391a9b1b429b14178f79536a8f7fe4307854e1530f0b599627d9436ee.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/d84f400391a9b1b429b14178f79536a8f7fe4307854e1530f0b599627d9436ee.jpg)

![dc66d993d26f6e8f0c7b3f4d1c17ccbc32a22ed21d2f26bb8a441cd695c7c540.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/dc66d993d26f6e8f0c7b3f4d1c17ccbc32a22ed21d2f26bb8a441cd695c7c540.jpg)

![f02ee7af1bfaaa9ab0e6ba16c7f348a6d7e32311be38cbe6dc060659dfa7d0b3.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/f02ee7af1bfaaa9ab0e6ba16c7f348a6d7e32311be38cbe6dc060659dfa7d0b3.jpg)

![f19ba6855fd04ec40e373c74eed9070dc28d83bc304b031aa5fbe40923d0f7d2.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/f19ba6855fd04ec40e373c74eed9070dc28d83bc304b031aa5fbe40923d0f7d2.jpg)

![f9064e777cf2ecc7d4c31316904c156ddc1525b0c1965fe7387d7de621584c8c.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/images/f9064e777cf2ecc7d4c31316904c156ddc1525b0c1965fe7387d7de621584c8c.jpg)

### Tables

![02f1e5f04fdba6015f1418f47684489cf970e4f14a86dc1557b7477637d8a92c.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/02f1e5f04fdba6015f1418f47684489cf970e4f14a86dc1557b7477637d8a92c.jpg)

![0c67f78e583c15bd868bf0036e0c52fa8213f4ca067c931aa7da2dbd532022d0.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/0c67f78e583c15bd868bf0036e0c52fa8213f4ca067c931aa7da2dbd532022d0.jpg)

![1d61c3606ea11c68d5753514e9739afa0bfe0ade70870ef109978457576ec5cb.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/1d61c3606ea11c68d5753514e9739afa0bfe0ade70870ef109978457576ec5cb.jpg)

![216729ad99dd4ac9b21cb657d2bb1b33e7b3c03f3852cd22f49c47a34145b86d.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/216729ad99dd4ac9b21cb657d2bb1b33e7b3c03f3852cd22f49c47a34145b86d.jpg)

![326c76eeca9e64332375335905b725addcbc31a46885291d33edc7a8624025d1.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/326c76eeca9e64332375335905b725addcbc31a46885291d33edc7a8624025d1.jpg)

![3993b818d61a6e6615b54bc723c2da8ae5c08952dd35f05b33852801dc404d7e.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/3993b818d61a6e6615b54bc723c2da8ae5c08952dd35f05b33852801dc404d7e.jpg)

![54b4c5fdb9ddea4b1500819eef652a247bbd6481f01268210a4dd6a1a71ff4db.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/54b4c5fdb9ddea4b1500819eef652a247bbd6481f01268210a4dd6a1a71ff4db.jpg)

![600c0c0fc164270f85101ed55968841331f3d0997b41e316cfbc716a44f836a6.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/600c0c0fc164270f85101ed55968841331f3d0997b41e316cfbc716a44f836a6.jpg)

![73f286ef173f470563650c5bb82bce97ad711f89e752e8300b7de9beb7154725.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/73f286ef173f470563650c5bb82bce97ad711f89e752e8300b7de9beb7154725.jpg)

![74411eefc93fe522be1f176e967be43dd3e1e64cf9dee1bdbd52746395103161.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/74411eefc93fe522be1f176e967be43dd3e1e64cf9dee1bdbd52746395103161.jpg)

![849a82e86fb2ee1eeb967efa98c6d2d55deea14cf238bfe2866765dae61ab0bf.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/849a82e86fb2ee1eeb967efa98c6d2d55deea14cf238bfe2866765dae61ab0bf.jpg)

![8d785cad578979c7039185099cdaeec39c1bfc655696de401ece8861abb6020f.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/8d785cad578979c7039185099cdaeec39c1bfc655696de401ece8861abb6020f.jpg)

![90e72d861cda33c35a22dfd86011e250ccc7a60757198598d5abae8fbe3e5a6f.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/90e72d861cda33c35a22dfd86011e250ccc7a60757198598d5abae8fbe3e5a6f.jpg)

![97a3adcb5482f3403c30583b36a8c3ebbe655931ec4b8f61beda4abc28783f0f.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/97a3adcb5482f3403c30583b36a8c3ebbe655931ec4b8f61beda4abc28783f0f.jpg)

![b4a5f92ebaae809ac9943375f4d7efdccc1f14fe5fd8d359b927bc6b44ef2aea.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/b4a5f92ebaae809ac9943375f4d7efdccc1f14fe5fd8d359b927bc6b44ef2aea.jpg)

![c502afae86d6063423e32f34d497658fcc56af72c826b09f233dec4944561ef6.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/c502afae86d6063423e32f34d497658fcc56af72c826b09f233dec4944561ef6.jpg)

![cf78764b6ff4bdb03bdc101a7d5abd1718ffb338fc1ea40dc11d7e410df20ed1.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/cf78764b6ff4bdb03bdc101a7d5abd1718ffb338fc1ea40dc11d7e410df20ed1.jpg)

![d49e4ce9f8eaf43fff1dee3fbe0bb3a594fe3aa804fe75c6df62e7278b5fd937.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/d49e4ce9f8eaf43fff1dee3fbe0bb3a594fe3aa804fe75c6df62e7278b5fd937.jpg)

![f0935ed54e78cf7a86f8f9b15d475bf80698de152219baa1d190c63d3311e082.jpg](../iclr_results/2394_A%20Training-Free%20Sub-quadratic%20Cost%20Transformer%20Model%20Serving%20Framework%20with%20Hierarchically%20Pruned%20At/tables/f0935ed54e78cf7a86f8f9b15d475bf80698de152219baa1d190c63d3311e082.jpg)

## ReSi: A Comprehensive Benchmark for Representational Similarity Measures


### Images

![0adea450073122fe82bd8e858b58adc39ac924005fa383c84032af045930ed03.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/0adea450073122fe82bd8e858b58adc39ac924005fa383c84032af045930ed03.jpg)

![1883b4313fe315864438fbf3501f9c28dd332d526a6c0f7a5d23e53abe2646b8.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/1883b4313fe315864438fbf3501f9c28dd332d526a6c0f7a5d23e53abe2646b8.jpg)

![195deee76cf1f026915f6ac54e85714806487c0ff6653bd2960efcbd12bf97fe.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/195deee76cf1f026915f6ac54e85714806487c0ff6653bd2960efcbd12bf97fe.jpg)

![316aaf4206e78c125e31195c6bba151240ae8e7c904fd2bf5f041f3314b7f62a.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/316aaf4206e78c125e31195c6bba151240ae8e7c904fd2bf5f041f3314b7f62a.jpg)

![7f2388eb0463b9218c3456df1e90960559fcb76d81d6a367dfeb5ee4705ef415.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/7f2388eb0463b9218c3456df1e90960559fcb76d81d6a367dfeb5ee4705ef415.jpg)

![8749f4817a663bc455f4602545b2c7c72819ccff3965eb92e6284943fc454ade.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/8749f4817a663bc455f4602545b2c7c72819ccff3965eb92e6284943fc454ade.jpg)

![a4f3182bf77060cbcf4c1f6768e404c3a1f8592aa37ec220c587f0e3c8bebeab.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/a4f3182bf77060cbcf4c1f6768e404c3a1f8592aa37ec220c587f0e3c8bebeab.jpg)

![a775b2b19f3cc8c356c838f0913b9ac6d1fe7ebb968c8b58bffa71c03bccdeb0.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/a775b2b19f3cc8c356c838f0913b9ac6d1fe7ebb968c8b58bffa71c03bccdeb0.jpg)

![a9e8acff6ad0727d3be9f8e24bed078a66ea7fa20f6564b5334a206fcc583555.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/a9e8acff6ad0727d3be9f8e24bed078a66ea7fa20f6564b5334a206fcc583555.jpg)

![c037887e69d0618a503707c39412877a6bb913a802c3072dc34c2bb0da88e339.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/c037887e69d0618a503707c39412877a6bb913a802c3072dc34c2bb0da88e339.jpg)

![c1499231c5075bf0a79c6d174b45a3ea9bd7139b9c0b7e5d3d7b18a72b4aa723.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/c1499231c5075bf0a79c6d174b45a3ea9bd7139b9c0b7e5d3d7b18a72b4aa723.jpg)

![cd304d138cb16335e203157e442638ec6f4f161caca8370802b5750e54a1eb8a.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/cd304d138cb16335e203157e442638ec6f4f161caca8370802b5750e54a1eb8a.jpg)

![d62726f461cafc117106cd20ddf18478ad2f8a6412f91a005cda0167fc8b011d.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/d62726f461cafc117106cd20ddf18478ad2f8a6412f91a005cda0167fc8b011d.jpg)

![dce106e792f88939774328e905af1eb135ed7ee1991d590dd4fc808249660403.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/dce106e792f88939774328e905af1eb135ed7ee1991d590dd4fc808249660403.jpg)

![efca272f0c4be184091a721bb0756eb1081e989fcc0b9517eb0f4f9fb1a082f2.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/images/efca272f0c4be184091a721bb0756eb1081e989fcc0b9517eb0f4f9fb1a082f2.jpg)

### Tables

![083ce7d83a563a70b635db74bc25a882e92686a8ed14021b6b2b5dced30960ca.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/083ce7d83a563a70b635db74bc25a882e92686a8ed14021b6b2b5dced30960ca.jpg)

![14fed2c614efc288b75e6ed6b0ce67987da3bd9decaffeb9fbb70127d99fca32.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/14fed2c614efc288b75e6ed6b0ce67987da3bd9decaffeb9fbb70127d99fca32.jpg)

![1cbde1a8cca506ca62dc2345ba60f974b15965b761e1693c11e0e4286356a1ce.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/1cbde1a8cca506ca62dc2345ba60f974b15965b761e1693c11e0e4286356a1ce.jpg)

![1e0a23788c0dc59c337044a03b6ea8f4f1820d8e9a01a9d8b084266a31dcbb44.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/1e0a23788c0dc59c337044a03b6ea8f4f1820d8e9a01a9d8b084266a31dcbb44.jpg)

![23737f0c59e73cd35246f51b0800b2a0b90fbf66af53ae59fe5897fa6bb283b1.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/23737f0c59e73cd35246f51b0800b2a0b90fbf66af53ae59fe5897fa6bb283b1.jpg)

![2d5458892d51f65edd59f8ca5352ee1b0d28679bfbfb872e46e1838c0076133f.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/2d5458892d51f65edd59f8ca5352ee1b0d28679bfbfb872e46e1838c0076133f.jpg)

![2e076322bd65d25b78729157cd10b06b363398dda0fd2a17b93dd7a9bae66d55.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/2e076322bd65d25b78729157cd10b06b363398dda0fd2a17b93dd7a9bae66d55.jpg)

![3b8db5f5e00762e3a6d218f09055efd2346ddb6423c9b10129fc90133ad36f1d.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/3b8db5f5e00762e3a6d218f09055efd2346ddb6423c9b10129fc90133ad36f1d.jpg)

![46375b975c34802945ea0ffe77e2ec4558e3c841968d993e9efe708afa0436df.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/46375b975c34802945ea0ffe77e2ec4558e3c841968d993e9efe708afa0436df.jpg)

![491189728e5e0f7f993aed80d9791eeeab6b053338726057ddadc72d88770a79.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/491189728e5e0f7f993aed80d9791eeeab6b053338726057ddadc72d88770a79.jpg)

![546276346ed291c0fbea475a36c8a92a0bd7563013c9486b9ab882066ede3f7f.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/546276346ed291c0fbea475a36c8a92a0bd7563013c9486b9ab882066ede3f7f.jpg)

![5c979efd9b3889bda15ac44472020229ffed17cfe9e89db823ba52100bc2eff9.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/5c979efd9b3889bda15ac44472020229ffed17cfe9e89db823ba52100bc2eff9.jpg)

![5fbba3072544aca6d0a0cb1c4b949c17ffd5f1d987c67e63fef3234ba55392cd.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/5fbba3072544aca6d0a0cb1c4b949c17ffd5f1d987c67e63fef3234ba55392cd.jpg)

![78e7be1dbad0f7a91930cb40531c136670f8f51e34dd9e5edfb17416611bcd97.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/78e7be1dbad0f7a91930cb40531c136670f8f51e34dd9e5edfb17416611bcd97.jpg)

![80460b40b6514547e1c2972416cea3f0f6cf6c4ca5fe32bb5e1c80370cae95cc.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/80460b40b6514547e1c2972416cea3f0f6cf6c4ca5fe32bb5e1c80370cae95cc.jpg)

![85fcd934763b50f4ca75f11f76fbd4ef8872ba54dd7377017d0bc6e4725c0a8e.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/85fcd934763b50f4ca75f11f76fbd4ef8872ba54dd7377017d0bc6e4725c0a8e.jpg)

![9045de8717e6001c052df21b18610560175bacf8a69b99432a759efc0574eea8.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/9045de8717e6001c052df21b18610560175bacf8a69b99432a759efc0574eea8.jpg)

![9789cba44425280900661edc0f835e86b0b47686e1a4f99708afae1f26b075ef.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/9789cba44425280900661edc0f835e86b0b47686e1a4f99708afae1f26b075ef.jpg)

![9b915d5ea911c46d8c15fb141c150f142ce31201d2e1e10584413aba8ecee6f6.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/9b915d5ea911c46d8c15fb141c150f142ce31201d2e1e10584413aba8ecee6f6.jpg)

![ae9e39d4c4897ac82449576cfe4a9eb559c906a88f3d9bbd46628213fcd650db.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/ae9e39d4c4897ac82449576cfe4a9eb559c906a88f3d9bbd46628213fcd650db.jpg)

![b7e4b3cd666e0781647899d53103ab342a2b3d935e6b07d59d4035d4b06e6322.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/b7e4b3cd666e0781647899d53103ab342a2b3d935e6b07d59d4035d4b06e6322.jpg)

![c621d65187e67351590b4bc07b1db5f23a34d312b934be721fd88f0e091e45fd.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/c621d65187e67351590b4bc07b1db5f23a34d312b934be721fd88f0e091e45fd.jpg)

![cdba5e0dea8e1af408b3a08286ddc006249ceadae2ee866a8e91d1858947c0ca.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/cdba5e0dea8e1af408b3a08286ddc006249ceadae2ee866a8e91d1858947c0ca.jpg)

![ce51fe85f7e7f9c7f85e011563bb5712738564837fe66995b4140f743e18b425.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/ce51fe85f7e7f9c7f85e011563bb5712738564837fe66995b4140f743e18b425.jpg)

![d0873ab5e1430f97f1ac1221e87b6afba6232b0563c0185226d799166c54fe03.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/d0873ab5e1430f97f1ac1221e87b6afba6232b0563c0185226d799166c54fe03.jpg)

![da86e98c1f8eb6cf820e7359bb32ade3fae97a3c21d622acb04925452b69644d.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/da86e98c1f8eb6cf820e7359bb32ade3fae97a3c21d622acb04925452b69644d.jpg)

![de6ef4cef8070484d6306e6edad35705e2a87c0696fe12d700fe18bd258bb78c.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/de6ef4cef8070484d6306e6edad35705e2a87c0696fe12d700fe18bd258bb78c.jpg)

![e14fbc53256b03dab98dfb73063d4a9ecaebb18f436a5da4b83a25416f6f9e2b.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/e14fbc53256b03dab98dfb73063d4a9ecaebb18f436a5da4b83a25416f6f9e2b.jpg)

![e61c1a0dc912406486889bb7779bad55a16cb3622b3b9c407332b6bf1c6e6c6d.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/e61c1a0dc912406486889bb7779bad55a16cb3622b3b9c407332b6bf1c6e6c6d.jpg)

![ea555470de26816c4ef5f40da1b9bca14a91d731b5b0ff8aa4bdc90e8533a862.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/ea555470de26816c4ef5f40da1b9bca14a91d731b5b0ff8aa4bdc90e8533a862.jpg)

![f9117520c729a64cd5b40c266ce86c0b919358644e8754d5ae40bee45b9e107b.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/f9117520c729a64cd5b40c266ce86c0b919358644e8754d5ae40bee45b9e107b.jpg)

![fb5083f9b3bd571ae2ca372945b7e46519ae5d524cdaf773bc77c30731053043.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/fb5083f9b3bd571ae2ca372945b7e46519ae5d524cdaf773bc77c30731053043.jpg)

![fd93e0d01f85bdb8d82584edd3e445d6c15b9279ede2c0d8a4044d7bc96009b5.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/fd93e0d01f85bdb8d82584edd3e445d6c15b9279ede2c0d8a4044d7bc96009b5.jpg)

![fe665dd25259bae1fce8ec24d1754e6192c068269d50552003ba7de67200dd99.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/fe665dd25259bae1fce8ec24d1754e6192c068269d50552003ba7de67200dd99.jpg)

![feca770d0796b2b3b8be53b1856b2a03212db1e00b3a89c1c5c99731737c3c64.jpg](../iclr_results/2395_ReSi_%20A%20Comprehensive%20Benchmark%20for%20Representational%20Similarity%20Measures/tables/feca770d0796b2b3b8be53b1856b2a03212db1e00b3a89c1c5c99731737c3c64.jpg)

## Autoregressive Pretraining with Mamba in Vision


### Images

![614f18c9b9375eecf058487be01743181aa1516942f4e527ff44818ad5dbdf12.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/images/614f18c9b9375eecf058487be01743181aa1516942f4e527ff44818ad5dbdf12.jpg)

![68329ea7c9376f7239bf4a31e887403dd7c7e11de9388ab55b44b103e9622090.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/images/68329ea7c9376f7239bf4a31e887403dd7c7e11de9388ab55b44b103e9622090.jpg)

![a3defed606c472fbb04dd35e1ecee221f6bf61b5ffb92ff5caf177463773e1b8.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/images/a3defed606c472fbb04dd35e1ecee221f6bf61b5ffb92ff5caf177463773e1b8.jpg)

![dd7c32255e467b9ac4897b3d34ea3ada0e0193af0760e4726b7cc2307885cb8e.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/images/dd7c32255e467b9ac4897b3d34ea3ada0e0193af0760e4726b7cc2307885cb8e.jpg)

### Tables

![27b961731bcc5fc334ef5dbbf74ccd5d718a53d50e0efd0ef2fc97e1bea1524f.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/27b961731bcc5fc334ef5dbbf74ccd5d718a53d50e0efd0ef2fc97e1bea1524f.jpg)

![315ad8eda33a150187b2702afad9b98fd022f7f12b2d3bd2f10838b04d987986.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/315ad8eda33a150187b2702afad9b98fd022f7f12b2d3bd2f10838b04d987986.jpg)

![52792cc37993d0a1cfca010e4759fd7e209410584bd24418ca0afd8737c4741c.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/52792cc37993d0a1cfca010e4759fd7e209410584bd24418ca0afd8737c4741c.jpg)

![6cf183d6e32160ad50cf8e3232359c6f74c72061446477e538219276f04e211d.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/6cf183d6e32160ad50cf8e3232359c6f74c72061446477e538219276f04e211d.jpg)

![6dc7f5799522ab3479af599201768fd31b59ede827aabda52534ea77c42a4518.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/6dc7f5799522ab3479af599201768fd31b59ede827aabda52534ea77c42a4518.jpg)

![6fc155ddc92c7b5d03336a35586f86f9847a46d29a695335fc7796a6be4dbe91.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/6fc155ddc92c7b5d03336a35586f86f9847a46d29a695335fc7796a6be4dbe91.jpg)

![86c2a3704207d676cedd222c2795c5f59071c3200ebc793ff98f3608a413e419.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/86c2a3704207d676cedd222c2795c5f59071c3200ebc793ff98f3608a413e419.jpg)

![b7ba4e46d487bfe5ac9e8e56fa0a25cdbd3179486c9d4c979f51c720285f87ce.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/b7ba4e46d487bfe5ac9e8e56fa0a25cdbd3179486c9d4c979f51c720285f87ce.jpg)

![c040b5b4cd1f4620de11777c4fed0526bd3b5ba883c598c06989ba06ca4a0260.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/c040b5b4cd1f4620de11777c4fed0526bd3b5ba883c598c06989ba06ca4a0260.jpg)

![c20ce12dfa2b48a3706b3e3a3010d9343fe7e1443a5ae428a75e801135b33a66.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/c20ce12dfa2b48a3706b3e3a3010d9343fe7e1443a5ae428a75e801135b33a66.jpg)

![f1975d85e094dcea10a0a26b94f8aa95d752f4932ae00cc95ccf143b6019a3ee.jpg](../iclr_results/2396_Autoregressive%20Pretraining%20with%20Mamba%20in%20Vision/tables/f1975d85e094dcea10a0a26b94f8aa95d752f4932ae00cc95ccf143b6019a3ee.jpg)

## Improved Training Technique for Latent Consistency Models


### Images

![046c43398944e6944697d0989d219f58dc3ef2367303e1355365be45290c2ce3.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/046c43398944e6944697d0989d219f58dc3ef2367303e1355365be45290c2ce3.jpg)

![0713156e1fa4b1ca45026940e9b7652990d645e269319167b6b5d8df89c6f831.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/0713156e1fa4b1ca45026940e9b7652990d645e269319167b6b5d8df89c6f831.jpg)

![1a29bb9280e24e4d40a5bb9f97e6455f1ebfe78947ea392478dead215f5ebfdf.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/1a29bb9280e24e4d40a5bb9f97e6455f1ebfe78947ea392478dead215f5ebfdf.jpg)

![1cbc332a8d92c26e5d5e641ab85ce72bc8ecbdf63c8dcb18989615b388346ae5.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/1cbc332a8d92c26e5d5e641ab85ce72bc8ecbdf63c8dcb18989615b388346ae5.jpg)

![366299f44667e672f3f93d183b42d1bb3f495f22d2f1b7fa474faa0be2ff83c4.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/366299f44667e672f3f93d183b42d1bb3f495f22d2f1b7fa474faa0be2ff83c4.jpg)

![5a9ffe354d505670de64353238a3b69cc1f4b728ff8935eb2da1150431e14298.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/5a9ffe354d505670de64353238a3b69cc1f4b728ff8935eb2da1150431e14298.jpg)

![6484dda90a99dd7aa9fec6911d4aca59513d429d50737e851b8728f178a697f3.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/6484dda90a99dd7aa9fec6911d4aca59513d429d50737e851b8728f178a697f3.jpg)

![6ff40ea03f0f5bf4b930dcbd32be1645df24de9414a71e5f1ade09e3f7267a31.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/6ff40ea03f0f5bf4b930dcbd32be1645df24de9414a71e5f1ade09e3f7267a31.jpg)

![8bd2a4107179023d60f23395b0b17755270348752f596493c6a45672cb46e916.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/8bd2a4107179023d60f23395b0b17755270348752f596493c6a45672cb46e916.jpg)

![99d83ec34579903af578d1fa39928ffa1e4d58b4f569eb517b268591d38b5540.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/99d83ec34579903af578d1fa39928ffa1e4d58b4f569eb517b268591d38b5540.jpg)

![a6c009e96516033945081fc9bda799c10f264ba03cca482cc8fb783616d5137a.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/a6c009e96516033945081fc9bda799c10f264ba03cca482cc8fb783616d5137a.jpg)

![a8323e7765209d053e5e4d1975e83775c75263c277bdc0fd7bc2f09b3a423de1.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/a8323e7765209d053e5e4d1975e83775c75263c277bdc0fd7bc2f09b3a423de1.jpg)

![e6d3733d2ccd63f70b24e41132759aae4783963048bd865dba1d531b9908d807.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/images/e6d3733d2ccd63f70b24e41132759aae4783963048bd865dba1d531b9908d807.jpg)

### Tables

![8125c90824c26b278d67b2839a8a25826f2c2bdde352c7398a6f0d87e2500c88.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/tables/8125c90824c26b278d67b2839a8a25826f2c2bdde352c7398a6f0d87e2500c88.jpg)

![890e4f550bdab0f4ed8ab77086ba92d53bbe3385790a9acfbdddf8f53770cca2.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/tables/890e4f550bdab0f4ed8ab77086ba92d53bbe3385790a9acfbdddf8f53770cca2.jpg)

![97cfe7f358fb8e2b12443e6b06826da1a16af3c826a856ab88c968c604762d27.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/tables/97cfe7f358fb8e2b12443e6b06826da1a16af3c826a856ab88c968c604762d27.jpg)

![e037a7e8710e2ce1d0c7aeb7911d9d861ce49c75931e7d4859c94c309293de5c.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/tables/e037a7e8710e2ce1d0c7aeb7911d9d861ce49c75931e7d4859c94c309293de5c.jpg)

![ed17b508b09a84ecd9c40e908f02e1210e64ca45c2e542f82cb4befb1ba06671.jpg](../iclr_results/2397_Improved%20Training%20Technique%20for%20Latent%20Consistency%20Models/tables/ed17b508b09a84ecd9c40e908f02e1210e64ca45c2e542f82cb4befb1ba06671.jpg)

## Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization


### Images

![25bcbe15561ae641d757c7f06f7fc66aa9d5e5474854f2bab95d39340ae05bd3.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/25bcbe15561ae641d757c7f06f7fc66aa9d5e5474854f2bab95d39340ae05bd3.jpg)

![326bb81dfc52bbdcedd030342443ea198367633e67c0bcb1085a42d74478056b.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/326bb81dfc52bbdcedd030342443ea198367633e67c0bcb1085a42d74478056b.jpg)

![41ca5089872d8019d552ce818084ee64ad9fdb93fce5da694e7e3e252f9c5ecf.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/41ca5089872d8019d552ce818084ee64ad9fdb93fce5da694e7e3e252f9c5ecf.jpg)

![4fe33b12c1284075ae4534e8ebe463ecfc14ba62534f032f596f1748bccd1cb7.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/4fe33b12c1284075ae4534e8ebe463ecfc14ba62534f032f596f1748bccd1cb7.jpg)

![548cf1e457fc6bc10d1d3b628a8829b19ebd1f6709ee4371946bf8e94d615114.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/548cf1e457fc6bc10d1d3b628a8829b19ebd1f6709ee4371946bf8e94d615114.jpg)

![6a55ff7aa850a2eb69e4e521da02b1496504830e909f2ec59e351e38c91d11df.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/6a55ff7aa850a2eb69e4e521da02b1496504830e909f2ec59e351e38c91d11df.jpg)

![858195fe01eb476c1d9243d79f70cfa9525bb8b0783612fb98e9bc6a1dabf2ec.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/858195fe01eb476c1d9243d79f70cfa9525bb8b0783612fb98e9bc6a1dabf2ec.jpg)

![93b425e6a0547276945c50ee4a2270b35de88efafc1fd47b0373129a9d93cbb3.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/93b425e6a0547276945c50ee4a2270b35de88efafc1fd47b0373129a9d93cbb3.jpg)

![94640336ef7640e5efc496baee603b6c94a4bdb3898043a18d27ceb9d1dfa13a.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/94640336ef7640e5efc496baee603b6c94a4bdb3898043a18d27ceb9d1dfa13a.jpg)

![a4bc2fcabb2b77d97fb38d0ec5e72bbb4ae9018dd5780f08af03e8fb961a1b26.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/a4bc2fcabb2b77d97fb38d0ec5e72bbb4ae9018dd5780f08af03e8fb961a1b26.jpg)

![b6a74a661088c13088e2f2b0987b4b1b77d0a7ede297d7d62a112bb2a021d969.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/b6a74a661088c13088e2f2b0987b4b1b77d0a7ede297d7d62a112bb2a021d969.jpg)

![cc9552a2453ca1c48c0f2f184de641012601b57ef9d93e7e29167b831b837b6c.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/cc9552a2453ca1c48c0f2f184de641012601b57ef9d93e7e29167b831b837b6c.jpg)

![e5d36b534e527f2612afa3b73e48b4836ed96d3af29b6a6a85c7027d6eb5f5b5.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/e5d36b534e527f2612afa3b73e48b4836ed96d3af29b6a6a85c7027d6eb5f5b5.jpg)

![fff01469e145f2fd6073d5d447193bb90bb38fcff1b0158b690e96111f4562d6.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/images/fff01469e145f2fd6073d5d447193bb90bb38fcff1b0158b690e96111f4562d6.jpg)

### Tables

![5c5eb04b02c955ac78964ed335b6bcd81f9493aeba45f0ddda65e786acb86324.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/tables/5c5eb04b02c955ac78964ed335b6bcd81f9493aeba45f0ddda65e786acb86324.jpg)

![5dbb52154923adc4e57507920dce034023ebe20b31a9116d4ff6b52633118605.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/tables/5dbb52154923adc4e57507920dce034023ebe20b31a9116d4ff6b52633118605.jpg)

![7541cba659d683bb45eaaf8b7414e19c18b740dcfd32feed1a692e95a5a37883.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/tables/7541cba659d683bb45eaaf8b7414e19c18b740dcfd32feed1a692e95a5a37883.jpg)

![a1843dab0b808d2ff746b60dd78f10e625d45e070703fbb6cb9a874d26219437.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/tables/a1843dab0b808d2ff746b60dd78f10e625d45e070703fbb6cb9a874d26219437.jpg)

![c26cec62bc6d86160cfdc5ba586b20dc5d60ac8d085d995b24069ca241cb71b2.jpg](../iclr_results/2398_Mitigating%20Reward%20Over-Optimization%20in%20RLHF%20via%20Behavior-Supported%20Regularization/tables/c26cec62bc6d86160cfdc5ba586b20dc5d60ac8d085d995b24069ca241cb71b2.jpg)

## Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional, Black-box Systems


### Images

![21245244b4f52426194b1a79a4700de7b18ce5ca246a60b890ae029c56b10a98.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/21245244b4f52426194b1a79a4700de7b18ce5ca246a60b890ae029c56b10a98.jpg)

![229a8c1485efb746816030d18fdced038a502484fe7baf2c41f172a4f76722a6.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/229a8c1485efb746816030d18fdced038a502484fe7baf2c41f172a4f76722a6.jpg)

![2416640d060026c5b97a210ce52f2f9f44083a25ccf28ff803694a903b1e4bb9.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/2416640d060026c5b97a210ce52f2f9f44083a25ccf28ff803694a903b1e4bb9.jpg)

![2d4cadcf12f2500f3c4c76dabaa38a16e38291407797a59e54327041887a7912.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/2d4cadcf12f2500f3c4c76dabaa38a16e38291407797a59e54327041887a7912.jpg)

![3544671ac06e0baef83c628bba0c0299ed49dbe0ef60ee21e6ba6cbccd38ed42.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/3544671ac06e0baef83c628bba0c0299ed49dbe0ef60ee21e6ba6cbccd38ed42.jpg)

![645bf0208590f1275d013f79043962fd76c43db309a4ae65cb10c8401a77f793.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/645bf0208590f1275d013f79043962fd76c43db309a4ae65cb10c8401a77f793.jpg)

![794962cd171bba72d9bb2a165fe1e0b824ce30c473b964936226442ea432ebc1.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/794962cd171bba72d9bb2a165fe1e0b824ce30c473b964936226442ea432ebc1.jpg)

![84ef3fc2426b285a989042e14596fcc5bef8e6f58cd1ec976ab65c26fc834f30.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/84ef3fc2426b285a989042e14596fcc5bef8e6f58cd1ec976ab65c26fc834f30.jpg)

![92b28454295b73e2561ea5038c13a2f6c429e958c6a273e922002497d2c7ff26.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/92b28454295b73e2561ea5038c13a2f6c429e958c6a273e922002497d2c7ff26.jpg)

![98908e23916fb4f4e844c5f68dbd97b1fed2381a79f4514f76037eba74e47d1b.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/98908e23916fb4f4e844c5f68dbd97b1fed2381a79f4514f76037eba74e47d1b.jpg)

![9b1c321a910cb3f8eea8d50ea18e692c592353f998f435dbc36f93955380d248.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/9b1c321a910cb3f8eea8d50ea18e692c592353f998f435dbc36f93955380d248.jpg)

![a853a65e62569b102d05a92c8fefb1f1baff550dd5846b9e849e28ad0a822f01.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/a853a65e62569b102d05a92c8fefb1f1baff550dd5846b9e849e28ad0a822f01.jpg)

![d799b748e0489e6956bcca1da263ec00bc63fefef33716ddcb15a8c59ba9a690.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/d799b748e0489e6956bcca1da263ec00bc63fefef33716ddcb15a8c59ba9a690.jpg)

![e331c6247ce5d087760ec123cad5ca50b5f654fa53f51697483892571c0e2731.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/e331c6247ce5d087760ec123cad5ca50b5f654fa53f51697483892571c0e2731.jpg)

![edc66b48e814f5d0321b2964d81b0b6473835562a0b9bca08c215c1e62e32d60.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/edc66b48e814f5d0321b2964d81b0b6473835562a0b9bca08c215c1e62e32d60.jpg)

![f328d975464252c470427f7858678f6c5078252361a7975ef1b40dc0763d3e79.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/images/f328d975464252c470427f7858678f6c5078252361a7975ef1b40dc0763d3e79.jpg)

### Tables

![08d01b87e84caea1304f39ba0aa0d8b7a1aba2e6f66ce63139a50d070d5aa035.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/tables/08d01b87e84caea1304f39ba0aa0d8b7a1aba2e6f66ce63139a50d070d5aa035.jpg)

![16b5120b908d8e29c1b4aff1c383496c4faeecf34cd9a4bf1e848268370d06df.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/tables/16b5120b908d8e29c1b4aff1c383496c4faeecf34cd9a4bf1e848268370d06df.jpg)

![9ac3c297d2803b8ee6fa161277b96ff1b15857841aacb464204d83662245f814.jpg](../iclr_results/2399_Gaussian%20Ensemble%20Belief%20Propagation%20for%20Efficient%20Inference%20in%20High-Dimensional%2C%20Black-box%20Systems/tables/9ac3c297d2803b8ee6fa161277b96ff1b15857841aacb464204d83662245f814.jpg)

## MS-Diffusion: Multi-subject Zero-shot Image Personalization with Layout Guidance


### Images

![016b97b343c21c686ebd627728cab776fd63063b81338cc4683946a1b70338af.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/016b97b343c21c686ebd627728cab776fd63063b81338cc4683946a1b70338af.jpg)

![06994592d209c235752d7a5548171d79f23adb4dbf36c4790eeb31b2505e5ad8.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/06994592d209c235752d7a5548171d79f23adb4dbf36c4790eeb31b2505e5ad8.jpg)

![0da6907702149b1b4854b41d61e98dff71e23fd34f5dda67f8a0c1e433783175.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/0da6907702149b1b4854b41d61e98dff71e23fd34f5dda67f8a0c1e433783175.jpg)

![1b8043007f08f9d00f90aacf36786297adc3201c5fefdba8c41dfc32ce77fa1f.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/1b8043007f08f9d00f90aacf36786297adc3201c5fefdba8c41dfc32ce77fa1f.jpg)

![39c8a0a4ee435ef98fb44c350f6fb31ca3d24be2571c5fb70bf8764892a7acca.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/39c8a0a4ee435ef98fb44c350f6fb31ca3d24be2571c5fb70bf8764892a7acca.jpg)

![3ce84515c59f500c963fbbc63a0ac48de79d8d7eb56ecb0833b4fe33430ed296.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/3ce84515c59f500c963fbbc63a0ac48de79d8d7eb56ecb0833b4fe33430ed296.jpg)

![4a9814018a76009f6b68cba9e14fd900e33021f0dab102ccd00edb4b0234031b.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/4a9814018a76009f6b68cba9e14fd900e33021f0dab102ccd00edb4b0234031b.jpg)

![4d5821877fa4d2666fecdf79a4636a0afcf8813b47e7724bf4f67b12caf7c811.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/4d5821877fa4d2666fecdf79a4636a0afcf8813b47e7724bf4f67b12caf7c811.jpg)

![4ddaa6a8181f8147f7493baf352c56f4726b5773b173e51b7c7ca8ff33093986.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/4ddaa6a8181f8147f7493baf352c56f4726b5773b173e51b7c7ca8ff33093986.jpg)

![4e5bd4c958de2e217b4f77c6c8657a5b80b849bc9f8ac16c4051eb6f7b7e6661.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/4e5bd4c958de2e217b4f77c6c8657a5b80b849bc9f8ac16c4051eb6f7b7e6661.jpg)

![4e9129ad7176b386c5f72c67c20a808272be66e5972f795bd85d933904827c15.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/4e9129ad7176b386c5f72c67c20a808272be66e5972f795bd85d933904827c15.jpg)

![5378cdac941ff0d64280e03208b5e5304c70e05fc4b69d552f229bffcd56e29d.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/5378cdac941ff0d64280e03208b5e5304c70e05fc4b69d552f229bffcd56e29d.jpg)

![5bb01941e20848a40d7fe2050f203f16061a55403bf93c7f9edb055cb9eff729.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/5bb01941e20848a40d7fe2050f203f16061a55403bf93c7f9edb055cb9eff729.jpg)

![60bb58db5bef43f8e5e3d5c68b8094bd525c8ccbecbe520118719cee3039f8db.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/60bb58db5bef43f8e5e3d5c68b8094bd525c8ccbecbe520118719cee3039f8db.jpg)

![6130f2832887bc536f044784da3971846af962a572ffbeca2e9c72ffb3ed979d.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/6130f2832887bc536f044784da3971846af962a572ffbeca2e9c72ffb3ed979d.jpg)

![652a996c48a6b9556e8d2ca6a704efd4974797ab098828b9208543e5624e5f88.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/652a996c48a6b9556e8d2ca6a704efd4974797ab098828b9208543e5624e5f88.jpg)

![65804ebaf11ecc8b601f5d7018ce3df688e402785a1d29bb91701b1478c6bd09.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/65804ebaf11ecc8b601f5d7018ce3df688e402785a1d29bb91701b1478c6bd09.jpg)

![732edbc69720dc957822e3e6658fd7a2c102978ddca6e564d3dc7aeb5f94935d.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/732edbc69720dc957822e3e6658fd7a2c102978ddca6e564d3dc7aeb5f94935d.jpg)

![818951f1aca022a0c14a591798e9fa1fe704164c29072f5643119d5690994706.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/818951f1aca022a0c14a591798e9fa1fe704164c29072f5643119d5690994706.jpg)

![91e16e8a891b001a48bf6afb4db56f723ecd8e48070abf9668de7dd818717511.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/91e16e8a891b001a48bf6afb4db56f723ecd8e48070abf9668de7dd818717511.jpg)

![956ac43f7e3f8e8ba576157d44ce661d07d97867e81afd2c6e70e3cd69df002d.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/956ac43f7e3f8e8ba576157d44ce661d07d97867e81afd2c6e70e3cd69df002d.jpg)

![99882cb7e7297f28d638ac1a3262a487cab7777940fd790cb17d7975039453be.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/99882cb7e7297f28d638ac1a3262a487cab7777940fd790cb17d7975039453be.jpg)

![9d903b5a0414f80c679bfc5f063bd478e2528c36feeb0d57de51ca1d0baa1ebe.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/9d903b5a0414f80c679bfc5f063bd478e2528c36feeb0d57de51ca1d0baa1ebe.jpg)

![9ec925b11f8495d61fb5cb72dedfed094387951a1e447aff37c4956e190bfa8d.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/9ec925b11f8495d61fb5cb72dedfed094387951a1e447aff37c4956e190bfa8d.jpg)

![a22cc1f32be680cba00c8a9d7628b30c1286b0ee027b93b06f7c8525ca15cd71.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/a22cc1f32be680cba00c8a9d7628b30c1286b0ee027b93b06f7c8525ca15cd71.jpg)

![a3464d50354b99bc01e1bf669c9e3d3f92e2a059763a6853ba5dc386ee4dbd37.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/a3464d50354b99bc01e1bf669c9e3d3f92e2a059763a6853ba5dc386ee4dbd37.jpg)

![accd2bede427d8d717e55c83e21925b7846cf453132a23234d366d0135a1a61f.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/accd2bede427d8d717e55c83e21925b7846cf453132a23234d366d0135a1a61f.jpg)

![af55e55cf6e61a275449cc8663db891e8b34dcb3b642262fdd026e5ce7996b2d.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/af55e55cf6e61a275449cc8663db891e8b34dcb3b642262fdd026e5ce7996b2d.jpg)

![b391c97667a35650273472aba7044a3e02a85e5571c109cc2c8400905838bc60.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/b391c97667a35650273472aba7044a3e02a85e5571c109cc2c8400905838bc60.jpg)

![b453f1d640ffe7abcb9c29b44ca72873a6578a8346d2378d818dbb66959e918b.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/b453f1d640ffe7abcb9c29b44ca72873a6578a8346d2378d818dbb66959e918b.jpg)

![b5961c7402e225b3d3a14942bcab9e1d01f22b2abd3d1c8d6dbb679c20d85abc.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/b5961c7402e225b3d3a14942bcab9e1d01f22b2abd3d1c8d6dbb679c20d85abc.jpg)

![b79aa84473fcb403657b4a1dcc394033a8e753ab34fed7d589cffdb58c9da19b.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/b79aa84473fcb403657b4a1dcc394033a8e753ab34fed7d589cffdb58c9da19b.jpg)

![b980f8f93cd0997aee54a147a9f9ec1484f23255c446a1c628d0b8701655b45a.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/b980f8f93cd0997aee54a147a9f9ec1484f23255c446a1c628d0b8701655b45a.jpg)

![b9893a43701440f82461c5a19621b8bcd77055b4335fa04138300a86b38031a7.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/b9893a43701440f82461c5a19621b8bcd77055b4335fa04138300a86b38031a7.jpg)

![ca3c24eb9dfba9d2f31872ff6d4b5eeeb61f8843960dbe592faecd4a3d86f39a.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/ca3c24eb9dfba9d2f31872ff6d4b5eeeb61f8843960dbe592faecd4a3d86f39a.jpg)

![d2a2f6ca60e7f2fe9141e1839e3c871f359659fc27edee005d18f164fb97c7ba.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/d2a2f6ca60e7f2fe9141e1839e3c871f359659fc27edee005d18f164fb97c7ba.jpg)

![d2f4e862a98c28d8b70dafcfaf8ca10d46b572a923fa5d34fc33e1373cb0b1d7.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/d2f4e862a98c28d8b70dafcfaf8ca10d46b572a923fa5d34fc33e1373cb0b1d7.jpg)

![d3d999ae06288745592463e6b5f3adb5b899b76e59d28afbf90506d8030bfb5e.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/d3d999ae06288745592463e6b5f3adb5b899b76e59d28afbf90506d8030bfb5e.jpg)

![d5ed83dc692c0dbfd2ed796f7da3becb3780d08e4dcdd4e6ca318744fd062550.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/d5ed83dc692c0dbfd2ed796f7da3becb3780d08e4dcdd4e6ca318744fd062550.jpg)

![d634c82649048f43b0efb59a105c502d12df956004242b96659664566e11b6fb.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/d634c82649048f43b0efb59a105c502d12df956004242b96659664566e11b6fb.jpg)

![dd3f43b04093b841cd80e1f8da9d31327a8748d13533e3e5b8899e2341bae2d6.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/dd3f43b04093b841cd80e1f8da9d31327a8748d13533e3e5b8899e2341bae2d6.jpg)

![e65d1c861a5c27963887d60205488b576d8d1d7156239e4e11594d4d46fd12af.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/e65d1c861a5c27963887d60205488b576d8d1d7156239e4e11594d4d46fd12af.jpg)

![ea7b9fce13e8af28651dca1d2b654204da2a7c0ded419a410222c5bb13acf6df.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/ea7b9fce13e8af28651dca1d2b654204da2a7c0ded419a410222c5bb13acf6df.jpg)

![ee8376c31f3b60e6c2cac30c20ef05e589e0ff2c9c9475cf88ab4ebe201849af.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/ee8376c31f3b60e6c2cac30c20ef05e589e0ff2c9c9475cf88ab4ebe201849af.jpg)

![f6592119dedb2ac9e7c519a1552037607de3bc9ad269383c0a80f2fbb5615ac7.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/f6592119dedb2ac9e7c519a1552037607de3bc9ad269383c0a80f2fbb5615ac7.jpg)

![f8744597cde7a5848193ad3e94f10f7109d230fd8fec4f139d35495cca280b84.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/images/f8744597cde7a5848193ad3e94f10f7109d230fd8fec4f139d35495cca280b84.jpg)

### Tables

![44ae0421cb3407621ac9d1a31241ec9a7bcffec93fad866bc274bdfe3b298724.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/tables/44ae0421cb3407621ac9d1a31241ec9a7bcffec93fad866bc274bdfe3b298724.jpg)

![8b040ec94ffdf78c2127c573e9b19b769ca47b48159dda84828cf295ef39627c.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/tables/8b040ec94ffdf78c2127c573e9b19b769ca47b48159dda84828cf295ef39627c.jpg)

![eeb613c8f8dca32950638907b26cab3f00a7946ff55a85a38007fc6430120d0b.jpg](../iclr_results/2400_MS-Diffusion_%20Multi-subject%20Zero-shot%20Image%20Personalization%20with%20Layout%20Guidance/tables/eeb613c8f8dca32950638907b26cab3f00a7946ff55a85a38007fc6430120d0b.jpg)

## Generalization and Distributed Learning of GFlowNets


### Images

![02dfac1cb346dbc1bea2f3a70a14c03212dc6a6d7d0254345deed99bb32e6dad.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/02dfac1cb346dbc1bea2f3a70a14c03212dc6a6d7d0254345deed99bb32e6dad.jpg)

![0d6db0c5014f22b26e6e78450751e89b37362b73c54acde0864b680d97a8a62f.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/0d6db0c5014f22b26e6e78450751e89b37362b73c54acde0864b680d97a8a62f.jpg)

![192a0f14f927c04c69eb457c86bb4d00ad44f1703c13ce105dbb4246d462c2cd.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/192a0f14f927c04c69eb457c86bb4d00ad44f1703c13ce105dbb4246d462c2cd.jpg)

![2b3c02398b85e355136a605401798eb36c27fdf811187a0dd1fc96c2c5b1765a.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/2b3c02398b85e355136a605401798eb36c27fdf811187a0dd1fc96c2c5b1765a.jpg)

![35bc6c6c6ae79a8fe16c9258e95a9c5147c48f2253a273a2a2fe26c900276116.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/35bc6c6c6ae79a8fe16c9258e95a9c5147c48f2253a273a2a2fe26c900276116.jpg)

![39474349b45f2975c84ab375efe8ad3ac5578ee427db5ae6a6c5a4cf6d300abf.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/39474349b45f2975c84ab375efe8ad3ac5578ee427db5ae6a6c5a4cf6d300abf.jpg)

![54699d9d55eac93bf5d11f072b1cff57eaa09a90876a2ed4da9882adb159d163.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/54699d9d55eac93bf5d11f072b1cff57eaa09a90876a2ed4da9882adb159d163.jpg)

![5fb9199cf37f03b5eded0d6374ef1daaa1325348bc1c3f73f003db68abf5e086.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/5fb9199cf37f03b5eded0d6374ef1daaa1325348bc1c3f73f003db68abf5e086.jpg)

![62a4e7ba4d85dc3edf52efd79f48b2b6418b860be4b69c096deaa568a3287b5f.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/62a4e7ba4d85dc3edf52efd79f48b2b6418b860be4b69c096deaa568a3287b5f.jpg)

![664adff714f80dc672c9c73ddf0181695888e4e990e55da4f707d48b16ef6fec.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/664adff714f80dc672c9c73ddf0181695888e4e990e55da4f707d48b16ef6fec.jpg)

![672c42b6bb039a1ff7f53334218afc5993d4a030223be6849d6489bcb02033e3.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/672c42b6bb039a1ff7f53334218afc5993d4a030223be6849d6489bcb02033e3.jpg)

![6bf2aaeece7a22e3e035a09547ea30b3b98db9fbae44594c0f3ea58d40bbfbc6.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/6bf2aaeece7a22e3e035a09547ea30b3b98db9fbae44594c0f3ea58d40bbfbc6.jpg)

![ae953694656aa1cf00c1d8e1f834b5c2bd6a7a4dda9d2109cd5e30a10b9af665.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/ae953694656aa1cf00c1d8e1f834b5c2bd6a7a4dda9d2109cd5e30a10b9af665.jpg)

![c9ab40c85d7b78e7660fb6e58ef0f6b0696ea671fafe25450588396139755525.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/c9ab40c85d7b78e7660fb6e58ef0f6b0696ea671fafe25450588396139755525.jpg)

![cf5d1cf0aab6323a67e394545ae8bc3c56ba02371244212c7378ddadc3e8b789.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/cf5d1cf0aab6323a67e394545ae8bc3c56ba02371244212c7378ddadc3e8b789.jpg)

![d53b1ff2d2a33d7b6ab761ea9399fbae2b51994198abdb58ffe1a212e1e1120d.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/d53b1ff2d2a33d7b6ab761ea9399fbae2b51994198abdb58ffe1a212e1e1120d.jpg)

![f023cd3156d9cf6ff57e90d4a5cbfd353acf9ff5cb5aa9cec4a791fbd4216e96.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/images/f023cd3156d9cf6ff57e90d4a5cbfd353acf9ff5cb5aa9cec4a791fbd4216e96.jpg)

### Tables

![7520c5cccb3636f272569c9070be2882f3825bc2e4fb4a90fa0c665af7d90d3d.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/tables/7520c5cccb3636f272569c9070be2882f3825bc2e4fb4a90fa0c665af7d90d3d.jpg)

![8e7daf566f73ce48e5e84c5eee0bc39b0be3df70803c760245901a323fd4063e.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/tables/8e7daf566f73ce48e5e84c5eee0bc39b0be3df70803c760245901a323fd4063e.jpg)

![ddc7bb7d43fdca47a74d8d3c61602ff619dab0c7bd87b65c08161c990d55e94d.jpg](../iclr_results/2401_Generalization%20and%20Distributed%20Learning%20of%20GFlowNets/tables/ddc7bb7d43fdca47a74d8d3c61602ff619dab0c7bd87b65c08161c990d55e94d.jpg)

## Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors


### Images

![0ca4b4eab4e4042e413b6f789d2ac3714e4c347bdf365d529ed6df136dfa5919.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/images/0ca4b4eab4e4042e413b6f789d2ac3714e4c347bdf365d529ed6df136dfa5919.jpg)

![510bf75bc61158afb93524c434cfaed7d4ef968dec86dc6f7f45c841414ab954.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/images/510bf75bc61158afb93524c434cfaed7d4ef968dec86dc6f7f45c841414ab954.jpg)

![71c030a3c30c4549041874b08c03c8174617692b9e1dce08b0da084218c78f31.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/images/71c030a3c30c4549041874b08c03c8174617692b9e1dce08b0da084218c78f31.jpg)

![b5a1b096dd4c6e28c4ccf7b26d89865b74dd9c4ad46468ca09fc447ac139f28e.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/images/b5a1b096dd4c6e28c4ccf7b26d89865b74dd9c4ad46468ca09fc447ac139f28e.jpg)

![d0da7e5bb5458c6ff7981a98e5be01c78159c1674ab6e7bdfdee45e41edfcac2.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/images/d0da7e5bb5458c6ff7981a98e5be01c78159c1674ab6e7bdfdee45e41edfcac2.jpg)

![f09f51ed586e0e4c9ec9b8c81e278a51865bc6f6443bdff3d2f51f74ec1589a5.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/images/f09f51ed586e0e4c9ec9b8c81e278a51865bc6f6443bdff3d2f51f74ec1589a5.jpg)

### Tables

![110c6974ae9583a4aae57db440ab4d9b03f6850d467afbecfb3947a3976305c1.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/110c6974ae9583a4aae57db440ab4d9b03f6850d467afbecfb3947a3976305c1.jpg)

![1d35274fe9acdb1b5707037451826168ef9ae0f67826c510ac4e9b593c69589f.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/1d35274fe9acdb1b5707037451826168ef9ae0f67826c510ac4e9b593c69589f.jpg)

![38df9ca21efaf13b2ab636a3b4921ac11308cfe8bce89167774362c9edfd26da.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/38df9ca21efaf13b2ab636a3b4921ac11308cfe8bce89167774362c9edfd26da.jpg)

![5503e3a51497e06e189ce956fda7920bd2ac0e912f0f93f503fb749e2418835f.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/5503e3a51497e06e189ce956fda7920bd2ac0e912f0f93f503fb749e2418835f.jpg)

![6294d1818757bb9a0e26e4f2b049003a9dc027bc43844f5f2a693989ea556033.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/6294d1818757bb9a0e26e4f2b049003a9dc027bc43844f5f2a693989ea556033.jpg)

![67daa9f22c778fa602f2e33833e4930a624fcc1700b8aea09fe5ebcf95d99c4c.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/67daa9f22c778fa602f2e33833e4930a624fcc1700b8aea09fe5ebcf95d99c4c.jpg)

![798157cc76dbc4b8ad48c5b0634e2530886399ae658ce03609d0e79a785656ec.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/798157cc76dbc4b8ad48c5b0634e2530886399ae658ce03609d0e79a785656ec.jpg)

![9024125ac4aa04e553219feed3b49deb20c851d560639ed07188a330e28effa1.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/9024125ac4aa04e553219feed3b49deb20c851d560639ed07188a330e28effa1.jpg)

![9a2881c4d8deca3f46271160fe0393cc6c9d23e930609a308bf2e87f88100adc.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/9a2881c4d8deca3f46271160fe0393cc6c9d23e930609a308bf2e87f88100adc.jpg)

![a0470aef132998f71dbb621b4ccd8a7b3c6cd6d81470740f44526bdccc49cf9a.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/a0470aef132998f71dbb621b4ccd8a7b3c6cd6d81470740f44526bdccc49cf9a.jpg)

![aade186a71a71291c0f1c85ade6582543a8604ded4ba86d764a164e8b1deb6ac.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/aade186a71a71291c0f1c85ade6582543a8604ded4ba86d764a164e8b1deb6ac.jpg)

![c941b0e98347a1070c2583d512e865f325599dc5af41612dfe88ec6b78765c73.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/c941b0e98347a1070c2583d512e865f325599dc5af41612dfe88ec6b78765c73.jpg)

![d3e1e2782ed672e59748bd5a84fa1501ebed67aca98a0fe1fe7091fce06a3d1a.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/d3e1e2782ed672e59748bd5a84fa1501ebed67aca98a0fe1fe7091fce06a3d1a.jpg)

![de7e4a510ccd4ad4c5d5110aeac087a8f61c1b836773bc952dac3402f7759b4d.jpg](../iclr_results/2402_Humanizing%20the%20Machine_%20Proxy%20Attacks%20to%20Mislead%20LLM%20Detectors/tables/de7e4a510ccd4ad4c5d5110aeac087a8f61c1b836773bc952dac3402f7759b4d.jpg)

## Unlocking Efficient, Scalable, and Continual Knowledge Editing with Basis-Level Representation Fine-Tuning


### Images

![00a2b951b66875e164707e1bd43e1d711172db204b3b24dbce2d535858f2abc6.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/images/00a2b951b66875e164707e1bd43e1d711172db204b3b24dbce2d535858f2abc6.jpg)

![470db7b21a6b8eb7bdfffd0cd6412b3d6981d86145970da46dc7bea2a857fc92.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/images/470db7b21a6b8eb7bdfffd0cd6412b3d6981d86145970da46dc7bea2a857fc92.jpg)

![4a987cf37a221dabdc2aa278959faefca257c7e9db18c26866e0780c84f54239.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/images/4a987cf37a221dabdc2aa278959faefca257c7e9db18c26866e0780c84f54239.jpg)

![60c95e208ea05a10812a8150eb744f2c3dbd7f8581ea49270e3f8942f28efeaa.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/images/60c95e208ea05a10812a8150eb744f2c3dbd7f8581ea49270e3f8942f28efeaa.jpg)

### Tables

![16fa2ba0f4b6e2fbca2fc2f93f14f53221047e607cb2739e91174c3c96ea3422.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/16fa2ba0f4b6e2fbca2fc2f93f14f53221047e607cb2739e91174c3c96ea3422.jpg)

![757e8530e946cfa7bd0dee83bb052ad16d6e6d3289e964feb6e047c30bc02bfb.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/757e8530e946cfa7bd0dee83bb052ad16d6e6d3289e964feb6e047c30bc02bfb.jpg)

![7912ea35a7751aaa67c910922e2178b05f038e9b4be1045ea9b2e361cf5f5052.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/7912ea35a7751aaa67c910922e2178b05f038e9b4be1045ea9b2e361cf5f5052.jpg)

![a57dd77c087a0a3a9a664756d4cfc162574c5396a00d0a72a0a6ea7f98dfd5e3.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/a57dd77c087a0a3a9a664756d4cfc162574c5396a00d0a72a0a6ea7f98dfd5e3.jpg)

![b5baf55402a0b6fafd35cd7ecc5c148d1c2b9778c7dd0599cbcd22ec91727e53.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/b5baf55402a0b6fafd35cd7ecc5c148d1c2b9778c7dd0599cbcd22ec91727e53.jpg)

![c0159b63400bb9542bef6fb386014ce9726c7b3cbb27ef4a5b30ee1ff91a2aea.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/c0159b63400bb9542bef6fb386014ce9726c7b3cbb27ef4a5b30ee1ff91a2aea.jpg)

![c6454f1062c54258d183c72135a51f174c68043717662be96b03acaac5a436a2.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/c6454f1062c54258d183c72135a51f174c68043717662be96b03acaac5a436a2.jpg)

![cac1ea8ef2392a18f65f99c87628bee0f95c14af441ec2d7f2307015ac3e58d2.jpg](../iclr_results/2403_Unlocking%20Efficient%2C%20Scalable%2C%20and%20Continual%20Knowledge%20Editing%20with%20Basis-Level%20Representation%20Fine-/tables/cac1ea8ef2392a18f65f99c87628bee0f95c14af441ec2d7f2307015ac3e58d2.jpg)

## RTop-K: Ultra-Fast Row-Wise Top-K Selection for Neural Network Acceleration on GPUs


### Images

![046ecac93bbfe79671d2998353291c5bc9315fbdf4bff2f477084a5d64481640.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/images/046ecac93bbfe79671d2998353291c5bc9315fbdf4bff2f477084a5d64481640.jpg)

![2d61de8479fc9e291a02aede4d15e956dd92c730073bb7c88da757f33e65d793.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/images/2d61de8479fc9e291a02aede4d15e956dd92c730073bb7c88da757f33e65d793.jpg)

![3cdad48a610d5559d3a778163f75cd0374892176b479100141c5a876f371d8e8.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/images/3cdad48a610d5559d3a778163f75cd0374892176b479100141c5a876f371d8e8.jpg)

![82d75ed8c83d41073badbe74767efba461ff00ef7f24bc128591c04e3eab7c3a.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/images/82d75ed8c83d41073badbe74767efba461ff00ef7f24bc128591c04e3eab7c3a.jpg)

![88749cae441385a49537d33e2b87e1b8c8ed57d6678fc9a3330c432c05ba8597.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/images/88749cae441385a49537d33e2b87e1b8c8ed57d6678fc9a3330c432c05ba8597.jpg)

![b5bd92bdb1f0cfe4b1c8400666ba75ccb3d6db4cf329dde400ee650e37b6a1ce.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/images/b5bd92bdb1f0cfe4b1c8400666ba75ccb3d6db4cf329dde400ee650e37b6a1ce.jpg)

![c7ab1a97b5338b739653568e72de4bf5586c7f887da3d937f7f611a3e023c8fa.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/images/c7ab1a97b5338b739653568e72de4bf5586c7f887da3d937f7f611a3e023c8fa.jpg)

### Tables

![17d420b198b8c7b1edc56134383d275ce03ac5103ee1bc4bc01ef25ae87e013c.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/tables/17d420b198b8c7b1edc56134383d275ce03ac5103ee1bc4bc01ef25ae87e013c.jpg)

![270743a990aeda2f3c316c244b6ed0b27453992b44ae04b0925cbec415bc27b9.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/tables/270743a990aeda2f3c316c244b6ed0b27453992b44ae04b0925cbec415bc27b9.jpg)

![2e082a355d065a4650b2bbcd4c7114db347f30bd87e9ee877384952e5a377750.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/tables/2e082a355d065a4650b2bbcd4c7114db347f30bd87e9ee877384952e5a377750.jpg)

![a51d6fc1cdffd99f931017d538d4429853f5a5726ec818e33f875b4448edbb62.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/tables/a51d6fc1cdffd99f931017d538d4429853f5a5726ec818e33f875b4448edbb62.jpg)

![ada5bb607627482da2e6848fcb498d123373271cd6ec81de3b0b7904463e0f2d.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/tables/ada5bb607627482da2e6848fcb498d123373271cd6ec81de3b0b7904463e0f2d.jpg)

![d85eab28cb844950530cb7b76320ad3f5c41c956cc6a5b8ea6e1fb39106cfbeb.jpg](../iclr_results/2404_RTop-K_%20Ultra-Fast%20Row-Wise%20Top-K%20Selection%20for%20Neural%20Network%20Acceleration%20on%20GPUs/tables/d85eab28cb844950530cb7b76320ad3f5c41c956cc6a5b8ea6e1fb39106cfbeb.jpg)

## Learning Robust Representations with Long-Term Information for Generalization in Visual Reinforcement Learning


### Images

![062962abd6db217e731a0e201d3b7b01f72496149a94aebe4cf5ff59bc0d3508.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/062962abd6db217e731a0e201d3b7b01f72496149a94aebe4cf5ff59bc0d3508.jpg)

![189006788a3cbfa11f888c84af1eb906beb3cf5cbc5ad72e6d969b2d3849a3f2.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/189006788a3cbfa11f888c84af1eb906beb3cf5cbc5ad72e6d969b2d3849a3f2.jpg)

![207b531a995f19f3b6d9382afe8d06c61d3197c4a4178c515d59f843b3ffda1b.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/207b531a995f19f3b6d9382afe8d06c61d3197c4a4178c515d59f843b3ffda1b.jpg)

![21d27815fbf1ec82d9395ca5b4f31ade3dc20cf33f2f1c54851e40b5b1485b56.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/21d27815fbf1ec82d9395ca5b4f31ade3dc20cf33f2f1c54851e40b5b1485b56.jpg)

![3edee6cc52e99eaa3c631940a3158dac0f7f621110b75178040f1b207e43b19f.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/3edee6cc52e99eaa3c631940a3158dac0f7f621110b75178040f1b207e43b19f.jpg)

![a4982f3d5a287ebc6b18aed872334b8f488ecf9abf0f82bcf065f45cd30a5f4c.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/a4982f3d5a287ebc6b18aed872334b8f488ecf9abf0f82bcf065f45cd30a5f4c.jpg)

![b2d4d7ca72e91463a258f551c2d6934ebf4c1fdb373b886cdc16163cf7f35026.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/b2d4d7ca72e91463a258f551c2d6934ebf4c1fdb373b886cdc16163cf7f35026.jpg)

![b8c22928ad8b72a2f5c62a26e5562ee86f5df9cebc43c04782fc6f5aa72ac77e.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/b8c22928ad8b72a2f5c62a26e5562ee86f5df9cebc43c04782fc6f5aa72ac77e.jpg)

![e56a9e32922c16b6209c7f4ddd7a8a2b3707e9edabfcb489692e105588ed3dc0.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/images/e56a9e32922c16b6209c7f4ddd7a8a2b3707e9edabfcb489692e105588ed3dc0.jpg)

### Tables

![17c8ff3d9cf0c17e1665eb6c287139fecf3be6cb22ae12e1ae8c3d27a980946e.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/17c8ff3d9cf0c17e1665eb6c287139fecf3be6cb22ae12e1ae8c3d27a980946e.jpg)

![767122ce764bc10c734e100944ddc47c87940a6ff574140cbac4e72a97d90ce5.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/767122ce764bc10c734e100944ddc47c87940a6ff574140cbac4e72a97d90ce5.jpg)

![77de6d7bab23efb8b768191a1471a841d3fb3e4aebbeac6c044d2193e0970fa3.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/77de6d7bab23efb8b768191a1471a841d3fb3e4aebbeac6c044d2193e0970fa3.jpg)

![7b5d078cded8985c2991dfde5df8e6e45bcc628ebaea49be9bd867a24a6623a9.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/7b5d078cded8985c2991dfde5df8e6e45bcc628ebaea49be9bd867a24a6623a9.jpg)

![9e225dbd96bb0a683b3171fc7ac1e9996da8350781bd2c51914036adbdddb40f.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/9e225dbd96bb0a683b3171fc7ac1e9996da8350781bd2c51914036adbdddb40f.jpg)

![a1f618e2b750dbe749ca8406e7f26e14cdc86553aca164d6cf56b1ddf7f3ea9f.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/a1f618e2b750dbe749ca8406e7f26e14cdc86553aca164d6cf56b1ddf7f3ea9f.jpg)

![a574228a4f86da6ec16c1b4a9f1c5ec0947cdd66161b1ef6e6ca3fa5b7b59b75.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/a574228a4f86da6ec16c1b4a9f1c5ec0947cdd66161b1ef6e6ca3fa5b7b59b75.jpg)

![d030f52ac496f84c275ef58113475f4bc8726ed7deaf3f374614d2b0003c59cb.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/d030f52ac496f84c275ef58113475f4bc8726ed7deaf3f374614d2b0003c59cb.jpg)

![d7e1892e9e40710963b52517344ffeaf525e3452ae5de56c5bca47626c18eefa.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/d7e1892e9e40710963b52517344ffeaf525e3452ae5de56c5bca47626c18eefa.jpg)

![e25ea7ea410ce4503d944cc6d40afcda6d747aed32d9a688552edede455823fb.jpg](../iclr_results/2405_Learning%20Robust%20Representations%20with%20Long-Term%20Information%20for%20Generalization%20in%20Visual%20Reinforcemen/tables/e25ea7ea410ce4503d944cc6d40afcda6d747aed32d9a688552edede455823fb.jpg)

## Magnetic Preference Optimization: Achieving Last-iterate Convergence for Language Model Alignment


### Images

![07b92731a76a7f3f503d72bdf6bc631eb6ca682618b52f2a98deee3827cab618.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/07b92731a76a7f3f503d72bdf6bc631eb6ca682618b52f2a98deee3827cab618.jpg)

![1360b854be1f7dd0716c83795c013932d6a2255c1144ab58783dfeb2dbf3dcc5.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/1360b854be1f7dd0716c83795c013932d6a2255c1144ab58783dfeb2dbf3dcc5.jpg)

![1ea4673eb9632386b1a07397a1cc45a410fb9c6bd0234a080f650809faa929c7.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/1ea4673eb9632386b1a07397a1cc45a410fb9c6bd0234a080f650809faa929c7.jpg)

![29e3b347511f91e79c27095bbbccd401412c9c28ae3946f7fd49a9d9d9a442a5.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/29e3b347511f91e79c27095bbbccd401412c9c28ae3946f7fd49a9d9d9a442a5.jpg)

![4651e02bf0238bdaad3fd5aebfc2aee5edccc8f7926ecea2cf0ed03719817514.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/4651e02bf0238bdaad3fd5aebfc2aee5edccc8f7926ecea2cf0ed03719817514.jpg)

![49e8631ab1434cf7ec9f01c9532e9d6cce6124666ef998cff7a5d39be99f4fe3.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/49e8631ab1434cf7ec9f01c9532e9d6cce6124666ef998cff7a5d39be99f4fe3.jpg)

![8ab17a6e7fb2e095692575562a98194d890bc56791ee643a31347049a112d12c.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/8ab17a6e7fb2e095692575562a98194d890bc56791ee643a31347049a112d12c.jpg)

![8d1e531eeba39eb5e17d52af398ee6029fe9ac50b43e4ce7f9cff5132a93cbea.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/8d1e531eeba39eb5e17d52af398ee6029fe9ac50b43e4ce7f9cff5132a93cbea.jpg)

![919fdca0100185dd11b8fe44c45a571a30724679a5e43fa240bdff41e1a3bb91.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/919fdca0100185dd11b8fe44c45a571a30724679a5e43fa240bdff41e1a3bb91.jpg)

![9904093a5df1e00b641c905adaf0f7214ab7e70225e92567591e56be7411f37b.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/9904093a5df1e00b641c905adaf0f7214ab7e70225e92567591e56be7411f37b.jpg)

![c9aa63b8f28487a69ffb3f79e08238549e3ad508048f2c2bd5ff8193ff6fed3b.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/c9aa63b8f28487a69ffb3f79e08238549e3ad508048f2c2bd5ff8193ff6fed3b.jpg)

![faa06529e0c523129300f1308ff449f3292a2dd8a36f3f47ed6032da33ccfebc.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/images/faa06529e0c523129300f1308ff449f3292a2dd8a36f3f47ed6032da33ccfebc.jpg)

### Tables

![195b87707746820039ae47cd615a46b524753eb3439a455976df94895ef32809.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/195b87707746820039ae47cd615a46b524753eb3439a455976df94895ef32809.jpg)

![2f4f7dc0f91b111847984d6c64106df4473cf8d6be49491906699a23bf59d268.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/2f4f7dc0f91b111847984d6c64106df4473cf8d6be49491906699a23bf59d268.jpg)

![516e57bab4af2fbe159ffc16c8b4c471189dcdc5ba738aa0680730b71663cfef.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/516e57bab4af2fbe159ffc16c8b4c471189dcdc5ba738aa0680730b71663cfef.jpg)

![56770d1b56f6d8924674f64c63c0360c8ee4ada5ecaea408348bffebe4c76a5e.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/56770d1b56f6d8924674f64c63c0360c8ee4ada5ecaea408348bffebe4c76a5e.jpg)

![5dcfba28e1e9e161e10dbcc53640ed00130dd928ac6989eae953a7a8db8cbc52.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/5dcfba28e1e9e161e10dbcc53640ed00130dd928ac6989eae953a7a8db8cbc52.jpg)

![9615f916f979300b78eda7d74a25b2dd127a252c07d6fd48d3a6619457d1380b.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/9615f916f979300b78eda7d74a25b2dd127a252c07d6fd48d3a6619457d1380b.jpg)

![a0ca7c27799c46d14b36f78abb5472f75a0638a4f39ff055bf262f9a587c029c.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/a0ca7c27799c46d14b36f78abb5472f75a0638a4f39ff055bf262f9a587c029c.jpg)

![baf09a7cb595c0dff42a5a799d4313956c9cc5080dfd3dc22bba009f8f8e27fd.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/baf09a7cb595c0dff42a5a799d4313956c9cc5080dfd3dc22bba009f8f8e27fd.jpg)

![cad3e28c76c037be59b5901de47d5d5eca65d462d3f6ece105262e3e3c2f81fa.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/cad3e28c76c037be59b5901de47d5d5eca65d462d3f6ece105262e3e3c2f81fa.jpg)

![cb010fedf0d449034ccfb220aaa22965a4a792c9890c9fe61ec7db54944f5802.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/cb010fedf0d449034ccfb220aaa22965a4a792c9890c9fe61ec7db54944f5802.jpg)

![d84b69836f696c3f3be32b4cd975ab577f3418dc52a120219dd74968c3aca0d6.jpg](../iclr_results/2406_Magnetic%20Preference%20Optimization_%20Achieving%20Last-iterate%20Convergence%20for%20Language%20Model%20Alignment/tables/d84b69836f696c3f3be32b4cd975ab577f3418dc52a120219dd74968c3aca0d6.jpg)

## Select before Act: Spatially Decoupled Action Repetition for Continuous Control


### Images

![0e57b81d0c7eda0d21a03b0eb04b49d004664faa0d6e582cf7e4f081907cb03b.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/images/0e57b81d0c7eda0d21a03b0eb04b49d004664faa0d6e582cf7e4f081907cb03b.jpg)

![7034f52a319444fd9da6a6365367d8ca6c63aebb26dd03d3f13d6c1f07af73ee.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/images/7034f52a319444fd9da6a6365367d8ca6c63aebb26dd03d3f13d6c1f07af73ee.jpg)

![838764ba7f34d713beb465af51ecd53b18dab2a0e66ae26eed6b78d559cb3d0e.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/images/838764ba7f34d713beb465af51ecd53b18dab2a0e66ae26eed6b78d559cb3d0e.jpg)

![a8621f64718027e1cb174491af28b5494707da19dcb618eea2ab035c3d8af1c9.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/images/a8621f64718027e1cb174491af28b5494707da19dcb618eea2ab035c3d8af1c9.jpg)

![b7f55ce46ea656927a2e42250056ff1f4cc2f314139e8169725419b854b072ad.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/images/b7f55ce46ea656927a2e42250056ff1f4cc2f314139e8169725419b854b072ad.jpg)

### Tables

![5cf19a42945ff587a06009c2f6ffc04c9b7bc8d329e1ea0749458de9e2d11fb0.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/tables/5cf19a42945ff587a06009c2f6ffc04c9b7bc8d329e1ea0749458de9e2d11fb0.jpg)

![80b8455f013287c77884b2b4ffaaeaf9057f1f545f7461278807d1c191a17511.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/tables/80b8455f013287c77884b2b4ffaaeaf9057f1f545f7461278807d1c191a17511.jpg)

![c56512152c35f1ecb1c60456b77dbbbb18789610645f67bb1907b0c9ac7d84b2.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/tables/c56512152c35f1ecb1c60456b77dbbbb18789610645f67bb1907b0c9ac7d84b2.jpg)

![deab32dfef741fae66ac989d5f447be29f7f09045f7d8c05f2c51de678601bec.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/tables/deab32dfef741fae66ac989d5f447be29f7f09045f7d8c05f2c51de678601bec.jpg)

![e603577312cf31c63431a6b0f06a8bd65e136fa9f3aa07246ec290898116c61a.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/tables/e603577312cf31c63431a6b0f06a8bd65e136fa9f3aa07246ec290898116c61a.jpg)

![e880af9d5fa58ff03e8891b411990e4d3e4c0b3c0d881bab4e4ab1a04e14fc69.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/tables/e880af9d5fa58ff03e8891b411990e4d3e4c0b3c0d881bab4e4ab1a04e14fc69.jpg)

![eeee7b0e1370d1ed89329caf298c871e262b3dd28047e057e4772285ca5488c0.jpg](../iclr_results/2407_Select%20before%20Act_%20Spatially%20Decoupled%20Action%20Repetition%20for%20Continuous%20Control/tables/eeee7b0e1370d1ed89329caf298c871e262b3dd28047e057e4772285ca5488c0.jpg)

## Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models


### Images

![136088f38eb68321f7d04c551673d450b445b8dd3cf7d6bf9e667b0b6d8c1165.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/136088f38eb68321f7d04c551673d450b445b8dd3cf7d6bf9e667b0b6d8c1165.jpg)

![1a009f35d495595cb1045ac81b983aafe699e551fb72c5de8c0fab97f1aca9bb.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/1a009f35d495595cb1045ac81b983aafe699e551fb72c5de8c0fab97f1aca9bb.jpg)

![1d0e6dee9d322fe26f369b1984f51a2d88b01110a113bb34162e3781b31a6c6f.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/1d0e6dee9d322fe26f369b1984f51a2d88b01110a113bb34162e3781b31a6c6f.jpg)

![2412c0daa8bfd8a17cc13f962e9894e30349fe0f96f9dddaf1398479726bc39a.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/2412c0daa8bfd8a17cc13f962e9894e30349fe0f96f9dddaf1398479726bc39a.jpg)

![31ebcf862d5dae34642512329665976d026e82ee1d991f28fe44c3a5d450f522.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/31ebcf862d5dae34642512329665976d026e82ee1d991f28fe44c3a5d450f522.jpg)

![35a799bb4a50c25806509297b951118d285b139af2a0742423c194ee9f45b5eb.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/35a799bb4a50c25806509297b951118d285b139af2a0742423c194ee9f45b5eb.jpg)

![412f8110ae9ce36cf4eebec28823b4e854859183259cc72fde99296ba519ed93.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/412f8110ae9ce36cf4eebec28823b4e854859183259cc72fde99296ba519ed93.jpg)

![4456e8e9e29666b7f617f6e37ea3d4a4df1a48900baaf83f8ecfd13a3510b081.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/4456e8e9e29666b7f617f6e37ea3d4a4df1a48900baaf83f8ecfd13a3510b081.jpg)

![4dacf8236ab19b7195aa483dbb879a968e1a88e7c21fa7f2b64986f31507b611.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/4dacf8236ab19b7195aa483dbb879a968e1a88e7c21fa7f2b64986f31507b611.jpg)

![4e703ea8a7619d214019d4a75b3b0ac633912819a1e33c98a58f6a5e761ee438.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/4e703ea8a7619d214019d4a75b3b0ac633912819a1e33c98a58f6a5e761ee438.jpg)

![5641312feb9691b888735bdce27ba2ee1a6cfbca342b9ecc06dba57db63e0dd8.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/5641312feb9691b888735bdce27ba2ee1a6cfbca342b9ecc06dba57db63e0dd8.jpg)

![7e1d041dd26b7ac4e5926a1b64dc0d4af1c2ebff37239f6e1deda9ec502d0984.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/7e1d041dd26b7ac4e5926a1b64dc0d4af1c2ebff37239f6e1deda9ec502d0984.jpg)

![886fba4133bdfac361ac9ec797da5ef07fbdf1ef2a52e46a04e3249206be54b2.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/886fba4133bdfac361ac9ec797da5ef07fbdf1ef2a52e46a04e3249206be54b2.jpg)

![993857a4842e64d8a85cf9ca2ad3806b0e6c2b48f06b4d035b8beb5d886062ef.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/993857a4842e64d8a85cf9ca2ad3806b0e6c2b48f06b4d035b8beb5d886062ef.jpg)

![a2d07da4c0b99137deb332bfb8ae9e2def663da34b3c14fb37d48ae69ac840a4.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/a2d07da4c0b99137deb332bfb8ae9e2def663da34b3c14fb37d48ae69ac840a4.jpg)

![b58b07aa3ee3963a5944482842bc0c8b3deb0b42af2e8b39fed4533af283514d.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/b58b07aa3ee3963a5944482842bc0c8b3deb0b42af2e8b39fed4533af283514d.jpg)

![b5949271593ade332b1ccd351c66fe71b9b38601d0df5a553b60d505bbf7d119.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/b5949271593ade332b1ccd351c66fe71b9b38601d0df5a553b60d505bbf7d119.jpg)

![cdf79fece413a396957956107f4d17bfdec0aa15c6236646867ea2e5d382a614.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/cdf79fece413a396957956107f4d17bfdec0aa15c6236646867ea2e5d382a614.jpg)

![d69a7c1c28ab4f2b55b15a2a54f83cd510fa8c61292b0dd89f18129f05a310a8.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/d69a7c1c28ab4f2b55b15a2a54f83cd510fa8c61292b0dd89f18129f05a310a8.jpg)

![e5cce827f667c069e329fd106759c1aeef3ff6dbc36cd146db2bc581b089257b.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/e5cce827f667c069e329fd106759c1aeef3ff6dbc36cd146db2bc581b089257b.jpg)

![e983be47bdb324be257743b4bd8ab7f595198f90f357c84ad5d85769a8108217.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/images/e983be47bdb324be257743b4bd8ab7f595198f90f357c84ad5d85769a8108217.jpg)

### Tables

![1ac5f27e19c9ec94e336089d2d1ebfdc1ff4dd1ecd10046124f44492cf174782.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/tables/1ac5f27e19c9ec94e336089d2d1ebfdc1ff4dd1ecd10046124f44492cf174782.jpg)

![3abc19c83a1ee32952121bab0e418bc8700bab618009856bf0be53e42f151777.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/tables/3abc19c83a1ee32952121bab0e418bc8700bab618009856bf0be53e42f151777.jpg)

![67f44d1bd37553c095198db744e8cf85f77a8bab63fb4260a9efaa66b1b84238.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/tables/67f44d1bd37553c095198db744e8cf85f77a8bab63fb4260a9efaa66b1b84238.jpg)

![897b47f14df5d1b08b5cf13586945cbdf3392485a9da60fe0cd1f56fbcb92dee.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/tables/897b47f14df5d1b08b5cf13586945cbdf3392485a9da60fe0cd1f56fbcb92dee.jpg)

![cfaba9e82b57100f977fe587694e8a9f7f979c9f1b83b04d78284db5104b378c.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/tables/cfaba9e82b57100f977fe587694e8a9f7f979c9f1b83b04d78284db5104b378c.jpg)

![e49f839686b9823e8eeacb563f134eea38860bb2a248344965dbedab3874d432.jpg](../iclr_results/2408_Can%20Video%20LLMs%20Refuse%20to%20Answer_%20Alignment%20for%20Answerability%20in%20Video%20Large%20Language%20Models/tables/e49f839686b9823e8eeacb563f134eea38860bb2a248344965dbedab3874d432.jpg)

## Diffusion Models Are Real-Time Game Engines


### Images

![0b6836a13e50b99504354b643419c38490d3bee7a7488fad8b2fb1c51507febe.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/0b6836a13e50b99504354b643419c38490d3bee7a7488fad8b2fb1c51507febe.jpg)

![0d6ba6931a00d4619db1e0231949428b4d01befecab47fae1a649a27fb19d9e4.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/0d6ba6931a00d4619db1e0231949428b4d01befecab47fae1a649a27fb19d9e4.jpg)

![1d03dcdc82031d82aa27caabbe88dc4c27e84f49c0bbdb3785487abeee9bd71f.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/1d03dcdc82031d82aa27caabbe88dc4c27e84f49c0bbdb3785487abeee9bd71f.jpg)

![295c31d2bba33c575542ee3a3e7fa0045f27befd99ff71048a39de302741e5e5.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/295c31d2bba33c575542ee3a3e7fa0045f27befd99ff71048a39de302741e5e5.jpg)

![39a1bc6d0e8de7f9f4130a2937d653b2875d5f1d9f1e5f637c36342f94cdb8f7.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/39a1bc6d0e8de7f9f4130a2937d653b2875d5f1d9f1e5f637c36342f94cdb8f7.jpg)

![43a5abf6b3de2d7988421c68a66b0fa7b8aaa2675486ace8a7993dfaad5356f2.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/43a5abf6b3de2d7988421c68a66b0fa7b8aaa2675486ace8a7993dfaad5356f2.jpg)

![56e9520859d14c4efe25b3b5f26659a67a4c2e93ee676d8f6130d6941c716ff2.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/56e9520859d14c4efe25b3b5f26659a67a4c2e93ee676d8f6130d6941c716ff2.jpg)

![5a00ea7dafa4538a13e713a4f0bef49074217158b77b33a8dd0c9f697502fe72.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/5a00ea7dafa4538a13e713a4f0bef49074217158b77b33a8dd0c9f697502fe72.jpg)

![5bbc9380aaa83ee645c8728646ca6f6e89f4224cc9170fa2bbc067c5187d1879.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/5bbc9380aaa83ee645c8728646ca6f6e89f4224cc9170fa2bbc067c5187d1879.jpg)

![7a9b99d32e9faa3d47b3a6f2125b8fda28912a40bd1ddb048833926c67f24aa7.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/7a9b99d32e9faa3d47b3a6f2125b8fda28912a40bd1ddb048833926c67f24aa7.jpg)

![903c31ececceedcdd02594dd4ab15be82700010231997698f64201b86592c7c3.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/903c31ececceedcdd02594dd4ab15be82700010231997698f64201b86592c7c3.jpg)

![98dd1b3e2cd87c73c0e166808f50ccfb736e59bb96611473f638fd6f902e7ade.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/98dd1b3e2cd87c73c0e166808f50ccfb736e59bb96611473f638fd6f902e7ade.jpg)

![b852241fdd45cefe3bc3f9272bba8c903107a209be9a7ecfd0504d1281c0726f.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/b852241fdd45cefe3bc3f9272bba8c903107a209be9a7ecfd0504d1281c0726f.jpg)

![b9a15b5d6120d8dd4091440ab8be2aa47541a2862a56d6db16f24f140aa558ce.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/b9a15b5d6120d8dd4091440ab8be2aa47541a2862a56d6db16f24f140aa558ce.jpg)

![c6622903f4a70610bb5814dff68c392f80db27ef1b6d1e9f99dbfb617bd44d73.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/c6622903f4a70610bb5814dff68c392f80db27ef1b6d1e9f99dbfb617bd44d73.jpg)

![ec7be5c6ee259a91c690f3d0bda615230920a3740a7c8d57df58a793454aa5f9.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/ec7be5c6ee259a91c690f3d0bda615230920a3740a7c8d57df58a793454aa5f9.jpg)

![f3dcf2495a6aa046ce3e3e0a396347d83424f61310e3c0035427d8d031391305.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/f3dcf2495a6aa046ce3e3e0a396347d83424f61310e3c0035427d8d031391305.jpg)

![fc54fb208d5283933148bea2232583d749f40aeda057b3cd82ff486089f62183.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/images/fc54fb208d5283933148bea2232583d749f40aeda057b3cd82ff486089f62183.jpg)

### Tables

![5ee18fdb6a3d1ee65dc98027e3bab0419ad1146f21ece886ca4f2a0f1d029875.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/tables/5ee18fdb6a3d1ee65dc98027e3bab0419ad1146f21ece886ca4f2a0f1d029875.jpg)

![6b7cb8a2ddce9e7321772897424188ffce8f226aee2b605dd03e9adbbeece74f.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/tables/6b7cb8a2ddce9e7321772897424188ffce8f226aee2b605dd03e9adbbeece74f.jpg)

![c0fa087188917ec466029c3c6d280ef0c74d6342b3a2226cf0992d26150dde25.jpg](../iclr_results/2409_Diffusion%20Models%20Are%20Real-Time%20Game%20Engines/tables/c0fa087188917ec466029c3c6d280ef0c74d6342b3a2226cf0992d26150dde25.jpg)

## Swift Hydra:  Self-Reinforcing Generative Framework for Anomaly Detection with Multiple Mamba Models


### Images

![1992ecd313c24d7c53743b05d37f8da5f5bcafda61270ecec7f76c9176192be3.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/1992ecd313c24d7c53743b05d37f8da5f5bcafda61270ecec7f76c9176192be3.jpg)

![30ee6d83e809a2921092dfd55bcbb169b033b3efd09dd6dee79cbf97a6fdbb7f.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/30ee6d83e809a2921092dfd55bcbb169b033b3efd09dd6dee79cbf97a6fdbb7f.jpg)

![64324425f73fea2e05c25442b968e367443362314e0a22c979c757c937c469e0.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/64324425f73fea2e05c25442b968e367443362314e0a22c979c757c937c469e0.jpg)

![77ea4d04e981dbdbcc25ac90b91818ea3229d859efd364523e3d4a17755c54a5.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/77ea4d04e981dbdbcc25ac90b91818ea3229d859efd364523e3d4a17755c54a5.jpg)

![b7242751ac02dd6254f937b19973bbb9c88748e9423e4e723ba15253b6c84595.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/b7242751ac02dd6254f937b19973bbb9c88748e9423e4e723ba15253b6c84595.jpg)

![c0ac49dbfd773c5f17444a64a465c1f39889c8b0892d686cfcbf0736b6eb6793.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/c0ac49dbfd773c5f17444a64a465c1f39889c8b0892d686cfcbf0736b6eb6793.jpg)

![c995d4e496ea5f97d0d76c5718b5616fd051f2f58dad49e7f45e6ee71d429996.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/c995d4e496ea5f97d0d76c5718b5616fd051f2f58dad49e7f45e6ee71d429996.jpg)

![fd5402162a998c2adcbf74f12ac8966d1d343f864502bb371d6628673de5e00d.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/images/fd5402162a998c2adcbf74f12ac8966d1d343f864502bb371d6628673de5e00d.jpg)

### Tables

![01203ee6b14afc255c607f24af3f1a9079555f945548cb7b07cf6a44c6720586.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/01203ee6b14afc255c607f24af3f1a9079555f945548cb7b07cf6a44c6720586.jpg)

![025b86854abb17e9262a5d996a23337f42f5dae2f2b1c8fa29a4c35c8b0cf750.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/025b86854abb17e9262a5d996a23337f42f5dae2f2b1c8fa29a4c35c8b0cf750.jpg)

![12e61daf5cc40928edfd1b17d3ac1b72654bcd2285c27854ef61668ac84781d3.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/12e61daf5cc40928edfd1b17d3ac1b72654bcd2285c27854ef61668ac84781d3.jpg)

![1d7a83183b999ee22b2fb57b101f764638cfc91a12431c465fb859899e35886c.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/1d7a83183b999ee22b2fb57b101f764638cfc91a12431c465fb859899e35886c.jpg)

![33d08ae94e11099f65804cc196f1345a63acfda1961cec24bd88bee2454e5961.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/33d08ae94e11099f65804cc196f1345a63acfda1961cec24bd88bee2454e5961.jpg)

![3493c10a5f088cab28b3c058f912ec91261cd07a80d3e4dd28c50dca601f8df3.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/3493c10a5f088cab28b3c058f912ec91261cd07a80d3e4dd28c50dca601f8df3.jpg)

![34a04a08223c0189f11f73143dceba8613909495d2744be25f049b586307d8e4.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/34a04a08223c0189f11f73143dceba8613909495d2744be25f049b586307d8e4.jpg)

![3948dca515c64d81fe9dc350ad0eaa8c90e71c1ef1a4e871b5421d6a29d88761.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/3948dca515c64d81fe9dc350ad0eaa8c90e71c1ef1a4e871b5421d6a29d88761.jpg)

![3e49a6cb7ae621d4329716b431d5392e040b69a14492d5f99118a9bb4d4ac8c9.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/3e49a6cb7ae621d4329716b431d5392e040b69a14492d5f99118a9bb4d4ac8c9.jpg)

![506443b31f927e1ca49c1169a1b6fcd0d827028b579c667550b01a26d523b57e.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/506443b31f927e1ca49c1169a1b6fcd0d827028b579c667550b01a26d523b57e.jpg)

![5e7506bb5e5de07840c050948c6b76942c71ef58b2a627685a93002bb20eedbd.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/5e7506bb5e5de07840c050948c6b76942c71ef58b2a627685a93002bb20eedbd.jpg)

![87d718c19aae7852352b9707ab4d147a84e4cc1eeb63f251b358cd12447c9a5c.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/87d718c19aae7852352b9707ab4d147a84e4cc1eeb63f251b358cd12447c9a5c.jpg)

![8bdfd7648b758ed8530c03b9c0721c2a4182e81f2067139ac3ce0f8de00a3529.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/8bdfd7648b758ed8530c03b9c0721c2a4182e81f2067139ac3ce0f8de00a3529.jpg)

![953ef07ac0ee5d00a46eb2a326644946a64cd90ee0adc3cf966f6ce83f419100.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/953ef07ac0ee5d00a46eb2a326644946a64cd90ee0adc3cf966f6ce83f419100.jpg)

![a2257bb38f3af515b6d50c4da34b22e198cd2e0119de156084da3b2aba3de357.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/a2257bb38f3af515b6d50c4da34b22e198cd2e0119de156084da3b2aba3de357.jpg)

![a24e3fc8507aa040c126f302cb02563995c375a62cbd3459eb97c0a8494c8b8c.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/a24e3fc8507aa040c126f302cb02563995c375a62cbd3459eb97c0a8494c8b8c.jpg)

![b26b8f60ae0ad88cd13bc8e0a3aa187a1746af0c1bc3b36a88132f654931bcd7.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/b26b8f60ae0ad88cd13bc8e0a3aa187a1746af0c1bc3b36a88132f654931bcd7.jpg)

![cc476f2c41ea6380ef3d315f2b7306406bae76a704cc146201624192e274332d.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/cc476f2c41ea6380ef3d315f2b7306406bae76a704cc146201624192e274332d.jpg)

![f92f80f13818ccf11b4f4a4cd497a30c98220d14e291e8dfa1640e87951f3c9e.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/f92f80f13818ccf11b4f4a4cd497a30c98220d14e291e8dfa1640e87951f3c9e.jpg)

![fe8a7f9f401394f0d30dd00ce855502d1827b0918e96b927efea08ae67b63de0.jpg](../iclr_results/2410_Swift%20Hydra_%20%20Self-Reinforcing%20Generative%20Framework%20for%20Anomaly%20Detection%20with%20Multiple%20Mamba%20Models/tables/fe8a7f9f401394f0d30dd00ce855502d1827b0918e96b927efea08ae67b63de0.jpg)

## QuaDiM: A Conditional Diffusion Model For Quantum State Property Estimation


### Images

![0c48fd95889413a2083d9a63c3bb5541cb779fdb32d6bc897b0a572e3fafcae6.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/images/0c48fd95889413a2083d9a63c3bb5541cb779fdb32d6bc897b0a572e3fafcae6.jpg)

![210e8eb5c9ca48907d6d9beb33ca61d4d5633b274d55e4c88fd63d52e0b7ddb8.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/images/210e8eb5c9ca48907d6d9beb33ca61d4d5633b274d55e4c88fd63d52e0b7ddb8.jpg)

![4cf82c79227b53dc06d26eff3e2f1d0169b234fc6fbb104157d0c88631ec727f.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/images/4cf82c79227b53dc06d26eff3e2f1d0169b234fc6fbb104157d0c88631ec727f.jpg)

![6f7746f50b3d2c0f259000ecee0d0602febf3f82959823a1b7707ae24bad39d1.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/images/6f7746f50b3d2c0f259000ecee0d0602febf3f82959823a1b7707ae24bad39d1.jpg)

![91c8b9b0dbd3ff9fbff1c97c67ffa790485bdaf3c4d31b7815451984c1e4715c.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/images/91c8b9b0dbd3ff9fbff1c97c67ffa790485bdaf3c4d31b7815451984c1e4715c.jpg)

![bfde74905ee44696145b71255dca1b6e50ea7d77224e7ac8ef172f841f8e04eb.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/images/bfde74905ee44696145b71255dca1b6e50ea7d77224e7ac8ef172f841f8e04eb.jpg)

### Tables

![0eb60569527696a1417d53713e16363f961d5141d9e65e161c45c72d1fa265fe.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/0eb60569527696a1417d53713e16363f961d5141d9e65e161c45c72d1fa265fe.jpg)

![1c75770b1719693b068e5507c58c0fe70d0759d9946475aa13ea84e794242303.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/1c75770b1719693b068e5507c58c0fe70d0759d9946475aa13ea84e794242303.jpg)

![26e9370238d5e2c8e193f3b1dc711d23f2d566b4c88a1e44ef6edfb6579db4a9.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/26e9370238d5e2c8e193f3b1dc711d23f2d566b4c88a1e44ef6edfb6579db4a9.jpg)

![2cff2253804b83283ba21cd0a8561326182e9f89e1598d2dc8b1317f24b6680a.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/2cff2253804b83283ba21cd0a8561326182e9f89e1598d2dc8b1317f24b6680a.jpg)

![4fb0d06bb999d64602c60472509307d1e2e63e5e6de8d0fe8e4b8cebe106cc47.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/4fb0d06bb999d64602c60472509307d1e2e63e5e6de8d0fe8e4b8cebe106cc47.jpg)

![8052ab80d941f36fe3ebe73977fe05713e289b9b39f6f8179dcd6c6bc86d4224.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/8052ab80d941f36fe3ebe73977fe05713e289b9b39f6f8179dcd6c6bc86d4224.jpg)

![8fb50b9da2938843f54244b1c3a09be446519b27e07e786010101001e09511ab.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/8fb50b9da2938843f54244b1c3a09be446519b27e07e786010101001e09511ab.jpg)

![c60048b3f392543680710a4358a59257a71b50c1f0010782668c3bd2104f4a03.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/c60048b3f392543680710a4358a59257a71b50c1f0010782668c3bd2104f4a03.jpg)

![cb896c8c495553394381009fa4acf0c1f321d2e2357bfba718ddba55e334b49c.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/cb896c8c495553394381009fa4acf0c1f321d2e2357bfba718ddba55e334b49c.jpg)

![e83a2d873e0b28bbe0e0769f77205815a6909b49daf010a380feb3e6be88b79a.jpg](../iclr_results/2411_QuaDiM_%20A%20Conditional%20Diffusion%20Model%20For%20Quantum%20State%20Property%20Estimation/tables/e83a2d873e0b28bbe0e0769f77205815a6909b49daf010a380feb3e6be88b79a.jpg)

## Expected Sliced Transport Plans


### Images

![267ecf508c40634c8670f119852a07a032825887d5070e4962ebb1521c4e3442.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/267ecf508c40634c8670f119852a07a032825887d5070e4962ebb1521c4e3442.jpg)

![39eefe996ade62179d070508e677dc28ddf04df9a71f7284a066d357864fe490.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/39eefe996ade62179d070508e677dc28ddf04df9a71f7284a066d357864fe490.jpg)

![52d1eca8fd6841e25fd4a09ee6b813e550ccc40afbe7b6c2b5be4f2eb3e2bf96.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/52d1eca8fd6841e25fd4a09ee6b813e550ccc40afbe7b6c2b5be4f2eb3e2bf96.jpg)

![738f4481aa20fb3f6d70661843d7d51c53bd8acfc6d1891a04fb24c97ae01947.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/738f4481aa20fb3f6d70661843d7d51c53bd8acfc6d1891a04fb24c97ae01947.jpg)

![84a7fc000c8ecec42f450706c497b244494a8dce4a641d8584e530176b69e0ca.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/84a7fc000c8ecec42f450706c497b244494a8dce4a641d8584e530176b69e0ca.jpg)

![95444b75421667a1374a06685151d784720e4c8eb45dfcadc7e7d884f4165da4.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/95444b75421667a1374a06685151d784720e4c8eb45dfcadc7e7d884f4165da4.jpg)

![be18419ca4f1f915cab222aa36c650f9b67fe49c910d6002381f2214c6a03d8a.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/be18419ca4f1f915cab222aa36c650f9b67fe49c910d6002381f2214c6a03d8a.jpg)

![e07abcbfd12e5c7a07113ee2348a8253b3526175ca4779727100602ac9a8e505.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/e07abcbfd12e5c7a07113ee2348a8253b3526175ca4779727100602ac9a8e505.jpg)

![f2f3fa183dd4d0865bf5612034281ad7447815d9b6b0deae02abb096ab34e0b3.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/images/f2f3fa183dd4d0865bf5612034281ad7447815d9b6b0deae02abb096ab34e0b3.jpg)

### Tables

![7615862217c6bc0212d6e973847ac1cf674ef4ae0cf90b7ac42302dd8b3c57d3.jpg](../iclr_results/2412_Expected%20Sliced%20Transport%20Plans/tables/7615862217c6bc0212d6e973847ac1cf674ef4ae0cf90b7ac42302dd8b3c57d3.jpg)

## B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners


### Images

![0806c5fc78f23b5d2bc0939c6d1ac12437c144e1e11faaf27f4f694d01461a66.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/0806c5fc78f23b5d2bc0939c6d1ac12437c144e1e11faaf27f4f694d01461a66.jpg)

![1dd0bdee21051da588e4bc26411b9ecea5847c54c6dafc2c45c5d41a628a3778.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/1dd0bdee21051da588e4bc26411b9ecea5847c54c6dafc2c45c5d41a628a3778.jpg)

![78110b86fbdbf0f819d18ce0371f5f400fc1b96f1b31da2b13019f292ec722ff.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/78110b86fbdbf0f819d18ce0371f5f400fc1b96f1b31da2b13019f292ec722ff.jpg)

![7cc1209d0a7521ff374253663047756687f460800c464c237b9a2dd40b5b0547.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/7cc1209d0a7521ff374253663047756687f460800c464c237b9a2dd40b5b0547.jpg)

![ad15a988a6b5b77ed95ab7d6d34fb66217ce3bbcb78d18751185e28a63315032.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/ad15a988a6b5b77ed95ab7d6d34fb66217ce3bbcb78d18751185e28a63315032.jpg)

![cc3b35c20b50f09fdbd9a80d6fc6eff5a8ae36f401d815352ac38db4192f72d8.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/cc3b35c20b50f09fdbd9a80d6fc6eff5a8ae36f401d815352ac38db4192f72d8.jpg)

![dc40c6a2b0b5e3d304882472b5ac12e2d9d9364f2b4376f4f78a3c5f2d197bf8.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/dc40c6a2b0b5e3d304882472b5ac12e2d9d9364f2b4376f4f78a3c5f2d197bf8.jpg)

![fb2773cb9fb23e637cafede464bbdcb8c8e533c73ff1e8bc74c947ed094db308.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/images/fb2773cb9fb23e637cafede464bbdcb8c8e533c73ff1e8bc74c947ed094db308.jpg)

### Tables

![0bd345d56c078d81efffd15c52928977a2fd072caad496eb916b2bee040df50a.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/tables/0bd345d56c078d81efffd15c52928977a2fd072caad496eb916b2bee040df50a.jpg)

![609f3e6e0b5f0fffc53883938540c1ef6ff8dfea9da3dade5a417720d9fc50c8.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/tables/609f3e6e0b5f0fffc53883938540c1ef6ff8dfea9da3dade5a417720d9fc50c8.jpg)

![6b2391d32a6c9faeb9390576805d6982a3a35f8c970025b95f25420e65124ebb.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/tables/6b2391d32a6c9faeb9390576805d6982a3a35f8c970025b95f25420e65124ebb.jpg)

![883f9519caa2fd3b6a35ebf66b925e7802e36c33f503ce3b33406396c24e2bc3.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/tables/883f9519caa2fd3b6a35ebf66b925e7802e36c33f503ce3b33406396c24e2bc3.jpg)

![ac1a4f43cbd2ceb55bab143c9f3ddfda1124bf6c44ba753d8f7d1c45312c42d3.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/tables/ac1a4f43cbd2ceb55bab143c9f3ddfda1124bf6c44ba753d8f7d1c45312c42d3.jpg)

![f83aa2c35ba1f1c6e659ef13bdbd971c35ff4f5f5c400c22e6d49c38ee894e67.jpg](../iclr_results/2413_B-STaR_%20Monitoring%20and%20Balancing%20Exploration%20and%20Exploitation%20in%20Self-Taught%20Reasoners/tables/f83aa2c35ba1f1c6e659ef13bdbd971c35ff4f5f5c400c22e6d49c38ee894e67.jpg)

## Provable Benefit of Annealed Langevin Monte Carlo for Non-log-concave Sampling


### Images

![ba0a79dcdf64b5f55e1ad3aeb3f0e1172f6d4f00f9188f43cb821761b4fd96dd.jpg](../iclr_results/2414_Provable%20Benefit%20of%20Annealed%20Langevin%20Monte%20Carlo%20for%20Non-log-concave%20Sampling/images/ba0a79dcdf64b5f55e1ad3aeb3f0e1172f6d4f00f9188f43cb821761b4fd96dd.jpg)

![ed5b1726723651a90b8301f4cbf9ee1d35c9ff18e976100c47b5bba659d1eae2.jpg](../iclr_results/2414_Provable%20Benefit%20of%20Annealed%20Langevin%20Monte%20Carlo%20for%20Non-log-concave%20Sampling/images/ed5b1726723651a90b8301f4cbf9ee1d35c9ff18e976100c47b5bba659d1eae2.jpg)

### Tables

![0fb79849aabae618cea8df4b61dfc89c20d9c663eb019496572bdf50a3bcd3d2.jpg](../iclr_results/2414_Provable%20Benefit%20of%20Annealed%20Langevin%20Monte%20Carlo%20for%20Non-log-concave%20Sampling/tables/0fb79849aabae618cea8df4b61dfc89c20d9c663eb019496572bdf50a3bcd3d2.jpg)

## Modeling Fine-Grained Hand-Object Dynamics for Egocentric Video Representation Learning


### Images

![135a5f2cebd25ad7757e2149718e66ebbcf95730bddbfee403741513ccfb9019.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/135a5f2cebd25ad7757e2149718e66ebbcf95730bddbfee403741513ccfb9019.jpg)

![1db8479579c9019251e3f0a85df3ea1e384455be2e0bb032937e6160fcd28078.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/1db8479579c9019251e3f0a85df3ea1e384455be2e0bb032937e6160fcd28078.jpg)

![5c045ecd5fa2f8ee83a9c53f2f89252a2727a8d03bb5f6e3aa0c5db1da25e339.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/5c045ecd5fa2f8ee83a9c53f2f89252a2727a8d03bb5f6e3aa0c5db1da25e339.jpg)

![656901934816c1a62dd506e4c660a3e33a1c83531d01ab92056eb1a0fcc93ba9.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/656901934816c1a62dd506e4c660a3e33a1c83531d01ab92056eb1a0fcc93ba9.jpg)

![74d1ea1eada0fcbdb762e4722899eb7d8b4a2590eb43277d575ac25ffb88062c.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/74d1ea1eada0fcbdb762e4722899eb7d8b4a2590eb43277d575ac25ffb88062c.jpg)

![8461f6a367fab014c9ca812fb9cfdce5b2cbdf32cc68bd5629cd736800184bc2.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/8461f6a367fab014c9ca812fb9cfdce5b2cbdf32cc68bd5629cd736800184bc2.jpg)

![93045f32abc0772b3cf6bc0031764f5ef044b5f1c2d372704b6bd4d09b2c104e.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/93045f32abc0772b3cf6bc0031764f5ef044b5f1c2d372704b6bd4d09b2c104e.jpg)

![a939abc5fafb98c2ffa017331ed529b8a76653e26a20d390d5fb8418fe76d33d.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/a939abc5fafb98c2ffa017331ed529b8a76653e26a20d390d5fb8418fe76d33d.jpg)

![aa27bcae0d4152ffd70376d126fff917967c084a28d236bd5627f0ce9233379f.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/aa27bcae0d4152ffd70376d126fff917967c084a28d236bd5627f0ce9233379f.jpg)

![b4a27e7b5dff5046b5849b4f700bbddbd028e9b385f414cecd18d437f8b5dc34.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/b4a27e7b5dff5046b5849b4f700bbddbd028e9b385f414cecd18d437f8b5dc34.jpg)

![b7d62c57cbf13dc29240257b225d4e530e0ae151e362d8035a7e762ffd8c3e70.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/b7d62c57cbf13dc29240257b225d4e530e0ae151e362d8035a7e762ffd8c3e70.jpg)

![c36f627fc1ed82b1416f6badb74785806ef5ed30da6bddab1567b1903c72bd6c.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/c36f627fc1ed82b1416f6badb74785806ef5ed30da6bddab1567b1903c72bd6c.jpg)

![d117b4bfa9fb5dd16c8a3b075f0524b963db026179d0d6f249e49ba39e4f9bd7.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/d117b4bfa9fb5dd16c8a3b075f0524b963db026179d0d6f249e49ba39e4f9bd7.jpg)

![dcf3e9cb2ed9e030883aff47d19a28d43ba439e6ae1a6c4cc0842cb89da18294.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/dcf3e9cb2ed9e030883aff47d19a28d43ba439e6ae1a6c4cc0842cb89da18294.jpg)

![f73f6996beb22c0ef76b8a36f7d78d4270a21aff765cd69ec80b7f19b365982e.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/f73f6996beb22c0ef76b8a36f7d78d4270a21aff765cd69ec80b7f19b365982e.jpg)

![fb993d479ca237fd9fcd2069d13054b63fbace32e06927332be9abaf5f013cfc.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/images/fb993d479ca237fd9fcd2069d13054b63fbace32e06927332be9abaf5f013cfc.jpg)

### Tables

![354fca79ba0c663e918864cf42f402b16e1a57b1186d8ec6e1ef611375e03402.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/354fca79ba0c663e918864cf42f402b16e1a57b1186d8ec6e1ef611375e03402.jpg)

![3f7cc9bb8948afe354095516aacb19eeb5d41a87722187603e6176fe1ed39e55.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/3f7cc9bb8948afe354095516aacb19eeb5d41a87722187603e6176fe1ed39e55.jpg)

![62c385eb5958dc8d959f9d19a4928f02e210881323412dacfaa74953e740e703.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/62c385eb5958dc8d959f9d19a4928f02e210881323412dacfaa74953e740e703.jpg)

![6a25cd5f07d23a83bd7f0e9c62d9b0c19f8c61ea7d07f8f4ef1da183550c5eb1.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/6a25cd5f07d23a83bd7f0e9c62d9b0c19f8c61ea7d07f8f4ef1da183550c5eb1.jpg)

![798e42feaecfd4d660b5893c221f9364426ef10f416077ed4329191d2b7f437e.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/798e42feaecfd4d660b5893c221f9364426ef10f416077ed4329191d2b7f437e.jpg)

![92f373988da753cccc8188bae18649ffedb7ae3869958ba1884b844a0ae4dfe2.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/92f373988da753cccc8188bae18649ffedb7ae3869958ba1884b844a0ae4dfe2.jpg)

![9da91eb87233acffa8dde2c9af37999d797d161741a3a001ac794429aa3e13e8.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/9da91eb87233acffa8dde2c9af37999d797d161741a3a001ac794429aa3e13e8.jpg)

![ae03d6f16f9b414481492cb4ebf303a715f9f6328b93275341cf16fee9eb6f84.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/ae03d6f16f9b414481492cb4ebf303a715f9f6328b93275341cf16fee9eb6f84.jpg)

![afca613949495021d81eb5f7a45cbacf5f39d04987aaf36414ae94bba41c5c15.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/afca613949495021d81eb5f7a45cbacf5f39d04987aaf36414ae94bba41c5c15.jpg)

![b45d054c42758d527b6ccd7071fc6ee6dc443d158f91b9af328ebbec2507c996.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/b45d054c42758d527b6ccd7071fc6ee6dc443d158f91b9af328ebbec2507c996.jpg)

![c194433adf866549a6526b316669190e44b2d267349664cc3c1c5f46a08c9043.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/c194433adf866549a6526b316669190e44b2d267349664cc3c1c5f46a08c9043.jpg)

![d15b70237021ef108902e230fda38c90142f164062bd7eaa686f83b0c6984f30.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/d15b70237021ef108902e230fda38c90142f164062bd7eaa686f83b0c6984f30.jpg)

![d4d039f3115f0cc1ab2e1919156516b416a9cbef801a55e1980b5a85095fd158.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/d4d039f3115f0cc1ab2e1919156516b416a9cbef801a55e1980b5a85095fd158.jpg)

![ef774b845f545bb87511e5ed606996de49b6cc52a1ecd6caf545e814cbba0741.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/ef774b845f545bb87511e5ed606996de49b6cc52a1ecd6caf545e814cbba0741.jpg)

![f46ea375d7501165cf411c9abb017dc009e38ba2a149769bfd422884b9b1f042.jpg](../iclr_results/2415_Modeling%20Fine-Grained%20Hand-Object%20Dynamics%20for%20Egocentric%20Video%20Representation%20Learning/tables/f46ea375d7501165cf411c9abb017dc009e38ba2a149769bfd422884b9b1f042.jpg)

## ACES: Automatic Cohort Extraction System for Event-Stream Datasets


### Images

![29facce44cd976399f70af00b3ad1f1e26b8b3b5374152945fea8041a75771c0.jpg](../iclr_results/2416_ACES_%20Automatic%20Cohort%20Extraction%20System%20for%20Event-Stream%20Datasets/images/29facce44cd976399f70af00b3ad1f1e26b8b3b5374152945fea8041a75771c0.jpg)

![9ca5d304238992d45718b9dfbd45471f3a7542c565613d29897176d64ddde574.jpg](../iclr_results/2416_ACES_%20Automatic%20Cohort%20Extraction%20System%20for%20Event-Stream%20Datasets/images/9ca5d304238992d45718b9dfbd45471f3a7542c565613d29897176d64ddde574.jpg)

![a3a40850dc8a6c9578a52b664bff4b6ea07318c249f7ef2b5f4587af350c768b.jpg](../iclr_results/2416_ACES_%20Automatic%20Cohort%20Extraction%20System%20for%20Event-Stream%20Datasets/images/a3a40850dc8a6c9578a52b664bff4b6ea07318c249f7ef2b5f4587af350c768b.jpg)

![be3be3d5f2b774c7e7c3e84ddf0977e489d5dcf9b42df2f8840e067bb2e113af.jpg](../iclr_results/2416_ACES_%20Automatic%20Cohort%20Extraction%20System%20for%20Event-Stream%20Datasets/images/be3be3d5f2b774c7e7c3e84ddf0977e489d5dcf9b42df2f8840e067bb2e113af.jpg)

### Tables

![3c331d66d7c94bd74f8b31a908b1e343616fe22489c8688a42fe68765c601703.jpg](../iclr_results/2416_ACES_%20Automatic%20Cohort%20Extraction%20System%20for%20Event-Stream%20Datasets/tables/3c331d66d7c94bd74f8b31a908b1e343616fe22489c8688a42fe68765c601703.jpg)

![5dc10bb924cefeabcc4d090f56f5b0bdf1e5fcadbbd2536454959ee7bd5dddc3.jpg](../iclr_results/2416_ACES_%20Automatic%20Cohort%20Extraction%20System%20for%20Event-Stream%20Datasets/tables/5dc10bb924cefeabcc4d090f56f5b0bdf1e5fcadbbd2536454959ee7bd5dddc3.jpg)

![b92c89c799d728f5d46737994ef8e90b58f31c974431a86a7ec10d8493f5494a.jpg](../iclr_results/2416_ACES_%20Automatic%20Cohort%20Extraction%20System%20for%20Event-Stream%20Datasets/tables/b92c89c799d728f5d46737994ef8e90b58f31c974431a86a7ec10d8493f5494a.jpg)

## GaussianAnything: Interactive Point Cloud Flow Matching for 3D Generation

### Images

![3f250364958a6cad99f54afed346bf16563c1dd10c582aebcc04528eb90b00f6.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/3f250364958a6cad99f54afed346bf16563c1dd10c582aebcc04528eb90b00f6.jpg)

![5443f20be148d8c69e500712e4807fab14f8d9cfe6f068ac7f5c39533b4d7325.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/5443f20be148d8c69e500712e4807fab14f8d9cfe6f068ac7f5c39533b4d7325.jpg)

![596d2b5a9ed448293a37de3da8caae9318c4cf7909696b7f457d8ad5f27651b4.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/596d2b5a9ed448293a37de3da8caae9318c4cf7909696b7f457d8ad5f27651b4.jpg)

![60bc523cac992d89e8ba1aaf12be708bc7106d4780cdf222b0b3b8c90071b687.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/60bc523cac992d89e8ba1aaf12be708bc7106d4780cdf222b0b3b8c90071b687.jpg)

![6fa1005fdde3d9e10eb7106b489552c9da73f4f7a38335006d5054d15e0fd5b4.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/6fa1005fdde3d9e10eb7106b489552c9da73f4f7a38335006d5054d15e0fd5b4.jpg)

![70c18d2b5bef187641422f1ef406dacea6c88009c2dcf5b12fe172511992f617.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/70c18d2b5bef187641422f1ef406dacea6c88009c2dcf5b12fe172511992f617.jpg)

![726e051468ccd2ed4adc7581cffea941cc126c67e68f8ca8cf3ecd282ed2dbc0.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/726e051468ccd2ed4adc7581cffea941cc126c67e68f8ca8cf3ecd282ed2dbc0.jpg)

![9998b01b13ceeafb90c46090d4e2ede14eb4ecaafd98d1cee54e8785b026dcdd.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/9998b01b13ceeafb90c46090d4e2ede14eb4ecaafd98d1cee54e8785b026dcdd.jpg)

![aaa62a6671457abe6d7f92c451702383a7a023e699d9c587b18dcb416f1f167e.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/aaa62a6671457abe6d7f92c451702383a7a023e699d9c587b18dcb416f1f167e.jpg)

![abdc3379819ce0a2bd55f70d8fa979a24c2ced797293dd825174ca1aa1a5aeab.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/abdc3379819ce0a2bd55f70d8fa979a24c2ced797293dd825174ca1aa1a5aeab.jpg)

![ad568792cfa63a1c59fedb937e8519be6451160c1e4fee303884a22f033d5d3f.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/images/ad568792cfa63a1c59fedb937e8519be6451160c1e4fee303884a22f033d5d3f.jpg)

### Tables

![a51e749394f95e256b5baf3ab79e2931e4e0dd3fb3e8ceec2a978317498805e0.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/tables/a51e749394f95e256b5baf3ab79e2931e4e0dd3fb3e8ceec2a978317498805e0.jpg)

![a9db1eb37741d5ee740472e1965b693683cdc463e5b24b676005cace558821f1.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/tables/a9db1eb37741d5ee740472e1965b693683cdc463e5b24b676005cace558821f1.jpg)

![be28c79190175509357daad040b3e1c5c36435cfd05e275c2afa347a65256d17.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/tables/be28c79190175509357daad040b3e1c5c36435cfd05e275c2afa347a65256d17.jpg)

![d94bddf0daf01282af6acc2a9bdf266050ca993238e0ad31b7484358c910a080.jpg](../iclr_results/2417_GaussianAnything_%20Interactive%20Point%20Cloud%20Flow%20Matching%20for%203D%20Generation/tables/d94bddf0daf01282af6acc2a9bdf266050ca993238e0ad31b7484358c910a080.jpg)
