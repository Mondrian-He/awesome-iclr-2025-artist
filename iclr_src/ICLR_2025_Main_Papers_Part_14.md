# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 14 of 100

## 目录 (Table of Contents)

1. [AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials](#AgentTrek-Agent-Trajectory-Synthesis-via-Guiding-Replay-with-Web-Tutorials)
2. [Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences](#Bayesian-Optimization-of-Antibodies-Informed-by-a-Generative-Model-of-Evolving-Sequences)
3. [Scaling FP8 training to trillion-token LLMs](#Scaling-FP8-training-to-trillion-token-LLMs)
4. [Student-Informed Teacher Training](#Student-Informed-Teacher-Training)
5. [Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation](#Stabilizing-Reinforcement-Learning-in-Differentiable-Multiphysics-Simulation)
6. [Estimating the Probabilities of Rare Outputs in Language Models](#Estimating-the-Probabilities-of-Rare-Outputs-in-Language-Models)
7. [Imputation for prediction: beware of diminishing returns.](#Imputation-for-prediction-beware-of-diminishing-returns)
8. [Physics-aligned field reconstruction with diffusion bridge](#Physics-aligned-field-reconstruction-with-diffusion-bridge)
9. [Rethinking Reward Model Evaluation: Are We Barking up the Wrong Tree?](#Rethinking-Reward-Model-Evaluation-Are-We-Barking-up-the-Wrong-Tree)
10. [$R^2$-Guard: Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning](#R2-Guard-Robust-Reasoning-Enabled-LLM-Guardrail-via-Knowledge-Enhanced-Logical-Reasoning)
11. [Test-time Adaptation for Cross-modal Retrieval with Query Shift](#Test-time-Adaptation-for-Cross-modal-Retrieval-with-Query-Shift)
12. [Severing Spurious Correlations with Data Pruning](#Severing-Spurious-Correlations-with-Data-Pruning)
13. [Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance](#Rare-to-Frequent-Unlocking-Compositional-Generation-Power-of-Diffusion-Models-on-Rare-Concepts-with-LLM-Guidance)
14. [Lightweight Neural App Control](#Lightweight-Neural-App-Control)
15. [DeLLMa: Decision Making Under Uncertainty with Large Language Models](#DeLLMa-Decision-Making-Under-Uncertainty-with-Large-Language-Models)
16. [Multi-Robot Motion Planning with Diffusion Models](#Multi-Robot-Motion-Planning-with-Diffusion-Models)
17. [ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks](#ConFIG-Towards-Conflict-free-Training-of-Physics-Informed-Neural-Networks)
18. [MagicPIG: LSH Sampling for Efficient LLM Generation](#MagicPIG-LSH-Sampling-for-Efficient-LLM-Generation)
19. [Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](#Rewarding-Progress-Scaling-Automated-Process-Verifiers-for-LLM-Reasoning)
20. [Hymba: A Hybrid-head Architecture for Small Language Models](#Hymba-A-Hybrid-head-Architecture-for-Small-Language-Models)
21. [AutoCGP: Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations](#AutoCGP-Closed-Loop-Concept-Guided-Policies-from-Unlabeled-Demonstrations)
22. [A CLIP-Powered Framework for Robust and Generalizable Data Selection](#A-CLIP-Powered-Framework-for-Robust-and-Generalizable-Data-Selection)
23. [OSDA Agent: Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agents](#OSDA-Agent-Leveraging-Large-Language-Models-for-De-Novo-Design-of-Organic-Structure-Directing-Agents)
24. [Sample then Identify: A General Framework for Risk Control and Assessment in Multimodal Large Language Models](#Sample-then-Identify-A-General-Framework-for-Risk-Control-and-Assessment-in-Multimodal-Large-Language-Models)
25. [Conditional Diffusion with Ordinal Regression: Longitudinal Data Generation for Neurodegenerative Disease Studies](#Conditional-Diffusion-with-Ordinal-Regression-Longitudinal-Data-Generation-for-Neurodegenerative-Disease-Studies)
26. [SplatFormer: Point Transformer for Robust 3D Gaussian Splatting](#SplatFormer-Point-Transformer-for-Robust-3D-Gaussian-Splatting)
27. [When do GFlowNets learn the right distribution?](#When-do-GFlowNets-learn-the-right-distribution)
28. [On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent](#On-the-Optimization-and-Generalization-of-Two-layer-Transformers-with-Sign-Gradient-Descent)
29. [Towards a Unified and Verified Understanding of Group-Operation Networks](#Towards-a-Unified-and-Verified-Understanding-of-Group-Operation-Networks)
30. [DenseMatcher: Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo](#DenseMatcher-Learning-3D-Semantic-Correspondence-for-Category-Level-Manipulation-from-a-Single-Demo)
31. [Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL](#Dont-flatten-tokenize-Unlocking-the-key-to-SoftMoEs-efficacy-in-deep-RL)
32. [Quality Measures for Dynamic Graph Generative Models](#Quality-Measures-for-Dynamic-Graph-Generative-Models)
33. [Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Feedback from Pre-training Demonstrations](#Direct-Post-Training-Preference-Alignment-for-Multi-Agent-Motion-Generation-Model-Using-Implicit-Feedback-from-Pre-training-Demonstrations)
34. [Better Instruction-Following Through Minimum Bayes Risk](#Better-Instruction-Following-Through-Minimum-Bayes-Risk)
35. [LiFT: Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning](#LiFT-Learning-to-Fine-Tune-via-Bayesian-Parameter-Efficient-Meta-Fine-Tuning)
36. [Theory on Mixture-of-Experts in Continual Learning](#Theory-on-Mixture-of-Experts-in-Continual-Learning)

---


## AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials

### Images

![f48de67af7cd0cb2d5edd6444b1a0cfde12ad8360abeb98774a4d4737c285f83.jpg](../iclr_results/505_Improved Approximation Algorithms for $k$-Submodular Maximization via Multilinear Extension/images/f48de67af7cd0cb2d5edd6444b1a0cfde12ad8360abeb98774a4d4737c285f83.jpg)

### Tables

![25363cbe8a7c319fb0712593da378b4056cbd55ee1ce81ed9bcf9516791c466b.jpg](../iclr_results/505_Improved Approximation Algorithms for $k$-Submodular Maximization via Multilinear Extension/tables/25363cbe8a7c319fb0712593da378b4056cbd55ee1ce81ed9bcf9516791c466b.jpg)

## AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials


### Images

![1662171ebe82f6a95c1ed67a2c69714ac8a00dbc36e50288b41fcb31e31d9fef.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/1662171ebe82f6a95c1ed67a2c69714ac8a00dbc36e50288b41fcb31e31d9fef.jpg)

![59dc24877ede8a01d422fc92fd0267113939526747c8cca84b84603e6977a97e.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/59dc24877ede8a01d422fc92fd0267113939526747c8cca84b84603e6977a97e.jpg)

![5c35bc739b505a9da88fa3bbf477e2144956ad1c6b2337c18dd0a35b7e56805e.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/5c35bc739b505a9da88fa3bbf477e2144956ad1c6b2337c18dd0a35b7e56805e.jpg)

![6ef675aaa954ffb135763ff3d9c55926c355c17e48ca17b304f6e5fc4ee637fc.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/6ef675aaa954ffb135763ff3d9c55926c355c17e48ca17b304f6e5fc4ee637fc.jpg)

![7b82273954eb07e903a0c9b038f591a0eef91317ec84a4af5cbd9e816e35fccf.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/7b82273954eb07e903a0c9b038f591a0eef91317ec84a4af5cbd9e816e35fccf.jpg)

![7fb4878d442d4c064299907dea32d3d9fb9aa6e6dfb29303a1100be4eac189ab.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/7fb4878d442d4c064299907dea32d3d9fb9aa6e6dfb29303a1100be4eac189ab.jpg)

![89ef603fa06d36e3e2070857e4f785bab45c1cac9f33b90c617f9aa0f2011548.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/89ef603fa06d36e3e2070857e4f785bab45c1cac9f33b90c617f9aa0f2011548.jpg)

![90e4f1315ae1d62f5820513a9006a6689bd9806c08cb2c6431174eedc0a13a28.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/90e4f1315ae1d62f5820513a9006a6689bd9806c08cb2c6431174eedc0a13a28.jpg)

![9ad3204bdc7dc23a0153a6adf1402b54cd2a98a41d7cadcbb83ab5668d74db5a.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/9ad3204bdc7dc23a0153a6adf1402b54cd2a98a41d7cadcbb83ab5668d74db5a.jpg)

![a621a0e68a602303d50e4539cb3f7d7978716c3804d060b948cddbc62d73ef40.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/a621a0e68a602303d50e4539cb3f7d7978716c3804d060b948cddbc62d73ef40.jpg)

![b4fce72eb07878c49a880753dc90b4e13d7f41f4db487e4a342b661009a30131.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/b4fce72eb07878c49a880753dc90b4e13d7f41f4db487e4a342b661009a30131.jpg)

![c50256d257b8021c681f7cbadef9d932440fc233a9a64462e98fcca378f1f5dd.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/c50256d257b8021c681f7cbadef9d932440fc233a9a64462e98fcca378f1f5dd.jpg)

![f1106595bd4c911d704b1d40847d3fc4d1b3bd4d2e910c9c181ed2fbca020b67.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/f1106595bd4c911d704b1d40847d3fc4d1b3bd4d2e910c9c181ed2fbca020b67.jpg)

![f288d066dbc6df4f8551b71210f97e0cfa0dbbf358cb415947a1d6633b8001e5.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/f288d066dbc6df4f8551b71210f97e0cfa0dbbf358cb415947a1d6633b8001e5.jpg)

![f5934ab5a41c6e4e58cbddc9fa2ab778cab07d614ae745f9a2c06d24589bdc99.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/images/f5934ab5a41c6e4e58cbddc9fa2ab778cab07d614ae745f9a2c06d24589bdc99.jpg)

### Tables

![1ae33b2c19994e7cb010b5177e5c1cbb52daa0cb2d79b8cc59e05654ba28c950.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/1ae33b2c19994e7cb010b5177e5c1cbb52daa0cb2d79b8cc59e05654ba28c950.jpg)

![3b8da4932e998617fb938806690d5b68124576f96f1d1df456a51556a6bfc09d.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/3b8da4932e998617fb938806690d5b68124576f96f1d1df456a51556a6bfc09d.jpg)

![409de0e8f5f76beb94824fedf938cf46b87c7121d975286ececbb480aa7721ee.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/409de0e8f5f76beb94824fedf938cf46b87c7121d975286ececbb480aa7721ee.jpg)

![5f7a4270439abcf628014ad1b6e1bc2a689b4ade605caab4a72090b247cfaa30.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/5f7a4270439abcf628014ad1b6e1bc2a689b4ade605caab4a72090b247cfaa30.jpg)

![66eb7a8773a9f8e97d689429c68f6fac5e301fc3a1ef79a24c1f0c7de6c8925a.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/66eb7a8773a9f8e97d689429c68f6fac5e301fc3a1ef79a24c1f0c7de6c8925a.jpg)

![6c8c935e7d4965c7d4ac74cc31e36b056fc058fc37e0ef669462d734402d86db.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/6c8c935e7d4965c7d4ac74cc31e36b056fc058fc37e0ef669462d734402d86db.jpg)

![7bf99ca6ad58d86aca0acfce7fa6ca556d93fad10bef9afedb472a018bd50211.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/7bf99ca6ad58d86aca0acfce7fa6ca556d93fad10bef9afedb472a018bd50211.jpg)

![cb36c9847069c96aa63bfc3216dc6ed3752d57412511a77935eb6f074983b6ed.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/cb36c9847069c96aa63bfc3216dc6ed3752d57412511a77935eb6f074983b6ed.jpg)

![ce97eddd068acdd78936fe4b43df576ca9bcb9a56d42f032b36a7d777ae2e02d.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/ce97eddd068acdd78936fe4b43df576ca9bcb9a56d42f032b36a7d777ae2e02d.jpg)

![dd7f1ccb4b62a0110c6d954e14d4f48f54fb51dd52c1868c60073e1f44cd7cb8.jpg](../iclr_results/506_AgentTrek_ Agent Trajectory Synthesis via Guiding Replay with Web Tutorials/tables/dd7f1ccb4b62a0110c6d954e14d4f48f54fb51dd52c1868c60073e1f44cd7cb8.jpg)

## Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences


### Images

![0a1dfeb733e6b307698104f36024372ad518e017561d23eed4f25a64789affb1.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/0a1dfeb733e6b307698104f36024372ad518e017561d23eed4f25a64789affb1.jpg)

![2b3413e48568de44615ffce3577b20bafc3e5b09f05aadeb680c756632cd0a85.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/2b3413e48568de44615ffce3577b20bafc3e5b09f05aadeb680c756632cd0a85.jpg)

![38679a7b8cc1489c9a72ba176f487997add34d17764b886c546f8870717cc589.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/38679a7b8cc1489c9a72ba176f487997add34d17764b886c546f8870717cc589.jpg)

![38d8ca3e7735d8329a4312771126edfd0c9f78a30efab229a541d2005037d1c1.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/38d8ca3e7735d8329a4312771126edfd0c9f78a30efab229a541d2005037d1c1.jpg)

![4103c727b748d7cc1522d849e28878b4d1467c00e8698126777c53779fb17bd3.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/4103c727b748d7cc1522d849e28878b4d1467c00e8698126777c53779fb17bd3.jpg)

![69df8ad999c8ab87a1a4d4c546e3f308a56bb8994aa88b5405eb679519c185d5.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/69df8ad999c8ab87a1a4d4c546e3f308a56bb8994aa88b5405eb679519c185d5.jpg)

![b51bcd5817ffa23242f141d6fa3a98884bcf7b49deab88f0fadd7a99040c006d.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/b51bcd5817ffa23242f141d6fa3a98884bcf7b49deab88f0fadd7a99040c006d.jpg)

![b5aa98f32895f39a3963f9a7fba5fec22bd16cc1fb2651fcb4d58f81796c110d.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/b5aa98f32895f39a3963f9a7fba5fec22bd16cc1fb2651fcb4d58f81796c110d.jpg)

![c11450f5ab775e3a3ff445a5bb699017eedb72d83919bd8e420f3d12404491a6.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/c11450f5ab775e3a3ff445a5bb699017eedb72d83919bd8e420f3d12404491a6.jpg)

![d6a74f42e37a275a99e0143b699ba124b76b5fbd1412cc181c2676430b3433a7.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/d6a74f42e37a275a99e0143b699ba124b76b5fbd1412cc181c2676430b3433a7.jpg)

![e27bc1582606ff2e3ec5ce246e97adf4541a641271bf0e588853eeed3fa7348d.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/e27bc1582606ff2e3ec5ce246e97adf4541a641271bf0e588853eeed3fa7348d.jpg)

![eb6c6f9fce8bfd4b035080cde2b3816af07e2efd98bac54079ae307e28bcf657.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/eb6c6f9fce8bfd4b035080cde2b3816af07e2efd98bac54079ae307e28bcf657.jpg)

![ee838cd9832fda5da0a3e5b03a4cc7631e38782535a83db5d2af4d9c0ba5828e.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/ee838cd9832fda5da0a3e5b03a4cc7631e38782535a83db5d2af4d9c0ba5828e.jpg)

![eefc7a44aafa0d857ec22132e4c871f3fcdcda3037cd2698de0f4a5e7e61217e.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/eefc7a44aafa0d857ec22132e4c871f3fcdcda3037cd2698de0f4a5e7e61217e.jpg)

![f889f0cfd1b05ab2d84c29486fae252298d2303b27e1cf8119ba292640e2dcde.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/f889f0cfd1b05ab2d84c29486fae252298d2303b27e1cf8119ba292640e2dcde.jpg)

![f9a244d40d81ecb567b24bd42994a57628648d4e8116508be05dfa11c384ddfe.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/images/f9a244d40d81ecb567b24bd42994a57628648d4e8116508be05dfa11c384ddfe.jpg)

### Tables

![dbbec2dcc37fb6077f17c5bbed6e69bbe00236a25dab07f827d0eb64e49e12b3.jpg](../iclr_results/507_Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences/tables/dbbec2dcc37fb6077f17c5bbed6e69bbe00236a25dab07f827d0eb64e49e12b3.jpg)

## Scaling FP8 training to trillion-token LLMs


### Images

![3466b178eb6001776654f61470138cfbef59f8336f9ff2e347beeedd4d17b09d.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/images/3466b178eb6001776654f61470138cfbef59f8336f9ff2e347beeedd4d17b09d.jpg)

![3e303ce08c7c3bd3c49449d67c21d7b28300a4efe1f739c475a3e60e1f9ff0a8.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/images/3e303ce08c7c3bd3c49449d67c21d7b28300a4efe1f739c475a3e60e1f9ff0a8.jpg)

![463313286080cdf9484c03f6c364c0ea76e96c5c735f6f7e86beba8757029b49.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/images/463313286080cdf9484c03f6c364c0ea76e96c5c735f6f7e86beba8757029b49.jpg)

![5a05c80a55b5fcc3e549f0009e38cc8a30281713a3e4919488e3badf1ca6e4bd.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/images/5a05c80a55b5fcc3e549f0009e38cc8a30281713a3e4919488e3badf1ca6e4bd.jpg)

![f33133a344ede71c26dee218e10140a420667d93e9000f559f920047f10c42cc.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/images/f33133a344ede71c26dee218e10140a420667d93e9000f559f920047f10c42cc.jpg)

![f76c6e2db0870746eb54da8824d7024ca466071abaab5425bf331d6517129cde.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/images/f76c6e2db0870746eb54da8824d7024ca466071abaab5425bf331d6517129cde.jpg)

### Tables

![60e9d4d75fa650c6b857999b09fd02d517c3d4510dc14bc6831c113559c09880.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/tables/60e9d4d75fa650c6b857999b09fd02d517c3d4510dc14bc6831c113559c09880.jpg)

![674ad94d7c68890860647c03d8e61ecb565cd612918aed8db696b1f5eb6ac861.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/tables/674ad94d7c68890860647c03d8e61ecb565cd612918aed8db696b1f5eb6ac861.jpg)

![800c25c0542680cbf9f38bb20f66837508037802720508b7a385b779539da09b.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/tables/800c25c0542680cbf9f38bb20f66837508037802720508b7a385b779539da09b.jpg)

![d52eb1cd68be7c259471ee41231227268f39ffea50571f1a708aad6684d7a190.jpg](../iclr_results/508_Scaling FP8 training to trillion-token LLMs/tables/d52eb1cd68be7c259471ee41231227268f39ffea50571f1a708aad6684d7a190.jpg)

## Student-Informed Teacher Training


### Images

![0b105145c20037cfc3c07347e70d1b449a99d7bcb67ef78134a305590814be8f.jpg](../iclr_results/509_Student-Informed Teacher Training/images/0b105145c20037cfc3c07347e70d1b449a99d7bcb67ef78134a305590814be8f.jpg)

![2cd248d4c5a974d9c83179ff94504ab34d4a84824cad3cf75b50592e8a058e98.jpg](../iclr_results/509_Student-Informed Teacher Training/images/2cd248d4c5a974d9c83179ff94504ab34d4a84824cad3cf75b50592e8a058e98.jpg)

![695c9f79684d23ffc88a2b780ffacbdcb52d8d3173032533765fc3170d080e77.jpg](../iclr_results/509_Student-Informed Teacher Training/images/695c9f79684d23ffc88a2b780ffacbdcb52d8d3173032533765fc3170d080e77.jpg)

![b2a934002a11ce867f9d6ff037038bf3b97b493f6312c229c1387ab9ad168b2b.jpg](../iclr_results/509_Student-Informed Teacher Training/images/b2a934002a11ce867f9d6ff037038bf3b97b493f6312c229c1387ab9ad168b2b.jpg)

![bda6a8c556c6874d98d85a313fc60d9c5c52404958bb99e266202729d861f430.jpg](../iclr_results/509_Student-Informed Teacher Training/images/bda6a8c556c6874d98d85a313fc60d9c5c52404958bb99e266202729d861f430.jpg)

![c2cc97a8dadc9f3a3a80b08922ce21d81497b7967fe11809da1e58e37ad1fdfe.jpg](../iclr_results/509_Student-Informed Teacher Training/images/c2cc97a8dadc9f3a3a80b08922ce21d81497b7967fe11809da1e58e37ad1fdfe.jpg)

### Tables

![36bff39f5613604a60f4e615c76ee002ebabdcd69565a118ea640f749c3ef968.jpg](../iclr_results/509_Student-Informed Teacher Training/tables/36bff39f5613604a60f4e615c76ee002ebabdcd69565a118ea640f749c3ef968.jpg)

![4deced1c59336ab0abd38ecdffb3c1e316d75b5ec2b031f982db5a8e49a6ee30.jpg](../iclr_results/509_Student-Informed Teacher Training/tables/4deced1c59336ab0abd38ecdffb3c1e316d75b5ec2b031f982db5a8e49a6ee30.jpg)

![4f58f948906beebef1f03c48ad2b9e350c0f0386d0a0994653bf6a2e6a8fe0ec.jpg](../iclr_results/509_Student-Informed Teacher Training/tables/4f58f948906beebef1f03c48ad2b9e350c0f0386d0a0994653bf6a2e6a8fe0ec.jpg)

![6b04217549d2d73c8fc51254a49ccc00eb8a9f5863b20fa9a2bccf48ff805ffe.jpg](../iclr_results/509_Student-Informed Teacher Training/tables/6b04217549d2d73c8fc51254a49ccc00eb8a9f5863b20fa9a2bccf48ff805ffe.jpg)

![b26055c03c1be4d31da6a78aa40acd06e95b9c05c9cf5a6daba11e6d950f6b4b.jpg](../iclr_results/509_Student-Informed Teacher Training/tables/b26055c03c1be4d31da6a78aa40acd06e95b9c05c9cf5a6daba11e6d950f6b4b.jpg)

## Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation


### Images

![0cddb0170a49667bc4f14d27f7ce38223891af49f68586c206def8e8a1ccb623.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/0cddb0170a49667bc4f14d27f7ce38223891af49f68586c206def8e8a1ccb623.jpg)

![17d4f0945e7e57dccebcda9c8461e243f24813943f999b4e8b1b3059c1ef03b6.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/17d4f0945e7e57dccebcda9c8461e243f24813943f999b4e8b1b3059c1ef03b6.jpg)

![29c818edead1c510bda573e829adf99998018d98adba36e7ed06b65c49a63b28.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/29c818edead1c510bda573e829adf99998018d98adba36e7ed06b65c49a63b28.jpg)

![3b73554561ebf6919890773b77618649a52b389377a794bdbcc0bb2a21bd810e.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/3b73554561ebf6919890773b77618649a52b389377a794bdbcc0bb2a21bd810e.jpg)

![43a6ab488029b80765dc5d8a8b7e53cd5f2f9054ac40cd587513e9f9d28dc3c4.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/43a6ab488029b80765dc5d8a8b7e53cd5f2f9054ac40cd587513e9f9d28dc3c4.jpg)

![4af72fd83485543ee97b9e1e3d3c66880c36568720a1ea124099ff598a3ee05f.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/4af72fd83485543ee97b9e1e3d3c66880c36568720a1ea124099ff598a3ee05f.jpg)

![5357053df200f7effcdc67f894f37a6dc24526a825b43bd479a601c9bec386cd.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/5357053df200f7effcdc67f894f37a6dc24526a825b43bd479a601c9bec386cd.jpg)

![7434757ae52d8aafbf7149e7da87bcc9e57b405086f602a2949252659f13871d.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/7434757ae52d8aafbf7149e7da87bcc9e57b405086f602a2949252659f13871d.jpg)

![918d8546ede807596380ebf0160959e3cfdb560536e5ecba6bcbac55b2e4d24a.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/918d8546ede807596380ebf0160959e3cfdb560536e5ecba6bcbac55b2e4d24a.jpg)

![9438c09e491a78438502bd060c8b7d57876f933ad544f9c3e6f5d28e5dd156ec.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/9438c09e491a78438502bd060c8b7d57876f933ad544f9c3e6f5d28e5dd156ec.jpg)

![e7c83ed36c1fd4971b12e3377596cea99eba744dd0e308ee50b771199c0c6a4e.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/e7c83ed36c1fd4971b12e3377596cea99eba744dd0e308ee50b771199c0c6a4e.jpg)

![f609872b4dba6b694acceebbd0c5d1f48c321e517b693f17e795eec2f32191b8.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/f609872b4dba6b694acceebbd0c5d1f48c321e517b693f17e795eec2f32191b8.jpg)

![feb26b85a238aeaee88deaa9442645fe07cb65b735f27864bc32c79b76976962.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/images/feb26b85a238aeaee88deaa9442645fe07cb65b735f27864bc32c79b76976962.jpg)

### Tables

![02c83c0b51dd454bdb0ca7091fe0c36a9879a728696b55a9484a3b680b79890c.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/02c83c0b51dd454bdb0ca7091fe0c36a9879a728696b55a9484a3b680b79890c.jpg)

![0bade9c0f8ea249c72e8d3e0c858c631e43b2f64ed5bfc632fc2688ebf75a223.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/0bade9c0f8ea249c72e8d3e0c858c631e43b2f64ed5bfc632fc2688ebf75a223.jpg)

![15294e4436c8e07d990ce3f448a39ed63437a758d747e75f9d83d1788e083a39.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/15294e4436c8e07d990ce3f448a39ed63437a758d747e75f9d83d1788e083a39.jpg)

![24931f8eb07c0301eea452ea8d75d1cff6aeaeb6d6c36ab00c55e413b8c55884.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/24931f8eb07c0301eea452ea8d75d1cff6aeaeb6d6c36ab00c55e413b8c55884.jpg)

![2af7c73e0656dec1e1a6cb05b3cca5b9e1640296c11a5f282e8bf8695a21aa7c.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/2af7c73e0656dec1e1a6cb05b3cca5b9e1640296c11a5f282e8bf8695a21aa7c.jpg)

![3d642ccb0ffeb6bc7286aac9a6de4ca413fed25521bc0b075f9a2a2a192614c7.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/3d642ccb0ffeb6bc7286aac9a6de4ca413fed25521bc0b075f9a2a2a192614c7.jpg)

![4baa969fd90cd422fe5bd7d0dc1aa2a191cd01c9097f6a444a27285a63295f8b.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/4baa969fd90cd422fe5bd7d0dc1aa2a191cd01c9097f6a444a27285a63295f8b.jpg)

![51d62b51587e9c52431fbf9c0737ea823722798cad40c6e9c54d8652c83e3d16.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/51d62b51587e9c52431fbf9c0737ea823722798cad40c6e9c54d8652c83e3d16.jpg)

![5f07080eee8d75256c82081b411eaa383fd7d16ea192962d1540e1457322cd60.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/5f07080eee8d75256c82081b411eaa383fd7d16ea192962d1540e1457322cd60.jpg)

![60e155ccea3be3f4980e76a7afcf45985f224c01e23e1db9061485128e6c16b1.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/60e155ccea3be3f4980e76a7afcf45985f224c01e23e1db9061485128e6c16b1.jpg)

![708d0d80c573cb4632cd6f7c1a4c997fd580d8328950d34b9d12e6ddebbe502f.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/708d0d80c573cb4632cd6f7c1a4c997fd580d8328950d34b9d12e6ddebbe502f.jpg)

![7456301aa4ca96ff72419656bf1e309623cd332498ffb033a791b9794c77900f.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/7456301aa4ca96ff72419656bf1e309623cd332498ffb033a791b9794c77900f.jpg)

![9c9ee2a69469b989b61307e5a57831d7f54f513e210074399671b265352036f8.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/9c9ee2a69469b989b61307e5a57831d7f54f513e210074399671b265352036f8.jpg)

![d515be3ae3e40de126cec6bbc58397613f6183e824deb142626dea460a7b32cc.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/d515be3ae3e40de126cec6bbc58397613f6183e824deb142626dea460a7b32cc.jpg)

![dc8a6cceaaab78324abb4267b56c44417fcdfa5aeecf30a1bd118b4c82e91371.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/dc8a6cceaaab78324abb4267b56c44417fcdfa5aeecf30a1bd118b4c82e91371.jpg)

![dded33c694314763fbba65c88b999b5df75ce6858d057c829e52159eb9f7a267.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/dded33c694314763fbba65c88b999b5df75ce6858d057c829e52159eb9f7a267.jpg)

![ef9eaa676304b5ef309ce8378ceee3f13154179c62f2ddd116d8fa510dcc26be.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/ef9eaa676304b5ef309ce8378ceee3f13154179c62f2ddd116d8fa510dcc26be.jpg)

![f7cc1340a00d0a6307d48d89a022ebe11f209cadbf7de9267d3ffacd79247468.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/f7cc1340a00d0a6307d48d89a022ebe11f209cadbf7de9267d3ffacd79247468.jpg)

![f863115401fa6662d62177718ccf463affdf00edb7e0d2d130c442b5dc613865.jpg](../iclr_results/510_Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation/tables/f863115401fa6662d62177718ccf463affdf00edb7e0d2d130c442b5dc613865.jpg)

## Estimating the Probabilities of Rare Outputs in Language Models


### Images

![20e5499ac94fedaeea2174501f7ed8e1d3c90b114d57e004794aeb279fa0cd47.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/20e5499ac94fedaeea2174501f7ed8e1d3c90b114d57e004794aeb279fa0cd47.jpg)

![4ef31921e5f9f1a57955a761f7c99634b5375d5d67a7013c748b881405699967.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/4ef31921e5f9f1a57955a761f7c99634b5375d5d67a7013c748b881405699967.jpg)

![5696daa49709c480bd2092fea87a7b8eefaf0544c4393ebe56b1ea9d78c0a72e.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/5696daa49709c480bd2092fea87a7b8eefaf0544c4393ebe56b1ea9d78c0a72e.jpg)

![5954ed511bfce184fc5830801fd6dbed593f3cebe2f293c6d339f489b00acc2e.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/5954ed511bfce184fc5830801fd6dbed593f3cebe2f293c6d339f489b00acc2e.jpg)

![5ed936dffa6a51ba854e3b72d4b2c059dca37be5fc78aafd73ec631d5312acb4.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/5ed936dffa6a51ba854e3b72d4b2c059dca37be5fc78aafd73ec631d5312acb4.jpg)

![6e7d2e863891300188cef1cd1c9688997472dc26601dcbbad227ef01db1cd854.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/6e7d2e863891300188cef1cd1c9688997472dc26601dcbbad227ef01db1cd854.jpg)

![94db847ae4b655c5e0cab30ccc3e398a3794faa436f716ba42ce7b47a82117d8.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/94db847ae4b655c5e0cab30ccc3e398a3794faa436f716ba42ce7b47a82117d8.jpg)

![d4a62eeae1d388666a8a0c5586c87c03c876c76111305308fb5f4e07ad5bafce.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/d4a62eeae1d388666a8a0c5586c87c03c876c76111305308fb5f4e07ad5bafce.jpg)

![df5ea96b5d27d025745953e51924570887d80d2f0fa653dfe934677f7c1de343.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/images/df5ea96b5d27d025745953e51924570887d80d2f0fa653dfe934677f7c1de343.jpg)

### Tables

![022e64789f32971ef7e70586b3a91580411f0227aeaf2841f8cc0918aaf771ba.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/022e64789f32971ef7e70586b3a91580411f0227aeaf2841f8cc0918aaf771ba.jpg)

![2a8f64e1852b39e2feaaa6ee4b0985b0e5b8274732911502de6342fb69aa808e.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/2a8f64e1852b39e2feaaa6ee4b0985b0e5b8274732911502de6342fb69aa808e.jpg)

![4f6d1b8e6b49d9ea7c84b88db8eeb85ca63fb38835e0896d5ff786ed1eb32521.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/4f6d1b8e6b49d9ea7c84b88db8eeb85ca63fb38835e0896d5ff786ed1eb32521.jpg)

![50e10b1dedf7d05442262b57e2aa38a230099a9292a4895c37ca1e5e94bb8e0a.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/50e10b1dedf7d05442262b57e2aa38a230099a9292a4895c37ca1e5e94bb8e0a.jpg)

![554c35038b86c57eb6080e27dbafef8d1c2b0a5370e8259449d3413dd2061860.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/554c35038b86c57eb6080e27dbafef8d1c2b0a5370e8259449d3413dd2061860.jpg)

![73cff8620f8f68b71d83cbf3bfc6f8692eab5103b6c100859378b1326f09a458.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/73cff8620f8f68b71d83cbf3bfc6f8692eab5103b6c100859378b1326f09a458.jpg)

![7cf4db2523990d4d48b16bf1aefe10581b006b62542d4c0697e4bd3ab3070d9b.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/7cf4db2523990d4d48b16bf1aefe10581b006b62542d4c0697e4bd3ab3070d9b.jpg)

![9d9a8a83883c61004e7a3673b6476322a855c3c87487f4f7dc1efcb8fbe1c6a8.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/9d9a8a83883c61004e7a3673b6476322a855c3c87487f4f7dc1efcb8fbe1c6a8.jpg)

![e39b2d2a2e7ebf47b5480d37f43297491965e2816acfdcc0d31b086c58395360.jpg](../iclr_results/511_Estimating the Probabilities of Rare Outputs in Language Models/tables/e39b2d2a2e7ebf47b5480d37f43297491965e2816acfdcc0d31b086c58395360.jpg)

## Imputation for prediction: beware of diminishing returns.


### Images

![0c96cb9ed76fa7987202a016f9482741940ecc0293f9466d126e5eb5983f2329.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/0c96cb9ed76fa7987202a016f9482741940ecc0293f9466d126e5eb5983f2329.jpg)

![10aea38563ae5c6653f5bda82726859c56d14376fc634453db3d625de7cd1263.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/10aea38563ae5c6653f5bda82726859c56d14376fc634453db3d625de7cd1263.jpg)

![12513472778432b39af7290a88e817c4b9e1102a0d9b0bbdb032f73333c6a6f2.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/12513472778432b39af7290a88e817c4b9e1102a0d9b0bbdb032f73333c6a6f2.jpg)

![1565fd30fdc6dc844756b0142d309a62adee625f725484b1d38c5324aa31bbd9.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/1565fd30fdc6dc844756b0142d309a62adee625f725484b1d38c5324aa31bbd9.jpg)

![2d2c266d6cfd4dbe026a9731a64a096a2636fb6145513670d9354e1226cbd92e.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/2d2c266d6cfd4dbe026a9731a64a096a2636fb6145513670d9354e1226cbd92e.jpg)

![379c9cca97a8ef698b05e08590ae426a11745254e62bf177826806bd8cdf165b.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/379c9cca97a8ef698b05e08590ae426a11745254e62bf177826806bd8cdf165b.jpg)

![3b3dcabb5e8329e8181c6fdecc3e07a7d13edfcec6d452163119097ada0cf744.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/3b3dcabb5e8329e8181c6fdecc3e07a7d13edfcec6d452163119097ada0cf744.jpg)

![428ed3b5dd5395487ef5d8a3de782c18e17dc378ac950ca56c09fa86096feda9.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/428ed3b5dd5395487ef5d8a3de782c18e17dc378ac950ca56c09fa86096feda9.jpg)

![45fc9c94448260ffbd8667020a159ec40f8ab21d0c5458bdf80b2b3d22fd43b4.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/45fc9c94448260ffbd8667020a159ec40f8ab21d0c5458bdf80b2b3d22fd43b4.jpg)

![483e79d80de41b8a724b38c10273e87592131f06e765a6e7274a3b1d8bec36f2.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/483e79d80de41b8a724b38c10273e87592131f06e765a6e7274a3b1d8bec36f2.jpg)

![55e55990276f44c80f9ef1a737f44ce9d4504ce228e5d9cfdc79584c59da945e.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/55e55990276f44c80f9ef1a737f44ce9d4504ce228e5d9cfdc79584c59da945e.jpg)

![5a9a23947cc37b6e1a16a2f317d0b43ef3eadb67c8cdcdf147116c62deb9194a.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/5a9a23947cc37b6e1a16a2f317d0b43ef3eadb67c8cdcdf147116c62deb9194a.jpg)

![5b9706ff5e52b35d3c836e963bff144be5358286f965e91823210b58c42cbe99.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/5b9706ff5e52b35d3c836e963bff144be5358286f965e91823210b58c42cbe99.jpg)

![5e4a98d323c1434e3c1821072e8a96940d65379a8d47930072b8ef7009a1f6e4.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/5e4a98d323c1434e3c1821072e8a96940d65379a8d47930072b8ef7009a1f6e4.jpg)

![68d7f50b5f8f560eafb7f5131396aa7a243b8b3dae63d25f07bad0a3b11e3e66.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/68d7f50b5f8f560eafb7f5131396aa7a243b8b3dae63d25f07bad0a3b11e3e66.jpg)

![7597f8c02c4b18045d296be18ba26d1a8e14a3dd24c7a5e00c69e3968feacdf2.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/7597f8c02c4b18045d296be18ba26d1a8e14a3dd24c7a5e00c69e3968feacdf2.jpg)

![837917b3b4a2cf710b8b2e24673f45423335df47e9f0e87a7af047bc091c4104.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/837917b3b4a2cf710b8b2e24673f45423335df47e9f0e87a7af047bc091c4104.jpg)

![842adc33940a5b3f56e7fba195db0717bbdd1f759ef4c8ccd36fc278a3a247f0.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/842adc33940a5b3f56e7fba195db0717bbdd1f759ef4c8ccd36fc278a3a247f0.jpg)

![9142a9ebcd7ec43ca3a514479bf2f936e7f10a56d03f20c537328ff151b2f79c.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/9142a9ebcd7ec43ca3a514479bf2f936e7f10a56d03f20c537328ff151b2f79c.jpg)

![91f4bda8e67aa03daf9bb1c485f9c78187f792c896b17fd444c25921f01ff5f2.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/91f4bda8e67aa03daf9bb1c485f9c78187f792c896b17fd444c25921f01ff5f2.jpg)

![9650d9e3cd1ed0cf07fcc948e4dd37c80b2c10e8fd50304084e0d36578c334dc.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/9650d9e3cd1ed0cf07fcc948e4dd37c80b2c10e8fd50304084e0d36578c334dc.jpg)

![9bdbe6fc590aa7ac7cc963c49677609bc2df869c8defb354f513effc3784bd34.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/9bdbe6fc590aa7ac7cc963c49677609bc2df869c8defb354f513effc3784bd34.jpg)

![9eb51fb100eeb08fc5b41b8bf72e8f811ec0c33ff58e34148a8d22b86d48db3d.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/9eb51fb100eeb08fc5b41b8bf72e8f811ec0c33ff58e34148a8d22b86d48db3d.jpg)

![a513a646c835818112a61f21002869fe100404797f40cabcc34cbec12fb598ec.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/a513a646c835818112a61f21002869fe100404797f40cabcc34cbec12fb598ec.jpg)

![a5b885ed4ed5506a6a2feca81c51eeb735b4ae3c138c8990a0d689a96f02ed68.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/a5b885ed4ed5506a6a2feca81c51eeb735b4ae3c138c8990a0d689a96f02ed68.jpg)

![a5d317a9c23d07ea04c21dccfa7c1889be0f346147da80dcd4fb8cf39fe0466c.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/a5d317a9c23d07ea04c21dccfa7c1889be0f346147da80dcd4fb8cf39fe0466c.jpg)

![ad0bcb4dfab31e8d7211cde63752c99ca903e4457ccc775e932b16d651a3b808.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/ad0bcb4dfab31e8d7211cde63752c99ca903e4457ccc775e932b16d651a3b808.jpg)

![b21e5ab8a14f11d8485be2c7c388c9a56c5713913260669f49b25f12827d90d1.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/b21e5ab8a14f11d8485be2c7c388c9a56c5713913260669f49b25f12827d90d1.jpg)

![bc190209f97eb25caefc64b66c238aa2c764c3ce9751445f4a03c992935ff954.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/bc190209f97eb25caefc64b66c238aa2c764c3ce9751445f4a03c992935ff954.jpg)

![caebf38d29aaf4a962c1f67ee5cc33a1dbddb863124213f4d1ed8563ef8382a9.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/caebf38d29aaf4a962c1f67ee5cc33a1dbddb863124213f4d1ed8563ef8382a9.jpg)

![d0c1e562982e38e04724d55fd3e253fa12ce72ea2e8a168cb6fae8bd787dbeae.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/d0c1e562982e38e04724d55fd3e253fa12ce72ea2e8a168cb6fae8bd787dbeae.jpg)

![db4ed6b01d06aa032f765ae27eb8bcd4055dc3343357e5c8d0d71f46d85d9322.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/db4ed6b01d06aa032f765ae27eb8bcd4055dc3343357e5c8d0d71f46d85d9322.jpg)

![e4ff60eaca0401e389f06a86c753e5947fed429fad113a5e60895228240df4b6.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/e4ff60eaca0401e389f06a86c753e5947fed429fad113a5e60895228240df4b6.jpg)

![ec008ee1bfb53735cbd5ac9180ccf0c25636d31ffa2aa84bc774e92102c5fdd9.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/ec008ee1bfb53735cbd5ac9180ccf0c25636d31ffa2aa84bc774e92102c5fdd9.jpg)

![f209f7353d6c96bc05b7c83c8c1d02f4a0c3aece69a41b442145d8f11b802899.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./images/f209f7353d6c96bc05b7c83c8c1d02f4a0c3aece69a41b442145d8f11b802899.jpg)

### Tables

![27f22959733718bcea56305942aaa6268724b475108dc50f0c4e84e427ed34ec.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./tables/27f22959733718bcea56305942aaa6268724b475108dc50f0c4e84e427ed34ec.jpg)

![dc739526692973eea2c25a0cdbca377caad9eed48a636352e0ba077a1a368cec.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./tables/dc739526692973eea2c25a0cdbca377caad9eed48a636352e0ba077a1a368cec.jpg)

![ea1f9f1999b2d8d9b55f8c48c8fc4e3f178c0446b70fec8b3f5ffb8f9368ebff.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./tables/ea1f9f1999b2d8d9b55f8c48c8fc4e3f178c0446b70fec8b3f5ffb8f9368ebff.jpg)

![f48e5672fa46ca2440dd487939d30bb17307fa952503e7e57c105bfe8305e2e1.jpg](../iclr_results/512_Imputation for prediction_ beware of diminishing returns./tables/f48e5672fa46ca2440dd487939d30bb17307fa952503e7e57c105bfe8305e2e1.jpg)

## Physics-aligned field reconstruction with diffusion bridge


### Images

![395c730110d0eb2d70b260c6e94f8802d0cc9c35b639410f8f377d6a216e371e.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/395c730110d0eb2d70b260c6e94f8802d0cc9c35b639410f8f377d6a216e371e.jpg)

![4168a4920534818b79f2f2d7ad94e1196de68ea644e0f075338a1ccd392dba08.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/4168a4920534818b79f2f2d7ad94e1196de68ea644e0f075338a1ccd392dba08.jpg)

![43a35655e7bd55a752a52e81ddf9e17c574b425623774981d996b6c6b750f26f.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/43a35655e7bd55a752a52e81ddf9e17c574b425623774981d996b6c6b750f26f.jpg)

![48e5697f68b2a7fb5f89fa7747d743fcec8cd764638805a283ca8f6bd6bbb3b4.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/48e5697f68b2a7fb5f89fa7747d743fcec8cd764638805a283ca8f6bd6bbb3b4.jpg)

![5400238dd6edd8542784a2a91189878330fc0a40f02527da954cfd1c5776584d.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/5400238dd6edd8542784a2a91189878330fc0a40f02527da954cfd1c5776584d.jpg)

![58cd30bb50823a3ab1a1ed91ef3944a0d7d7d53f6498aea0308ba7bddd963142.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/58cd30bb50823a3ab1a1ed91ef3944a0d7d7d53f6498aea0308ba7bddd963142.jpg)

![5cd46633e5f81ae1da8d825b2774773a21518e50f5a02201a0a08370adfc1025.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/5cd46633e5f81ae1da8d825b2774773a21518e50f5a02201a0a08370adfc1025.jpg)

![5f343a637eb122e31bac1ef286f5f79000b4da6e8df5e5d7e10404b22200edbf.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/5f343a637eb122e31bac1ef286f5f79000b4da6e8df5e5d7e10404b22200edbf.jpg)

![652907df504edc954c28a9e681e05a945e3073d122c20717870cc8eae1627232.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/652907df504edc954c28a9e681e05a945e3073d122c20717870cc8eae1627232.jpg)

![6c80d36efc8bb1f3afbdcda2ac4783bb6b250607b537fc4d2627f03f94f822e8.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/6c80d36efc8bb1f3afbdcda2ac4783bb6b250607b537fc4d2627f03f94f822e8.jpg)

![732e0647a18ab5ea0c4ea1ca2c96730296b3b429944d0bf8dfd1b7b221b4a51d.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/732e0647a18ab5ea0c4ea1ca2c96730296b3b429944d0bf8dfd1b7b221b4a51d.jpg)

![7aa772e35dea91e29a0bedd878e5a135b13a51ed6f56d2ef06cc0b257a0b21b0.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/7aa772e35dea91e29a0bedd878e5a135b13a51ed6f56d2ef06cc0b257a0b21b0.jpg)

![7b4d1e101329852321152aef94a8dbd575b7be90878615dab26c21b2497568bb.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/7b4d1e101329852321152aef94a8dbd575b7be90878615dab26c21b2497568bb.jpg)

![7dbd3f84919ad9001f65cd90512b6e0dca6148d7f0d67fa46ee8d88b114db779.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/7dbd3f84919ad9001f65cd90512b6e0dca6148d7f0d67fa46ee8d88b114db779.jpg)

![7f35651e0e19b8c91b8c7b319a157ff58fb075d26173184519ea358f5617eaf1.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/7f35651e0e19b8c91b8c7b319a157ff58fb075d26173184519ea358f5617eaf1.jpg)

![abc2e184193167f98f6abe556aca5f023de9fa033687c45cc5c038eae4d4ae11.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/abc2e184193167f98f6abe556aca5f023de9fa033687c45cc5c038eae4d4ae11.jpg)

![d640d34b1eeec41b9d9a10a6f1c590f75795895778f4562e18fd6b913fccb233.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/d640d34b1eeec41b9d9a10a6f1c590f75795895778f4562e18fd6b913fccb233.jpg)

![db6400f0d18237cab0569de02771cc1088e85276e2814390cae323be60e7ce97.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/db6400f0d18237cab0569de02771cc1088e85276e2814390cae323be60e7ce97.jpg)

![e081c46bd9f270dfe6d2f85bcc52641b7b35c7b62472c39f09c59ee60912a81b.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/e081c46bd9f270dfe6d2f85bcc52641b7b35c7b62472c39f09c59ee60912a81b.jpg)

![e4cad1261cb38ee7ef7fce7056a16965aa364bb37dca078987d1b8731042904c.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/e4cad1261cb38ee7ef7fce7056a16965aa364bb37dca078987d1b8731042904c.jpg)

![e5fba90d7975cd8b1a3ea09b999ea1e3d054b4501c0079d901eab71f25595322.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/images/e5fba90d7975cd8b1a3ea09b999ea1e3d054b4501c0079d901eab71f25595322.jpg)

### Tables

![0165227338a81f4a407d360000c7408244e9a50d59110eb103ea5edd2ad0217e.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/0165227338a81f4a407d360000c7408244e9a50d59110eb103ea5edd2ad0217e.jpg)

![06cd0ac9e3fac67d429cf9b441493c52394a21387bcd65dbf033f33f6cab8625.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/06cd0ac9e3fac67d429cf9b441493c52394a21387bcd65dbf033f33f6cab8625.jpg)

![10df34e315a9e1829af6c6b280cd7fca43386f1316a6e2c446e76e3048a61afd.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/10df34e315a9e1829af6c6b280cd7fca43386f1316a6e2c446e76e3048a61afd.jpg)

![2f16f02d3024e9da3b977eb967bb55df40594e886f82d696842001e5e15646f2.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/2f16f02d3024e9da3b977eb967bb55df40594e886f82d696842001e5e15646f2.jpg)

![549d72674f99c503164bfc3498d74ea30699f07bcf67ff740c61c479b286a46f.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/549d72674f99c503164bfc3498d74ea30699f07bcf67ff740c61c479b286a46f.jpg)

![5ef18af4d811649964c75a4886bdfb5a0d40e4db526a438877ff14d7600c1ea5.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/5ef18af4d811649964c75a4886bdfb5a0d40e4db526a438877ff14d7600c1ea5.jpg)

![9e639918f6d3139c0b6606e8efc1207ca964694c4814a5ddbc27e1eae011bc8c.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/9e639918f6d3139c0b6606e8efc1207ca964694c4814a5ddbc27e1eae011bc8c.jpg)

![a20b221d96e20e5729e577aa90bc93b04f962dff1ac32ce9ede78c77ca849c83.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/a20b221d96e20e5729e577aa90bc93b04f962dff1ac32ce9ede78c77ca849c83.jpg)

![c45de621857230dfa09a51a3e7ca9e302ee978e11fb78b56569efbd02f9e1795.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/c45de621857230dfa09a51a3e7ca9e302ee978e11fb78b56569efbd02f9e1795.jpg)

![e21016669c35ea2e54f88d3b146a050046c865c7602973914394caad02c9e5ed.jpg](../iclr_results/513_Physics-aligned field reconstruction with diffusion bridge/tables/e21016669c35ea2e54f88d3b146a050046c865c7602973914394caad02c9e5ed.jpg)

## Rethinking Reward Model Evaluation: Are We Barking up the Wrong Tree?


### Images

![013ff1c10c9d9429107311c96c7ff0d2562654aba6c60750e06890eb7df2bd76.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/013ff1c10c9d9429107311c96c7ff0d2562654aba6c60750e06890eb7df2bd76.jpg)

![0bfc287996f9d5a6387577fd1f15140715db6cdf2776455150ed285122fad7a3.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/0bfc287996f9d5a6387577fd1f15140715db6cdf2776455150ed285122fad7a3.jpg)

![0f32361ef0be01c34cfdbda7d91d675acfbeaa078ce46a6099e3d4ea47260093.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/0f32361ef0be01c34cfdbda7d91d675acfbeaa078ce46a6099e3d4ea47260093.jpg)

![28938ed298e936f834b92cda0c4080383a5e3b7b8aa09644039db1f2de0a8fba.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/28938ed298e936f834b92cda0c4080383a5e3b7b8aa09644039db1f2de0a8fba.jpg)

![30b46fada41bc5b7b6697ff8088796fc83090f6f98b6029f9d557dd33dbddbc3.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/30b46fada41bc5b7b6697ff8088796fc83090f6f98b6029f9d557dd33dbddbc3.jpg)

![558f9788b7b791e45907f434bdb10c717f656e465606070839a6cd4744aedfb6.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/558f9788b7b791e45907f434bdb10c717f656e465606070839a6cd4744aedfb6.jpg)

![55c887793448988e36fd745cb8596251d860da1be7b1e3da1169320274959c88.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/55c887793448988e36fd745cb8596251d860da1be7b1e3da1169320274959c88.jpg)

![63ebbf33ba813a8efee6a3753b444e7ff3539abbaebe12cd12f9dfab9385bdaa.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/63ebbf33ba813a8efee6a3753b444e7ff3539abbaebe12cd12f9dfab9385bdaa.jpg)

![657cd7b8267311f5ebba370e65d4b5977469499902415d77214140c25a72feae.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/657cd7b8267311f5ebba370e65d4b5977469499902415d77214140c25a72feae.jpg)

![7869bd026e088710b6c7a1ba09fedf4d2f69a13a15315398dc3d5a0c47109be3.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/7869bd026e088710b6c7a1ba09fedf4d2f69a13a15315398dc3d5a0c47109be3.jpg)

![7be8e781d28226507a20a015b6022f455258513ec0fc5d14191ce1160d6ac209.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/7be8e781d28226507a20a015b6022f455258513ec0fc5d14191ce1160d6ac209.jpg)

![7e3288216b2930e8f1b5e5b93bef5bb2466467ddb29cf4dd03a1033a4f8b626d.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/7e3288216b2930e8f1b5e5b93bef5bb2466467ddb29cf4dd03a1033a4f8b626d.jpg)

![8d1333afe775f924c28d03c84c2e570d848156847d97ca554bedc272685fe863.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/8d1333afe775f924c28d03c84c2e570d848156847d97ca554bedc272685fe863.jpg)

![9f86652c98e1d53bc9fd79e650043c41d18e68cad73243f71f0a875e110f5b06.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/9f86652c98e1d53bc9fd79e650043c41d18e68cad73243f71f0a875e110f5b06.jpg)

![a62065f47a8626aa1f86753d2126cb0ae41e64035c56e71025653777ec95e7e3.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/a62065f47a8626aa1f86753d2126cb0ae41e64035c56e71025653777ec95e7e3.jpg)

![b089c9151c93f4c068ade5a4cabd07b3ee648f1397290a109f9d042e717c4059.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/b089c9151c93f4c068ade5a4cabd07b3ee648f1397290a109f9d042e717c4059.jpg)

![b976dc9604c570351973adcdaf4dc137201310380643de2b22200f27245197e6.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/b976dc9604c570351973adcdaf4dc137201310380643de2b22200f27245197e6.jpg)

![e00adb19787357af7c794b9fc94c27839b350a149fd330dcd46b6cb2d9099d70.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/e00adb19787357af7c794b9fc94c27839b350a149fd330dcd46b6cb2d9099d70.jpg)

![e84e1d3338e81d24b1c1a5293244134079fd1240e81bf45ece6aaa2d1f14c158.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/e84e1d3338e81d24b1c1a5293244134079fd1240e81bf45ece6aaa2d1f14c158.jpg)

![f1430c6bc20c5ddb5337cdd91c04f6460b15e343c07ed420f3361190779c4471.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/images/f1430c6bc20c5ddb5337cdd91c04f6460b15e343c07ed420f3361190779c4471.jpg)

### Tables

![0a0986eae4a186aa4e2880844e31e385dfac687c1d69eba4796e3198a763157a.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/tables/0a0986eae4a186aa4e2880844e31e385dfac687c1d69eba4796e3198a763157a.jpg)

![1ce0ae0db25508529df1bdf6f138324d2cddc5ca47d051991ef12fa9d40c8b3c.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/tables/1ce0ae0db25508529df1bdf6f138324d2cddc5ca47d051991ef12fa9d40c8b3c.jpg)

![4bf3108fef54fbbe53b4fcfd444ec39843adce93a4ab88f6e67376e43af98b87.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/tables/4bf3108fef54fbbe53b4fcfd444ec39843adce93a4ab88f6e67376e43af98b87.jpg)

![68afbd8787181ba04bbc99c45292c4c8296f5ee8e578904db7e924d1aa049930.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/tables/68afbd8787181ba04bbc99c45292c4c8296f5ee8e578904db7e924d1aa049930.jpg)

![e3c38ec86d07d28d0dd0509fac7609ebd80808f4471d636aff94b29996bbf8e4.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/tables/e3c38ec86d07d28d0dd0509fac7609ebd80808f4471d636aff94b29996bbf8e4.jpg)

![eeeef5e90fcdd9ded199cc193f6454c5f4dbc5af6067312d8df0f223d29d8023.jpg](../iclr_results/514_Rethinking Reward Model Evaluation_ Are We Barking up the Wrong Tree_/tables/eeeef5e90fcdd9ded199cc193f6454c5f4dbc5af6067312d8df0f223d29d8023.jpg)

## $R^2$-Guard: Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning


### Images

![1dde6d7cde6111889a3f3f314f23978b811188d2b19620a0e81d777c8ff4ae57.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/images/1dde6d7cde6111889a3f3f314f23978b811188d2b19620a0e81d777c8ff4ae57.jpg)

![4b436a6aba0438cbda380e017f676d2d97b5eb2b4d990fb4d0e603303a2297d8.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/images/4b436a6aba0438cbda380e017f676d2d97b5eb2b4d990fb4d0e603303a2297d8.jpg)

![71889552f66a9328a469bfb6d7ee65ecfde865216f486cfdde7754436050597f.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/images/71889552f66a9328a469bfb6d7ee65ecfde865216f486cfdde7754436050597f.jpg)

![ee62c3392aad37e1d50ba72cdae9ad2ae65c682fc4cce419797c7f771ac8c818.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/images/ee62c3392aad37e1d50ba72cdae9ad2ae65c682fc4cce419797c7f771ac8c818.jpg)

![fc501a3136ebd19c7b4fe11227bf526e4cf55e761ebf10b5faae056750a5287b.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/images/fc501a3136ebd19c7b4fe11227bf526e4cf55e761ebf10b5faae056750a5287b.jpg)

### Tables

![08b40313f6d3e0ad9198a4e9023bd1cef26daea2fded8993d0350169d8b81776.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/08b40313f6d3e0ad9198a4e9023bd1cef26daea2fded8993d0350169d8b81776.jpg)

![0f38347b104cc4d3dc5d73f7bba1420b80933ce46b811eceec8b74f783392943.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/0f38347b104cc4d3dc5d73f7bba1420b80933ce46b811eceec8b74f783392943.jpg)

![126ea059e37ff88ee2b2b61063d43c78fa05dc089c3ae84421669f121913d847.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/126ea059e37ff88ee2b2b61063d43c78fa05dc089c3ae84421669f121913d847.jpg)

![31fb3eaceec454bac20b06b09eb19b5fefd6fb5c678c8ebc5f567184ace2c349.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/31fb3eaceec454bac20b06b09eb19b5fefd6fb5c678c8ebc5f567184ace2c349.jpg)

![566c77ec01e72f1b73ab7fff2d020dfeb5f39897a3f2c7f7275f046aa196d189.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/566c77ec01e72f1b73ab7fff2d020dfeb5f39897a3f2c7f7275f046aa196d189.jpg)

![830516658deb451eef2517416abb4be29a544be11a98d92776616916cf3b0350.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/830516658deb451eef2517416abb4be29a544be11a98d92776616916cf3b0350.jpg)

![a0f4e7b7b6ba241de28b6ab5bc82323161a48aa245b9caac2f49ea8f9d47a951.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/a0f4e7b7b6ba241de28b6ab5bc82323161a48aa245b9caac2f49ea8f9d47a951.jpg)

![b7cc43779fab0d6907742dbd678d4b4f39a40361d8d98e472d16222a21276822.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/b7cc43779fab0d6907742dbd678d4b4f39a40361d8d98e472d16222a21276822.jpg)

![c24e960feca4feb1593e1f47339b51b81bed764e49865701585324a3f90c3ef6.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/c24e960feca4feb1593e1f47339b51b81bed764e49865701585324a3f90c3ef6.jpg)

![ffba7c9ec706d809e4a0cf587a2079a17b87c88b85bd5430d05815ba91c46200.jpg](../iclr_results/515_$R^2$-Guard_ Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning/tables/ffba7c9ec706d809e4a0cf587a2079a17b87c88b85bd5430d05815ba91c46200.jpg)

## Test-time Adaptation for Cross-modal Retrieval with Query Shift


### Images

![14f8baef3dd2f9d717212e143b0a3bf0473c74e2e32618d2b7776c3d4dbdb5b3.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/14f8baef3dd2f9d717212e143b0a3bf0473c74e2e32618d2b7776c3d4dbdb5b3.jpg)

![1c49dffc0a692ba922e64c69669f958498589f9808247ddd80d0be5a56ee88af.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/1c49dffc0a692ba922e64c69669f958498589f9808247ddd80d0be5a56ee88af.jpg)

![1e7193ab9a70629c7d2fb059aab6699a6ce6eddb8e2f381b91eebd4005dd632e.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/1e7193ab9a70629c7d2fb059aab6699a6ce6eddb8e2f381b91eebd4005dd632e.jpg)

![1f748a804684feeb2facecb739b7d3c3dbbdb3494c3db113aaa7e723788ad969.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/1f748a804684feeb2facecb739b7d3c3dbbdb3494c3db113aaa7e723788ad969.jpg)

![1f9d07347bdee62a8448bf7d3a70288b2cf52eaf3be93097efbae4ab081ef908.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/1f9d07347bdee62a8448bf7d3a70288b2cf52eaf3be93097efbae4ab081ef908.jpg)

![453c2b11dd73da06057b38dfec9b64762ba47214d3d4105c4ebc3d0dd6a05675.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/453c2b11dd73da06057b38dfec9b64762ba47214d3d4105c4ebc3d0dd6a05675.jpg)

![73c53885a57e15426464f50c62e3e435d554febd9794a3be5daceca15e125e8f.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/73c53885a57e15426464f50c62e3e435d554febd9794a3be5daceca15e125e8f.jpg)

![75a906f193e47a505b473bdacf17d3dfc6b24b3ccb25caa79c4ad70468d115aa.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/75a906f193e47a505b473bdacf17d3dfc6b24b3ccb25caa79c4ad70468d115aa.jpg)

![8244b009cea8550964d5e053b2f47be2750571753d7fa7f7705adb81b0d2c1ac.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/8244b009cea8550964d5e053b2f47be2750571753d7fa7f7705adb81b0d2c1ac.jpg)

![9e450964560977522951116a1c8bf946ea09bdc8a49c1fcf817f331531398655.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/9e450964560977522951116a1c8bf946ea09bdc8a49c1fcf817f331531398655.jpg)

![a3a5e0684404f2163f1564c3c6368ac6e496df07cc465a7b5ff8f69fdbe4ce30.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/a3a5e0684404f2163f1564c3c6368ac6e496df07cc465a7b5ff8f69fdbe4ce30.jpg)

![ba87d9086f03ced3472dc4825adb9efd2e1d574eed0ecef2f99e074919f278b5.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/ba87d9086f03ced3472dc4825adb9efd2e1d574eed0ecef2f99e074919f278b5.jpg)

![c92c0bb27b99cf8632998c76055b6dd3ec73b290a523205bb2c7cade757525d7.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/c92c0bb27b99cf8632998c76055b6dd3ec73b290a523205bb2c7cade757525d7.jpg)

![cb39cbf220a0a59b8a78ac77a665fee13ab6a2adad45a1b69a2fd7f6e711efb8.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/cb39cbf220a0a59b8a78ac77a665fee13ab6a2adad45a1b69a2fd7f6e711efb8.jpg)

![d922ebeb4515012b5b2c53a04f08eb12eb5f9bd721949c5aec4d910e88cda24a.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/images/d922ebeb4515012b5b2c53a04f08eb12eb5f9bd721949c5aec4d910e88cda24a.jpg)

### Tables

![0baa74ca5513e36a9e40fbfe6edf842f62eda8187eba6f7eaef699ce98348ef4.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/0baa74ca5513e36a9e40fbfe6edf842f62eda8187eba6f7eaef699ce98348ef4.jpg)

![1570d53930dd3ddb8da17ab6b426563c8448ad8bee582756ca202332a5b61ed5.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/1570d53930dd3ddb8da17ab6b426563c8448ad8bee582756ca202332a5b61ed5.jpg)

![1a6bfbf638fb28ce4b81a92b7d500093d5c05395db0248bf615a2baa697c862a.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/1a6bfbf638fb28ce4b81a92b7d500093d5c05395db0248bf615a2baa697c862a.jpg)

![1be0319c39b28b977f314f6ebc448dbc4d567b78dece0473c0f9ec26bf3ae082.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/1be0319c39b28b977f314f6ebc448dbc4d567b78dece0473c0f9ec26bf3ae082.jpg)

![1f26b88f53706d5bcda667a5ff7e4ad6d82293ea53517dc2fd11284bc9d886cd.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/1f26b88f53706d5bcda667a5ff7e4ad6d82293ea53517dc2fd11284bc9d886cd.jpg)

![2e05a80ae10fc7764b2f8bc4805b899eefac2343b27eb0c2501bd266e0b7774f.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/2e05a80ae10fc7764b2f8bc4805b899eefac2343b27eb0c2501bd266e0b7774f.jpg)

![31a76ed798ff71b13d9854fc75331a3fcc895b4c6aff6ca097e21184264cbc83.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/31a76ed798ff71b13d9854fc75331a3fcc895b4c6aff6ca097e21184264cbc83.jpg)

![5a62d963146f47239ea0d3da926422c7688f0f47da14e5b92f66722bb2122af6.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/5a62d963146f47239ea0d3da926422c7688f0f47da14e5b92f66722bb2122af6.jpg)

![64f7e3119e72072226f0f10097da9902fb7c868120109cbe7efb0281bbf58b30.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/64f7e3119e72072226f0f10097da9902fb7c868120109cbe7efb0281bbf58b30.jpg)

![6834c1650f1f7a8cb9a57c3d41ea6fafc1161daf037108265157dbd6805e3a3a.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/6834c1650f1f7a8cb9a57c3d41ea6fafc1161daf037108265157dbd6805e3a3a.jpg)

![6fb0ef45f4faf0aebcb2670fcbc324869d8d78c8698bbca837b176ebf56f395c.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/6fb0ef45f4faf0aebcb2670fcbc324869d8d78c8698bbca837b176ebf56f395c.jpg)

![6fd4c05ad5a3a4070560c9368d679dad3eda9d3c56a8a52db7a454005981cc30.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/6fd4c05ad5a3a4070560c9368d679dad3eda9d3c56a8a52db7a454005981cc30.jpg)

![799fba811db1eae526ff5cae142b2ba6e906c00c2dcc16fde46e655fb6c70521.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/799fba811db1eae526ff5cae142b2ba6e906c00c2dcc16fde46e655fb6c70521.jpg)

![90e7125d11b7a049b9c843e83ed0e84072d99e645ed9d681815adc6a39b3cd5a.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/90e7125d11b7a049b9c843e83ed0e84072d99e645ed9d681815adc6a39b3cd5a.jpg)

![b0fdb9f2b13c71d825b1bd74ea6b378061258d8a41b0abfc293349a96d0232f0.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/b0fdb9f2b13c71d825b1bd74ea6b378061258d8a41b0abfc293349a96d0232f0.jpg)

![dd13c48302dea977fd2ed324ebd8e8abe59cfa5d4d68b348f111f29c27a44e6f.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/dd13c48302dea977fd2ed324ebd8e8abe59cfa5d4d68b348f111f29c27a44e6f.jpg)

![efd649b24483af040d09ca116befa3fe573a37a73819b0854a638caaee29f237.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/efd649b24483af040d09ca116befa3fe573a37a73819b0854a638caaee29f237.jpg)

![efffe23956bdfc14309cd505d8920c2d5714228b2ec19eec7915131c73e9fd50.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/efffe23956bdfc14309cd505d8920c2d5714228b2ec19eec7915131c73e9fd50.jpg)

![f709c6b201eb40e15dda81d3959b50e016f6ae4b08b7827da7ad4ba9e9dc18d9.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/f709c6b201eb40e15dda81d3959b50e016f6ae4b08b7827da7ad4ba9e9dc18d9.jpg)

![f72498a24c4a15ee8294ef00c7c9fd940907849b0f4dc6e7d2f34027704ce2a3.jpg](../iclr_results/516_Test-time Adaptation for Cross-modal Retrieval with Query Shift/tables/f72498a24c4a15ee8294ef00c7c9fd940907849b0f4dc6e7d2f34027704ce2a3.jpg)

## Severing Spurious Correlations with Data Pruning


### Images

![0aa3b3eec635b3f442d3e7bb29c6813263a6c5e16ca8b3c4bcfd622b3770124a.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/0aa3b3eec635b3f442d3e7bb29c6813263a6c5e16ca8b3c4bcfd622b3770124a.jpg)

![0dcf801805812616704640a85af00e9863ef98d84971269ad422be18f762db6f.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/0dcf801805812616704640a85af00e9863ef98d84971269ad422be18f762db6f.jpg)

![18337acb91c8481808ed8b7f588742d4afdc6f591df18f77f6126744e46dfc4a.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/18337acb91c8481808ed8b7f588742d4afdc6f591df18f77f6126744e46dfc4a.jpg)

![2a27751091f591ade7599d3a7238cc294e950f75f875e31621b91dbd238421e6.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/2a27751091f591ade7599d3a7238cc294e950f75f875e31621b91dbd238421e6.jpg)

![4fa1033d0d3f7be2bab2671348e7c2c9149d6a3d6e40b8076619b349085d1b81.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/4fa1033d0d3f7be2bab2671348e7c2c9149d6a3d6e40b8076619b349085d1b81.jpg)

![5ef0e5fc6d106aeceecd4a687220ae664df3dd9769734ef5d0bc6fa580f1a227.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/5ef0e5fc6d106aeceecd4a687220ae664df3dd9769734ef5d0bc6fa580f1a227.jpg)

![62405a5b59ac9d57e95959dd0458218b9374b7392a1c5ba0b8168af61f462f05.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/62405a5b59ac9d57e95959dd0458218b9374b7392a1c5ba0b8168af61f462f05.jpg)

![649bc612c1e3cba30124ebad55f748aaf0e858c2e50c4e9cc75267212dc14253.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/649bc612c1e3cba30124ebad55f748aaf0e858c2e50c4e9cc75267212dc14253.jpg)

![88ba5b3a03206076644281138474765e079488c95195a03dc552fa881706a8bf.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/88ba5b3a03206076644281138474765e079488c95195a03dc552fa881706a8bf.jpg)

![9a169026a96b0bbd8653779d825ba6fdec6bd25eb4cf43b3bae5ba25b048a4ed.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/9a169026a96b0bbd8653779d825ba6fdec6bd25eb4cf43b3bae5ba25b048a4ed.jpg)

![c2a0744c8b4d4388a3b127935ec3ccf46ab2465dc917b6f93bfa2fe257e85815.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/c2a0744c8b4d4388a3b127935ec3ccf46ab2465dc917b6f93bfa2fe257e85815.jpg)

![e3b1f30ce0de85cbe4aec10b7c0cb7ce5e2fa6f760316f82a6bab10209c8ef4b.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/images/e3b1f30ce0de85cbe4aec10b7c0cb7ce5e2fa6f760316f82a6bab10209c8ef4b.jpg)

### Tables

![263570fae3a43bf8e45341edd5f09349dfccbd3d30617d025e2375eefde4cd19.jpg](../iclr_results/517_Severing Spurious Correlations with Data Pruning/tables/263570fae3a43bf8e45341edd5f09349dfccbd3d30617d025e2375eefde4cd19.jpg)

## Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance


### Images

![25ecc941c5935deb5220968752fb20f85618eeeed2ede9f49dc6fbba6670f129.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/25ecc941c5935deb5220968752fb20f85618eeeed2ede9f49dc6fbba6670f129.jpg)

![25fc8ccfe357dee8ef02c2b71fc751f742a3305f6180f70493a2fa8e5947ae31.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/25fc8ccfe357dee8ef02c2b71fc751f742a3305f6180f70493a2fa8e5947ae31.jpg)

![2869a4da79e2e8c6951d90365ffa5e585a2f71420b39e7d02c52809a42204543.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/2869a4da79e2e8c6951d90365ffa5e585a2f71420b39e7d02c52809a42204543.jpg)

![2b61c95c37c00e6ad98d05471b913f65a9b4df2f72da0c545928053197fef849.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/2b61c95c37c00e6ad98d05471b913f65a9b4df2f72da0c545928053197fef849.jpg)

![3142b7e3d5f638d2a9171398aaa9a9d8b38a0148708e366b1989ed4b2ef66702.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/3142b7e3d5f638d2a9171398aaa9a9d8b38a0148708e366b1989ed4b2ef66702.jpg)

![39a973810a2fa8ae316c1c96889ea8eda387cca265e872fd76f447081d97d4ad.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/39a973810a2fa8ae316c1c96889ea8eda387cca265e872fd76f447081d97d4ad.jpg)

![3dc32b02665ca936b2b68eeca3f67e288ff47c01cca4affe5174145249c86d1c.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/3dc32b02665ca936b2b68eeca3f67e288ff47c01cca4affe5174145249c86d1c.jpg)

![3ee9eb19c0d1a08da75a8d9a59abf30f6b987f9e3da444d9fa0e4fa31e164b88.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/3ee9eb19c0d1a08da75a8d9a59abf30f6b987f9e3da444d9fa0e4fa31e164b88.jpg)

![40e224c146a6afd13eda740d294ecdf356abf42ed86ec5fbfe7ef1ac93a7b8b6.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/40e224c146a6afd13eda740d294ecdf356abf42ed86ec5fbfe7ef1ac93a7b8b6.jpg)

![461366ddc2d4cb32cc002ae763fc9a6fd42969ab39bdf516b812f3a36f7ab019.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/461366ddc2d4cb32cc002ae763fc9a6fd42969ab39bdf516b812f3a36f7ab019.jpg)

![4c3e3a78845e46d2d3adcba852a75c6ed9615b60424ca3154de3c0aa98e99d3a.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/4c3e3a78845e46d2d3adcba852a75c6ed9615b60424ca3154de3c0aa98e99d3a.jpg)

![52e967cb735f21ce10a289cbc00939141697a5c7cd369df0c39edd128e5cb4fe.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/52e967cb735f21ce10a289cbc00939141697a5c7cd369df0c39edd128e5cb4fe.jpg)

![6c331bcbdc0f487bb28d4e4bcdc056d6b9d753f47e97774e9d41161bd5c03af2.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/6c331bcbdc0f487bb28d4e4bcdc056d6b9d753f47e97774e9d41161bd5c03af2.jpg)

![6fe648bae71c77c0b67eb92d6f786243b0858e4300fd1014641f7d4f9dec4e57.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/6fe648bae71c77c0b67eb92d6f786243b0858e4300fd1014641f7d4f9dec4e57.jpg)

![7505fa762b3090cd59ff4f2e30c8d831d3f0cbbe696899815b29367a022f30f6.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/7505fa762b3090cd59ff4f2e30c8d831d3f0cbbe696899815b29367a022f30f6.jpg)

![7912c239ba287923613add2a9752e9d9c13d236d889c93b16c0b6fa050d110b9.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/7912c239ba287923613add2a9752e9d9c13d236d889c93b16c0b6fa050d110b9.jpg)

![7feb0597486c439c3d30dbbb7f922bd7eed67ac85a12e370d5e3b3798b169f6f.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/7feb0597486c439c3d30dbbb7f922bd7eed67ac85a12e370d5e3b3798b169f6f.jpg)

![8bb31e6290665459004374189fb5bce0fe2cdddc9237983691f4e433e11faaa5.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/8bb31e6290665459004374189fb5bce0fe2cdddc9237983691f4e433e11faaa5.jpg)

![96db2eec79a13e657bc45043c35868867738ae4bd232975d202fd8d9db56d689.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/96db2eec79a13e657bc45043c35868867738ae4bd232975d202fd8d9db56d689.jpg)

![abc3e1bfb8dead64e042fe38af07f754166d8a3ca7f504f38162abbd2dfb03c7.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/abc3e1bfb8dead64e042fe38af07f754166d8a3ca7f504f38162abbd2dfb03c7.jpg)

![c06a13cb903454444b4061633afb5d700aecd5f176405104e48647e2c6d66432.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/c06a13cb903454444b4061633afb5d700aecd5f176405104e48647e2c6d66432.jpg)

![cd0d75cb85547a1c6b95b1e6e66d3b4a561a4e7fffcfba5f864deda33e00887e.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/cd0d75cb85547a1c6b95b1e6e66d3b4a561a4e7fffcfba5f864deda33e00887e.jpg)

![d262d72558de5608d62d52cb1ea664659d5c045d76604c0158d43a59bd0c785f.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/d262d72558de5608d62d52cb1ea664659d5c045d76604c0158d43a59bd0c785f.jpg)

![d5f0eda466e1c275725173d1fca72b4cd58ef84fa66c098e7b09ea18cf396d1e.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/d5f0eda466e1c275725173d1fca72b4cd58ef84fa66c098e7b09ea18cf396d1e.jpg)

![e2a3504d79d8802eb98693a2ff2de965a379531e0aa05234dbb490ff27edeabd.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/e2a3504d79d8802eb98693a2ff2de965a379531e0aa05234dbb490ff27edeabd.jpg)

![f70d4f7cf266aa6a9f8d30fec9534cee983a13ca20fce6092dd0cad870eab9aa.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/f70d4f7cf266aa6a9f8d30fec9534cee983a13ca20fce6092dd0cad870eab9aa.jpg)

![fb51c2a2a64cdd2724b3e5c5257144e01abb9d81f658f12d81db397d853514e2.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/images/fb51c2a2a64cdd2724b3e5c5257144e01abb9d81f658f12d81db397d853514e2.jpg)

### Tables

![05112de8be2b3f43c23bb091c1150e58a30cb273650259f2f3854aadae9ff0df.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/05112de8be2b3f43c23bb091c1150e58a30cb273650259f2f3854aadae9ff0df.jpg)

![11b63056228969cd71ec06fb79ab946979f6fea05d5f2904aa312f7c9cdf0ef3.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/11b63056228969cd71ec06fb79ab946979f6fea05d5f2904aa312f7c9cdf0ef3.jpg)

![5092a2d9e1c67f11102706ffd3a893f2d2465ab9bf05b84d632a236ac6a2bc79.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/5092a2d9e1c67f11102706ffd3a893f2d2465ab9bf05b84d632a236ac6a2bc79.jpg)

![51cd5907ff5265b593cf8c2518b0a2bed5fc72cb4a40de92c610c06477064509.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/51cd5907ff5265b593cf8c2518b0a2bed5fc72cb4a40de92c610c06477064509.jpg)

![5de50b1c070b59c711120c915c2e86405612f4a15ac43835de9149d9aad47175.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/5de50b1c070b59c711120c915c2e86405612f4a15ac43835de9149d9aad47175.jpg)

![6005211ae438cbe0ba314625fbb227f4f7160662070e23628e79a492578155f6.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/6005211ae438cbe0ba314625fbb227f4f7160662070e23628e79a492578155f6.jpg)

![6f03243458c6a1521850653fddbddc527b49577a48d1e2eb6e71148c81dcbcda.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/6f03243458c6a1521850653fddbddc527b49577a48d1e2eb6e71148c81dcbcda.jpg)

![75871c41b8c26a341ae0c918622bdb7e6b4aa1b5827213f388c36250408a2517.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/75871c41b8c26a341ae0c918622bdb7e6b4aa1b5827213f388c36250408a2517.jpg)

![7adfdc4d1ef01165c2754b7d8f11d118ea9f7ed0045ce5311548dd8a105ba6f4.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/7adfdc4d1ef01165c2754b7d8f11d118ea9f7ed0045ce5311548dd8a105ba6f4.jpg)

![98d80c775b17530bbb8da3aa65df4c7c2831944c660512ed6686f8293b5535a6.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/98d80c775b17530bbb8da3aa65df4c7c2831944c660512ed6686f8293b5535a6.jpg)

![a4660d43fa008cf8dfd48cf2d6aaa3d474913f230f64cca7343429709b68dd2b.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/a4660d43fa008cf8dfd48cf2d6aaa3d474913f230f64cca7343429709b68dd2b.jpg)

![cda0ca335267fbe3cb090ddcf6fb31c8cface633c13a551a0d73eed57e7de0b4.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/cda0ca335267fbe3cb090ddcf6fb31c8cface633c13a551a0d73eed57e7de0b4.jpg)

![e5abc5a057ef88264f337d32fedb26017516a04e5885b5a3e88ca00be7526155.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/e5abc5a057ef88264f337d32fedb26017516a04e5885b5a3e88ca00be7526155.jpg)

![eae793365be20abb9eb836a6a1ea8a9aa8766e51c3e405ff1ac10289e72e9f66.jpg](../iclr_results/518_Rare-to-Frequent_ Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with/tables/eae793365be20abb9eb836a6a1ea8a9aa8766e51c3e405ff1ac10289e72e9f66.jpg)

## Lightweight Neural App Control


### Images

![1df2ee1ba4ca4838f3aaf4bb622d1e1d035a70a5df30d3fe0d7d6a0a2a826c67.jpg](../iclr_results/519_Lightweight Neural App Control/images/1df2ee1ba4ca4838f3aaf4bb622d1e1d035a70a5df30d3fe0d7d6a0a2a826c67.jpg)

![31c20025304bf80abbfe374135233459fafdf7e972482f13ae1545a744e8b999.jpg](../iclr_results/519_Lightweight Neural App Control/images/31c20025304bf80abbfe374135233459fafdf7e972482f13ae1545a744e8b999.jpg)

![36d857bc237e32b8b6609eba0c2f6e222fd5ec0e5b1c7d8d1cfca9363c31234d.jpg](../iclr_results/519_Lightweight Neural App Control/images/36d857bc237e32b8b6609eba0c2f6e222fd5ec0e5b1c7d8d1cfca9363c31234d.jpg)

![5448bd744d1db6ac995a41c0404d4ffd7e3ed5cf054e4b96e3db608e04275714.jpg](../iclr_results/519_Lightweight Neural App Control/images/5448bd744d1db6ac995a41c0404d4ffd7e3ed5cf054e4b96e3db608e04275714.jpg)

![89ebe557ea3ded94c5779ff7faa6afd1016ce2fdabf264edd81058c09aa86490.jpg](../iclr_results/519_Lightweight Neural App Control/images/89ebe557ea3ded94c5779ff7faa6afd1016ce2fdabf264edd81058c09aa86490.jpg)

![b25f8048730b85a3c6b8d941e8597c3302760ca188708ecb782ac2e54f4afee4.jpg](../iclr_results/519_Lightweight Neural App Control/images/b25f8048730b85a3c6b8d941e8597c3302760ca188708ecb782ac2e54f4afee4.jpg)

![ee59b3726d992c164184d53ef9a93ca41c380c72773cd79f37a3b79731362bfa.jpg](../iclr_results/519_Lightweight Neural App Control/images/ee59b3726d992c164184d53ef9a93ca41c380c72773cd79f37a3b79731362bfa.jpg)

![f138201104184398a48828f8c06fecc863403e50ca2ea3b33d312a760f92bd44.jpg](../iclr_results/519_Lightweight Neural App Control/images/f138201104184398a48828f8c06fecc863403e50ca2ea3b33d312a760f92bd44.jpg)

### Tables

![1d40123faf248eebd30fe5f4d923187be7fb4c9d11e82f37b1993ae51b19e036.jpg](../iclr_results/519_Lightweight Neural App Control/tables/1d40123faf248eebd30fe5f4d923187be7fb4c9d11e82f37b1993ae51b19e036.jpg)

![231790b69373ed9526904f708e55b5aa6cfc891305ebfbbd336c06e951513c80.jpg](../iclr_results/519_Lightweight Neural App Control/tables/231790b69373ed9526904f708e55b5aa6cfc891305ebfbbd336c06e951513c80.jpg)

![59b2452df887f1ebb4fce3d491292c84421481cd562843d06d1911d3c8ad9a17.jpg](../iclr_results/519_Lightweight Neural App Control/tables/59b2452df887f1ebb4fce3d491292c84421481cd562843d06d1911d3c8ad9a17.jpg)

![8cb78c33e5a350eb4d77f4efb4e53a1ff05645f4a2d5e97fa2455a3d95e0d277.jpg](../iclr_results/519_Lightweight Neural App Control/tables/8cb78c33e5a350eb4d77f4efb4e53a1ff05645f4a2d5e97fa2455a3d95e0d277.jpg)

![ab386f30b92b49f747fc006c3f9ec19d68b1199701f3e0ee55aec984bf6ea386.jpg](../iclr_results/519_Lightweight Neural App Control/tables/ab386f30b92b49f747fc006c3f9ec19d68b1199701f3e0ee55aec984bf6ea386.jpg)

![ed8d900898ddd0bf3d744e22ca5369756801f40d3ec36af38f6bcf364eb4beea.jpg](../iclr_results/519_Lightweight Neural App Control/tables/ed8d900898ddd0bf3d744e22ca5369756801f40d3ec36af38f6bcf364eb4beea.jpg)

## DeLLMa: Decision Making Under Uncertainty with Large Language Models


### Images

![0750b4f6abcff401d89de744949300b2d3e3e2cfa537f4c2ee5d89ec940aa4ca.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/0750b4f6abcff401d89de744949300b2d3e3e2cfa537f4c2ee5d89ec940aa4ca.jpg)

![13eec128f52a019e37344263a8945de521a2c80e2d3f59099073c41967210006.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/13eec128f52a019e37344263a8945de521a2c80e2d3f59099073c41967210006.jpg)

![3a856091a9dc378f6c57f898731b23c5323c0d0647829c35aa278f0e83c9afc5.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/3a856091a9dc378f6c57f898731b23c5323c0d0647829c35aa278f0e83c9afc5.jpg)

![4737385ef8d9bcf1afd8af14ac459599ac3355f4b2319707198576dfcfae020c.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/4737385ef8d9bcf1afd8af14ac459599ac3355f4b2319707198576dfcfae020c.jpg)

![5c4e2bc6a7813e8d16be09fe21879090f21dd55bf0a16bc71451158ae5839445.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/5c4e2bc6a7813e8d16be09fe21879090f21dd55bf0a16bc71451158ae5839445.jpg)

![69cbd4e037fe55a54998c10419fe2573451b26b7782e6ba05342ec5bc1f13d78.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/69cbd4e037fe55a54998c10419fe2573451b26b7782e6ba05342ec5bc1f13d78.jpg)

![7cd367f35b4b8e12e5bf90168ef604daf666fd5eb0fbd42581e6035128ae4a59.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/7cd367f35b4b8e12e5bf90168ef604daf666fd5eb0fbd42581e6035128ae4a59.jpg)

![96738bd776b89b2f3deb79221b627f5964df456d81d7b9983418e6abcaacf526.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/96738bd776b89b2f3deb79221b627f5964df456d81d7b9983418e6abcaacf526.jpg)

![d1e38c415832d9c164800c2cbda3f9fe3bc9f739d2e71ed70e2f34bc2064aa50.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/d1e38c415832d9c164800c2cbda3f9fe3bc9f739d2e71ed70e2f34bc2064aa50.jpg)

![dce3d0b16eaf996241451471fbdc90b4fefff1e78afe811c6ac6cf2f86b0d933.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/images/dce3d0b16eaf996241451471fbdc90b4fefff1e78afe811c6ac6cf2f86b0d933.jpg)

### Tables

![4075b2f0e300ad07722edc4c37a9db95dd77be67bc55a64501db9d2c9b46916e.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/tables/4075b2f0e300ad07722edc4c37a9db95dd77be67bc55a64501db9d2c9b46916e.jpg)

![5b7640871bf348a60fb6a3b3b65b2acce7a8ec606b4cee1607f95194f33ab3bd.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/tables/5b7640871bf348a60fb6a3b3b65b2acce7a8ec606b4cee1607f95194f33ab3bd.jpg)

![a1d3640429f11c38581d649b1b3af3d7807b1dd4ba7ef9df63f9297df8c9edc7.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/tables/a1d3640429f11c38581d649b1b3af3d7807b1dd4ba7ef9df63f9297df8c9edc7.jpg)

![bd0446f6fd3e5a3ef99a992aa7a7159d29a321b098ff785fd27ae6ac0c0526cc.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/tables/bd0446f6fd3e5a3ef99a992aa7a7159d29a321b098ff785fd27ae6ac0c0526cc.jpg)

![c5e502e315cef0ed68a318cbd533f02a5e776a609d35b5c0f1e9265a30edfaa3.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/tables/c5e502e315cef0ed68a318cbd533f02a5e776a609d35b5c0f1e9265a30edfaa3.jpg)

![eb36f2749992ac706c37390a569de79c61119a1e194db64dec31ac02b73ba689.jpg](../iclr_results/520_DeLLMa_ Decision Making Under Uncertainty with Large Language Models/tables/eb36f2749992ac706c37390a569de79c61119a1e194db64dec31ac02b73ba689.jpg)

## Multi-Robot Motion Planning with Diffusion Models


### Images

![21ba535e580bc8d98319b42b7f412df848976b8f2a27fc9650df1521f871c6e1.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/21ba535e580bc8d98319b42b7f412df848976b8f2a27fc9650df1521f871c6e1.jpg)

![3d50d787733c0933381c86c0a0db519b758b24a822a3c760fcb4cad9e9ca025b.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/3d50d787733c0933381c86c0a0db519b758b24a822a3c760fcb4cad9e9ca025b.jpg)

![63f73b93ea53397f8a291403de687d199527a8fb48259bfea8a553c6ce6005df.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/63f73b93ea53397f8a291403de687d199527a8fb48259bfea8a553c6ce6005df.jpg)

![6c741c8d065538295e95ee04445f1b746a8d558d860b499055121b4de3560f31.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/6c741c8d065538295e95ee04445f1b746a8d558d860b499055121b4de3560f31.jpg)

![74cde2423ee9b78fea7f5d3e7b68b18eedce58149b87f0a43635afa4f968baa5.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/74cde2423ee9b78fea7f5d3e7b68b18eedce58149b87f0a43635afa4f968baa5.jpg)

![77956cf815fde6cdc0ff423b87644892cd3fb0777ee3fd7845784feca011e325.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/77956cf815fde6cdc0ff423b87644892cd3fb0777ee3fd7845784feca011e325.jpg)

![7a1973670fa30af993b5ab5a98b98855d0b66bb14f207e7a76bcf40c73d2663d.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/7a1973670fa30af993b5ab5a98b98855d0b66bb14f207e7a76bcf40c73d2663d.jpg)

![7aa1dfbd6aafc6cea5d78ed65f24248615ba291cd1cbcbfec8cc4d024a1676db.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/7aa1dfbd6aafc6cea5d78ed65f24248615ba291cd1cbcbfec8cc4d024a1676db.jpg)

![c7488d29907f8994e032a7052c6bd80b699f500a4d53b41869a475a84037dc72.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/images/c7488d29907f8994e032a7052c6bd80b699f500a4d53b41869a475a84037dc72.jpg)

### Tables

![4ddff22b64d49c0d22ed297c4e5adf340e24b8a05a2146f095a59ed3db18cb23.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/tables/4ddff22b64d49c0d22ed297c4e5adf340e24b8a05a2146f095a59ed3db18cb23.jpg)

![8f4a01fb2f7f4e9966e182260778ebeae5a0899e5434676e5dea5f6f317529c7.jpg](../iclr_results/521_Multi-Robot Motion Planning with Diffusion Models/tables/8f4a01fb2f7f4e9966e182260778ebeae5a0899e5434676e5dea5f6f317529c7.jpg)

## ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks


### Images

![004da36a3013fd5d7ee56b2d796162051d40b0f62eb5bb5df5228c2e537ee564.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/004da36a3013fd5d7ee56b2d796162051d40b0f62eb5bb5df5228c2e537ee564.jpg)

![0be64f837f4c66342cb3299ef50029b05eae96d5579e43d3031200339a48ffde.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/0be64f837f4c66342cb3299ef50029b05eae96d5579e43d3031200339a48ffde.jpg)

![1f885fea746aa1ebcf92e76fc20995a811963caada843a541432bde0ab945e82.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/1f885fea746aa1ebcf92e76fc20995a811963caada843a541432bde0ab945e82.jpg)

![20e6db16d24af65834f878404b5b20ebed12d2dafe0ff216007f7347830173e7.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/20e6db16d24af65834f878404b5b20ebed12d2dafe0ff216007f7347830173e7.jpg)

![2f5396a3f7daefbb20514f337294310d82c890f8e804b9ca24fb02f71038ff2e.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/2f5396a3f7daefbb20514f337294310d82c890f8e804b9ca24fb02f71038ff2e.jpg)

![30537f1221949ce5e7ec4ece718a5ae707e82fbea8de6bd557029c993155148f.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/30537f1221949ce5e7ec4ece718a5ae707e82fbea8de6bd557029c993155148f.jpg)

![3bb51ec2a41ea445376e5748c14c989820f4e4eeebbdd32adda127c2dd16c66b.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/3bb51ec2a41ea445376e5748c14c989820f4e4eeebbdd32adda127c2dd16c66b.jpg)

![4b2d85ab12ce03369c8ec449cdb908d04e4bbf96943da4c4085de46a9098577c.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/4b2d85ab12ce03369c8ec449cdb908d04e4bbf96943da4c4085de46a9098577c.jpg)

![5570362c8954b9b699dcc988089ac887b30dee6230729a287833e848d8516e60.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/5570362c8954b9b699dcc988089ac887b30dee6230729a287833e848d8516e60.jpg)

![57546e883915e795eedb0c3db3d5c4eb9ff2a34b7ea822725c88fa8f72770c77.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/57546e883915e795eedb0c3db3d5c4eb9ff2a34b7ea822725c88fa8f72770c77.jpg)

![57f059b20113a7fc7c8fd17599843c34a41a83c0f232573663366c86dff76150.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/57f059b20113a7fc7c8fd17599843c34a41a83c0f232573663366c86dff76150.jpg)

![66c49c4d7ef7903e6687142e36035ca1bcb6c3fa4510cbc7183315446a781295.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/66c49c4d7ef7903e6687142e36035ca1bcb6c3fa4510cbc7183315446a781295.jpg)

![84b0b29c72af76ec9d18ff39e8f032ca826dc2cbd45f7ce1353265b9168b4411.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/84b0b29c72af76ec9d18ff39e8f032ca826dc2cbd45f7ce1353265b9168b4411.jpg)

![a9170eb7d859a7c30cb0c2e458a22648c0948776a4b88239db7a13ce53dca333.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/a9170eb7d859a7c30cb0c2e458a22648c0948776a4b88239db7a13ce53dca333.jpg)

![b1648526a98af3ddc3a401c2fa61567d233a89c3b872b471c554cd770acf3bc8.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/b1648526a98af3ddc3a401c2fa61567d233a89c3b872b471c554cd770acf3bc8.jpg)

![b1ba12afe212de5f3b2cdf0de60f3770e14ec6b6c996336bf9c01eb521b2fee4.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/b1ba12afe212de5f3b2cdf0de60f3770e14ec6b6c996336bf9c01eb521b2fee4.jpg)

![c46a1bc6af99522e474d9c63d4d07ab618d5efbae0a201119f7d52baf18eed6e.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/c46a1bc6af99522e474d9c63d4d07ab618d5efbae0a201119f7d52baf18eed6e.jpg)

![ca23378f097ce5750a2284704805aa9d9de7abc96ab2ad144607615311f89320.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/ca23378f097ce5750a2284704805aa9d9de7abc96ab2ad144607615311f89320.jpg)

![cf1457992dca00b9316360dd7d9eb2c8aa954cc028f3d945f2f4c33ec58bf69a.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/cf1457992dca00b9316360dd7d9eb2c8aa954cc028f3d945f2f4c33ec58bf69a.jpg)

![d55b9e4e0729aaf031799ae9ede168cacfd9d2a6073250be8de6017aca7c435f.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/d55b9e4e0729aaf031799ae9ede168cacfd9d2a6073250be8de6017aca7c435f.jpg)

![dfbc847649352a395d038056aa6666462492d5086cd3d9ba8acd1b33f570d5ba.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/dfbc847649352a395d038056aa6666462492d5086cd3d9ba8acd1b33f570d5ba.jpg)

![e38169a316f9791d73111483ddff9b41fe803e83b62321be71188c80179c3bf6.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/e38169a316f9791d73111483ddff9b41fe803e83b62321be71188c80179c3bf6.jpg)

![e7cb14dce9da59a6069f7dbe4a0fea6cb0436696bacd8ea076307397b1c3490b.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/e7cb14dce9da59a6069f7dbe4a0fea6cb0436696bacd8ea076307397b1c3490b.jpg)

![f6277f880b7e5cc3b946a3db2274e70a82e79e004696a00af13e4da969d2b8d7.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/f6277f880b7e5cc3b946a3db2274e70a82e79e004696a00af13e4da969d2b8d7.jpg)

![f87b15ea84f8d9c002f352895d43096bf718d71738137f73523c47f03f4270c0.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/f87b15ea84f8d9c002f352895d43096bf718d71738137f73523c47f03f4270c0.jpg)

![fe5ef9c5e4ee7c6435b12df657b39059cbd1fc5a815093b3e17e408a8fc649b6.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/images/fe5ef9c5e4ee7c6435b12df657b39059cbd1fc5a815093b3e17e408a8fc649b6.jpg)

### Tables

![14aebaf90d6aa10edd1086e224fb1a03f06f953841a9d2bb7fe3d339a71bd286.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/14aebaf90d6aa10edd1086e224fb1a03f06f953841a9d2bb7fe3d339a71bd286.jpg)

![24a94477d4ea9eafcbad570308a5d8c8da071dba2825f04b4797939039ff2225.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/24a94477d4ea9eafcbad570308a5d8c8da071dba2825f04b4797939039ff2225.jpg)

![2c67d8cae66bacacd9d98bbfa9ba513d5ae0b039ef96ab9eb9ddbb86e2921dc6.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/2c67d8cae66bacacd9d98bbfa9ba513d5ae0b039ef96ab9eb9ddbb86e2921dc6.jpg)

![2d3937ccb199001e2f458ae596aaa79f9b0b39b8bd463583f1fb1924fce79c11.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/2d3937ccb199001e2f458ae596aaa79f9b0b39b8bd463583f1fb1924fce79c11.jpg)

![2d4c2ef33973144d255919a46a77f9477c3e80c586159364074eb772a46f084a.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/2d4c2ef33973144d255919a46a77f9477c3e80c586159364074eb772a46f084a.jpg)

![341c3572ea20defb393500f8d135b3d5fc214201db37e5e57fc566eaa7fba553.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/341c3572ea20defb393500f8d135b3d5fc214201db37e5e57fc566eaa7fba553.jpg)

![3e4de46fa6efeea3dd6912038c10133ae92712c6630ec65e8659a1dae62c6c4f.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/3e4de46fa6efeea3dd6912038c10133ae92712c6630ec65e8659a1dae62c6c4f.jpg)

![41a8dd6eaef3387feca73bf8fdd5948f8aede52cbb996f5a08cd64ce87c8a6f8.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/41a8dd6eaef3387feca73bf8fdd5948f8aede52cbb996f5a08cd64ce87c8a6f8.jpg)

![4f0c4c8df27c45c33ec18bd3e72188578451f7a8d04a2d9e813cf33bdec16da5.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/4f0c4c8df27c45c33ec18bd3e72188578451f7a8d04a2d9e813cf33bdec16da5.jpg)

![5c469e3865432dedd03a61a1a89bc34b7ffebe2645ff92aee63324d4833cfd9e.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/5c469e3865432dedd03a61a1a89bc34b7ffebe2645ff92aee63324d4833cfd9e.jpg)

![683a09776978b1065aff5b1054446a9baaeeffa5cdc58ac2233047c9ecce61a7.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/683a09776978b1065aff5b1054446a9baaeeffa5cdc58ac2233047c9ecce61a7.jpg)

![89446f2db6fddf9dbcacb14f14435ba139fff4a5b56cc6aeef663f89cccbe84d.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/89446f2db6fddf9dbcacb14f14435ba139fff4a5b56cc6aeef663f89cccbe84d.jpg)

![8bda20dd1a8805a3a483bf7ca9a41ba1a6eb6b287e45c32cdb27e10c60588545.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/8bda20dd1a8805a3a483bf7ca9a41ba1a6eb6b287e45c32cdb27e10c60588545.jpg)

![8c97e2ab5633cc72e158bc7770a9cf058b3cb861e80bd4ee532f3ce6fd22352a.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/8c97e2ab5633cc72e158bc7770a9cf058b3cb861e80bd4ee532f3ce6fd22352a.jpg)

![8c9d85619ad56427f0e41e1ca25d1a3c0527a6f1b0218ee6d7cd0a36c13f0f77.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/8c9d85619ad56427f0e41e1ca25d1a3c0527a6f1b0218ee6d7cd0a36c13f0f77.jpg)

![969909616f2cbed712996f125d43e1f9ac5307067619393dd77d551e1b94c87a.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/969909616f2cbed712996f125d43e1f9ac5307067619393dd77d551e1b94c87a.jpg)

![9a7778b889c77128a653cbe5c5e645585aad551a90d824e649b6492c11b3ff93.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/9a7778b889c77128a653cbe5c5e645585aad551a90d824e649b6492c11b3ff93.jpg)

![9beb351525c52d9b2d3e4c83d03582bc4fe1a82e8e232523c4c624b024fae7ea.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/9beb351525c52d9b2d3e4c83d03582bc4fe1a82e8e232523c4c624b024fae7ea.jpg)

![bb3e5cbd26274a30127d468206ee664e6793c31051dd5568224ed53a1d60d79b.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/bb3e5cbd26274a30127d468206ee664e6793c31051dd5568224ed53a1d60d79b.jpg)

![c21d3b43841f36c44417e11c95b70dea30d9e3a9fb990a21acc8a2d2d6c98095.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/c21d3b43841f36c44417e11c95b70dea30d9e3a9fb990a21acc8a2d2d6c98095.jpg)

![c516567bcde695b7b88f88bc8020750755221461256c524a08563ff25acc7d57.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/c516567bcde695b7b88f88bc8020750755221461256c524a08563ff25acc7d57.jpg)

![d537b83d849d516f8444c3d6091be5bd98385764265a9c70f287c27c0a3a63e6.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/d537b83d849d516f8444c3d6091be5bd98385764265a9c70f287c27c0a3a63e6.jpg)

![d5a6d610b1cb0f98104922298ef17112e64ff9f902803af852e56c008e66a482.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/d5a6d610b1cb0f98104922298ef17112e64ff9f902803af852e56c008e66a482.jpg)

![d6e63acc585628765a67da383ca80440950afef6199741596be716faaf6553de.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/d6e63acc585628765a67da383ca80440950afef6199741596be716faaf6553de.jpg)

![dfb42added60d68e4933879a24731a13a93fea03931020815a36899cde6c82c7.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/dfb42added60d68e4933879a24731a13a93fea03931020815a36899cde6c82c7.jpg)

![e0f146d145c85ce3ff9e5718c8b78c60a66944261e703de15c97583672342b5b.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/e0f146d145c85ce3ff9e5718c8b78c60a66944261e703de15c97583672342b5b.jpg)

![ee52528fb5f5b2c39dabfb1dc64048fd7312eb6460b41751d80ca35e1649da3c.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/ee52528fb5f5b2c39dabfb1dc64048fd7312eb6460b41751d80ca35e1649da3c.jpg)

![f51106aa91f0bc2e5dda2f7bafe1caeec64cc94bf3ab79d5e4aab98fb31b081a.jpg](../iclr_results/522_ConFIG_ Towards Conflict-free Training of Physics Informed Neural Networks/tables/f51106aa91f0bc2e5dda2f7bafe1caeec64cc94bf3ab79d5e4aab98fb31b081a.jpg)

## MagicPIG: LSH Sampling for Efficient LLM Generation


### Images

![0036d40b1ff14e502ea69e191179d0f19185f4f84ae1e97a9ac26d7169188035.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/0036d40b1ff14e502ea69e191179d0f19185f4f84ae1e97a9ac26d7169188035.jpg)

![35d0a5df63e18cc74c183f8e7d36470c9c4149b90998da097e3b5e6806a95ce9.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/35d0a5df63e18cc74c183f8e7d36470c9c4149b90998da097e3b5e6806a95ce9.jpg)

![5b5ab72c84c83b6a39c440e9861fcfba534f87a4fa3b56aeafac19bca8ba7372.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/5b5ab72c84c83b6a39c440e9861fcfba534f87a4fa3b56aeafac19bca8ba7372.jpg)

![5c0a4d3fa433de454a600d78c8a27679efdf1422eab184e42a40c2e5eab7336b.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/5c0a4d3fa433de454a600d78c8a27679efdf1422eab184e42a40c2e5eab7336b.jpg)

![5dd5f75974b4a7569fbdf11a656968da4962afaa90fcf17be4ba0d84370f4d1e.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/5dd5f75974b4a7569fbdf11a656968da4962afaa90fcf17be4ba0d84370f4d1e.jpg)

![5ebb5e67a5ea18a89d5462107b9c60f0517cb84f85321d88ed1ac079a8430e8f.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/5ebb5e67a5ea18a89d5462107b9c60f0517cb84f85321d88ed1ac079a8430e8f.jpg)

![cbb0c87048de9cea5b6adc88eb24e26c074116eb6dce639d09048dd7c6ed2939.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/cbb0c87048de9cea5b6adc88eb24e26c074116eb6dce639d09048dd7c6ed2939.jpg)

![ce1c71666042837dc3dc4457b19b483795ed173cb1b35cf22ba728a8177ea6c5.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/ce1c71666042837dc3dc4457b19b483795ed173cb1b35cf22ba728a8177ea6c5.jpg)

![d6ab8ef53a0683cc954221cd3f3de08163091e3ffcff81b00595f9865b14b50f.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/d6ab8ef53a0683cc954221cd3f3de08163091e3ffcff81b00595f9865b14b50f.jpg)

![f1c753279666ef4b44c03775d36ff677ae9d3f0689a6b558dabadfca8f57ba46.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/f1c753279666ef4b44c03775d36ff677ae9d3f0689a6b558dabadfca8f57ba46.jpg)

![fc281fd1c655fda18ba7329da57004a01aa47b27339cde1dcf37b0b1d7319035.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/images/fc281fd1c655fda18ba7329da57004a01aa47b27339cde1dcf37b0b1d7319035.jpg)

### Tables

![02f6ab99d81ff4fc24be240faf82ea56b0eea9219aa5abf97c1626de4106643c.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/02f6ab99d81ff4fc24be240faf82ea56b0eea9219aa5abf97c1626de4106643c.jpg)

![03c1ea9dfbf881cf2ec6f1f19d0d2516f014d2ee554a0e80a1bf165ea267004a.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/03c1ea9dfbf881cf2ec6f1f19d0d2516f014d2ee554a0e80a1bf165ea267004a.jpg)

![190b1e18765999ec1be948bfef6dc08a6399b63dc1c15a7d5f4c4bf18b0767cc.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/190b1e18765999ec1be948bfef6dc08a6399b63dc1c15a7d5f4c4bf18b0767cc.jpg)

![1927f3b7e3e141ba2f8ebc4e85560cca3be530476432cd834d9bbcd7869eb04e.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/1927f3b7e3e141ba2f8ebc4e85560cca3be530476432cd834d9bbcd7869eb04e.jpg)

![47c10e40d08d20954d1a1805fc64f7e0ed53c487bd0b5fbb02f6ebc6e09cb56e.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/47c10e40d08d20954d1a1805fc64f7e0ed53c487bd0b5fbb02f6ebc6e09cb56e.jpg)

![aebf05a3dc729819f27dd8b90fe724c21b4272bf5f0b5e00e82e1fac2ba67a51.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/aebf05a3dc729819f27dd8b90fe724c21b4272bf5f0b5e00e82e1fac2ba67a51.jpg)

![b2b6fa3509a5753ccb9d6a0c122d0f76e43a1a87560cf164a66f602cb47ad7c6.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/b2b6fa3509a5753ccb9d6a0c122d0f76e43a1a87560cf164a66f602cb47ad7c6.jpg)

![c7969bb1193ca34f2a515caa4345a01a0cf8915fbab1ba3bc0fb284fab29aaa7.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/c7969bb1193ca34f2a515caa4345a01a0cf8915fbab1ba3bc0fb284fab29aaa7.jpg)

![d15e3a7e54323e3acec46fb9fc555b18501a2fd6ff0d8c46afb1bbe8e5950441.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/d15e3a7e54323e3acec46fb9fc555b18501a2fd6ff0d8c46afb1bbe8e5950441.jpg)

![d2c482a9ff011f5defdee4e1a69af4c9dfead2b4ed4f5d7a1a957b0215c5a6a3.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/d2c482a9ff011f5defdee4e1a69af4c9dfead2b4ed4f5d7a1a957b0215c5a6a3.jpg)

![d5973360daeedb5496e07b17058519b5fb1d55b486dae98d392c72b4f082c00a.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/d5973360daeedb5496e07b17058519b5fb1d55b486dae98d392c72b4f082c00a.jpg)

![fe10f86c4dd85614400ff6d840c94253a1ba8dd0bda81285d5b3d8da09fa2492.jpg](../iclr_results/523_MagicPIG_ LSH Sampling for Efficient LLM Generation/tables/fe10f86c4dd85614400ff6d840c94253a1ba8dd0bda81285d5b3d8da09fa2492.jpg)

## Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning


### Images

![03b76e532ce8b63dfd43d106ec774c094f6b8033c06462fc64705903fddf9770.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/03b76e532ce8b63dfd43d106ec774c094f6b8033c06462fc64705903fddf9770.jpg)

![1e1cfda0ebdd2c67730d6410b0324e6d93b68a7d48ce520d9af76a5b800a38b3.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/1e1cfda0ebdd2c67730d6410b0324e6d93b68a7d48ce520d9af76a5b800a38b3.jpg)

![1ef489f6074ac2b3fa435a2e3fcb656b876c1bf861c0072c4875efbf2e74088b.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/1ef489f6074ac2b3fa435a2e3fcb656b876c1bf861c0072c4875efbf2e74088b.jpg)

![32d10f584635583ba13ccaed6b1ce7b3985c10e86b863bd2c28ab57f0df3e448.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/32d10f584635583ba13ccaed6b1ce7b3985c10e86b863bd2c28ab57f0df3e448.jpg)

![361065f01b6a35865b3f7fa0d4dbc1e06d863d0442d51046df4d42419fae4059.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/361065f01b6a35865b3f7fa0d4dbc1e06d863d0442d51046df4d42419fae4059.jpg)

![5857adbb2f538e0022f396b6919631963be0d973359d42420fb2ae975c467a4b.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/5857adbb2f538e0022f396b6919631963be0d973359d42420fb2ae975c467a4b.jpg)

![5b67ed1f89cf6f0f0f607718e16ec2acb7a6d4296d9ef845cb3e775ca577794d.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/5b67ed1f89cf6f0f0f607718e16ec2acb7a6d4296d9ef845cb3e775ca577794d.jpg)

![76aad1eb15e730921515b352fb1d0f61eb7851c5b23cc927a1bbe9c678537ff6.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/76aad1eb15e730921515b352fb1d0f61eb7851c5b23cc927a1bbe9c678537ff6.jpg)

![914ab982802371415fce4430de31e9717265d0ded846b293fa6a004bd89154f8.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/914ab982802371415fce4430de31e9717265d0ded846b293fa6a004bd89154f8.jpg)

![916612176cddc201f87dfc58e727bce2bd02489f526c38c8052591449cda8c66.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/916612176cddc201f87dfc58e727bce2bd02489f526c38c8052591449cda8c66.jpg)

![9327977f10dc7a512a83088f00076251236d246770f30bb2e490e45452a7f855.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/9327977f10dc7a512a83088f00076251236d246770f30bb2e490e45452a7f855.jpg)

![bf15fa611014086441ed589c8d7436d7976c9d2e5b86c95fda3502cecf238a22.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/bf15fa611014086441ed589c8d7436d7976c9d2e5b86c95fda3502cecf238a22.jpg)

![db9ecd6dff7c28fb02a1b245efd7d6b96e77a601faf20861fc08073bc36afb70.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/db9ecd6dff7c28fb02a1b245efd7d6b96e77a601faf20861fc08073bc36afb70.jpg)

![e0e2c66470447147d314356bfbd54d197e4357fb5047a7e4b789d371892b2e50.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/e0e2c66470447147d314356bfbd54d197e4357fb5047a7e4b789d371892b2e50.jpg)

![e97d9566684bd7e1e05c4a9259b5e4d0fee25e5fec6f80e9b614d963033ad69a.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/e97d9566684bd7e1e05c4a9259b5e4d0fee25e5fec6f80e9b614d963033ad69a.jpg)

![ec58929cd007386ec6175e1daa8cf750f5fe936fbcae63839202dd484059e8c7.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/ec58929cd007386ec6175e1daa8cf750f5fe936fbcae63839202dd484059e8c7.jpg)

![f6a79cf07889469942115fb2acb6996ca4f9a1008c5391228e5f19b62fb15145.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/f6a79cf07889469942115fb2acb6996ca4f9a1008c5391228e5f19b62fb15145.jpg)

![f96026ffa2729576dc508794f0410a6e45b1d956753f7703e83471d704f524ec.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/f96026ffa2729576dc508794f0410a6e45b1d956753f7703e83471d704f524ec.jpg)

![f9bc3e635f95587f930cfc0b7d0ebeb603d9c5fd2d675805537493a5484abc54.jpg](../iclr_results/525_Rewarding Progress_ Scaling Automated Process Verifiers for LLM Reasoning/images/f9bc3e635f95587f930cfc0b7d0ebeb603d9c5fd2d675805537493a5484abc54.jpg)

## Hymba: A Hybrid-head Architecture for Small Language Models


### Images

![0a1e5934676234c820b6f77c83370ff60db8de9df5afc742661fd7aa166b7577.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/0a1e5934676234c820b6f77c83370ff60db8de9df5afc742661fd7aa166b7577.jpg)

![2e8a97ccada3ee5655468ccf5e1477df64bf85d4a6ec834d3fbf0f400ac3cdb2.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/2e8a97ccada3ee5655468ccf5e1477df64bf85d4a6ec834d3fbf0f400ac3cdb2.jpg)

![47f5572dd49d8912fe8d72abecc003e9ed47604ae52220b1714ab734f37d00ae.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/47f5572dd49d8912fe8d72abecc003e9ed47604ae52220b1714ab734f37d00ae.jpg)

![487eef17637f72b16a48486f987c9d582433b8e0f5dfebe5e5c624e44a9ac2b7.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/487eef17637f72b16a48486f987c9d582433b8e0f5dfebe5e5c624e44a9ac2b7.jpg)

![4989c7cb2887908b3dd225c086a7bc51991c3059ff663cf6dc7f0fd4a10f5c9d.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/4989c7cb2887908b3dd225c086a7bc51991c3059ff663cf6dc7f0fd4a10f5c9d.jpg)

![5ab02cf716773223d838609e9f19a5bc772fc850139f8dda42fe5072330ff90d.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/5ab02cf716773223d838609e9f19a5bc772fc850139f8dda42fe5072330ff90d.jpg)

![7082d7e770375fe870f3d6401cec40779399accc02d5d0c5669ef8308a98f030.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/7082d7e770375fe870f3d6401cec40779399accc02d5d0c5669ef8308a98f030.jpg)

![71808b5ed6622a72b9ab462c97279c0a5ea31f53e93ed89511a7cd8714e46de9.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/71808b5ed6622a72b9ab462c97279c0a5ea31f53e93ed89511a7cd8714e46de9.jpg)

![8e23ac8e7be2069903680dc9a54002d3dd8dd25d2a06f513732752ef1bb05014.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/8e23ac8e7be2069903680dc9a54002d3dd8dd25d2a06f513732752ef1bb05014.jpg)

![a5429f70d33eb0f951ed452e7d50061dbb9c3230a3fb5a32d8ee4703b63ebe11.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/a5429f70d33eb0f951ed452e7d50061dbb9c3230a3fb5a32d8ee4703b63ebe11.jpg)

![a9f67bce711e84b003c5410a1cf82e2c907d43177770507b70101a65810a06c2.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/a9f67bce711e84b003c5410a1cf82e2c907d43177770507b70101a65810a06c2.jpg)

![c0095cf999a1068fdc2377ce392881400245c6bcf961da6228dbc195d555e6ea.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/c0095cf999a1068fdc2377ce392881400245c6bcf961da6228dbc195d555e6ea.jpg)

![c2e0d88237652632bc2302dc2970937f01158525c3ef8d7b0bf23ffc2dd18517.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/c2e0d88237652632bc2302dc2970937f01158525c3ef8d7b0bf23ffc2dd18517.jpg)

![d31ff20ffc287d8457dddb2fc86ade1fd151deb69a335b81be564d2ec95ebe0b.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/images/d31ff20ffc287d8457dddb2fc86ade1fd151deb69a335b81be564d2ec95ebe0b.jpg)

### Tables

![4c3a628d2ff4b5b57395c90d0fe1f6e307d77b537899c9f92616b2c88ab915ac.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/4c3a628d2ff4b5b57395c90d0fe1f6e307d77b537899c9f92616b2c88ab915ac.jpg)

![4d0b491298910c3fd159d2a180d3e29da131b9233bcad1766255195d88ccb66f.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/4d0b491298910c3fd159d2a180d3e29da131b9233bcad1766255195d88ccb66f.jpg)

![55af2404d2c9151c3122f3012cbaa5b191ced6a0f1da0b853e7e3efad3a902fa.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/55af2404d2c9151c3122f3012cbaa5b191ced6a0f1da0b853e7e3efad3a902fa.jpg)

![5647c6ea19ab04c1a2a7c23f04b683db61c743f6353021040ed6655e2eb11330.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/5647c6ea19ab04c1a2a7c23f04b683db61c743f6353021040ed6655e2eb11330.jpg)

![67054387f2d887a6967a7169564987d9c706ff4221a16e233d9a72d6fd782210.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/67054387f2d887a6967a7169564987d9c706ff4221a16e233d9a72d6fd782210.jpg)

![76777c3bc076146dfac7c20ca34844032ec3219f959baffd3ec1244e2ad48012.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/76777c3bc076146dfac7c20ca34844032ec3219f959baffd3ec1244e2ad48012.jpg)

![7e1509ff9bb3862a1648a420019f082fc1e2ba4cb712ab0ba50fc57504efc372.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/7e1509ff9bb3862a1648a420019f082fc1e2ba4cb712ab0ba50fc57504efc372.jpg)

![a91036063971eb1d10d401cf70018d501915d881ee05540c3ef64a28285d4727.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/a91036063971eb1d10d401cf70018d501915d881ee05540c3ef64a28285d4727.jpg)

![d57b99c748051ce79befce248dbef24460510867d1ccda6255e0fa7a38d81ea6.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/d57b99c748051ce79befce248dbef24460510867d1ccda6255e0fa7a38d81ea6.jpg)

![dc48ee945ac7afc6fe9f942da21659d0f48443c19eeb81ffecdb23962cdd8889.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/dc48ee945ac7afc6fe9f942da21659d0f48443c19eeb81ffecdb23962cdd8889.jpg)

![e33f7b81a28eaef3a94efc6c00b7f8b0318e31ab0bafa0c583fa9a410a3883bc.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/e33f7b81a28eaef3a94efc6c00b7f8b0318e31ab0bafa0c583fa9a410a3883bc.jpg)

![ea1d622dc3545909b6f92f65d0716e832ca791cd4708f4790e445ff66b4db655.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/ea1d622dc3545909b6f92f65d0716e832ca791cd4708f4790e445ff66b4db655.jpg)

![f3f209016d9e34e088c43f965e2625ff472cebd077f6604edd35f37f303679db.jpg](../iclr_results/526_Hymba_ A Hybrid-head Architecture for Small Language Models/tables/f3f209016d9e34e088c43f965e2625ff472cebd077f6604edd35f37f303679db.jpg)

## AutoCGP: Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations


### Images

![0c434b136b9e91ba1bd289fc81c243f393b40d48c768635c36a72f45c86a2576.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/0c434b136b9e91ba1bd289fc81c243f393b40d48c768635c36a72f45c86a2576.jpg)

![1ac342efe8ec25a395693e046f7ec57fdae5f5fe48092204cef0883b70eac955.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/1ac342efe8ec25a395693e046f7ec57fdae5f5fe48092204cef0883b70eac955.jpg)

![1efe0046c68c9433cf9a9611c6ae0f41d799631313d7d3821ef317f08016a4df.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/1efe0046c68c9433cf9a9611c6ae0f41d799631313d7d3821ef317f08016a4df.jpg)

![2f126ff27785694592f67ad924e828389aa062a955f8fc5d2d649398715300e3.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/2f126ff27785694592f67ad924e828389aa062a955f8fc5d2d649398715300e3.jpg)

![387acfeda22f840e02666a92c1d1638a722425b05c59df799519a9d16515a7ce.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/387acfeda22f840e02666a92c1d1638a722425b05c59df799519a9d16515a7ce.jpg)

![43d2748b09d5b549256ea192f9f2c73f119d545c1d229420df33be859905f7d4.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/43d2748b09d5b549256ea192f9f2c73f119d545c1d229420df33be859905f7d4.jpg)

![4b3983538d96ea3a26cd0e02310f6c1c2392eaca9ad0770b57d6ad705c83a1a2.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/4b3983538d96ea3a26cd0e02310f6c1c2392eaca9ad0770b57d6ad705c83a1a2.jpg)

![5df38b5f0c3e9c1929b8f6724f14ec2c45d17ddbf4683df4248b2444974e38e2.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/5df38b5f0c3e9c1929b8f6724f14ec2c45d17ddbf4683df4248b2444974e38e2.jpg)

![6325c7266170592548c9efcbe39acce8f0595adffc30e852552cd6e918b40eee.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/6325c7266170592548c9efcbe39acce8f0595adffc30e852552cd6e918b40eee.jpg)

![7d96c2f86d0d9816be41bdae227b8a973cdd656289e6aececb353381e41891bd.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/7d96c2f86d0d9816be41bdae227b8a973cdd656289e6aececb353381e41891bd.jpg)

![96273952493496f7ff09203c142dca5d7a3c0031d31c86382e9329328991dd57.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/96273952493496f7ff09203c142dca5d7a3c0031d31c86382e9329328991dd57.jpg)

![a5c432cfabdd6bcb1b016e7163692638bed2b336c349a46738a3f6195fa412c1.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/a5c432cfabdd6bcb1b016e7163692638bed2b336c349a46738a3f6195fa412c1.jpg)

![c68069316fcdeba95402f17b6e2a501d7e56b52d475882acec89501085b13753.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/c68069316fcdeba95402f17b6e2a501d7e56b52d475882acec89501085b13753.jpg)

![e9248540bc859d2f8b25084564487ab472ee5e13cb0b516c19c52459c0690743.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/e9248540bc859d2f8b25084564487ab472ee5e13cb0b516c19c52459c0690743.jpg)

![f4a360bd2559e10581d2a091f153a771a7a813e652b0c82b4a3b90cf6f025927.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/images/f4a360bd2559e10581d2a091f153a771a7a813e652b0c82b4a3b90cf6f025927.jpg)

### Tables

![07d12c81ac1c46910c0852abfedbc0d6d61b2b05b1164c4e2f424c28539c6e12.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/tables/07d12c81ac1c46910c0852abfedbc0d6d61b2b05b1164c4e2f424c28539c6e12.jpg)

![7669ac5bb502cf605432e32873bd6f8fd62b0d443207ef546fbe8fceb17f4beb.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/tables/7669ac5bb502cf605432e32873bd6f8fd62b0d443207ef546fbe8fceb17f4beb.jpg)

![8eb583f1dd796b0fce0bb6b2633caba16c2e8b57f37e90eb7b7d34f4b46a32df.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/tables/8eb583f1dd796b0fce0bb6b2633caba16c2e8b57f37e90eb7b7d34f4b46a32df.jpg)

![b666f521784a028b79a44d4655a2f1665a2063a9929059e90dc8bc599390ddad.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/tables/b666f521784a028b79a44d4655a2f1665a2063a9929059e90dc8bc599390ddad.jpg)

![eab93c9e2dce34dacee1a235c5bac869cbdbfb394ca9eeb48b563b1733301485.jpg](../iclr_results/527_AutoCGP_ Closed-Loop Concept-Guided Policies from Unlabeled Demonstrations/tables/eab93c9e2dce34dacee1a235c5bac869cbdbfb394ca9eeb48b563b1733301485.jpg)

## A CLIP-Powered Framework for Robust and Generalizable Data Selection


### Images

![180b37c49bc1f9d38d4a9f1a334854eec1faed64eaab940db4590319aa6e539c.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/180b37c49bc1f9d38d4a9f1a334854eec1faed64eaab940db4590319aa6e539c.jpg)

![25dfa83db8cabdc336506ce9b6acf71b095f3c6c8fbb9e8d3b83cbffed244e9b.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/25dfa83db8cabdc336506ce9b6acf71b095f3c6c8fbb9e8d3b83cbffed244e9b.jpg)

![494369cb8547b5b1c05eb7b347500dfae576ab802282a1c71cb3e1142b5f1bf4.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/494369cb8547b5b1c05eb7b347500dfae576ab802282a1c71cb3e1142b5f1bf4.jpg)

![49af9932e31a9a82be10fe14b1d1b99d2a7bbcdd0386236fc2fb7f32abb6edc2.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/49af9932e31a9a82be10fe14b1d1b99d2a7bbcdd0386236fc2fb7f32abb6edc2.jpg)

![659b32a8f34c0a104566964da1528511eec3374ec92b28dad6a944807fecc44a.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/659b32a8f34c0a104566964da1528511eec3374ec92b28dad6a944807fecc44a.jpg)

![831c9f5103f9910c6f61f9db9b338dd97b528abaa989ac13d56faf06e534c22e.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/831c9f5103f9910c6f61f9db9b338dd97b528abaa989ac13d56faf06e534c22e.jpg)

![8391387ab3945ae25e8cd543d7af5081915f92e20d6dc756f12f3b5d3804020e.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/8391387ab3945ae25e8cd543d7af5081915f92e20d6dc756f12f3b5d3804020e.jpg)

![c6aaf43f217291f2f7f9aef6cf33c0439381b39756177bc98e9886eab8b52304.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/c6aaf43f217291f2f7f9aef6cf33c0439381b39756177bc98e9886eab8b52304.jpg)

![cb3c8ca2be10f8f16f6cfb95400da2f2ddcde6fd8ac10debdbff10cb3b67c9de.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/cb3c8ca2be10f8f16f6cfb95400da2f2ddcde6fd8ac10debdbff10cb3b67c9de.jpg)

![e15069cc0d68d6fa59493151fecdcab29f07793101f5fb7553396f07bae4d386.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/images/e15069cc0d68d6fa59493151fecdcab29f07793101f5fb7553396f07bae4d386.jpg)

### Tables

![0685fbbfa428bf47e1862132b0ead86cf1a17020a6cf9fe88d58110001868ab9.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/0685fbbfa428bf47e1862132b0ead86cf1a17020a6cf9fe88d58110001868ab9.jpg)

![51fc2bf51b40744ab1d6930595b37067f95d00d6747bada8815663b057f1a538.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/51fc2bf51b40744ab1d6930595b37067f95d00d6747bada8815663b057f1a538.jpg)

![774677bc0337fea1e4ffffb4221fce2ea937403c8ebb16075d8e01e0648db3fd.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/774677bc0337fea1e4ffffb4221fce2ea937403c8ebb16075d8e01e0648db3fd.jpg)

![8f02f7dad9154ce279484cecc6f0d8bc35a5fe5a233bfdab9d3d7cdc7720de3d.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/8f02f7dad9154ce279484cecc6f0d8bc35a5fe5a233bfdab9d3d7cdc7720de3d.jpg)

![92373b9121321868ec78361db5ae4a19543fd1c26a37f15fe3931d80706aa5fe.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/92373b9121321868ec78361db5ae4a19543fd1c26a37f15fe3931d80706aa5fe.jpg)

![a0270a20a2acd342e5660d0a48db17c1cdbbd637d9d8411868f62faa6b4cead6.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/a0270a20a2acd342e5660d0a48db17c1cdbbd637d9d8411868f62faa6b4cead6.jpg)

![b2fd67a1e8f5dd628f0bbe564034cf6cd46984d7c8cd455253432e9f8b13a0a5.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/b2fd67a1e8f5dd628f0bbe564034cf6cd46984d7c8cd455253432e9f8b13a0a5.jpg)

![b9655cd71333aa0f41c400b87cda8c32bc0ce7dd60cab6d9b7ed78849c9d8335.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/b9655cd71333aa0f41c400b87cda8c32bc0ce7dd60cab6d9b7ed78849c9d8335.jpg)

![cbfa6d1fd003d078b9d2b86fdacaba25b89a7ebd6514d07b371696219723f06c.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/cbfa6d1fd003d078b9d2b86fdacaba25b89a7ebd6514d07b371696219723f06c.jpg)

![e38c6734ef208b57f53112f8e994886d4cb84b350f8bff4923f8d4b182884eaa.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/e38c6734ef208b57f53112f8e994886d4cb84b350f8bff4923f8d4b182884eaa.jpg)

![f030f992c5c4d68bb38e35f827cdae8b4a51e3030449fe64031ce9789e588955.jpg](../iclr_results/528_A CLIP-Powered Framework for Robust and Generalizable Data Selection/tables/f030f992c5c4d68bb38e35f827cdae8b4a51e3030449fe64031ce9789e588955.jpg)

## OSDA Agent: Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agents


### Images

![0969cacaa89d0e62741ef6489caaf2a4c457f8cba5f56954339dd6cf9c4819c3.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/0969cacaa89d0e62741ef6489caaf2a4c457f8cba5f56954339dd6cf9c4819c3.jpg)

![12265bb15ba9b6fd9d35e513c206a9c46ee0ac3ba5e5d0393135933bcd1f9dbc.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/12265bb15ba9b6fd9d35e513c206a9c46ee0ac3ba5e5d0393135933bcd1f9dbc.jpg)

![12dec36923828ad11acf5894b89eb0a9776b86a2ba120d792fb9cb6a45151a63.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/12dec36923828ad11acf5894b89eb0a9776b86a2ba120d792fb9cb6a45151a63.jpg)

![1e89013fbcdf970eed48f690cb04041ee87cb1b3f6356d5517c0fd7e6badcd21.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/1e89013fbcdf970eed48f690cb04041ee87cb1b3f6356d5517c0fd7e6badcd21.jpg)

![206218b5442fe706aa41907d1329a4c241a7b66e115ac4b91d97a6120f7ccad9.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/206218b5442fe706aa41907d1329a4c241a7b66e115ac4b91d97a6120f7ccad9.jpg)

![2737f91fd6e74995005bc612db2033b6d73c02b73d1f835228f7c3da8c12cae2.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/2737f91fd6e74995005bc612db2033b6d73c02b73d1f835228f7c3da8c12cae2.jpg)

![2c301aa575ce2f5c4ffb41f53085c62b8983f35cf9e3e913c933a4c8a9ee938c.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/2c301aa575ce2f5c4ffb41f53085c62b8983f35cf9e3e913c933a4c8a9ee938c.jpg)

![334c87505017f36ac96573da9e4802508864c9ceee027b0bbbaec774977c13a9.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/334c87505017f36ac96573da9e4802508864c9ceee027b0bbbaec774977c13a9.jpg)

![53c66438d1c9f7e61c99a542fb5d9d290da67b3502b040596eeb38895f242208.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/53c66438d1c9f7e61c99a542fb5d9d290da67b3502b040596eeb38895f242208.jpg)

![550b841f36cd4c6dc923333b50ad6436356ca5bd23d47bfdd373965feb617a8d.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/550b841f36cd4c6dc923333b50ad6436356ca5bd23d47bfdd373965feb617a8d.jpg)

![5dfdb1835a661cc487c33653dfcddea5c41fc6b79831cbc2ee2d1964fb262401.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/5dfdb1835a661cc487c33653dfcddea5c41fc6b79831cbc2ee2d1964fb262401.jpg)

![801aa8c562416ab4323d86fe9f70a4360b498b80e480ad3a6a2ccab55c70265e.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/801aa8c562416ab4323d86fe9f70a4360b498b80e480ad3a6a2ccab55c70265e.jpg)

![866f6b5dac1e011591223cdff6ccaccc93457c2eb8936c2173e76e29ce087a80.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/866f6b5dac1e011591223cdff6ccaccc93457c2eb8936c2173e76e29ce087a80.jpg)

![94d857f5ce14bf4337e850ab27b65415af5df74846408f77748748317428506f.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/94d857f5ce14bf4337e850ab27b65415af5df74846408f77748748317428506f.jpg)

![a07356f0e230247219e5e02f25c8d19f20aa4ea49def902d9e00137874023f99.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/a07356f0e230247219e5e02f25c8d19f20aa4ea49def902d9e00137874023f99.jpg)

![b92d8cd03545813924045ef6d25176768ac6a59251e15ac389877dcbeec69c67.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/b92d8cd03545813924045ef6d25176768ac6a59251e15ac389877dcbeec69c67.jpg)

![ba13e295ff943abb5523a085f9783d4566a8e2402031cd04145371ad4bc2598e.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/ba13e295ff943abb5523a085f9783d4566a8e2402031cd04145371ad4bc2598e.jpg)

![bfcd7e2941ede1132630f2ffd6e59b29414bf109b1f2e0067c69698bced71726.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/bfcd7e2941ede1132630f2ffd6e59b29414bf109b1f2e0067c69698bced71726.jpg)

![c2c31c2b26d0d373b7cbd915cbc5c793b8542406fbbc2319aca0a678c0337a47.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/c2c31c2b26d0d373b7cbd915cbc5c793b8542406fbbc2319aca0a678c0337a47.jpg)

![f77edc2444742119951da884ff8e2f5cab263bb02bfe9c2e686e12d150e4e2a3.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/images/f77edc2444742119951da884ff8e2f5cab263bb02bfe9c2e686e12d150e4e2a3.jpg)

### Tables

![2c891000ff5095df3340af7c5ca03d4397d79868e04b11e7ab27cbad2ce7bb47.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/tables/2c891000ff5095df3340af7c5ca03d4397d79868e04b11e7ab27cbad2ce7bb47.jpg)

![5890a7aaaacb341176d5a67f9408e4086d76f91198539209d2d39112c9e56114.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/tables/5890a7aaaacb341176d5a67f9408e4086d76f91198539209d2d39112c9e56114.jpg)

![863caa62caa84659c040c44f39bacee3c85a967adc21e4d333a766f67a2f95fd.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/tables/863caa62caa84659c040c44f39bacee3c85a967adc21e4d333a766f67a2f95fd.jpg)

![bef174840ead082e9e997033169afc77a52b49a03fa1e787a04f74f9bbcd7e41.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/tables/bef174840ead082e9e997033169afc77a52b49a03fa1e787a04f74f9bbcd7e41.jpg)

![d1527f0d80b690c48db93268b392b9ed21abbe9b27c3b10856314c239f09a1ea.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/tables/d1527f0d80b690c48db93268b392b9ed21abbe9b27c3b10856314c239f09a1ea.jpg)

![d630a64641dc51f38eb597884521b22e1a2f2d1690f4e9dec67db44ecca68402.jpg](../iclr_results/529_OSDA Agent_ Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agent/tables/d630a64641dc51f38eb597884521b22e1a2f2d1690f4e9dec67db44ecca68402.jpg)

## Sample then Identify: A General Framework for Risk Control and Assessment in Multimodal Large Language Models


### Images

![05aa4ce69524c1cd8f5e3ce173e4076724eeec52e994a585c403d86c7462f4a0.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/05aa4ce69524c1cd8f5e3ce173e4076724eeec52e994a585c403d86c7462f4a0.jpg)

![14e2c7d143e0015ad156ae23f1669f53d6984f191882a8b4f133f19ec4b86ce6.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/14e2c7d143e0015ad156ae23f1669f53d6984f191882a8b4f133f19ec4b86ce6.jpg)

![2aeba47475f7daa7adf222fda74d339eff4cc8a7108dc9ce20975808bc2fbe56.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/2aeba47475f7daa7adf222fda74d339eff4cc8a7108dc9ce20975808bc2fbe56.jpg)

![2f199229681be56db2e1275a4d8f78257f3822b7eaf0a3f40dab3c511a65ff8e.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/2f199229681be56db2e1275a4d8f78257f3822b7eaf0a3f40dab3c511a65ff8e.jpg)

![38a836e0d604f3a97b7ff3d21387c462d1800066a25265c10d1e88948b8b9435.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/38a836e0d604f3a97b7ff3d21387c462d1800066a25265c10d1e88948b8b9435.jpg)

![3eb7b0784ee0eb9f480a3691c930bbf68179b96890adf29fbc6c1307fd090147.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/3eb7b0784ee0eb9f480a3691c930bbf68179b96890adf29fbc6c1307fd090147.jpg)

![507eec0c6adf542c18a4e32b7501309dbc84650dc25ee655cda3727340be12a3.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/507eec0c6adf542c18a4e32b7501309dbc84650dc25ee655cda3727340be12a3.jpg)

![5fd78f8d025b9d84dc1a0efb1bbc550d9c3f39ddf7162472e13df4b5fc19461f.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/5fd78f8d025b9d84dc1a0efb1bbc550d9c3f39ddf7162472e13df4b5fc19461f.jpg)

![913aaa0041ec665ca44bec394c3bc62a5b4b9eedb1b0b5647bd4859b6bb84144.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/images/913aaa0041ec665ca44bec394c3bc62a5b4b9eedb1b0b5647bd4859b6bb84144.jpg)

### Tables

![0ba400254ef469517369c78c4ae9444ed3419c2526dc758514889a01ffdce4e4.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/tables/0ba400254ef469517369c78c4ae9444ed3419c2526dc758514889a01ffdce4e4.jpg)

![ac74792d3dbcdcc794c6ed397698ad33b654d89d49ae6448aea6fa0348e8cdcb.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/tables/ac74792d3dbcdcc794c6ed397698ad33b654d89d49ae6448aea6fa0348e8cdcb.jpg)

![d178ad0743da1ad2aa180aecf0a78885206dfd513b2bb38541214f98fa4b13c4.jpg](../iclr_results/530_Sample then Identify_ A General Framework for Risk Control and Assessment in Multimodal Large Langua/tables/d178ad0743da1ad2aa180aecf0a78885206dfd513b2bb38541214f98fa4b13c4.jpg)

## Conditional Diffusion with Ordinal Regression: Longitudinal Data Generation for Neurodegenerative Disease Studies


### Images

![0d0789beab4962989f3094c41144e5baf4952081ed23717eaccdc6380ca1b63b.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/0d0789beab4962989f3094c41144e5baf4952081ed23717eaccdc6380ca1b63b.jpg)

![0fe9a801c707c36856d0f075a9ebc86c74f22642e384134cc6bdce011b63eaed.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/0fe9a801c707c36856d0f075a9ebc86c74f22642e384134cc6bdce011b63eaed.jpg)

![58bf828363be836642c886480342ea0c2309cd6358d6cc5627d47b3433bc8e24.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/58bf828363be836642c886480342ea0c2309cd6358d6cc5627d47b3433bc8e24.jpg)

![5b755065d7fcc6092b3ccd2de31fc9050569e7b444cf4e465c0964b36f0adaac.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/5b755065d7fcc6092b3ccd2de31fc9050569e7b444cf4e465c0964b36f0adaac.jpg)

![7919043d45b8b59fc72b040ae38321f67328f1572c8fb98f92b6d34be2a49000.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/7919043d45b8b59fc72b040ae38321f67328f1572c8fb98f92b6d34be2a49000.jpg)

![a6ee1bac0548e1c038325e943ac5ba6781997826997f7ec26fde448e29bbc799.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/a6ee1bac0548e1c038325e943ac5ba6781997826997f7ec26fde448e29bbc799.jpg)

![b11fd96f5077727a530cba5ba6094705c64644f8500bd62840204087f5bccc81.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/b11fd96f5077727a530cba5ba6094705c64644f8500bd62840204087f5bccc81.jpg)

![cfb2e528e0633b15b0beefe2519ecc8014dfb23a2e171f1b9445204944d10280.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/images/cfb2e528e0633b15b0beefe2519ecc8014dfb23a2e171f1b9445204944d10280.jpg)

### Tables

![201def3be6fb088b32ac01e1049465714aa7ae62eb0da1ca812c1b21002d740e.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/tables/201def3be6fb088b32ac01e1049465714aa7ae62eb0da1ca812c1b21002d740e.jpg)

![988fb7431b26c5fce491d20d4905163b3cdb88e393d4cb5d4a87ec63d199c14c.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/tables/988fb7431b26c5fce491d20d4905163b3cdb88e393d4cb5d4a87ec63d199c14c.jpg)

![9a02011a5af1049ff5731055f4548369170e9b7857bddd4fcc97d1425ee8dc99.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/tables/9a02011a5af1049ff5731055f4548369170e9b7857bddd4fcc97d1425ee8dc99.jpg)

![c69cbe3ff9c07be3eece926072d92ca0aa753e8dd8ae4ef72d22e0208c6ad2bd.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/tables/c69cbe3ff9c07be3eece926072d92ca0aa753e8dd8ae4ef72d22e0208c6ad2bd.jpg)

![d2727e716be8f3c01e1e9e7f877579eecffeac50532de6274b832b3142972bc4.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/tables/d2727e716be8f3c01e1e9e7f877579eecffeac50532de6274b832b3142972bc4.jpg)

![dce92c4580c421101d13e92465bc9d4c63250bb1a5c8a7b58f5459f1fcb72f1f.jpg](../iclr_results/531_Conditional Diffusion with Ordinal Regression_ Longitudinal Data Generation for Neurodegenerative Di/tables/dce92c4580c421101d13e92465bc9d4c63250bb1a5c8a7b58f5459f1fcb72f1f.jpg)

## SplatFormer: Point Transformer for Robust 3D Gaussian Splatting


### Images

![095998d7ea98ca64e870de9c0b0ade79086f3b9da423d84b190b9093621c68e5.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/095998d7ea98ca64e870de9c0b0ade79086f3b9da423d84b190b9093621c68e5.jpg)

![208d8a9ea9f8e90a2c2e108a628848b0603c2487f92c826c2388247278b68367.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/208d8a9ea9f8e90a2c2e108a628848b0603c2487f92c826c2388247278b68367.jpg)

![26335f5c3de57bd9d1958d48114c76254e7fa4193a1719553558a24dd58257f6.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/26335f5c3de57bd9d1958d48114c76254e7fa4193a1719553558a24dd58257f6.jpg)

![52dd8cfe5f61ae5c161c680c6a0566e9fada39ba341687ba3aa92a72f7736c62.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/52dd8cfe5f61ae5c161c680c6a0566e9fada39ba341687ba3aa92a72f7736c62.jpg)

![5a9f8a563c9c240b86a65f01caa65267858ced52df4bcd17c5fdeb5f31b8e82b.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/5a9f8a563c9c240b86a65f01caa65267858ced52df4bcd17c5fdeb5f31b8e82b.jpg)

![65a3829b9f15499e53408996f9be895e1bb4af9e02cc0848ab04de2b0b52ed29.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/65a3829b9f15499e53408996f9be895e1bb4af9e02cc0848ab04de2b0b52ed29.jpg)

![7ad294ff93821373f94e0e75beb896ab85f7fa9f58ef2819584251db15083b8c.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/7ad294ff93821373f94e0e75beb896ab85f7fa9f58ef2819584251db15083b8c.jpg)

![88aa391d714ffd2b798af4d8d40407abcced3295a87e266c79a81b75048055be.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/88aa391d714ffd2b798af4d8d40407abcced3295a87e266c79a81b75048055be.jpg)

![8d27a534b3695c8a368e69e0b4e251b794f8462282ffdbe54c8aea76811d8075.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/8d27a534b3695c8a368e69e0b4e251b794f8462282ffdbe54c8aea76811d8075.jpg)

![a38c7345b94ff49f5cebeca7fdf39fb4e9c03e04f0e4f4a93a163ab3d4d8d972.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/a38c7345b94ff49f5cebeca7fdf39fb4e9c03e04f0e4f4a93a163ab3d4d8d972.jpg)

![b42b5b77ffcc44a59548555bf2cbd1147695789add50ca2885eadf566b00b830.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/b42b5b77ffcc44a59548555bf2cbd1147695789add50ca2885eadf566b00b830.jpg)

![c0d007d09ab1405b4ee2e3ebdc37b14e7b3a2df81525fffe8dadc666585c9926.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/c0d007d09ab1405b4ee2e3ebdc37b14e7b3a2df81525fffe8dadc666585c9926.jpg)

![ea99aac31cd08f952c9596d8fc12c2c5404282c49db0bceda0f00dca9f4ba7f2.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/ea99aac31cd08f952c9596d8fc12c2c5404282c49db0bceda0f00dca9f4ba7f2.jpg)

![fc8f37c192c1694de1b8dae7dd840d6deae91adf85f8c1909d5c320b977fd626.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/fc8f37c192c1694de1b8dae7dd840d6deae91adf85f8c1909d5c320b977fd626.jpg)

![fe326b13f704cf42c33bdf8a341ee66c02f58b9f9bdc5a0299ff6ac2c9337811.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/images/fe326b13f704cf42c33bdf8a341ee66c02f58b9f9bdc5a0299ff6ac2c9337811.jpg)

### Tables

![0496f4167daeaf3a546de0c6a1c836d69046db45ccdc197c2c81f317117b8e98.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/0496f4167daeaf3a546de0c6a1c836d69046db45ccdc197c2c81f317117b8e98.jpg)

![2ade6d7488e1ed0d60bfd404bf1bdffdf27803af5c4be3f83753091b600e61d2.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/2ade6d7488e1ed0d60bfd404bf1bdffdf27803af5c4be3f83753091b600e61d2.jpg)

![5885eca3e2dc69e72a0cc132fd7f1ef2b240d1ffb6c1d30b7a37fbbba7d34eae.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/5885eca3e2dc69e72a0cc132fd7f1ef2b240d1ffb6c1d30b7a37fbbba7d34eae.jpg)

![6c6643769e15ddf1728d4bdce0c1e4df8e7ae9270b9ff24a0a98c21b9244e59d.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/6c6643769e15ddf1728d4bdce0c1e4df8e7ae9270b9ff24a0a98c21b9244e59d.jpg)

![6f0097d6aec3fae09aee55a92d317c191f48ab0214334247d512cb043c91f6d2.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/6f0097d6aec3fae09aee55a92d317c191f48ab0214334247d512cb043c91f6d2.jpg)

![9278011e0a6eb187f00beb2acf6d19e7b1e41f5853c235f1380619c1b6a4c42e.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/9278011e0a6eb187f00beb2acf6d19e7b1e41f5853c235f1380619c1b6a4c42e.jpg)

![a7212573951bd562998e99ab89322feceeb227c54861d1c8d4db61ad4d33d1fc.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/a7212573951bd562998e99ab89322feceeb227c54861d1c8d4db61ad4d33d1fc.jpg)

![e85c54dd58bec27837145cbe4fa1e4ad46e8ac8695dad8ada094301a661d9737.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/e85c54dd58bec27837145cbe4fa1e4ad46e8ac8695dad8ada094301a661d9737.jpg)

![eb2ad32fc3ee77fa66ee25ce8eb89be6686deea8169dd6f92eb653534c92474a.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/eb2ad32fc3ee77fa66ee25ce8eb89be6686deea8169dd6f92eb653534c92474a.jpg)

![fdcd090a5ea0cc63197830e3d5788469532ce2efbc9fc919b98e085aac1cfc65.jpg](../iclr_results/532_SplatFormer_ Point Transformer for Robust 3D Gaussian Splatting/tables/fdcd090a5ea0cc63197830e3d5788469532ce2efbc9fc919b98e085aac1cfc65.jpg)

## When do GFlowNets learn the right distribution?


### Images

![004305afdd9de78a8a44a59c307297ad17657e4214392bfe9fd61d80b8155437.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/004305afdd9de78a8a44a59c307297ad17657e4214392bfe9fd61d80b8155437.jpg)

![04ca7dfffa2a6fb11c04959d371af378076200174dec7fa86966a8102a882adf.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/04ca7dfffa2a6fb11c04959d371af378076200174dec7fa86966a8102a882adf.jpg)

![28531e2c7f7d0e6543030ae0ddf7bd43094a20fee34600463aea00f82b468157.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/28531e2c7f7d0e6543030ae0ddf7bd43094a20fee34600463aea00f82b468157.jpg)

![30d8c84a6d6cb4098bc52a2cd07ee62ff5944d3bb2b1f77eeed46079dd373761.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/30d8c84a6d6cb4098bc52a2cd07ee62ff5944d3bb2b1f77eeed46079dd373761.jpg)

![3114d4b4ef854632b96752ac9d3e652b9354293b7726c3ab24a2377dd58460c4.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/3114d4b4ef854632b96752ac9d3e652b9354293b7726c3ab24a2377dd58460c4.jpg)

![7c1e24b053e7fc9097f6aa29c7e4ee82ee046316f0c7c2c9fd817f117378e610.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/7c1e24b053e7fc9097f6aa29c7e4ee82ee046316f0c7c2c9fd817f117378e610.jpg)

![7cb2f60f89029e867386ae71ef04a80be3f6cb447a84cb8fdbc3c22b6a2317bb.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/7cb2f60f89029e867386ae71ef04a80be3f6cb447a84cb8fdbc3c22b6a2317bb.jpg)

![a87d5e03bd521939f7ec3650424140f105e8faf90845f13841aa57ae6112f154.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/a87d5e03bd521939f7ec3650424140f105e8faf90845f13841aa57ae6112f154.jpg)

![aad602e3bf90fc0917291bdfff431204f14df5908ed7d7fdea84baccd0a8372e.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/aad602e3bf90fc0917291bdfff431204f14df5908ed7d7fdea84baccd0a8372e.jpg)

![b4af2e14eb678b2c8bd0a805be24cc1442abead8d54a77d3d02f6dc0bed59ea6.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/b4af2e14eb678b2c8bd0a805be24cc1442abead8d54a77d3d02f6dc0bed59ea6.jpg)

![b7a63f79dd46d1c4822e0fd6df51505fd8068c4f7817fc2ba063b907c68ec1c5.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/b7a63f79dd46d1c4822e0fd6df51505fd8068c4f7817fc2ba063b907c68ec1c5.jpg)

![bbaaac670ddad389eefd5233b4c0c3f9f4304c984f1f2426281952a00d72eda8.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/bbaaac670ddad389eefd5233b4c0c3f9f4304c984f1f2426281952a00d72eda8.jpg)

![be3b4856d789afe7273140e7a0e853e0ab14a38876858d78c4a485f6a8a8430d.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/images/be3b4856d789afe7273140e7a0e853e0ab14a38876858d78c4a485f6a8a8430d.jpg)

### Tables

![47bc3718fbded1c740b0b4996bd065b238adc644376204bde97b3f91f5740ed8.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/tables/47bc3718fbded1c740b0b4996bd065b238adc644376204bde97b3f91f5740ed8.jpg)

![dd65afc676137c1cb18f14e470854aacc65363fe3dc30c1ae125a639fb9cc8c8.jpg](../iclr_results/533_When do GFlowNets learn the right distribution_/tables/dd65afc676137c1cb18f14e470854aacc65363fe3dc30c1ae125a639fb9cc8c8.jpg)

## On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent


### Images

![0395f05984ea420774ce9e3628c39e1fd6ebf903f29ddb94d85d88bdf2be584c.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/0395f05984ea420774ce9e3628c39e1fd6ebf903f29ddb94d85d88bdf2be584c.jpg)

![03a2a7736db58c33cefb37ecd912e28d0a37babb38c6629ac577432e7a90806c.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/03a2a7736db58c33cefb37ecd912e28d0a37babb38c6629ac577432e7a90806c.jpg)

![0744837140fb97cd4c5723b6a1ce3d05211df11b53c3c1e24f2cf496cbe49264.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/0744837140fb97cd4c5723b6a1ce3d05211df11b53c3c1e24f2cf496cbe49264.jpg)

![0fee3b3119996ea9887a85e2b2928a7bc157e2328bd73043014391f9ec4d2b74.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/0fee3b3119996ea9887a85e2b2928a7bc157e2328bd73043014391f9ec4d2b74.jpg)

![11bafac37e57e5299bc3f85548c5b384dd450940595e4f734592da38f5cf7e3e.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/11bafac37e57e5299bc3f85548c5b384dd450940595e4f734592da38f5cf7e3e.jpg)

![1dd448593a3ccc2d59114338c57ba61d89d032973326c576276ec79ec58820e9.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/1dd448593a3ccc2d59114338c57ba61d89d032973326c576276ec79ec58820e9.jpg)

![288dc3fa036664ca197e53c04c7ffe5c2b69ca76521912bea4faf4c02759d14e.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/288dc3fa036664ca197e53c04c7ffe5c2b69ca76521912bea4faf4c02759d14e.jpg)

![4efc856503af42e51c5327981c5cbe2dc28ae10c321a5b756c730ce37abe96e0.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/4efc856503af42e51c5327981c5cbe2dc28ae10c321a5b756c730ce37abe96e0.jpg)

![59c34560d2146b6acd832ef77d21742a8d2ea40d54e3bb78e5b6b81ceb6218f5.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/59c34560d2146b6acd832ef77d21742a8d2ea40d54e3bb78e5b6b81ceb6218f5.jpg)

![5c11a9c962abfae3a235b20d618779dd7c8a7e3109ef5ee73b2fae1246dc5f42.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/5c11a9c962abfae3a235b20d618779dd7c8a7e3109ef5ee73b2fae1246dc5f42.jpg)

![63c360a4f364232101ab1daf5fcdbb28e4cce21af025ed85ff45bde964b1fc21.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/63c360a4f364232101ab1daf5fcdbb28e4cce21af025ed85ff45bde964b1fc21.jpg)

![6abbf6aa701310710557b7cf6d659e76ef77f0cb5c2ab84997fe941c0d6173ee.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/6abbf6aa701310710557b7cf6d659e76ef77f0cb5c2ab84997fe941c0d6173ee.jpg)

![76e424420a95833842dd645ef0a9bf921668700be853730842cf489dceca758e.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/76e424420a95833842dd645ef0a9bf921668700be853730842cf489dceca758e.jpg)

![b646199a9097943b754bdbca938e32d2a40bca336d0762a070367c20ebb41531.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/b646199a9097943b754bdbca938e32d2a40bca336d0762a070367c20ebb41531.jpg)

![b92aa649ef9fc8b68657684744045ac13396471970a485e16cdfd32ce648f8cc.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/b92aa649ef9fc8b68657684744045ac13396471970a485e16cdfd32ce648f8cc.jpg)

![bab35b871a5dfca1925693e19a539a1e44235d6cecf5447d68fa3b73b1c97669.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/bab35b871a5dfca1925693e19a539a1e44235d6cecf5447d68fa3b73b1c97669.jpg)

![d591dd973b3305712b3f25eccdf00a28d6f88f47dd541a82bbb31fe3339d862d.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/d591dd973b3305712b3f25eccdf00a28d6f88f47dd541a82bbb31fe3339d862d.jpg)

![d9ef33398118306b5d3eedd26cd1a85a65e37e02c0a2330c6c14950db619d661.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/d9ef33398118306b5d3eedd26cd1a85a65e37e02c0a2330c6c14950db619d661.jpg)

![f853a315f0d7b2de5116ebabc60f0f43007928c120938cc927e684369d6f089c.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/images/f853a315f0d7b2de5116ebabc60f0f43007928c120938cc927e684369d6f089c.jpg)

### Tables

![0e09f11537cb2df14f2546a4bfa0b871ecea1a1b7459c5cb9d5d1eab161cff1c.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/0e09f11537cb2df14f2546a4bfa0b871ecea1a1b7459c5cb9d5d1eab161cff1c.jpg)

![0e9dc544a3ae8e0921db01825534e1d41bda31ee390de9d9d4272faed29c82ce.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/0e9dc544a3ae8e0921db01825534e1d41bda31ee390de9d9d4272faed29c82ce.jpg)

![0f8a60831d4e0ed6cf9f5be66f9f0887577d9679d7ffcd454fdc0f5110c02c6b.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/0f8a60831d4e0ed6cf9f5be66f9f0887577d9679d7ffcd454fdc0f5110c02c6b.jpg)

![159f3fe3625a1e07c971f1fefc2d088844184af8cc269b4b15c28411a0b48984.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/159f3fe3625a1e07c971f1fefc2d088844184af8cc269b4b15c28411a0b48984.jpg)

![2795b4be9df5d8c9ffa42ca01e68265db4c12226d112762fc49f9d368e8311be.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/2795b4be9df5d8c9ffa42ca01e68265db4c12226d112762fc49f9d368e8311be.jpg)

![3b3b57496401f19f44bad3e02a444aab2e8fb5f7d5215b9759eaa22d91ea32a4.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/3b3b57496401f19f44bad3e02a444aab2e8fb5f7d5215b9759eaa22d91ea32a4.jpg)

![53ce58a353d09bb9b1ed63457f1efa7f4d020ea4883b8f5eb96beb03ba681d1e.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/53ce58a353d09bb9b1ed63457f1efa7f4d020ea4883b8f5eb96beb03ba681d1e.jpg)

![6d53657213109fdbeb9db42b813b1318509402b905b2918a0f28648f363791eb.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/6d53657213109fdbeb9db42b813b1318509402b905b2918a0f28648f363791eb.jpg)

![c97e3abcb3877b7279497b0820fdb1a2d9a748344d45cac79d847f909b7387a7.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/c97e3abcb3877b7279497b0820fdb1a2d9a748344d45cac79d847f909b7387a7.jpg)

![e4003879099d2e512bbda69cccb13dccd85154efa06de22db039334c37ef461c.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/e4003879099d2e512bbda69cccb13dccd85154efa06de22db039334c37ef461c.jpg)

![f79908b9654f29ee4b867ed34fe34b16e6d75de183a1f6b7fc29cea288432c2d.jpg](../iclr_results/534_On the Optimization and Generalization of Two-layer Transformers with Sign Gradient Descent/tables/f79908b9654f29ee4b867ed34fe34b16e6d75de183a1f6b7fc29cea288432c2d.jpg)

## Towards a Unified and Verified Understanding of Group-Operation Networks


### Images

![1987094e35251eff9cc7edadb5449d332271ddf0e7d48ef4af69ba1fa961c67c.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/1987094e35251eff9cc7edadb5449d332271ddf0e7d48ef4af69ba1fa961c67c.jpg)

![1bb4802325f2d8fe876c67fa0908fbac73868efcb91c398202a09d27b19509c7.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/1bb4802325f2d8fe876c67fa0908fbac73868efcb91c398202a09d27b19509c7.jpg)

![313aca84373b9a4595af83fbca930bde126a2e77c88a681396ad48161bc7de2c.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/313aca84373b9a4595af83fbca930bde126a2e77c88a681396ad48161bc7de2c.jpg)

![364d12be9b12e3fa67eaf4baebac0b07c201bc01eb735de4dd4b9c45bcc2cd7e.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/364d12be9b12e3fa67eaf4baebac0b07c201bc01eb735de4dd4b9c45bcc2cd7e.jpg)

![3afc1ed62cbd62512078366d43077784dd0e80f01a404f9124ba6ae46080d322.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/3afc1ed62cbd62512078366d43077784dd0e80f01a404f9124ba6ae46080d322.jpg)

![9080f85d70095a40c3a59a5d55c8fd582dbdab9326cedc8b4cce44999097fd8a.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/9080f85d70095a40c3a59a5d55c8fd582dbdab9326cedc8b4cce44999097fd8a.jpg)

![a3edd8e2808437d33a5bf28a7065bc39d0e17fb8e8c61b895880c9e41d0519ee.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/a3edd8e2808437d33a5bf28a7065bc39d0e17fb8e8c61b895880c9e41d0519ee.jpg)

![a507a75797eb26e3fc3e719ff849fedc5d0884edab118554bb719dacb19e7044.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/a507a75797eb26e3fc3e719ff849fedc5d0884edab118554bb719dacb19e7044.jpg)

![d4591bb470c8ff4941f8e152d335959e4e3e58f2f485d6ac2336edaf75fa7068.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/d4591bb470c8ff4941f8e152d335959e4e3e58f2f485d6ac2336edaf75fa7068.jpg)

![e42513a3c596991c5b2f125a27c71ea9630ba059ea37f72f5e33bee29a1b48c2.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/e42513a3c596991c5b2f125a27c71ea9630ba059ea37f72f5e33bee29a1b48c2.jpg)

![f16bc0a30bba907d55f43fb8cc167e1459ac7a084c0ba40edcf8678f08ba7914.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/images/f16bc0a30bba907d55f43fb8cc167e1459ac7a084c0ba40edcf8678f08ba7914.jpg)

### Tables

![5e1007434c74869ff6c098919648fc6eadebc58c2c736502582ca2f582960c77.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/tables/5e1007434c74869ff6c098919648fc6eadebc58c2c736502582ca2f582960c77.jpg)

![95a391bc6b16c43b43a928b12ffc24a4802cd827e835ade02a5690bed08bc804.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/tables/95a391bc6b16c43b43a928b12ffc24a4802cd827e835ade02a5690bed08bc804.jpg)

![b5accdb46e2af3a5b89666c5fbf95ca44a99b970b1bcba29100c1843a26569a3.jpg](../iclr_results/535_Towards a Unified and Verified Understanding of Group-Operation Networks/tables/b5accdb46e2af3a5b89666c5fbf95ca44a99b970b1bcba29100c1843a26569a3.jpg)

## DenseMatcher: Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo


### Images

![0b16b96b0671af7b004ba6ea5a4bbb8b9c806d00988a099a80e9b9091fc32727.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/0b16b96b0671af7b004ba6ea5a4bbb8b9c806d00988a099a80e9b9091fc32727.jpg)

![1e55009b60a408b38a7b3bac72781f6a977a5746365e2f0ffb8920ec4e520c28.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/1e55009b60a408b38a7b3bac72781f6a977a5746365e2f0ffb8920ec4e520c28.jpg)

![1f206e7a7b00a44d58be2b70fd22b92da8a22625d7903be303fc4cf1f7b4a915.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/1f206e7a7b00a44d58be2b70fd22b92da8a22625d7903be303fc4cf1f7b4a915.jpg)

![23246fe1e88e3e5f1f6f4f65025fba8cc13d66c2f3f37dc326fd5c144a9266a4.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/23246fe1e88e3e5f1f6f4f65025fba8cc13d66c2f3f37dc326fd5c144a9266a4.jpg)

![438fd70b911e934fca02f233fcdbb742de95f36e6a4a0b582f1bb9358b0449b1.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/438fd70b911e934fca02f233fcdbb742de95f36e6a4a0b582f1bb9358b0449b1.jpg)

![451da19c63956b9ff9d5cfd8dd97c1f493171c8995bc5e7a52ebe990f1f57145.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/451da19c63956b9ff9d5cfd8dd97c1f493171c8995bc5e7a52ebe990f1f57145.jpg)

![52cc1587f057ccedb7440f42fe1cdce4601ec82f0495e51ac33d22465c5564f7.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/52cc1587f057ccedb7440f42fe1cdce4601ec82f0495e51ac33d22465c5564f7.jpg)

![578fa5ef8310134db4b7a407f2760aedaa875b715958f91996eeb51762423f41.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/578fa5ef8310134db4b7a407f2760aedaa875b715958f91996eeb51762423f41.jpg)

![82404ad4e662677ede1c173a92e7b234db8dbe3270314ae1f9c1c407e8e0666b.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/82404ad4e662677ede1c173a92e7b234db8dbe3270314ae1f9c1c407e8e0666b.jpg)

![c5deff92fcddc3bc0d4b8508dffcace1020532090125d8b5a5542be00e095fa7.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/c5deff92fcddc3bc0d4b8508dffcace1020532090125d8b5a5542be00e095fa7.jpg)

![c85a60c21dd5d539f2a96f6b1738045d9f06e4c7e57bee1c5a9ddcf97222d241.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/c85a60c21dd5d539f2a96f6b1738045d9f06e4c7e57bee1c5a9ddcf97222d241.jpg)

![d201f6256b0519ca8737e721421307ddf57040097c698dfdb9bc3f18a0c0a6ac.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/d201f6256b0519ca8737e721421307ddf57040097c698dfdb9bc3f18a0c0a6ac.jpg)

![f91c6884df225e87c9ca197e69ea0908ca5404b155f65d5be5d9f4104c8358b9.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/images/f91c6884df225e87c9ca197e69ea0908ca5404b155f65d5be5d9f4104c8358b9.jpg)

### Tables

![2da56faba9f7614f4f08f145cd7618b4748dc19ac7f61bc3538e12bf9fd74b44.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/tables/2da56faba9f7614f4f08f145cd7618b4748dc19ac7f61bc3538e12bf9fd74b44.jpg)

![3830695efe1fc05a2064d240fc1335ab53e63509fe8937316ec0b78ffe7776ed.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/tables/3830695efe1fc05a2064d240fc1335ab53e63509fe8937316ec0b78ffe7776ed.jpg)

![3a6d5dd0f77f5cb5c3b9c738195e02e9df96b14259e977db47c2dd3f2f1b4268.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/tables/3a6d5dd0f77f5cb5c3b9c738195e02e9df96b14259e977db47c2dd3f2f1b4268.jpg)

![3b1ddcd3d29a67eecb7f56fd5e8c89aa10d9fa5af75541923a8b489d8904b6f7.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/tables/3b1ddcd3d29a67eecb7f56fd5e8c89aa10d9fa5af75541923a8b489d8904b6f7.jpg)

![5be8b3e8b90391903b59b5afbacdc8b79051035231f797ec5e1978c17fb73e46.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/tables/5be8b3e8b90391903b59b5afbacdc8b79051035231f797ec5e1978c17fb73e46.jpg)

![a656b836a383109a90e7452985a76667c31626265ec874f47991f522f15be582.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/tables/a656b836a383109a90e7452985a76667c31626265ec874f47991f522f15be582.jpg)

![c5909d865baa18310e982a8e304ba6eba4de7864de0bed947f212eea83dea208.jpg](../iclr_results/536_DenseMatcher_ Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo/tables/c5909d865baa18310e982a8e304ba6eba4de7864de0bed947f212eea83dea208.jpg)

## Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL


### Images

![0e4e039bc3dc3e71a2c664595e7edd7edd1cf6e6b92d92f9c3feff0ea40d41e8.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/0e4e039bc3dc3e71a2c664595e7edd7edd1cf6e6b92d92f9c3feff0ea40d41e8.jpg)

![11677afc1f93410db8fa6b7bcf593e0372c4a17305550b640f1cba41962d77d2.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/11677afc1f93410db8fa6b7bcf593e0372c4a17305550b640f1cba41962d77d2.jpg)

![124c28a4b1d3392e6389cbd2bf200e715651d5f34a5f78421d5dbeb0beae9b92.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/124c28a4b1d3392e6389cbd2bf200e715651d5f34a5f78421d5dbeb0beae9b92.jpg)

![1e8ac31dcfa4ccae6fcd624f0298ba70fa2bc11cf286aaaaf6f1c32a09698d58.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/1e8ac31dcfa4ccae6fcd624f0298ba70fa2bc11cf286aaaaf6f1c32a09698d58.jpg)

![243bdd0ac62ae09496743a7a93742081ee575c089a02ecd303ac70bfb97fa220.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/243bdd0ac62ae09496743a7a93742081ee575c089a02ecd303ac70bfb97fa220.jpg)

![2a256eb869cc9dab85ae215af3cbc78469e8f37f830d052eedfecd7a36187e34.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/2a256eb869cc9dab85ae215af3cbc78469e8f37f830d052eedfecd7a36187e34.jpg)

![2af76a6f8ea9555c0e4c72c64e637d7d1e6e6ea1cd5271dc6490c40f5b24fc0f.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/2af76a6f8ea9555c0e4c72c64e637d7d1e6e6ea1cd5271dc6490c40f5b24fc0f.jpg)

![2b49a33b364da8bca65ec471a4e76660c7b1c86aa4a17d1a14be1bf7767f614d.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/2b49a33b364da8bca65ec471a4e76660c7b1c86aa4a17d1a14be1bf7767f614d.jpg)

![3fbd673e02a4491d88513124ca2efb36f7d4e3398292de6393ad223fff1a98cf.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/3fbd673e02a4491d88513124ca2efb36f7d4e3398292de6393ad223fff1a98cf.jpg)

![4d2c573c0a38d77f3ba2db15b734aee7544b071497d167c5fc659bd21205fe3e.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/4d2c573c0a38d77f3ba2db15b734aee7544b071497d167c5fc659bd21205fe3e.jpg)

![4e007955b3feff3851165f347ce4e7dd356bd98fed9b9ff4c8835cd848a7ae42.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/4e007955b3feff3851165f347ce4e7dd356bd98fed9b9ff4c8835cd848a7ae42.jpg)

![52080b00121b1cf694099562033e70dd2d549a1942744a5162c1fac9c1b2e080.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/52080b00121b1cf694099562033e70dd2d549a1942744a5162c1fac9c1b2e080.jpg)

![58e085123a6e0ad94a6df5859f3b7f0f355091904284bdc477b77f5f6a37b602.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/58e085123a6e0ad94a6df5859f3b7f0f355091904284bdc477b77f5f6a37b602.jpg)

![5c6b402cc0b8dfcb98da3a7874569edf549bcbc8ec3d794369c25c5b81d4c061.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/5c6b402cc0b8dfcb98da3a7874569edf549bcbc8ec3d794369c25c5b81d4c061.jpg)

![782ab863f6b24b86a20c00558b783906df0c531f4ed8d2b5b726487fed4d365a.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/782ab863f6b24b86a20c00558b783906df0c531f4ed8d2b5b726487fed4d365a.jpg)

![8ec0b49ef4c26b5a378b082aba63d63b5445dc40ddaac938071e794b3cc2ab81.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/8ec0b49ef4c26b5a378b082aba63d63b5445dc40ddaac938071e794b3cc2ab81.jpg)

![925227375f9b90e170c2948ff28712e26b62fc3d41dd03864849152df74a45ea.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/925227375f9b90e170c2948ff28712e26b62fc3d41dd03864849152df74a45ea.jpg)

![a20ea1acd6b0db7133aaec742e8013681622c510ecec633d27a99fa794c038dd.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/a20ea1acd6b0db7133aaec742e8013681622c510ecec633d27a99fa794c038dd.jpg)

![bb0167127649ac46492658e590626e1e6f7d8ef8ceac0b016e1405aff3f774a1.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/bb0167127649ac46492658e590626e1e6f7d8ef8ceac0b016e1405aff3f774a1.jpg)

![d66d566ef04cd406c2be0983e3e52dde12638f6303a08e2c01b2ac2792d6db84.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/d66d566ef04cd406c2be0983e3e52dde12638f6303a08e2c01b2ac2792d6db84.jpg)

![efddf26415a058bdb26079f113f005223fb841cd421e3c9a00586c82caee8af4.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/efddf26415a058bdb26079f113f005223fb841cd421e3c9a00586c82caee8af4.jpg)

![f66dcab49b89fab9b0803878608a79099a4a8f69b79255193f11b83d3a31353a.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/f66dcab49b89fab9b0803878608a79099a4a8f69b79255193f11b83d3a31353a.jpg)

![f94265aa6547ab8797be8eaa596829759a90003ab6f8c40d04cdcd63237ba70a.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/f94265aa6547ab8797be8eaa596829759a90003ab6f8c40d04cdcd63237ba70a.jpg)

![fd653c5fce5c058acc49b81afa9dcfe0af2e49829fbe8459af4d13c9795ee702.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/images/fd653c5fce5c058acc49b81afa9dcfe0af2e49829fbe8459af4d13c9795ee702.jpg)

### Tables

![fb177d27a919b20c57f16e1c0d8a709d1cd6d5982692e5a8965eec19e3c25887.jpg](../iclr_results/537_Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL/tables/fb177d27a919b20c57f16e1c0d8a709d1cd6d5982692e5a8965eec19e3c25887.jpg)

## Quality Measures for Dynamic Graph Generative Models


### Images

![7a73f04b2237c8804a752779ca9d809109678f578fb968cf25acb4c99d1ef343.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/images/7a73f04b2237c8804a752779ca9d809109678f578fb968cf25acb4c99d1ef343.jpg)

### Tables

![29716ba4c6b2ad2fad405c972c8f6ed801ffb15294d64243f092794f05a49bc5.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/tables/29716ba4c6b2ad2fad405c972c8f6ed801ffb15294d64243f092794f05a49bc5.jpg)

![2fab2e563e7dfd8cc794f5282ae14bebf6d534f173a566b46dbc96d7a4dfefb0.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/tables/2fab2e563e7dfd8cc794f5282ae14bebf6d534f173a566b46dbc96d7a4dfefb0.jpg)

![90c043f4c992f9aa7b04a7d6902ce47cd0869ed63e32aa54d6303abee7ba2fa8.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/tables/90c043f4c992f9aa7b04a7d6902ce47cd0869ed63e32aa54d6303abee7ba2fa8.jpg)

![b44b1fff22d2eef643bcc6c2e7692cd3801270a3740b7c62373309159012258d.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/tables/b44b1fff22d2eef643bcc6c2e7692cd3801270a3740b7c62373309159012258d.jpg)

![d302158c7ab5aeeb949bb37ef5ecff68aa862d17d89c8805cee6be0151fb98d8.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/tables/d302158c7ab5aeeb949bb37ef5ecff68aa862d17d89c8805cee6be0151fb98d8.jpg)

![e0ac8c57b8274466ac662edd14207a9f0333b3328f2af24b23979583f04524ef.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/tables/e0ac8c57b8274466ac662edd14207a9f0333b3328f2af24b23979583f04524ef.jpg)

![f45a4c0fd77dbb93a23fb48c0f26aa5b892f3461811fe2ef67e621f9cd0b33b5.jpg](../iclr_results/538_Quality Measures for Dynamic Graph Generative Models/tables/f45a4c0fd77dbb93a23fb48c0f26aa5b892f3461811fe2ef67e621f9cd0b33b5.jpg)

## Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Feedback from Pre-training Demonstrations


### Images

![021be0065f2313c75d73584efdddf96cff0fa839c5cfb1fba0e24ab43f16e472.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/images/021be0065f2313c75d73584efdddf96cff0fa839c5cfb1fba0e24ab43f16e472.jpg)

![0b8513d047f2367de3ead7f044a13a811c08cbab6f5a4fb259da894280389662.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/images/0b8513d047f2367de3ead7f044a13a811c08cbab6f5a4fb259da894280389662.jpg)

![4a9dd77cc7a908374d2bf5cfa235427663506444a7917663f2d4cf6d57de8c2c.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/images/4a9dd77cc7a908374d2bf5cfa235427663506444a7917663f2d4cf6d57de8c2c.jpg)

![5e4838c1a7b15afc5e673c90c0090d98339a8c9f93c3c31ff605fc6c8dc94a32.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/images/5e4838c1a7b15afc5e673c90c0090d98339a8c9f93c3c31ff605fc6c8dc94a32.jpg)

![b1cb58a6fc3cc5ae3d7fa6a629f3cf538c41501865c9090749731a95c50c501e.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/images/b1cb58a6fc3cc5ae3d7fa6a629f3cf538c41501865c9090749731a95c50c501e.jpg)

![e1633c620cbb024b021d04f3b3d7159893627e2c6550468e730eaa52dddc7598.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/images/e1633c620cbb024b021d04f3b3d7159893627e2c6550468e730eaa52dddc7598.jpg)

![e70808615281fc9992fe3be0375177f36fb09374ac2e527c235d1bdba35e5344.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/images/e70808615281fc9992fe3be0375177f36fb09374ac2e527c235d1bdba35e5344.jpg)

### Tables

![496e756c7735a75d020f811099aff3bbbe9e3f99c8577497ae03ea80b6f27486.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/tables/496e756c7735a75d020f811099aff3bbbe9e3f99c8577497ae03ea80b6f27486.jpg)

![72bed24d42f7f336e403b4a702f2a0d2d76932fe45b1e288802b8485f5837c86.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/tables/72bed24d42f7f336e403b4a702f2a0d2d76932fe45b1e288802b8485f5837c86.jpg)

![e8cdd0cf34df68c5986dd12374d066cfb4ddc18cd67fcf17e6267769e73ece03.jpg](../iclr_results/539_Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Model Using Implicit Fee/tables/e8cdd0cf34df68c5986dd12374d066cfb4ddc18cd67fcf17e6267769e73ece03.jpg)

## Better Instruction-Following Through Minimum Bayes Risk


### Images

![3801dec0ed6240e1d4f07c0464cada387fdf9f88fbb7cfbb1a89b94a868079f0.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/3801dec0ed6240e1d4f07c0464cada387fdf9f88fbb7cfbb1a89b94a868079f0.jpg)

![788fdd2092583461ef94a54f05d66e0fa46edd68ffe251befa09296093c43d7b.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/788fdd2092583461ef94a54f05d66e0fa46edd68ffe251befa09296093c43d7b.jpg)

![8ad0b56eb5452dde2e23a98d5d07aef23cdc650a43643ba53e5f5404407291ca.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/8ad0b56eb5452dde2e23a98d5d07aef23cdc650a43643ba53e5f5404407291ca.jpg)

![8d7e2070316aa000ae30c688d2fed24cbc570cd07fff2262b37e13f64c854c0b.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/8d7e2070316aa000ae30c688d2fed24cbc570cd07fff2262b37e13f64c854c0b.jpg)

![bbec5d4563a7a4691fbd27b0c7d4d1cb803e12e177b920bf579df51deb2af7f8.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/bbec5d4563a7a4691fbd27b0c7d4d1cb803e12e177b920bf579df51deb2af7f8.jpg)

![c13838f0e0375c4ed0d05df78a8a08f1bbe9f46300c3938f6263c9d7c29e537a.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/c13838f0e0375c4ed0d05df78a8a08f1bbe9f46300c3938f6263c9d7c29e537a.jpg)

![c1a769d68c152e1ee4bb68df59b1bda4048934b945f0efa2126ea714e8013867.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/c1a769d68c152e1ee4bb68df59b1bda4048934b945f0efa2126ea714e8013867.jpg)

![f4d8203c3c668663ab817edcf26dcd11b770e14d4de3148aa1ce9270e5625e42.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/f4d8203c3c668663ab817edcf26dcd11b770e14d4de3148aa1ce9270e5625e42.jpg)

![f835f4750a257027508b7d105264b24999a9ece1716fd892d185cec2601acc6a.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/images/f835f4750a257027508b7d105264b24999a9ece1716fd892d185cec2601acc6a.jpg)

### Tables

![0dec9692ec62d632dfcc4ea962416bafc9c1ff075f8f77082dcc8380325fe4db.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/0dec9692ec62d632dfcc4ea962416bafc9c1ff075f8f77082dcc8380325fe4db.jpg)

![1037181341f28fdd814718ca632a04e40a9ba224c6defe2e910555a4a2774e28.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/1037181341f28fdd814718ca632a04e40a9ba224c6defe2e910555a4a2774e28.jpg)

![14a9436dcd56363ea087b9e3350cef4bbdd6798157ff60dbea4ba2b99c1a9c47.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/14a9436dcd56363ea087b9e3350cef4bbdd6798157ff60dbea4ba2b99c1a9c47.jpg)

![1980235e0bae9f77744acb7161f1a26d460ed33d6787e77a5da17d8fcb511066.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/1980235e0bae9f77744acb7161f1a26d460ed33d6787e77a5da17d8fcb511066.jpg)

![206f126f371f05c431647e426a4efd2dccaf31bef09e44c910b29e7a63a959d1.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/206f126f371f05c431647e426a4efd2dccaf31bef09e44c910b29e7a63a959d1.jpg)

![2b515a8b32311004d866625358a3d24805f7c6b3389ee4828ef11cf9068253bb.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/2b515a8b32311004d866625358a3d24805f7c6b3389ee4828ef11cf9068253bb.jpg)

![3a4b98e6bf4fea7fda69a6ea512c3c27d07b975b02f1fe138ff7d4d8d8a7a0e8.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/3a4b98e6bf4fea7fda69a6ea512c3c27d07b975b02f1fe138ff7d4d8d8a7a0e8.jpg)

![3c133b4994d3fb56e53e3e361e9b4cf5ea7c58cee952bee6a7d0e401cedb8f26.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/3c133b4994d3fb56e53e3e361e9b4cf5ea7c58cee952bee6a7d0e401cedb8f26.jpg)

![3e7ff6edc6212db50ed7ac2cd32a0ec4c5e178c290f742f7541ef5144d86652f.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/3e7ff6edc6212db50ed7ac2cd32a0ec4c5e178c290f742f7541ef5144d86652f.jpg)

![4458c374e125e5bf0185646d315d560db7702a0824055548fa9b83c576f5ac2b.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/4458c374e125e5bf0185646d315d560db7702a0824055548fa9b83c576f5ac2b.jpg)

![4debca65b7389fb815a72e6ebd7d662a068857cb5f441c664d840c3ae6aec384.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/4debca65b7389fb815a72e6ebd7d662a068857cb5f441c664d840c3ae6aec384.jpg)

![50005f86d766ef679cf6413ffaaa2248d545d1d762d885ea1b584722cb8c1ff8.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/50005f86d766ef679cf6413ffaaa2248d545d1d762d885ea1b584722cb8c1ff8.jpg)

![630931c70c042c51a078a9dd67773feacb6198d215ab4bd230c7405b3c23b6ca.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/630931c70c042c51a078a9dd67773feacb6198d215ab4bd230c7405b3c23b6ca.jpg)

![65ffb7c66d13c941c6a3526af8b859dd072aa2e0b06d98e515f8ffacf02d66cb.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/65ffb7c66d13c941c6a3526af8b859dd072aa2e0b06d98e515f8ffacf02d66cb.jpg)

![6feef5b4c0fd3b45e9c397a5e518650689c9819fd909e07063fb9954b936558a.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/6feef5b4c0fd3b45e9c397a5e518650689c9819fd909e07063fb9954b936558a.jpg)

![8307b2ce30805fe90cd86b6f4bf00c1ef9daface75d3ff6e4ead8e4c131d9730.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/8307b2ce30805fe90cd86b6f4bf00c1ef9daface75d3ff6e4ead8e4c131d9730.jpg)

![8d9557f55d4ef792270d870b6a5248251e6da8591dc8177023863a3d05eaa550.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/8d9557f55d4ef792270d870b6a5248251e6da8591dc8177023863a3d05eaa550.jpg)

![94c9fab941e7a2d4abeb93ac8936c781120c8268d5140425f700973114e49565.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/94c9fab941e7a2d4abeb93ac8936c781120c8268d5140425f700973114e49565.jpg)

![a3f9a4a0c67b4968e851ff6d84c8e021750ebd9d7333afb44c75489a7043cbcc.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/a3f9a4a0c67b4968e851ff6d84c8e021750ebd9d7333afb44c75489a7043cbcc.jpg)

![afb37dbd0b3e02ab209f10b9306483cfc8b208336808afe443277ba427f502b1.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/afb37dbd0b3e02ab209f10b9306483cfc8b208336808afe443277ba427f502b1.jpg)

![b61302a3a9615b106224de55bb6a92da59a136263ea67b8122f3550e1827c131.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/b61302a3a9615b106224de55bb6a92da59a136263ea67b8122f3550e1827c131.jpg)

![c5e0714399d78cdbfa272806b2ab2ec88be634063376c71d5265f31d3ec1b5fe.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/c5e0714399d78cdbfa272806b2ab2ec88be634063376c71d5265f31d3ec1b5fe.jpg)

![cacfc147d60b16d8c867258b09f19dada442d1c33fc667e4d338c05f7acc6572.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/cacfc147d60b16d8c867258b09f19dada442d1c33fc667e4d338c05f7acc6572.jpg)

![d167462291b759b4e850217502dcee676a105a4b3c4d09a31667882bc355dd24.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/d167462291b759b4e850217502dcee676a105a4b3c4d09a31667882bc355dd24.jpg)

![d8af3f26cc3bfb3dbfc968702bc26394bd33499849ee4dfc1217f2964ded0639.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/d8af3f26cc3bfb3dbfc968702bc26394bd33499849ee4dfc1217f2964ded0639.jpg)

![db22a14f134d859559542ddb0b3e983652385acfda06b47316e71e3644955d25.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/db22a14f134d859559542ddb0b3e983652385acfda06b47316e71e3644955d25.jpg)

![e1457d9901d110542fc60cee84cc919ef9a16c9feae8f73ccf953d5c9e776669.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/e1457d9901d110542fc60cee84cc919ef9a16c9feae8f73ccf953d5c9e776669.jpg)

![e5b23f958804514ecce59c46594e7d33a46165bcd4254f10639a391369c462a4.jpg](../iclr_results/540_Better Instruction-Following Through Minimum Bayes Risk/tables/e5b23f958804514ecce59c46594e7d33a46165bcd4254f10639a391369c462a4.jpg)

## LiFT: Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning


### Images

![0418869f18d36aa4be09a73f2192e4894273f0275d0f390cf749c601baa947c6.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/images/0418869f18d36aa4be09a73f2192e4894273f0275d0f390cf749c601baa947c6.jpg)

![4b77651ae08be82deca482df3d68a3514f3ea4565784516537710d96f0d247fd.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/images/4b77651ae08be82deca482df3d68a3514f3ea4565784516537710d96f0d247fd.jpg)

![70d1954cc04774826d735b23a8334076f042cd4a0e3661126ffb053ea5a4b689.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/images/70d1954cc04774826d735b23a8334076f042cd4a0e3661126ffb053ea5a4b689.jpg)

![84d07a3e39b9825ff2c6cb75c966e32f9ff30d20a5a244b2918eae8cb4d2c20b.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/images/84d07a3e39b9825ff2c6cb75c966e32f9ff30d20a5a244b2918eae8cb4d2c20b.jpg)

![a29422295cb73daa8e7aecd094c6e2cd8ddd0bb8105e1fc3a0bd9f6f619dd170.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/images/a29422295cb73daa8e7aecd094c6e2cd8ddd0bb8105e1fc3a0bd9f6f619dd170.jpg)

### Tables

![22de07bd44f7831dc3710d93467681620c0ff1664d5656bb16d6981b180c88ff.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/22de07bd44f7831dc3710d93467681620c0ff1664d5656bb16d6981b180c88ff.jpg)

![2ce2a1cd7e1dbb0ebdccac63ef2b0ed76e13de5e8a2500cb543e72540c820e81.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/2ce2a1cd7e1dbb0ebdccac63ef2b0ed76e13de5e8a2500cb543e72540c820e81.jpg)

![3f0b40dcd019953e00101458ba580786a7bc9156e45ffb9ec856fcd59527e4a0.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/3f0b40dcd019953e00101458ba580786a7bc9156e45ffb9ec856fcd59527e4a0.jpg)

![3f9e98b060e1139d47a88d68467eef5c0545cea507bd4682fdac49140b15f3df.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/3f9e98b060e1139d47a88d68467eef5c0545cea507bd4682fdac49140b15f3df.jpg)

![46a73541eaf9d506c567827cb09a913f020d07c25698520456f8b6cb6a8a7ad9.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/46a73541eaf9d506c567827cb09a913f020d07c25698520456f8b6cb6a8a7ad9.jpg)

![59bc53be4dbfcbbb7301abe8fca0a57533f58d94cc577cbd5a5e31fbbf2d4f2c.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/59bc53be4dbfcbbb7301abe8fca0a57533f58d94cc577cbd5a5e31fbbf2d4f2c.jpg)

![7a3b57b5aabf02ca09a0cf16cfa5a5e67f7910f42a2cab55847db9ec32f089ce.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/7a3b57b5aabf02ca09a0cf16cfa5a5e67f7910f42a2cab55847db9ec32f089ce.jpg)

![85981593893aaec3bdc267e76a0c11f1c7adeae33a5e2b44532f5fc8c70cc47f.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/85981593893aaec3bdc267e76a0c11f1c7adeae33a5e2b44532f5fc8c70cc47f.jpg)

![86355546df13a6e16b260050e22f8cd93615ad3afd57e12414a1a26c9ddcec8d.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/86355546df13a6e16b260050e22f8cd93615ad3afd57e12414a1a26c9ddcec8d.jpg)

![923c59600802cb0b25de8091d5843dbdf4be6ac7a239590110755e5d5ede4727.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/923c59600802cb0b25de8091d5843dbdf4be6ac7a239590110755e5d5ede4727.jpg)

![a081bab9982d2aa948ced80002e8276a6a39d26626a209a861570190363d101a.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/a081bab9982d2aa948ced80002e8276a6a39d26626a209a861570190363d101a.jpg)

![ca9494fc5a5944502e7d1d82dd6c1b8257141e965cb0bd21fa5122da8e0b82de.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/ca9494fc5a5944502e7d1d82dd6c1b8257141e965cb0bd21fa5122da8e0b82de.jpg)

![ccadbf97a0d23251ca2ab58049461c804771df0722b231f76c26534cd7d6cdf1.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/ccadbf97a0d23251ca2ab58049461c804771df0722b231f76c26534cd7d6cdf1.jpg)

![f69bd3a7d6d38e76adbeaccbfbcbd8e95cc6efd3f2cc0d7f22308a5f699470ba.jpg](../iclr_results/541_LiFT_ Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning/tables/f69bd3a7d6d38e76adbeaccbfbcbd8e95cc6efd3f2cc0d7f22308a5f699470ba.jpg)

## Theory on Mixture-of-Experts in Continual Learning

### Images

![62d76c2d16d8289da9bded38b270290198cf34a43cc35c550bdea62ba024fb71.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/62d76c2d16d8289da9bded38b270290198cf34a43cc35c550bdea62ba024fb71.jpg)

![73f2f6c81bb90dc66f243f172f776d964de7c2a4d5f93cfade16a9c351b324c5.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/73f2f6c81bb90dc66f243f172f776d964de7c2a4d5f93cfade16a9c351b324c5.jpg)

![9681c5b331bd0dc911a51b93cc3d208a8224b25525537d987cf6b822badac525.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/9681c5b331bd0dc911a51b93cc3d208a8224b25525537d987cf6b822badac525.jpg)

![9cf82ff87670026b5146ae940b82ea6e21c1eaf4a57386ecf3846b646fad9d8c.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/9cf82ff87670026b5146ae940b82ea6e21c1eaf4a57386ecf3846b646fad9d8c.jpg)

![a64f9a763f7ecf1e2f8fd998ccecd46e579447290203519b0ea9cb628f722c59.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/a64f9a763f7ecf1e2f8fd998ccecd46e579447290203519b0ea9cb628f722c59.jpg)

![c9b342ebf43673d5e1faea580515427d6fdddab8777b6328fdd34f22addb3cf8.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/c9b342ebf43673d5e1faea580515427d6fdddab8777b6328fdd34f22addb3cf8.jpg)

![e92bd767d061b5b68cdf69e002a7525ba9dcad4380c3cce7ef67bf19691fbdf9.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/e92bd767d061b5b68cdf69e002a7525ba9dcad4380c3cce7ef67bf19691fbdf9.jpg)

![ecbdeafa69eb8f01f249608b4ffac0c0fb91b84f34e3189c038499d30289a761.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/ecbdeafa69eb8f01f249608b4ffac0c0fb91b84f34e3189c038499d30289a761.jpg)

![fe48b5a1d35ed2ec6c2cf7deaed21e83798b27bf4cb6cbe8f94b085ebfab55cb.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/images/fe48b5a1d35ed2ec6c2cf7deaed21e83798b27bf4cb6cbe8f94b085ebfab55cb.jpg)

### Tables

![3b0a5dd967572eab10b5524fcee60dc7cbc3d94d6f855d3b99ee067469851c0a.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/tables/3b0a5dd967572eab10b5524fcee60dc7cbc3d94d6f855d3b99ee067469851c0a.jpg)

![797d527b658b94972e9050171c63ef7c2e88639407f7489155db3f006cc52a76.jpg](../iclr_results/542_Theory on Mixture-of-Experts in Continual Learning/tables/797d527b658b94972e9050171c63ef7c2e88639407f7489155db3f006cc52a76.jpg)
