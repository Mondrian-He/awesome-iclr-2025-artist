# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 5 of 100

## 目录 (Table of Contents)

1. [MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions](#MMQA-Evaluating-LLMs-with-Multi-Table-Multi-Hop-Complex-Questions)
2. [GridMix: Exploring Spatial Modulation for Neural Fields in PDE Modeling](#GridMix-Exploring-Spatial-Modulation-for-Neural-Fields-in-PDE-Modeling)
3. [More RLHF, More Trust? On The Impact of Preference Alignment On Trustworthiness](#More-RLHF-More-Trust-On-The-Impact-of-Preference-Alignment-On-Trustworthiness)
4. [Population Transformer: Learning Population-level Representations of Neural Activity](#Population-Transformer-Learning-Population-level-Representations-of-Neural-Activity)
5. [Inference Scaling for Long-Context Retrieval Augmented Generation](#Inference-Scaling-for-Long-Context-Retrieval-Augmented-Generation)
6. [Proxy Denoising for Source-Free Domain Adaptation](#Proxy-Denoising-for-Source-Free-Domain-Adaptation)
7. [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](#Turning-Up-the-Heat-Min-p-Sampling-for-Creative-and-Coherent-LLM-Outputs)
8. [Topological Blindspots: Understanding and Extending Topological Deep Learning Through the Lens of Expressivity](#Topological-Blindspots-Understanding-and-Extending-Topological-Deep-Learning-Through-the-Lens-of-Expressivity)
9. [Retrieval Head Mechanistically Explains Long-Context Factuality](#Retrieval-Head-Mechanistically-Explains-Long-Context-Factuality)
10. [MIND over Body: Adaptive Thinking using Dynamic Computation](#MIND-over-Body-Adaptive-Thinking-using-Dynamic-Computation)
11. [How much of my dataset did you use? Quantitative Data Usage Inference in Machine Learning](#How-much-of-my-dataset-did-you-use-Quantitative-Data-Usage-Inference-in-Machine-Learning)
12. [SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups](#SymmetricDiffusers-Learning-Discrete-Diffusion-on-Finite-Symmetric-Groups)
13. [Cut Your Losses in Large-Vocabulary Language Models](#Cut-Your-Losses-in-Large-Vocabulary-Language-Models)
14. [Interpreting Emergent Planning in Model-Free Reinforcement Learning](#Interpreting-Emergent-Planning-in-Model-Free-Reinforcement-Learning)
15. [Progressive Compression with Universally Quantized Diffusion Models](#Progressive-Compression-with-Universally-Quantized-Diffusion-Models)
16. [Training Language Models to Self-Correct via Reinforcement Learning](#Training-Language-Models-to-Self-Correct-via-Reinforcement-Learning)
17. [What should a neuron aim for? Designing local objective functions based on information theory](#What-should-a-neuron-aim-for-Designing-local-objective-functions-based-on-information-theory)
18. [Backtracking Improves Generation Safety](#Backtracking-Improves-Generation-Safety)
19. [Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment](#Spread-Preference-Annotation-Direct-Preference-Judgment-for-Efficient-LLM-Alignment)
20. [Global Convergence in Neural ODEs: Impact of Activation Functions](#Global-Convergence-in-Neural-ODEs-Impact-of-Activation-Functions)
21. [Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces](#Geometry-of-Neural-Reinforcement-Learning-in-Continuous-State-and-Action-Spaces)
22. [The Hidden Cost of Waiting for Accurate Predictions](#The-Hidden-Cost-of-Waiting-for-Accurate-Predictions)
23. [DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL](#DeepLTL-Learning-to-Efficiently-Satisfy-Complex-LTL-Specifications-for-Multi-Task-RL)
24. [The Complexity of Two-Team Polymatrix Games with Independent Adversaries](#The-Complexity-of-Two-Team-Polymatrix-Games-with-Independent-Adversaries)
25. [Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series](#Amortized-Control-of-Continuous-State-Space-Feynman-Kac-Model-for-Irregular-Time-Series)
26. [MoDeGPT: Modular Decomposition for Large Language Model Compression](#MoDeGPT-Modular-Decomposition-for-Large-Language-Model-Compression)
27. [Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities](#Do-Vision-Language-Models-Represent-Space-and-How-Evaluating-Spatial-Frame-of-Reference-under-Ambiguities)
28. [Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects](#Geometry-aware-RL-for-Manipulation-of-Varying-Shapes-and-Deformable-Objects)
29. [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](#MLE-bench-Evaluating-Machine-Learning-Agents-on-Machine-Learning-Engineering)
30. [Safety Alignment Should be Made More Than Just a Few Tokens Deep](#Safety-Alignment-Should-be-Made-More-Than-Just-a-Few-Tokens-Deep)
31. [SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning](#SD-LoRA-Scalable-Decoupled-Low-Rank-Adaptation-for-Class-Incremental-Learning)
32. [Prioritized Generative Replay](#Prioritized-Generative-Replay)
33. [A Probabilistic Perspective on Unlearning and Alignment for Large Language Models](#A-Probabilistic-Perspective-on-Unlearning-and-Alignment-for-Large-Language-Models)
34. [Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning](#Flat-Reward-in-Policy-Parameter-Space-Implies-Robust-Reinforcement-Learning)
35. [Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning](#Scaling-LLM-Test-Time-Compute-Optimally-Can-be-More-Effective-than-Scaling-Parameters-for-Reasoning)
36. [Compositional Entailment Learning for Hyperbolic Vision-Language Models](#Compositional-Entailment-Learning-for-Hyperbolic-Vision-Language-Models)

---


## MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions

### Images

![03e288dd84f8958ce7b880a53a75b24a19ed8b4e2e7539487282d6e4ec5ce903.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/03e288dd84f8958ce7b880a53a75b24a19ed8b4e2e7539487282d6e4ec5ce903.jpg)

![1feae79487618ddf512bd824448f13bac6484564ec6c9c7fd8204ad8d0c29c84.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/1feae79487618ddf512bd824448f13bac6484564ec6c9c7fd8204ad8d0c29c84.jpg)

![20f31ea4e68b177b2af8977e6160143b692aea99c51346559e6111e6e43dacf9.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/20f31ea4e68b177b2af8977e6160143b692aea99c51346559e6111e6e43dacf9.jpg)

![32b20ab0355bf43b447e45d6018506836cc5a6e004c2d9f369dad0fcdc783ec6.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/32b20ab0355bf43b447e45d6018506836cc5a6e004c2d9f369dad0fcdc783ec6.jpg)

![3bbe18e6c22b7cf2c6387ee2c95ae469828117697bf29aebbfd20bdec095ad7f.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/3bbe18e6c22b7cf2c6387ee2c95ae469828117697bf29aebbfd20bdec095ad7f.jpg)

![3eb7d75e45a94dbd186f074599be265340f9800817b5529468c3ce1be467835b.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/3eb7d75e45a94dbd186f074599be265340f9800817b5529468c3ce1be467835b.jpg)

![49b527882df359ed0d9928298660ac6a8b82f9275a11cd0e3ae66e7c172cd04c.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/49b527882df359ed0d9928298660ac6a8b82f9275a11cd0e3ae66e7c172cd04c.jpg)

![541f1a0ff15f385109cea9c8ee8ba9f581b68a3c74224cbf838237061038ff3c.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/541f1a0ff15f385109cea9c8ee8ba9f581b68a3c74224cbf838237061038ff3c.jpg)

![6990941e6f6ab7fa462d441b99d3865ce55a7ae3a2499042249cdadecd06f9ff.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/6990941e6f6ab7fa462d441b99d3865ce55a7ae3a2499042249cdadecd06f9ff.jpg)

![710044e32a73e6118cd54c8741a55bfa14aa82d3499fa054ad2b2823333ac134.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/710044e32a73e6118cd54c8741a55bfa14aa82d3499fa054ad2b2823333ac134.jpg)

![88355bd853a2ccfb2e3801376189155931061148cfcfd84bfeb1a9920e4683f7.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/88355bd853a2ccfb2e3801376189155931061148cfcfd84bfeb1a9920e4683f7.jpg)

![8e3931b7627a1ba6700adfa0c23fad91ba364338cccfe2d2ef24b2518911cc41.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/8e3931b7627a1ba6700adfa0c23fad91ba364338cccfe2d2ef24b2518911cc41.jpg)

![8e60e783dff32ba7be8450ab64dab8f72745bb4f86702ca81b6eaed6c51dba11.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/8e60e783dff32ba7be8450ab64dab8f72745bb4f86702ca81b6eaed6c51dba11.jpg)

![9187357d404dba4fabdb959b8cd518a052f7c4024a85ae9fcd09ace1574e544f.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/9187357d404dba4fabdb959b8cd518a052f7c4024a85ae9fcd09ace1574e544f.jpg)

![962e4ad86ab812fd222fe531d290a936ab580ef2eea4768ab366393478849fa9.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/962e4ad86ab812fd222fe531d290a936ab580ef2eea4768ab366393478849fa9.jpg)

![c66ac648a1e5282a93888f71e7230ec4adbfa34c32c1c32e0298b981b5a7c9ae.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/c66ac648a1e5282a93888f71e7230ec4adbfa34c32c1c32e0298b981b5a7c9ae.jpg)

![ec24147a39ce1485899bff68a77ab19990ff9be0e9e4f2e42c54c120f7761a70.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/ec24147a39ce1485899bff68a77ab19990ff9be0e9e4f2e42c54c120f7761a70.jpg)

![f11e4cb489829c5aa6cadac761122914d9634ed723677b570a9a9057ed3a22ac.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/f11e4cb489829c5aa6cadac761122914d9634ed723677b570a9a9057ed3a22ac.jpg)

![f27815022b19b90d2b172ad3561e88ddcd8f183ed9fed26ad8a3a1aa66c551fa.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/f27815022b19b90d2b172ad3561e88ddcd8f183ed9fed26ad8a3a1aa66c551fa.jpg)

![f9a2d5d8f05bf7f7e9bdd199960b17a92421e3c2ea807a8aaa83f9a822db6bcb.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/f9a2d5d8f05bf7f7e9bdd199960b17a92421e3c2ea807a8aaa83f9a822db6bcb.jpg)

![fa0365bef1b5584fde79f320eb31bfcbb2db400cfdc8e6ddc4cce41e0c581b3b.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/fa0365bef1b5584fde79f320eb31bfcbb2db400cfdc8e6ddc4cce41e0c581b3b.jpg)

### Tables

![06f6f7a5ff2c4a9755ca2d78f5e9f68b78f5052eb6b54fa19bc99de4730dc575.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/06f6f7a5ff2c4a9755ca2d78f5e9f68b78f5052eb6b54fa19bc99de4730dc575.jpg)

![119a1c6d3416510b14230e64bd643e19338d9754b350f02a72da46d3f928e15d.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/119a1c6d3416510b14230e64bd643e19338d9754b350f02a72da46d3f928e15d.jpg)

![266011ea677388d2c2ac566351ee877960cac1d9fe8f3ed54740986f2f3c4da0.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/266011ea677388d2c2ac566351ee877960cac1d9fe8f3ed54740986f2f3c4da0.jpg)

![3059ff2580aa96b443c8a5ea093b9f78ef67b988543bd75b667236ace0244543.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/3059ff2580aa96b443c8a5ea093b9f78ef67b988543bd75b667236ace0244543.jpg)

![35acb11496a16967aa790375fe161e980876b393d8e02b0124f101fb078239d3.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/35acb11496a16967aa790375fe161e980876b393d8e02b0124f101fb078239d3.jpg)

![50443de74989a149cb7c39d00cad53ae43aec678e724cb2a528e74f62e650ed9.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/50443de74989a149cb7c39d00cad53ae43aec678e724cb2a528e74f62e650ed9.jpg)

![62b77991999c9aa492a7fa906674a901fa7a1b2e762bdf3081dbe9a8138276ef.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/62b77991999c9aa492a7fa906674a901fa7a1b2e762bdf3081dbe9a8138276ef.jpg)

![6e14ad1ec0ca0a4a06340e2993194fb7874518122d8cb291a55e498eb84eecd0.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/6e14ad1ec0ca0a4a06340e2993194fb7874518122d8cb291a55e498eb84eecd0.jpg)

![8ebb8207fb37706157323dd71f1a0cd351dee675da862caebd493f853934efc3.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/8ebb8207fb37706157323dd71f1a0cd351dee675da862caebd493f853934efc3.jpg)

![9c35dc3d15cdd508eca7ff3e323bdbe664317f192e15575d409ca895130c454a.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/9c35dc3d15cdd508eca7ff3e323bdbe664317f192e15575d409ca895130c454a.jpg)

![e657d22619d23f6f05659a2e560a34a01c947018639079d52402a53ad01941a4.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/e657d22619d23f6f05659a2e560a34a01c947018639079d52402a53ad01941a4.jpg)

## MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions


### Images

![03b5b200b42b04629f810025de0cc1068884fb42e15b01a5323f7bceef73e0f8.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/03b5b200b42b04629f810025de0cc1068884fb42e15b01a5323f7bceef73e0f8.jpg)

![8995eea1b753dfce505faf40a84588272049da8c6fddf13d2526bf3eb0c53446.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/8995eea1b753dfce505faf40a84588272049da8c6fddf13d2526bf3eb0c53446.jpg)

![a2954073f69de99bd566d0740ee7012879444ca1f1f645ae2fecbc10f19fe587.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/a2954073f69de99bd566d0740ee7012879444ca1f1f645ae2fecbc10f19fe587.jpg)

![d412d106720ecde080cd9342da027818220521f76b9fd54741d951bc71b02eb3.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/d412d106720ecde080cd9342da027818220521f76b9fd54741d951bc71b02eb3.jpg)

![dc0efb0e0f7ad02bb4582804192c7a4c8517db9bb827ab8b1856a6252c83c8bf.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/dc0efb0e0f7ad02bb4582804192c7a4c8517db9bb827ab8b1856a6252c83c8bf.jpg)

### Tables

![09e62e9f480cabe63be3e3bbc47620b570075f9c0ab6de4483e741e41a365079.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/09e62e9f480cabe63be3e3bbc47620b570075f9c0ab6de4483e741e41a365079.jpg)

![3b380534259904368a09066d75a203da0e2c21b0faaf4fdaf585c16ce5ac8624.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/3b380534259904368a09066d75a203da0e2c21b0faaf4fdaf585c16ce5ac8624.jpg)

![3cf76892c42482bf467fe27adf6fb7e89c30da8acf4ed92cbe4a2d5fe1c348e3.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/3cf76892c42482bf467fe27adf6fb7e89c30da8acf4ed92cbe4a2d5fe1c348e3.jpg)

![7daf007b565f4955797787e25f703ebdc1cea7727023009984c7ca6ccc77b6ff.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/7daf007b565f4955797787e25f703ebdc1cea7727023009984c7ca6ccc77b6ff.jpg)

![a4163f472d515833c36cbc0f1224978f89a2954c6224fae2d2c86c01bd9dd1df.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/a4163f472d515833c36cbc0f1224978f89a2954c6224fae2d2c86c01bd9dd1df.jpg)

![a7995ce61e617f0bbf2234b00ee0f009997427b00a31d3133334d86b70372e67.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/a7995ce61e617f0bbf2234b00ee0f009997427b00a31d3133334d86b70372e67.jpg)

![cae21f492c88eb86b54b326642e7acac0e501756dc1bda72230575833082d6d2.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/cae21f492c88eb86b54b326642e7acac0e501756dc1bda72230575833082d6d2.jpg)

![d7ec6f4e932a3f49744874873186ac42e5164777a3dea9d47975c539471fe3b7.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/d7ec6f4e932a3f49744874873186ac42e5164777a3dea9d47975c539471fe3b7.jpg)

![e0c225111838e3dc38b9261082060afa2a040b5c639362556bfe9f3badf0d92c.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/e0c225111838e3dc38b9261082060afa2a040b5c639362556bfe9f3badf0d92c.jpg)

![f98770ae4db0964d5bbd7ff406e30515813fbe856b9b422fad9dc546f2963a52.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/f98770ae4db0964d5bbd7ff406e30515813fbe856b9b422fad9dc546f2963a52.jpg)

## GridMix: Exploring Spatial Modulation for Neural Fields in PDE Modeling


### Images

![02fab3a47781bba0185daa5468b17799d7722c5dd32b09cd491291c5d5a74bd3.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/02fab3a47781bba0185daa5468b17799d7722c5dd32b09cd491291c5d5a74bd3.jpg)

![2be275b1f2d4d2c63f2e1d38cf3b2ad70bfc70e73e1497bc6d69e697cbc98105.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/2be275b1f2d4d2c63f2e1d38cf3b2ad70bfc70e73e1497bc6d69e697cbc98105.jpg)

![3fc453c8ddc8faa1f5dabfcc46a9289c6a903895a89a5c9905565da082ac7b2e.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/3fc453c8ddc8faa1f5dabfcc46a9289c6a903895a89a5c9905565da082ac7b2e.jpg)

![45598a475c74892e6d0759584474a125512a56153a0b1c01ac693f05247ce04f.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/45598a475c74892e6d0759584474a125512a56153a0b1c01ac693f05247ce04f.jpg)

![b4dc9702aee179507a03c317cf5e2e9c26ba1583fd518b15b85721236cf8877c.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/b4dc9702aee179507a03c317cf5e2e9c26ba1583fd518b15b85721236cf8877c.jpg)

![d2c6ac627d0f89c22643206d8cc3c94143302752790615bc2dda9e56d8eddf93.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/d2c6ac627d0f89c22643206d8cc3c94143302752790615bc2dda9e56d8eddf93.jpg)

![d912b6eac9345a1a73880a627f3f952f43442dc849892c71256d402e8bc70839.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/d912b6eac9345a1a73880a627f3f952f43442dc849892c71256d402e8bc70839.jpg)

![f7eb617476ad7ca74682f951097ad54e1562f4b5e3066013afed1b9b516bd1e1.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/f7eb617476ad7ca74682f951097ad54e1562f4b5e3066013afed1b9b516bd1e1.jpg)

### Tables

![158caeaf1971069ce3323f7ce07520a290f13ce2857d33609f5e137683f5fe9a.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/158caeaf1971069ce3323f7ce07520a290f13ce2857d33609f5e137683f5fe9a.jpg)

![184a86cd1768c09f182f529b73a824e00817c7640c2198d4cb51bc1e1b0b7780.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/184a86cd1768c09f182f529b73a824e00817c7640c2198d4cb51bc1e1b0b7780.jpg)

![355ce52f18c9d919a30da2d45b5df72e8c8ea2ced7551f1da9ac832567059bab.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/355ce52f18c9d919a30da2d45b5df72e8c8ea2ced7551f1da9ac832567059bab.jpg)

![4a88fbd37b20b55bcb8b3b49fe27c4ff7f748434430864d6db78395eb4cdd014.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/4a88fbd37b20b55bcb8b3b49fe27c4ff7f748434430864d6db78395eb4cdd014.jpg)

![a4ac65f43fc71ffca82e07d8b6dedfedd0c2d08cd4e86e2ddcdad40f2810ed4b.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/a4ac65f43fc71ffca82e07d8b6dedfedd0c2d08cd4e86e2ddcdad40f2810ed4b.jpg)

![b1d80f7b39fc301fc7844784561fecd3f3c0904129a2b800ccf3980dfbb853ed.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/b1d80f7b39fc301fc7844784561fecd3f3c0904129a2b800ccf3980dfbb853ed.jpg)

![d53fb7e00b833a4d365c04261b9342384122274a356ca8cda7b86fec126fc2f5.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/d53fb7e00b833a4d365c04261b9342384122274a356ca8cda7b86fec126fc2f5.jpg)

## More RLHF, More Trust? On The Impact of Preference Alignment On Trustworthiness


### Images

![4e2998fae48494157c3087f7b4b8215f9e1b15cf65d49656d5c4ba449c741a1c.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/4e2998fae48494157c3087f7b4b8215f9e1b15cf65d49656d5c4ba449c741a1c.jpg)

![56f52fa5027255b3a1d094f635bfedecffffa47466ad995744deb6bc9bc866be.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/56f52fa5027255b3a1d094f635bfedecffffa47466ad995744deb6bc9bc866be.jpg)

![82df7ca1d29dd8240ceeae1f986d5acc771ccf0f8c6d300d3516616e8245054b.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/82df7ca1d29dd8240ceeae1f986d5acc771ccf0f8c6d300d3516616e8245054b.jpg)

![8368c589c959f9b145dff63443afaea2751c209c42123d72b3729e046dc9a316.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/8368c589c959f9b145dff63443afaea2751c209c42123d72b3729e046dc9a316.jpg)

![917f21bd43f677efc309ff4d7b5e3d9fd03fd1c7c5ec48a6c943d52c40c75696.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/917f21bd43f677efc309ff4d7b5e3d9fd03fd1c7c5ec48a6c943d52c40c75696.jpg)

![93216c243bebe1678a60d4921d2077fcd5e152a3989570d30ee5606fc1d54541.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/93216c243bebe1678a60d4921d2077fcd5e152a3989570d30ee5606fc1d54541.jpg)

![97fbcce6d09b8138d54009c3aaff891bb2fdb987f23fe29bcb9f08590614aea4.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/97fbcce6d09b8138d54009c3aaff891bb2fdb987f23fe29bcb9f08590614aea4.jpg)

![9c6976c198ec9054f0b0d71bad5a0931cf4361db4d42f7f784499b30a3ed529a.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/9c6976c198ec9054f0b0d71bad5a0931cf4361db4d42f7f784499b30a3ed529a.jpg)

![9e448bcbff1c18f4e334eab2329044bf7132077c660395d25da4961ce6467d9a.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/9e448bcbff1c18f4e334eab2329044bf7132077c660395d25da4961ce6467d9a.jpg)

![a501393907b0c4dd0fb8a8166257b0e1ab23f1128aaf50334668fc5573883271.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/a501393907b0c4dd0fb8a8166257b0e1ab23f1128aaf50334668fc5573883271.jpg)

![d9803e0c8b06b099f217787d9053430b254d52d1b91c55af49d2cda3e47fb1ab.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/d9803e0c8b06b099f217787d9053430b254d52d1b91c55af49d2cda3e47fb1ab.jpg)

![e3944eb6441eca406e140fae4dda0a70ef672cc79526a6c7c41178c96132f0bb.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/e3944eb6441eca406e140fae4dda0a70ef672cc79526a6c7c41178c96132f0bb.jpg)

### Tables

![4143bcb77d074a2039380ebbfccb91b6a64d047412e9d17276299aec2c386917.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/4143bcb77d074a2039380ebbfccb91b6a64d047412e9d17276299aec2c386917.jpg)

![79852e024cacb350892e171e0963115be88d57cda2fe5997593a3015c2dc643e.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/79852e024cacb350892e171e0963115be88d57cda2fe5997593a3015c2dc643e.jpg)

![abbbb730da7520b4060861746c47f409d1a960bb38e2f6915330bb9b3b225e53.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/abbbb730da7520b4060861746c47f409d1a960bb38e2f6915330bb9b3b225e53.jpg)

![ca2961f115a83ba7b0ecf7869f821f7a8a1bbae6bacb3537f718d6f6473e5a4e.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/ca2961f115a83ba7b0ecf7869f821f7a8a1bbae6bacb3537f718d6f6473e5a4e.jpg)

![db9e743d79992db4f865827a7c75c94a02c8815f3b5ffe9776d01c9a03851eb3.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/db9e743d79992db4f865827a7c75c94a02c8815f3b5ffe9776d01c9a03851eb3.jpg)

![eb97ff628bf71a7d2d4cc54e4f93c6c252f8df55841dd17f9aed0f76455c9488.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/eb97ff628bf71a7d2d4cc54e4f93c6c252f8df55841dd17f9aed0f76455c9488.jpg)

![f3b4e32e8a254bc0072c2d9bc16394013a5b1d7d9f479c250e4a3c5082d56948.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/f3b4e32e8a254bc0072c2d9bc16394013a5b1d7d9f479c250e4a3c5082d56948.jpg)

## Population Transformer: Learning Population-level Representations of Neural Activity


### Images

![0026827419f5faaf831b2061ae8082a164c1f38d4b756cdfce88f8b1d5b9124c.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/0026827419f5faaf831b2061ae8082a164c1f38d4b756cdfce88f8b1d5b9124c.jpg)

![02d9fd0df02819bd91d003342a0c4c63731eeb9982bf03128cb87e90437d2a28.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/02d9fd0df02819bd91d003342a0c4c63731eeb9982bf03128cb87e90437d2a28.jpg)

![133e8ad6f930f56a59eb047e15ab15f8c170989fdc769d3c21c6a908a6e79953.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/133e8ad6f930f56a59eb047e15ab15f8c170989fdc769d3c21c6a908a6e79953.jpg)

![17360ccc8177b4d9ee33406277908c10dd328a50e83584fd6149e0dba697ab7c.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/17360ccc8177b4d9ee33406277908c10dd328a50e83584fd6149e0dba697ab7c.jpg)

![211f7cc53c5d8b1603bb91200e083d9fb3f2c10c0ecafcea814f699f21fdd0f9.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/211f7cc53c5d8b1603bb91200e083d9fb3f2c10c0ecafcea814f699f21fdd0f9.jpg)

![32da252ed3206be3d825c24be5dff8c1303b04258f81452649cc9472644c36e0.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/32da252ed3206be3d825c24be5dff8c1303b04258f81452649cc9472644c36e0.jpg)

![36a26f5d3f40bb7d03ce94c0c6f2f7e60133096f361311975aed8c2beeef9930.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/36a26f5d3f40bb7d03ce94c0c6f2f7e60133096f361311975aed8c2beeef9930.jpg)

![5b78ddbd2a723701d743a9009c65a7daf1d2462a5dd3448c51b023da843e2bc0.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/5b78ddbd2a723701d743a9009c65a7daf1d2462a5dd3448c51b023da843e2bc0.jpg)

![667ce13f39690869020e351b9bae8fee5097ea53d1a44d4de79e267df209363f.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/667ce13f39690869020e351b9bae8fee5097ea53d1a44d4de79e267df209363f.jpg)

![84ef38a6bcfc55df3cf94b6c434c5a496032eef6bf66738265ded72add5e36e3.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/84ef38a6bcfc55df3cf94b6c434c5a496032eef6bf66738265ded72add5e36e3.jpg)

![89aedd78e2d3d263e59200be37e1d53300db4daae03b924978e07605e9d2548d.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/89aedd78e2d3d263e59200be37e1d53300db4daae03b924978e07605e9d2548d.jpg)

![89f78afa37bf85b3c85e1ea16a6aceb49652189afa4d55da2b0207213fb5d611.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/89f78afa37bf85b3c85e1ea16a6aceb49652189afa4d55da2b0207213fb5d611.jpg)

![94ff4d640d67ff9cb7a6ec011c3158e59df6c71381e903f2686a04ac8a3a5c2a.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/94ff4d640d67ff9cb7a6ec011c3158e59df6c71381e903f2686a04ac8a3a5c2a.jpg)

![97e01887d0712a305a15f4eea6141bc5e7c9a051e603307f655e03816d340705.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/97e01887d0712a305a15f4eea6141bc5e7c9a051e603307f655e03816d340705.jpg)

![cee4f8417e2dfc0e100cf4d06f76f762d88d0ccacd2833054980d924888d377b.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/cee4f8417e2dfc0e100cf4d06f76f762d88d0ccacd2833054980d924888d377b.jpg)

![e19dc5812c9f947ea11b2541927dd6036176914aa6ade7f31ad8d13e59e68e17.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/e19dc5812c9f947ea11b2541927dd6036176914aa6ade7f31ad8d13e59e68e17.jpg)

![e7d0c1526ea714671afff6102ae2b4f101f820c3503886f8d1fac0e24de4ce1f.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/e7d0c1526ea714671afff6102ae2b4f101f820c3503886f8d1fac0e24de4ce1f.jpg)

### Tables

![0d96f5c83e2b9d74f1d9c279b5764eaa6ec453ec71fcb6b7a2689413c66d9c05.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/0d96f5c83e2b9d74f1d9c279b5764eaa6ec453ec71fcb6b7a2689413c66d9c05.jpg)

![1c812769c3c699b5a9ceed93028082a9c5d763937a43fdba1fd11651cbf370d0.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/1c812769c3c699b5a9ceed93028082a9c5d763937a43fdba1fd11651cbf370d0.jpg)

![2992f7df585830ae4da8dc2bc20cf4f0be1e991fb699b677c9b73b37a4fea3ba.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/2992f7df585830ae4da8dc2bc20cf4f0be1e991fb699b677c9b73b37a4fea3ba.jpg)

![3fef66415bcf22ab56663b8ba485f0f3285e1f449f640625927a4a1c1baaf51b.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/3fef66415bcf22ab56663b8ba485f0f3285e1f449f640625927a4a1c1baaf51b.jpg)

![45c1ec550d7af1d7113da6e4b8556d0166546c29d9a11cba5d08f1168aeb8630.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/45c1ec550d7af1d7113da6e4b8556d0166546c29d9a11cba5d08f1168aeb8630.jpg)

![5139ceaac2979a6232102746c1c714971a30b0802d3915a03704f109865c6d3d.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/5139ceaac2979a6232102746c1c714971a30b0802d3915a03704f109865c6d3d.jpg)

![90ca54b6fdb663cec44bf2015270e7b796cc1c05e9b34f0209b8d97ab126f3c8.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/90ca54b6fdb663cec44bf2015270e7b796cc1c05e9b34f0209b8d97ab126f3c8.jpg)

![b8f77a8d5c02a94d5a959d6948044c4294f1af70462d36bd2cc0ce7328683ebe.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/b8f77a8d5c02a94d5a959d6948044c4294f1af70462d36bd2cc0ce7328683ebe.jpg)

## Inference Scaling for Long-Context Retrieval Augmented Generation


### Images

![28c2c94a3b8446e6714638bb23eb8b3388649531d437366bd3cfbaff2bbe3dd5.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/28c2c94a3b8446e6714638bb23eb8b3388649531d437366bd3cfbaff2bbe3dd5.jpg)

![2a71d46f49942c18341d0b78d0c52fdae14bcc006ecae4c382a5bc656c6c4c07.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/2a71d46f49942c18341d0b78d0c52fdae14bcc006ecae4c382a5bc656c6c4c07.jpg)

![32e74e07aeb0f3eea5db4dcbe079ee8cc07e7e1a488df01843ced36c1a76534d.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/32e74e07aeb0f3eea5db4dcbe079ee8cc07e7e1a488df01843ced36c1a76534d.jpg)

![479dc87b6803ffbc54323ff4ff16646ecef52089a2999a5cfb9ee353e22acb3e.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/479dc87b6803ffbc54323ff4ff16646ecef52089a2999a5cfb9ee353e22acb3e.jpg)

![48135f2ef8370d885ee2ab2e53c83b2bc9f96a8a000a0f1e9c414f417d64be0d.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/48135f2ef8370d885ee2ab2e53c83b2bc9f96a8a000a0f1e9c414f417d64be0d.jpg)

![4b088d47e0c44bc397ab50e9689a290f570ee401c47d277bd23e3f1130b0c7dc.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/4b088d47e0c44bc397ab50e9689a290f570ee401c47d277bd23e3f1130b0c7dc.jpg)

![71fa70a6c684dd95d107cd47eb4fbf8c1ca233acc9b0c1a0908b0145eef5d63b.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/71fa70a6c684dd95d107cd47eb4fbf8c1ca233acc9b0c1a0908b0145eef5d63b.jpg)

![73173d6babaeb5544b114e987381eefd35b0db7d8b5aa1c4bc2035da776b7ac6.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/73173d6babaeb5544b114e987381eefd35b0db7d8b5aa1c4bc2035da776b7ac6.jpg)

![87883559c36e332acdc25171ba31d81d106ca123ee3d27456e06d42ea43c2ef5.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/87883559c36e332acdc25171ba31d81d106ca123ee3d27456e06d42ea43c2ef5.jpg)

![921128ff15a36f4f80f1708d06888da7b6143b7d916bd6176125c183b6cfed86.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/921128ff15a36f4f80f1708d06888da7b6143b7d916bd6176125c183b6cfed86.jpg)

![999ddb862b2dd3e0219b6917788940f66ea58eae81ddecb0e39c8b033870ee7e.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/999ddb862b2dd3e0219b6917788940f66ea58eae81ddecb0e39c8b033870ee7e.jpg)

![a6d5cc7db2c27a98c47e5e41e2184c17927385e2849d21a121171ba4a4ddeb8a.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/a6d5cc7db2c27a98c47e5e41e2184c17927385e2849d21a121171ba4a4ddeb8a.jpg)

![e1ec600f05cb3676b7d9ce21de69e18e7f2213fc5503241511f8eff197274aae.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/e1ec600f05cb3676b7d9ce21de69e18e7f2213fc5503241511f8eff197274aae.jpg)

![fb465289991d6084a5c0123f65f471274c292ee944e7075ce42a349d9c3097e6.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/fb465289991d6084a5c0123f65f471274c292ee944e7075ce42a349d9c3097e6.jpg)

### Tables

![09007137aea978bd5d005cbe6f7f879d7e23243dac8b34d9af0fa528f05ef6d1.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/09007137aea978bd5d005cbe6f7f879d7e23243dac8b34d9af0fa528f05ef6d1.jpg)

![21eaa16d4d44a8fe7dce452c36d2ce6798dc26dd612b12921174565a73f9d866.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/21eaa16d4d44a8fe7dce452c36d2ce6798dc26dd612b12921174565a73f9d866.jpg)

![255ef9d9e495bff15419c614bca0836661d9c606dafdce09b214445d4850b92a.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/255ef9d9e495bff15419c614bca0836661d9c606dafdce09b214445d4850b92a.jpg)

![39d84d9416f7e1b75e5261fe1577ae7effbdb4f576d80ef7cc3eaea8e13137f9.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/39d84d9416f7e1b75e5261fe1577ae7effbdb4f576d80ef7cc3eaea8e13137f9.jpg)

![60d98f555ca6099b7898483feeb33763d668a32e44532be0c9fec15c8c28243c.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/60d98f555ca6099b7898483feeb33763d668a32e44532be0c9fec15c8c28243c.jpg)

![842088c31b7aeaf4f52f3cbd787b22b9cbfc595454622cb8c545500784b69fc3.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/842088c31b7aeaf4f52f3cbd787b22b9cbfc595454622cb8c545500784b69fc3.jpg)

![b7baa6923da013a5981394118126efb7b2c8d4506f55df3f6c6702ac7adf47d9.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/b7baa6923da013a5981394118126efb7b2c8d4506f55df3f6c6702ac7adf47d9.jpg)

![bbbf03a2d0095a2ea7e80382f6e5a906575ca95f124917dd07314feb3dc83fc4.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/bbbf03a2d0095a2ea7e80382f6e5a906575ca95f124917dd07314feb3dc83fc4.jpg)

## Proxy Denoising for Source-Free Domain Adaptation


### Images

![29cb841c31a342df913ab974e9c42d4890d31f495b283e8227064655e45b4036.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/29cb841c31a342df913ab974e9c42d4890d31f495b283e8227064655e45b4036.jpg)

![4df0a5c16a26fda98e97f2f7f36d0b1a49a688ec17faa958e9ffe52447119e8c.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/4df0a5c16a26fda98e97f2f7f36d0b1a49a688ec17faa958e9ffe52447119e8c.jpg)

![66b84c40dfe16d90cebff235ea454d7823fbb24c8cba5a8736d2d39f01586f43.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/66b84c40dfe16d90cebff235ea454d7823fbb24c8cba5a8736d2d39f01586f43.jpg)

![748071375be8a5fa9bca223888c0c4531e8fed89da437c2cf0e6f00be0c7cc6c.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/748071375be8a5fa9bca223888c0c4531e8fed89da437c2cf0e6f00be0c7cc6c.jpg)

![85e24e08e013cd945a816ba4e3d97c64706e21fe23b64656cebf8dbaf341dcf0.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/85e24e08e013cd945a816ba4e3d97c64706e21fe23b64656cebf8dbaf341dcf0.jpg)

### Tables

![22be632aac7d77a975a9a60772adc42beba6e49ae8caca2221acf25e6b7e7752.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/22be632aac7d77a975a9a60772adc42beba6e49ae8caca2221acf25e6b7e7752.jpg)

![24ec33dfaedef288923c083ac06d99f05cd30c4efbf8072d3af1db4b5a5b77fe.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/24ec33dfaedef288923c083ac06d99f05cd30c4efbf8072d3af1db4b5a5b77fe.jpg)

![3038b91dea79572714255b7c0c49b407b0234f8f2eb0e76233190183890cb87e.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/3038b91dea79572714255b7c0c49b407b0234f8f2eb0e76233190183890cb87e.jpg)

![37e501b1fe06a0b8a641621834241b86c752c23d9edca3a40f683d2410090bf0.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/37e501b1fe06a0b8a641621834241b86c752c23d9edca3a40f683d2410090bf0.jpg)

![48a679f7921c70758b74e1cb35b1670e885300c6265da41df0fc1217042ab17b.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/48a679f7921c70758b74e1cb35b1670e885300c6265da41df0fc1217042ab17b.jpg)

![4adf07fdb2b4ada31d9381ca3b19432405343da1ad00b6c7671ddf57396222b6.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/4adf07fdb2b4ada31d9381ca3b19432405343da1ad00b6c7671ddf57396222b6.jpg)

![4e0a25c74e578f65510ce1caf45120ce87da11fc980b3b39c3359bfd7bb4cd40.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/4e0a25c74e578f65510ce1caf45120ce87da11fc980b3b39c3359bfd7bb4cd40.jpg)

![4fd68616d21c9e3aba648c73f88075b80d4ce071e4f8fcc9bfced0a5817a3b36.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/4fd68616d21c9e3aba648c73f88075b80d4ce071e4f8fcc9bfced0a5817a3b36.jpg)

![6906c5603d470b88d355ae29cc12c53b7db1c9a3e41a4857299ed733d3d1c306.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/6906c5603d470b88d355ae29cc12c53b7db1c9a3e41a4857299ed733d3d1c306.jpg)

![82b11ffadfd25b835e12a1eaabddde8154f3cc76621b3ae28559b522995931fe.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/82b11ffadfd25b835e12a1eaabddde8154f3cc76621b3ae28559b522995931fe.jpg)

![946ac1ece91afbe3c18e58862b7aa2590fda735a8400292f5238f0eec59f8f8f.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/946ac1ece91afbe3c18e58862b7aa2590fda735a8400292f5238f0eec59f8f8f.jpg)

![979944e2faaaa0286bbe99c4ad316146d5aa5c5bf36e6f51f410081f6e018def.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/979944e2faaaa0286bbe99c4ad316146d5aa5c5bf36e6f51f410081f6e018def.jpg)

![9e9db8d031d500ff67336f9eea79ce32594728a37cf3cff5fa9f031ced6df0af.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/9e9db8d031d500ff67336f9eea79ce32594728a37cf3cff5fa9f031ced6df0af.jpg)

![a3cbbbd5ab4d09f7ccc71ffaaffc076792cf7cca0d2644b18216af7591c5c48a.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/a3cbbbd5ab4d09f7ccc71ffaaffc076792cf7cca0d2644b18216af7591c5c48a.jpg)

![aa930560d6fa20aacf57a9e6695d26150bc6207be7c5ce5bb92cf8a56391fd19.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/aa930560d6fa20aacf57a9e6695d26150bc6207be7c5ce5bb92cf8a56391fd19.jpg)

![b10b3fc9c6b46cf458615202ec8df517b37dea8da261558382602837dcdd9408.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/b10b3fc9c6b46cf458615202ec8df517b37dea8da261558382602837dcdd9408.jpg)

![d8dc9c103319d5a32eac39fb80049c9edc851ca204e11bc26381b0068db56e38.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/d8dc9c103319d5a32eac39fb80049c9edc851ca204e11bc26381b0068db56e38.jpg)

![e2f88c486e870369f8b0e31814f148224720e868e28fcdf9ce3bf111aaeae587.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/e2f88c486e870369f8b0e31814f148224720e868e28fcdf9ce3bf111aaeae587.jpg)

![e4256cefe52b8216111185c4556cd184594abcd6bca449dbd15ac6c7feba13b3.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/e4256cefe52b8216111185c4556cd184594abcd6bca449dbd15ac6c7feba13b3.jpg)

![e63991143f10947a46c65a43150ff4ff40d0f12150956596c411ea0b3ca782b4.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/e63991143f10947a46c65a43150ff4ff40d0f12150956596c411ea0b3ca782b4.jpg)

![ed451dbb6a376aafc3e9a95e397321306405ab46e7f26f9ad785be65b840ca8f.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/ed451dbb6a376aafc3e9a95e397321306405ab46e7f26f9ad785be65b840ca8f.jpg)

![eda266d9f198878ce2ca7081300438124b98d32d16ee095234d3c407cc2a7ca3.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/eda266d9f198878ce2ca7081300438124b98d32d16ee095234d3c407cc2a7ca3.jpg)

![f1f2130276e930d06f934656c40d76c11a096fa57de60b036acf395af716a4ef.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/f1f2130276e930d06f934656c40d76c11a096fa57de60b036acf395af716a4ef.jpg)

## Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs


### Images

![9e2ac9a6596e224675dab8e1286e2b9401bbdb549d0cc5622923f20734edd242.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/images/9e2ac9a6596e224675dab8e1286e2b9401bbdb549d0cc5622923f20734edd242.jpg)

![fa218f3a435d488296a93edd578f911d844fb3a7f761f9159fec0b8e1fc99856.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/images/fa218f3a435d488296a93edd578f911d844fb3a7f761f9159fec0b8e1fc99856.jpg)

### Tables

![05fa5bcc6cbb45dde997c5d70a292f8a4cb734e51372ac7463c38fab4a3e2435.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/05fa5bcc6cbb45dde997c5d70a292f8a4cb734e51372ac7463c38fab4a3e2435.jpg)

![1395fdb359b5fc4c971cd8cebdb94aabb1fe5f1ed5d70f7e8c7f943c0a869a0a.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1395fdb359b5fc4c971cd8cebdb94aabb1fe5f1ed5d70f7e8c7f943c0a869a0a.jpg)

![1768f61b97a8eed0323dd973e71a89661723d239a1a8cd805aa2a1b92cb59aac.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1768f61b97a8eed0323dd973e71a89661723d239a1a8cd805aa2a1b92cb59aac.jpg)

![18fc2fb31b572b4f6d2f9ff60c6e33282c5966034640ded6e31f2d58b6f2018d.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/18fc2fb31b572b4f6d2f9ff60c6e33282c5966034640ded6e31f2d58b6f2018d.jpg)

![1a771a6af61487dcc5f354a765f8b6b93c4ba7218aa90064e5fe5d1812b03cfd.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1a771a6af61487dcc5f354a765f8b6b93c4ba7218aa90064e5fe5d1812b03cfd.jpg)

![1f65577826aebfb2fdd138f7d43799ee19c8cfe0afd6f1900e1164ba15397708.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1f65577826aebfb2fdd138f7d43799ee19c8cfe0afd6f1900e1164ba15397708.jpg)

![24138fe87320eca49f2b3b727b6d8fa2c0c5e7aa52572e701344974923d55014.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/24138fe87320eca49f2b3b727b6d8fa2c0c5e7aa52572e701344974923d55014.jpg)

![27bf30d3d292c70731f2107820e9d1535f87686fe5885f23f918ee995b9f52df.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/27bf30d3d292c70731f2107820e9d1535f87686fe5885f23f918ee995b9f52df.jpg)

![3c21f351a5859cc558ac0d0c477b943419247621799e636eb060f6d5a4ae539b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/3c21f351a5859cc558ac0d0c477b943419247621799e636eb060f6d5a4ae539b.jpg)

![3e92ecb54298534af89e8901a0702ed0c4b5967d4679c8ac5288c50e240f129b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/3e92ecb54298534af89e8901a0702ed0c4b5967d4679c8ac5288c50e240f129b.jpg)

![405d3993336080130ae76068058c0bc715f87495df1dfd8388b398c09e2997e3.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/405d3993336080130ae76068058c0bc715f87495df1dfd8388b398c09e2997e3.jpg)

![4a9d15661844fa8a01354daaf7901a291b1e577de158784a939c8063130c5d43.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/4a9d15661844fa8a01354daaf7901a291b1e577de158784a939c8063130c5d43.jpg)

![4e1fb5496832db44917fecfffaf410ee303ad4ceb2cadb2e5328ec546d998b57.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/4e1fb5496832db44917fecfffaf410ee303ad4ceb2cadb2e5328ec546d998b57.jpg)

![65f09e97ebc9d2ac5f79ce8e8d4e4731c7eb547b19e557003cb4b9c740d3bea8.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/65f09e97ebc9d2ac5f79ce8e8d4e4731c7eb547b19e557003cb4b9c740d3bea8.jpg)

![66b21606f0462bee1513b36a4ab5a4d6eea0ae60dd05136f64ea2515108c4776.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/66b21606f0462bee1513b36a4ab5a4d6eea0ae60dd05136f64ea2515108c4776.jpg)

![7013673224fafec10e4a4c3fcbd5d2d55595c2cba11142f03e3943cb8a138b09.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/7013673224fafec10e4a4c3fcbd5d2d55595c2cba11142f03e3943cb8a138b09.jpg)

![7b9bbdeb27707b096b32031eadddfe9f03e21d2726da544314b7040935476668.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/7b9bbdeb27707b096b32031eadddfe9f03e21d2726da544314b7040935476668.jpg)

![8c1cc8ac8c099f9b1ad1835615ff1fd2fceebd29c489f74ee76a3ed9cd9334dd.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/8c1cc8ac8c099f9b1ad1835615ff1fd2fceebd29c489f74ee76a3ed9cd9334dd.jpg)

![8d6ef97b7b45970e91ec30e58ee69c748acc7bff8ffa4647c112b36336c28e4a.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/8d6ef97b7b45970e91ec30e58ee69c748acc7bff8ffa4647c112b36336c28e4a.jpg)

![9437c5c90a53eb0662ba6dd66136f18b04c02ecc8e1a781bcbf9afa8513822f8.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/9437c5c90a53eb0662ba6dd66136f18b04c02ecc8e1a781bcbf9afa8513822f8.jpg)

![97c72001865f38a2f5e2696669f2615e83808a713d1753b82d56b6c303ecc954.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/97c72001865f38a2f5e2696669f2615e83808a713d1753b82d56b6c303ecc954.jpg)

![9a77fce2174fa64223943b98272b04b2447b927d05f4dfc56e9ddd564c23a572.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/9a77fce2174fa64223943b98272b04b2447b927d05f4dfc56e9ddd564c23a572.jpg)

![b95a900fd2641f725246bbec37143ac5f8974e8c26d1bc8f5032c6c72174827a.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/b95a900fd2641f725246bbec37143ac5f8974e8c26d1bc8f5032c6c72174827a.jpg)

![c7f2602919bcf6154b8c9c6b4285d8221fb73b3b12e9821fbb1a4169f15a7117.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/c7f2602919bcf6154b8c9c6b4285d8221fb73b3b12e9821fbb1a4169f15a7117.jpg)

![ca36ebd55b6fbe22e995e9d894546dfea9425dfed43366988d3f37a1dfc31464.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/ca36ebd55b6fbe22e995e9d894546dfea9425dfed43366988d3f37a1dfc31464.jpg)

![d1515be9fe7b75d57d65bd0a18d5fdf0369cb85e75b4536783c7265bf3a99f9e.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/d1515be9fe7b75d57d65bd0a18d5fdf0369cb85e75b4536783c7265bf3a99f9e.jpg)

![e484dddfb22c8db98e693696f2c30361cb1b6310bd1998439cf9f8529b1572d9.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/e484dddfb22c8db98e693696f2c30361cb1b6310bd1998439cf9f8529b1572d9.jpg)

![ec2559d545c82bea6201e0b1fdd90b67515aedaba854fba1ff07a3f84387183b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/ec2559d545c82bea6201e0b1fdd90b67515aedaba854fba1ff07a3f84387183b.jpg)

![f3565ee719fbc86a48445fec062cc6b2200a175694caa5a100fdeaa0a8bde29b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/f3565ee719fbc86a48445fec062cc6b2200a175694caa5a100fdeaa0a8bde29b.jpg)

## Topological Blindspots: Understanding and Extending Topological Deep Learning Through the Lens of Expressivity


### Images

![028503409a7a35b476a8107743ba4dde3ed790b45093a567567f631ec9a3624c.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/028503409a7a35b476a8107743ba4dde3ed790b45093a567567f631ec9a3624c.jpg)

![21a5f4f0cc90ee1da62cbc5d2b787066f130c7852195f327e98ac2e17bab73af.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/21a5f4f0cc90ee1da62cbc5d2b787066f130c7852195f327e98ac2e17bab73af.jpg)

![3bcf7efe85971119469ea21f6550a31b959f2870b9eb6a5c6d7f463f66768a3b.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/3bcf7efe85971119469ea21f6550a31b959f2870b9eb6a5c6d7f463f66768a3b.jpg)

![3cbbca8f506b2c5113f13504e54d391be9fad7efd321c8134fd51c82373ebe57.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/3cbbca8f506b2c5113f13504e54d391be9fad7efd321c8134fd51c82373ebe57.jpg)

![4e61aff506ff5b221588e682cf608b68510a0c4a277bc4c9c1d5a9a012a78fe6.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/4e61aff506ff5b221588e682cf608b68510a0c4a277bc4c9c1d5a9a012a78fe6.jpg)

![6cbe45eeb96308600ef6a98798a5d72ad31f0fe9b7f91f672edd06fa7944eaa6.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/6cbe45eeb96308600ef6a98798a5d72ad31f0fe9b7f91f672edd06fa7944eaa6.jpg)

![7a3846835461e72dfbe7ca79c600ba3315198ac2b37ce07005cabab468f4c6d2.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/7a3846835461e72dfbe7ca79c600ba3315198ac2b37ce07005cabab468f4c6d2.jpg)

![7d247f0ff1d1fdc93da4aeb05338fe297c57d1bfad98592471aca6cdc8986461.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/7d247f0ff1d1fdc93da4aeb05338fe297c57d1bfad98592471aca6cdc8986461.jpg)

![93f95928bbc76bf0fb2a609acd8434935952a1a09f2a56235ac75a413d48aab3.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/93f95928bbc76bf0fb2a609acd8434935952a1a09f2a56235ac75a413d48aab3.jpg)

![9cdfa1fef2bcb63fba5064a94775e14058748faac2ecafa0e033c18399776d13.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/9cdfa1fef2bcb63fba5064a94775e14058748faac2ecafa0e033c18399776d13.jpg)

![9e310d33f094e795c20e599910051df560df7221176b28f6b3062b14b79ff652.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/9e310d33f094e795c20e599910051df560df7221176b28f6b3062b14b79ff652.jpg)

![a42e369e2ac7128c1ed3cc196727ff8594170117c5f66be89df1e1829ae7cd3c.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/a42e369e2ac7128c1ed3cc196727ff8594170117c5f66be89df1e1829ae7cd3c.jpg)

![b04d4d330997ac03c7c41e2e53b9b0e178a127515d7f531d650a0e291569d69c.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/b04d4d330997ac03c7c41e2e53b9b0e178a127515d7f531d650a0e291569d69c.jpg)

![b7e7a460a130beaec24841047c3a93563cd3b225a7043a6bac934b0007230332.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/b7e7a460a130beaec24841047c3a93563cd3b225a7043a6bac934b0007230332.jpg)

![e641645820d10987c66f8dea93b18dc7c011fa06c3a04ab1f699fa69640b8353.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/e641645820d10987c66f8dea93b18dc7c011fa06c3a04ab1f699fa69640b8353.jpg)

![e82ad587cc2e7145884a82663121a72cff228b6235b5cf162c87c3164552c416.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/e82ad587cc2e7145884a82663121a72cff228b6235b5cf162c87c3164552c416.jpg)

### Tables

![4f721a60bc26cec21bb3334fffda8eb910472c9b2a5bba91ad8a51edab7532cd.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/4f721a60bc26cec21bb3334fffda8eb910472c9b2a5bba91ad8a51edab7532cd.jpg)

![659b615c5d137ac13e250b4048e2741117e12e6fb99fa52e00cabf22ecf658c8.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/659b615c5d137ac13e250b4048e2741117e12e6fb99fa52e00cabf22ecf658c8.jpg)

![888d51fd5d45761e8ea990233078dac69ece8abb17d6bd8b4180690e7d041cd5.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/888d51fd5d45761e8ea990233078dac69ece8abb17d6bd8b4180690e7d041cd5.jpg)

![a6b7ef521f9af475047088096c97dd17f8513f6ef4dfb1fbbac36b9993a7bb28.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/a6b7ef521f9af475047088096c97dd17f8513f6ef4dfb1fbbac36b9993a7bb28.jpg)

![f333cb5041d1e046671502e79c27640ed48f2b7b9d8eaf8cc2ab881d25ed746a.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/f333cb5041d1e046671502e79c27640ed48f2b7b9d8eaf8cc2ab881d25ed746a.jpg)

## Retrieval Head Mechanistically Explains Long-Context Factuality


### Images

![06b380daed50da9ed1faae6fa7571eb949ad2bdf362703e84d0373dfcf82227f.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/06b380daed50da9ed1faae6fa7571eb949ad2bdf362703e84d0373dfcf82227f.jpg)

![07f9f47bb5ace396e6e5ccc9f0c8265c37b3948088c06f8f32754b7bd54ae59c.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/07f9f47bb5ace396e6e5ccc9f0c8265c37b3948088c06f8f32754b7bd54ae59c.jpg)

![0c66e07dd153efef2a345d9e1b54e8a30357384ef8da71328cddff8ef1dd1de2.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/0c66e07dd153efef2a345d9e1b54e8a30357384ef8da71328cddff8ef1dd1de2.jpg)

![14bbb593682c4ed6889d35d59c31d3c5187a9c57f8c28f43a99fb7371223efe9.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/14bbb593682c4ed6889d35d59c31d3c5187a9c57f8c28f43a99fb7371223efe9.jpg)

![52184b46989c51024b522d681bfd50b09e341d71325d304e8fce9f8a943b152f.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/52184b46989c51024b522d681bfd50b09e341d71325d304e8fce9f8a943b152f.jpg)

![56368c4858ecbf4a8dc74d99bc476423432d18fd0c0e92f012c79b7e3b686453.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/56368c4858ecbf4a8dc74d99bc476423432d18fd0c0e92f012c79b7e3b686453.jpg)

![563938d528f459f9caee435c379e09add1254220764a02c29b3de8fdc57ca766.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/563938d528f459f9caee435c379e09add1254220764a02c29b3de8fdc57ca766.jpg)

![6b7ac60d53f7b805bbed403c5d08bcda5b5c2be208853721c2f79d9d965a21cb.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/6b7ac60d53f7b805bbed403c5d08bcda5b5c2be208853721c2f79d9d965a21cb.jpg)

![7345be229e71a85dafd18a6547ddcc36ee02c0d89540101f5da1d2d78946cb29.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/7345be229e71a85dafd18a6547ddcc36ee02c0d89540101f5da1d2d78946cb29.jpg)

![873df00d38b53ccd13c17158d5c48e1ca0934673603b5608896457bb5eb7c43d.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/873df00d38b53ccd13c17158d5c48e1ca0934673603b5608896457bb5eb7c43d.jpg)

![a51a4ccd761014d05c0a44a2b96d129d1dccce9b1aac47b78472bd03b03e9e9f.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/a51a4ccd761014d05c0a44a2b96d129d1dccce9b1aac47b78472bd03b03e9e9f.jpg)

![c5f52e9e8230244ed25de2a0d175006f729b351743004537d176c029c7112b69.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/c5f52e9e8230244ed25de2a0d175006f729b351743004537d176c029c7112b69.jpg)

![c82012c8ab057bb445dfbd51edf6c287fd441a766e6a7563dbde6a65a0ce3447.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/c82012c8ab057bb445dfbd51edf6c287fd441a766e6a7563dbde6a65a0ce3447.jpg)

![cf9ee2000f874c9de9cd3182df3438a02877a2e516fcad1f257612ff3397bbf2.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/cf9ee2000f874c9de9cd3182df3438a02877a2e516fcad1f257612ff3397bbf2.jpg)

### Tables

![0d48b98dc8e58d430fcff540b4da91986015b83487a1aab4e8d23ae2be77ef83.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/tables/0d48b98dc8e58d430fcff540b4da91986015b83487a1aab4e8d23ae2be77ef83.jpg)

## MIND over Body: Adaptive Thinking using Dynamic Computation


### Images

![287858bd88f39f5aaae222acd99bde2d7777918d403e464381978746eebdd8ab.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/287858bd88f39f5aaae222acd99bde2d7777918d403e464381978746eebdd8ab.jpg)

![4f79daf8b422e9560867327686154a6b39f38536271c60771fb2c32a3b6352bf.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/4f79daf8b422e9560867327686154a6b39f38536271c60771fb2c32a3b6352bf.jpg)

![54f979b1c1471bb8e7fe5ea56f84977ddace30ba81bce3eb834686e3a2f8b4ff.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/54f979b1c1471bb8e7fe5ea56f84977ddace30ba81bce3eb834686e3a2f8b4ff.jpg)

![66371ea5cdd870d85ad5275be1f2ae6fdf1d60f83d5deb705d3cfbd50c66d349.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/66371ea5cdd870d85ad5275be1f2ae6fdf1d60f83d5deb705d3cfbd50c66d349.jpg)

![6749cd688e3a81b97362885c75570892e3b565fc0ed7b427e1bc8eca004e5596.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/6749cd688e3a81b97362885c75570892e3b565fc0ed7b427e1bc8eca004e5596.jpg)

![94cc16b9c955d830512d7085f620f8a864e9abcc8be07c112fa527e4b500ce25.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/94cc16b9c955d830512d7085f620f8a864e9abcc8be07c112fa527e4b500ce25.jpg)

![a01f988dcf3d653574045d7bf6d9312c8b451c116b2b95969abf2ac8cd0cd54d.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/a01f988dcf3d653574045d7bf6d9312c8b451c116b2b95969abf2ac8cd0cd54d.jpg)

![a2f69eba4fafd43682351d5a6d5d00b20881f21f87e906132a3e48aea5c89e28.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/a2f69eba4fafd43682351d5a6d5d00b20881f21f87e906132a3e48aea5c89e28.jpg)

![ba52e17ad0fd05b671ddad7ba5e61737a1a1815a1eec9662c827dcd166242adc.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/ba52e17ad0fd05b671ddad7ba5e61737a1a1815a1eec9662c827dcd166242adc.jpg)

![bfcc17f783fb2b1ecbaf3ce72c7a9c7c97e80505241450933f043e3c5b6f32c2.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/bfcc17f783fb2b1ecbaf3ce72c7a9c7c97e80505241450933f043e3c5b6f32c2.jpg)

![e2d5eefc4ee33215f2dab5de9d3f9b08ab0b8f35c60b83f8ed6d5b78618d7f98.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/e2d5eefc4ee33215f2dab5de9d3f9b08ab0b8f35c60b83f8ed6d5b78618d7f98.jpg)

### Tables

![008141789ca2d91e32ddd4970adcd8b8fc91b5fe48bda0dabf334872039f3fe0.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/008141789ca2d91e32ddd4970adcd8b8fc91b5fe48bda0dabf334872039f3fe0.jpg)

![083c1a5ae22185cf6a19867e3d2c42cb5848faa7ade51c04517f17d37bf756bf.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/083c1a5ae22185cf6a19867e3d2c42cb5848faa7ade51c04517f17d37bf756bf.jpg)

![4125ad532e57579cb7bb768c43c7c7daf859f7265d5054d12b7b5c44c104941a.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/4125ad532e57579cb7bb768c43c7c7daf859f7265d5054d12b7b5c44c104941a.jpg)

![5b9ad05d30bcb9212d382a364f3a58f8328b79af7d16857b00957c4df2e3af00.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/5b9ad05d30bcb9212d382a364f3a58f8328b79af7d16857b00957c4df2e3af00.jpg)

![5c4e10526d0912717f8fa2979312364e044aae3ac72f409f237d3ce9dea1d97c.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/5c4e10526d0912717f8fa2979312364e044aae3ac72f409f237d3ce9dea1d97c.jpg)

![66ce0136eaed9c8399062b216854a44ec6e312b9bdf80efd74fd30b31e2cf9a1.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/66ce0136eaed9c8399062b216854a44ec6e312b9bdf80efd74fd30b31e2cf9a1.jpg)

![6a8ab888f8866f9e67a505a5052ecd6701825528934a02b8ae66f17aa794f0cd.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/6a8ab888f8866f9e67a505a5052ecd6701825528934a02b8ae66f17aa794f0cd.jpg)

![796d381023ccc5ed3ac19496756a76203bc66aa04598b09e3e171d8e92a52fce.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/796d381023ccc5ed3ac19496756a76203bc66aa04598b09e3e171d8e92a52fce.jpg)

![8aba64769f771314c7a962ae7d063ccf14d2229dcf2b46c7deb3c76fcab6d496.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/8aba64769f771314c7a962ae7d063ccf14d2229dcf2b46c7deb3c76fcab6d496.jpg)

![9284b68ca70c41e3fed7bb0b1925e1a144283894197581718f1068fe1dfdc6b8.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/9284b68ca70c41e3fed7bb0b1925e1a144283894197581718f1068fe1dfdc6b8.jpg)

![a110a2803a4365b2a0cdc737c8670878e355021026149014cd1e35303d076a62.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/a110a2803a4365b2a0cdc737c8670878e355021026149014cd1e35303d076a62.jpg)

![d5d7a6014b829f54bd047fc71fea58a67eca9613209df258401ffc978eeba40f.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/d5d7a6014b829f54bd047fc71fea58a67eca9613209df258401ffc978eeba40f.jpg)

![d914bb01992902ff8a68ae23f8e3d7d8aa32667968e58f41333ceaab97b3633f.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/d914bb01992902ff8a68ae23f8e3d7d8aa32667968e58f41333ceaab97b3633f.jpg)

![da9e896cbe49f3357fde8044b93a698e7f44a1a5ff9bc66dd1105eb5160016ab.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/da9e896cbe49f3357fde8044b93a698e7f44a1a5ff9bc66dd1105eb5160016ab.jpg)

![e09c11e36edaa27c062f92a3c6905a0f1c1172e336e2eccb741bcb54c8c371a4.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/e09c11e36edaa27c062f92a3c6905a0f1c1172e336e2eccb741bcb54c8c371a4.jpg)

![eb253e6bd2a7cc5dd7527adcfcdcc99b3cdc6e2d0b9ed5f32fcb928d6dddcc53.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/eb253e6bd2a7cc5dd7527adcfcdcc99b3cdc6e2d0b9ed5f32fcb928d6dddcc53.jpg)

![eeaed12669049e5379e6d1e846571d251918882cf6274f9d46135494bc598c7d.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/eeaed12669049e5379e6d1e846571d251918882cf6274f9d46135494bc598c7d.jpg)

## How much of my dataset did you use? Quantitative Data Usage Inference in Machine Learning


### Images

![0d291cb30106439c7be63fe3828798ccc96512a14c239139adfa3abc69b207db.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/0d291cb30106439c7be63fe3828798ccc96512a14c239139adfa3abc69b207db.jpg)

![2f53419d75d1b2caf2826ae1e93e268142c6b0e7954ee3607d28eadc25fb0d55.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/2f53419d75d1b2caf2826ae1e93e268142c6b0e7954ee3607d28eadc25fb0d55.jpg)

![591dabb5f17d42e351d63edb22ac1b1ada8b13f7025e40b1151eba5082188588.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/591dabb5f17d42e351d63edb22ac1b1ada8b13f7025e40b1151eba5082188588.jpg)

![84c31ac02c485ab02fad939f9172a8b123f676a5444da748ebaad8a51b892a91.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/84c31ac02c485ab02fad939f9172a8b123f676a5444da748ebaad8a51b892a91.jpg)

![8a5c0586e641ad3a4cc177a599be1a2603cbcc2d7143b9889eec59b0eaa2e021.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/8a5c0586e641ad3a4cc177a599be1a2603cbcc2d7143b9889eec59b0eaa2e021.jpg)

![9c3d1810df5b6b2203466023ee76e4fe18656cf54d876ffa8617f013d8dbb31b.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/9c3d1810df5b6b2203466023ee76e4fe18656cf54d876ffa8617f013d8dbb31b.jpg)

![babe6a03758cf2d64ceff51674b9ac21c7d506ab931bb80370c2eb74554408ef.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/babe6a03758cf2d64ceff51674b9ac21c7d506ab931bb80370c2eb74554408ef.jpg)

![e23fde6c56eb4e6ddf76eea7f2707168783ad9b97bfb44793a5b41180e629dc0.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/e23fde6c56eb4e6ddf76eea7f2707168783ad9b97bfb44793a5b41180e629dc0.jpg)

![f84c357a5d8d1c32807c960cc32f8573df61876df2de37c6ddc69c098211a6cd.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/f84c357a5d8d1c32807c960cc32f8573df61876df2de37c6ddc69c098211a6cd.jpg)

![f86d3d620b7549fa66b65d73c9bbb3e22acf35e4197f16b21a0dfbdceb35867c.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/f86d3d620b7549fa66b65d73c9bbb3e22acf35e4197f16b21a0dfbdceb35867c.jpg)

### Tables

![2a75b0cc0e1ef213350fed612f8498f6166c59e91865b019b58fdf9c43267ebd.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/2a75b0cc0e1ef213350fed612f8498f6166c59e91865b019b58fdf9c43267ebd.jpg)

![3225ba20638dacac277a4e2be79eaead60b10b1435ef4e17f6e6a63ae633a398.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/3225ba20638dacac277a4e2be79eaead60b10b1435ef4e17f6e6a63ae633a398.jpg)

![bda891fe05f415e031cf04d3eaf0141ef7f93c7f14437ac53f5a0cd8d95b9514.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/bda891fe05f415e031cf04d3eaf0141ef7f93c7f14437ac53f5a0cd8d95b9514.jpg)

![dc2a3b798f7b0db2cdeedcfe9edd4dde1ba726d12b3e120cc1a30d3ec6ea8404.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/dc2a3b798f7b0db2cdeedcfe9edd4dde1ba726d12b3e120cc1a30d3ec6ea8404.jpg)

![e1ad51921553f9fe038a5678e42f24f565bf9a31fb61b2b0e16c0a9ac3c996dd.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/e1ad51921553f9fe038a5678e42f24f565bf9a31fb61b2b0e16c0a9ac3c996dd.jpg)

![e3197568c97f6dfe6c166f9b5095079999232f9564423b2202bfecb078b930e2.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/e3197568c97f6dfe6c166f9b5095079999232f9564423b2202bfecb078b930e2.jpg)

![e3295c8372519faaf7d8304349cdc65108c87e67b49963d0cb61fce06355d280.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/e3295c8372519faaf7d8304349cdc65108c87e67b49963d0cb61fce06355d280.jpg)

## SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups


### Images

![48f45f86fa0656447f6d22c7c65edaba2d835beaba6a6936ba86ca5374610f04.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/images/48f45f86fa0656447f6d22c7c65edaba2d835beaba6a6936ba86ca5374610f04.jpg)

![a069ba5a70804e07bc5eed3eb65235ebaaa2eac07779555279719c57adc7fd1a.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/images/a069ba5a70804e07bc5eed3eb65235ebaaa2eac07779555279719c57adc7fd1a.jpg)

![f59f285cfab9d78378e8d0455beb3f68cc4557620397f03a681168ce3b5f8c8e.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/images/f59f285cfab9d78378e8d0455beb3f68cc4557620397f03a681168ce3b5f8c8e.jpg)

### Tables

![1313fca8ddc6e106dcd542e45a0d21d91a1909152ddb7b4d454eac093077ce3b.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/1313fca8ddc6e106dcd542e45a0d21d91a1909152ddb7b4d454eac093077ce3b.jpg)

![1c184844bd33d92748a15cfb911229470595daba7f64925a11d10db8bfcb7933.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/1c184844bd33d92748a15cfb911229470595daba7f64925a11d10db8bfcb7933.jpg)

![1c1ecf85e919a8a546c26ff928f5935bbdb75444ab1d0d8e6f61252dea53a975.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/1c1ecf85e919a8a546c26ff928f5935bbdb75444ab1d0d8e6f61252dea53a975.jpg)

![36f96b9de4c0bf2e8c46221bf759ebe77250e27e78a1c3050f1ece1f4f7c81ce.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/36f96b9de4c0bf2e8c46221bf759ebe77250e27e78a1c3050f1ece1f4f7c81ce.jpg)

![38c7eadaf239f9e7d90d57fd9cd71b2af348fae42cce8134565b8c7c205e5553.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/38c7eadaf239f9e7d90d57fd9cd71b2af348fae42cce8134565b8c7c205e5553.jpg)

![4ff2685427b8de14782770ae85b9b7462acb79a525bbb9bdaeb8e297c6de6816.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/4ff2685427b8de14782770ae85b9b7462acb79a525bbb9bdaeb8e297c6de6816.jpg)

![7df70981d5401a1495311da893eb0115a055f60d396c6dcb93e7932c208585a3.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/7df70981d5401a1495311da893eb0115a055f60d396c6dcb93e7932c208585a3.jpg)

![94fa9aaf92260e967ff70cd9d3b17c970e7940d1f5772b3472949b3b8e1ba1a7.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/94fa9aaf92260e967ff70cd9d3b17c970e7940d1f5772b3472949b3b8e1ba1a7.jpg)

![95182a9e10480c3b2d42440d65ad944a12ef97d3edd0acab02f02223dfefe4fb.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/95182a9e10480c3b2d42440d65ad944a12ef97d3edd0acab02f02223dfefe4fb.jpg)

![9c408020edf89e33d506cea0cb1af524a6120634714a443eddc7b1029166d438.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/9c408020edf89e33d506cea0cb1af524a6120634714a443eddc7b1029166d438.jpg)

![a00b1ae36c335adfaa1eed98f4d4072b9ea1ed2c79d63ba5c384454d270efd46.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/a00b1ae36c335adfaa1eed98f4d4072b9ea1ed2c79d63ba5c384454d270efd46.jpg)

![e8cac9325f2b3d89954299f8db639a161b6757ba5fee849ba6916a5a3a4f6c3a.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/e8cac9325f2b3d89954299f8db639a161b6757ba5fee849ba6916a5a3a4f6c3a.jpg)

![f55114ced3fad26eb6dea6002ecaee91ecac7d4b0033f3fe4020fe929d3e7c96.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/f55114ced3fad26eb6dea6002ecaee91ecac7d4b0033f3fe4020fe929d3e7c96.jpg)

![fa147f9024bf7e64bba4f623aa197a1cc06e0259be8b8042541923556f327e95.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/fa147f9024bf7e64bba4f623aa197a1cc06e0259be8b8042541923556f327e95.jpg)

## Cut Your Losses in Large-Vocabulary Language Models


### Images

![1fe57519dad0f2313e7217ff3e7fc3841c5c63ad694b1099a5e200533b85a986.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/1fe57519dad0f2313e7217ff3e7fc3841c5c63ad694b1099a5e200533b85a986.jpg)

![36b3ab3a964ea79779db9b7aa1a251e6841a95d3aaf232d0e5236fcc0127bb77.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/36b3ab3a964ea79779db9b7aa1a251e6841a95d3aaf232d0e5236fcc0127bb77.jpg)

![64a05bbf2c8829ab9c3389d9f1df959b10e8a1ee4712278ea5de7a5079e59bf8.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/64a05bbf2c8829ab9c3389d9f1df959b10e8a1ee4712278ea5de7a5079e59bf8.jpg)

![84f586e51a67fd4b27ea124be8aa876ef586c837fe807f3d1940f6ee3c3f15cb.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/84f586e51a67fd4b27ea124be8aa876ef586c837fe807f3d1940f6ee3c3f15cb.jpg)

![95d5f0de2360b08d1f9615dad9922ceb80cc5111112a420a06650233b7f1963c.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/95d5f0de2360b08d1f9615dad9922ceb80cc5111112a420a06650233b7f1963c.jpg)

![9a35190106ebce0d570ba986197c90e30934b2320960f7d7dd66f6c0528b30a6.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/9a35190106ebce0d570ba986197c90e30934b2320960f7d7dd66f6c0528b30a6.jpg)

![c4ef1ac05c0fe9df5f589fa1fad84c9a7241f1d36a83a219c2a8c47a2a2a9307.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/c4ef1ac05c0fe9df5f589fa1fad84c9a7241f1d36a83a219c2a8c47a2a2a9307.jpg)

### Tables

![1f63097fe9ce12f0583faf9b38e4baadc7d4a6c6398e91475f76ad39a81ed3b1.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/1f63097fe9ce12f0583faf9b38e4baadc7d4a6c6398e91475f76ad39a81ed3b1.jpg)

![454b6538f37da4bf53382bfd9a947a78989503917ab5e6ab6ef18e8fe15ab9e6.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/454b6538f37da4bf53382bfd9a947a78989503917ab5e6ab6ef18e8fe15ab9e6.jpg)

![4a69cba34de8d167bb39bcb2edbf4e922b7a6cf04204f54c0b7e3937dbf22c36.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/4a69cba34de8d167bb39bcb2edbf4e922b7a6cf04204f54c0b7e3937dbf22c36.jpg)

![6a414752cc60b5c352300e1c5fa17129b645f844f8da94e53854c77cd0a93091.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/6a414752cc60b5c352300e1c5fa17129b645f844f8da94e53854c77cd0a93091.jpg)

![756563ad190dcb2d590f77b4d60751b7b3f35aad6888c426ba7bc01e56f16aea.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/756563ad190dcb2d590f77b4d60751b7b3f35aad6888c426ba7bc01e56f16aea.jpg)

![766757a12a2a6cbd42e54557c6cd70ce1a27a5a703f7146171847100a9c6966e.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/766757a12a2a6cbd42e54557c6cd70ce1a27a5a703f7146171847100a9c6966e.jpg)

![88303db36270dacca1cac10e5a30a0a37262533bedf78bbbf95e87964571b191.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/88303db36270dacca1cac10e5a30a0a37262533bedf78bbbf95e87964571b191.jpg)

![8fa39e4270bd92620a5b7f1de1a6627745c135cd7634ed17e8f74b0e5f00312f.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/8fa39e4270bd92620a5b7f1de1a6627745c135cd7634ed17e8f74b0e5f00312f.jpg)

![952d1a748a7b448ed4ba74626317e924aceee811fa147b64b55f67d4f37314a2.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/952d1a748a7b448ed4ba74626317e924aceee811fa147b64b55f67d4f37314a2.jpg)

![d17485c452baee7cc029e774ca6dab8e6fef40a7ee50cebb776678f390d05f57.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/d17485c452baee7cc029e774ca6dab8e6fef40a7ee50cebb776678f390d05f57.jpg)

## Interpreting Emergent Planning in Model-Free Reinforcement Learning


### Images

![01376b29a6f0cdb2178d890ebf40321c478d560d23a913633352daf7eff2a74c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/01376b29a6f0cdb2178d890ebf40321c478d560d23a913633352daf7eff2a74c.jpg)

![0229fcea13ac3f049803def424803f0ebf8c9feb15e68e2ce4a1818980b557dd.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0229fcea13ac3f049803def424803f0ebf8c9feb15e68e2ce4a1818980b557dd.jpg)

![05c3f0a7e49b4d13905eeb76cea299dd6e5d30234f507af619df54b37cf65e89.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/05c3f0a7e49b4d13905eeb76cea299dd6e5d30234f507af619df54b37cf65e89.jpg)

![05e2c9bbff6df46927079b52645d045a80e56361f9a75ced74de9a0b9f6817a0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/05e2c9bbff6df46927079b52645d045a80e56361f9a75ced74de9a0b9f6817a0.jpg)

![082362ebfc29a7b248559b76049a8c0ab16d8446adb3115d114e31a5dc850f2f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/082362ebfc29a7b248559b76049a8c0ab16d8446adb3115d114e31a5dc850f2f.jpg)

![0c1a825f53ec20f0b5903c13dad5537d9e13c1a2a8fa592e92f8d9ce659dfb33.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0c1a825f53ec20f0b5903c13dad5537d9e13c1a2a8fa592e92f8d9ce659dfb33.jpg)

![0c3c9243910269a176ad1dc04ab1c3ad83bae1400516b46cafb3ae37593042b3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0c3c9243910269a176ad1dc04ab1c3ad83bae1400516b46cafb3ae37593042b3.jpg)

![0c4ddcde80357a22f06bda3a797e3753cae50e504f481204560dc4a4b2381f01.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0c4ddcde80357a22f06bda3a797e3753cae50e504f481204560dc4a4b2381f01.jpg)

![0f064d62c8d12505185591422eeabddf3c35099886fbe09ae1668d35790ed372.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0f064d62c8d12505185591422eeabddf3c35099886fbe09ae1668d35790ed372.jpg)

![0f8be8c8da76c322bace52d3d952827fd6276240caaf8a2e04a592062c9adff9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0f8be8c8da76c322bace52d3d952827fd6276240caaf8a2e04a592062c9adff9.jpg)

![150bf70908a7f1bb116930284738179a10fadb8da50be4c1447e5099902c8ba5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/150bf70908a7f1bb116930284738179a10fadb8da50be4c1447e5099902c8ba5.jpg)

![153145122f7bee48c1d9e8e03b4b606cf55688c13969be51651ec0f82723b813.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/153145122f7bee48c1d9e8e03b4b606cf55688c13969be51651ec0f82723b813.jpg)

![15b571ee9ecb2e92c638d40f93f34f233fc99d43054b3f68452779d57a5d92b9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/15b571ee9ecb2e92c638d40f93f34f233fc99d43054b3f68452779d57a5d92b9.jpg)

![16270d034526f679319e7787beafed1a7a7303e10f05d9d518433596909c1325.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/16270d034526f679319e7787beafed1a7a7303e10f05d9d518433596909c1325.jpg)

![17147258dc9e13ba1d624656d18645ba127dd3459183356885f8e47bf80afc91.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/17147258dc9e13ba1d624656d18645ba127dd3459183356885f8e47bf80afc91.jpg)

![17d8db35194cb61c87f929b9ed8dcd7e873e2d13f4f4186a8c729663de7a929b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/17d8db35194cb61c87f929b9ed8dcd7e873e2d13f4f4186a8c729663de7a929b.jpg)

![18662e34c332c735dcb37def7622711ed74b77bcfc8f11524d9cb7e6a6f70743.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/18662e34c332c735dcb37def7622711ed74b77bcfc8f11524d9cb7e6a6f70743.jpg)

![1cf030f57606a6200859a0b28f91297c4d5e50022bf5bd3f204eaa28457047dc.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/1cf030f57606a6200859a0b28f91297c4d5e50022bf5bd3f204eaa28457047dc.jpg)

![1e95506b1dd0b66d04d9ec8df41791eaa70d1527156416c1136cdeef02be5849.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/1e95506b1dd0b66d04d9ec8df41791eaa70d1527156416c1136cdeef02be5849.jpg)

![1efe5ca1d974c3f5fc4b16912a26e286f108cd470593b001fb0015fe4f6c2b8d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/1efe5ca1d974c3f5fc4b16912a26e286f108cd470593b001fb0015fe4f6c2b8d.jpg)

![211cae2a10bd6edb294c9314886e06fb840269a6613d1cab63cd8e67560b39e1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/211cae2a10bd6edb294c9314886e06fb840269a6613d1cab63cd8e67560b39e1.jpg)

![26d55ad6b162f3676a33494febf2467bb4dfe8f7f6a8997485b3eb99695d69ea.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/26d55ad6b162f3676a33494febf2467bb4dfe8f7f6a8997485b3eb99695d69ea.jpg)

![2802b12131fc7496928c8c2372607732ea8b736b213ec27d064161408f8da24d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2802b12131fc7496928c8c2372607732ea8b736b213ec27d064161408f8da24d.jpg)

![2b7d07429cd65106564d66a18916cda8e31e486b06758f9c96df72b3deb65d33.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2b7d07429cd65106564d66a18916cda8e31e486b06758f9c96df72b3deb65d33.jpg)

![2d836d1ab25cbdb71b671ee4a03aca2e65a31ae0e8bb4b2886c901322519d1c9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2d836d1ab25cbdb71b671ee4a03aca2e65a31ae0e8bb4b2886c901322519d1c9.jpg)

![2ec25d89ca9e6bd0fa5b2ceb0e6dc6d45cc384b3b9dd564e6061fadab16333ab.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2ec25d89ca9e6bd0fa5b2ceb0e6dc6d45cc384b3b9dd564e6061fadab16333ab.jpg)

![304534e82c0e21d4e6629bcf4ea60e44fe912374b8d1b2eafd3cea63e35c6352.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/304534e82c0e21d4e6629bcf4ea60e44fe912374b8d1b2eafd3cea63e35c6352.jpg)

![328a91e72b1720eef85e4414cd594d00eed5dea70c7b0c2ba0c2b8fcffbd7dfa.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/328a91e72b1720eef85e4414cd594d00eed5dea70c7b0c2ba0c2b8fcffbd7dfa.jpg)

![330020ec57b3898219ee7e653c1117b630b349ad8c13fbca35ccb88b270cb158.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/330020ec57b3898219ee7e653c1117b630b349ad8c13fbca35ccb88b270cb158.jpg)

![35e6f236215c76e64647cfe1bf295cb54f31583404269d1fdefdc1d5ced52a3e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/35e6f236215c76e64647cfe1bf295cb54f31583404269d1fdefdc1d5ced52a3e.jpg)

![39d212f6259955aba4d50e7e575714c607d1a923addf8781ee79e6462b596711.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/39d212f6259955aba4d50e7e575714c607d1a923addf8781ee79e6462b596711.jpg)

![3d1f30667e18788c1272e5c5e2e48211e607497cc758001edf9556e38303afc7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/3d1f30667e18788c1272e5c5e2e48211e607497cc758001edf9556e38303afc7.jpg)

![3d5ceff7d271ad0d2e8a6779d166d46b433b7d09d8b3d1d86dfa0cbac4913da7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/3d5ceff7d271ad0d2e8a6779d166d46b433b7d09d8b3d1d86dfa0cbac4913da7.jpg)

![3fa9aacd23c9dea714e55e863acefb65db098731c40e65b62548273c8a42233e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/3fa9aacd23c9dea714e55e863acefb65db098731c40e65b62548273c8a42233e.jpg)

![4019df224b5caf3151851f13ad42417500e68eddccda1fb09eb521cbb01a4703.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4019df224b5caf3151851f13ad42417500e68eddccda1fb09eb521cbb01a4703.jpg)

![409a6d4eb25de2cd867bc79949f71e7c6db51abaac0e5d855578c4b1f6c179b3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/409a6d4eb25de2cd867bc79949f71e7c6db51abaac0e5d855578c4b1f6c179b3.jpg)

![40a97c685308c286aa10ece0aca0c7c2cf410c9705e484865e0aac406b94ba48.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/40a97c685308c286aa10ece0aca0c7c2cf410c9705e484865e0aac406b94ba48.jpg)

![40cb92e8a76219b5f578764751ff456f0131b2746581bddfaa8ab1d14c44d712.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/40cb92e8a76219b5f578764751ff456f0131b2746581bddfaa8ab1d14c44d712.jpg)

![40e80a28edb19730447bd019fc82ab035390391b4ac937c7da5fdc4dc5b27508.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/40e80a28edb19730447bd019fc82ab035390391b4ac937c7da5fdc4dc5b27508.jpg)

![41f77ccdc552b73d1506c91686d889eb2bb4624ada0bb3917a82b8d4b78f4db7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/41f77ccdc552b73d1506c91686d889eb2bb4624ada0bb3917a82b8d4b78f4db7.jpg)

![44c43668ef21ef87e922395323f74cb80ad855b45c87538ee6748a81c0cf6dc5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/44c43668ef21ef87e922395323f74cb80ad855b45c87538ee6748a81c0cf6dc5.jpg)

![4577fe1698b75fe1253661636157fefaafb14d0f9c0510ea856f981d351cbb09.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4577fe1698b75fe1253661636157fefaafb14d0f9c0510ea856f981d351cbb09.jpg)

![494ae637e69d7d5f93d6beb47e7221a844f41b30c617cb542ee139ca13c2b665.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/494ae637e69d7d5f93d6beb47e7221a844f41b30c617cb542ee139ca13c2b665.jpg)

![4ab34da8696a722e45a0fe9ca1b6fd2a4293cc290067f9c4eb5e890b68210de7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4ab34da8696a722e45a0fe9ca1b6fd2a4293cc290067f9c4eb5e890b68210de7.jpg)

![4ba8023dcd8f83b190343487c88d87e182b132899192d85a90a47291a79ef49b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4ba8023dcd8f83b190343487c88d87e182b132899192d85a90a47291a79ef49b.jpg)

![4c5cb4d137324820756dc94662624d93afaad55fad2dbc577b3fc26711117dae.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4c5cb4d137324820756dc94662624d93afaad55fad2dbc577b3fc26711117dae.jpg)

![4d723afc3cf0defb7cde71ffd5b436671d01b905e81ff08d4f31db824e9d5775.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4d723afc3cf0defb7cde71ffd5b436671d01b905e81ff08d4f31db824e9d5775.jpg)

![51923328281f488a50702e875900f34ab079f1e5195b2c8b43fce7f9ec3f5f4e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/51923328281f488a50702e875900f34ab079f1e5195b2c8b43fce7f9ec3f5f4e.jpg)

![54642fe1b5775e3b187aeefe3dd0d7019dd7fed049d59984ff3458a562566c30.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/54642fe1b5775e3b187aeefe3dd0d7019dd7fed049d59984ff3458a562566c30.jpg)

![56a94806c747fbd20f5dc94381227f92057cfcc8d270e503658dd4ae8b53a86b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/56a94806c747fbd20f5dc94381227f92057cfcc8d270e503658dd4ae8b53a86b.jpg)

![587e93535b6922c3cd2f6d21d3109e2111d8b298d4ae8cdd802fff0a79151a5d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/587e93535b6922c3cd2f6d21d3109e2111d8b298d4ae8cdd802fff0a79151a5d.jpg)

![610818dec51424c43c017bc292f42839660ca35d932ad4b271bcf57d88ebc6c0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/610818dec51424c43c017bc292f42839660ca35d932ad4b271bcf57d88ebc6c0.jpg)

![61b3f696bd8384aced3d250e9832da4ad3a8dc1f7a34de0b962ea06277e63d8d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/61b3f696bd8384aced3d250e9832da4ad3a8dc1f7a34de0b962ea06277e63d8d.jpg)

![6be1d66d59820a84356ca86ae2c459b3f4e591e05499b91f95b53248f517450e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6be1d66d59820a84356ca86ae2c459b3f4e591e05499b91f95b53248f517450e.jpg)

![6cf3aac069cee5992ceeb8d9d958c6f6a63658a49cef1338adbdc73b822a8839.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6cf3aac069cee5992ceeb8d9d958c6f6a63658a49cef1338adbdc73b822a8839.jpg)

![6d205eab200a39b4e2ad7a529c03e271c2f23de5ae289975d039098d9d994e63.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6d205eab200a39b4e2ad7a529c03e271c2f23de5ae289975d039098d9d994e63.jpg)

![6d2aa70882bdb1220f2713ee5b35472e926f8fda16da948fa034d4241fcceaae.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6d2aa70882bdb1220f2713ee5b35472e926f8fda16da948fa034d4241fcceaae.jpg)

![6f8c27d749e2c83386d40b4c9ca590ad6366767df1d31325c995c4d089decac5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6f8c27d749e2c83386d40b4c9ca590ad6366767df1d31325c995c4d089decac5.jpg)

![7176a2041ecad328a8d0023210954e39ab1d1140b819ec6a94a8bffa719cc930.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/7176a2041ecad328a8d0023210954e39ab1d1140b819ec6a94a8bffa719cc930.jpg)

![746d24843bb566aafb844ef1076682202d174128890698bed492dfb670a61e08.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/746d24843bb566aafb844ef1076682202d174128890698bed492dfb670a61e08.jpg)

![77ab030f65f825bbe1853bc5e8b242ef87d64876b3afc88b2be92ab42ae30ec2.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/77ab030f65f825bbe1853bc5e8b242ef87d64876b3afc88b2be92ab42ae30ec2.jpg)

![7b48952ffa502b5613c8c25c6e396a301489cbbf094ed0587acbe361b791d56a.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/7b48952ffa502b5613c8c25c6e396a301489cbbf094ed0587acbe361b791d56a.jpg)

![826d05d978fe680a31bd7df203422de4bd65569cdf98ef7ac0c76f64ebd0391d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/826d05d978fe680a31bd7df203422de4bd65569cdf98ef7ac0c76f64ebd0391d.jpg)

![82fb5709a7c150a324e3687f01241230924f288a518e62fec68f165812fd9a52.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/82fb5709a7c150a324e3687f01241230924f288a518e62fec68f165812fd9a52.jpg)

![86bc5f56f8c9f0ce542561ae4f38fe60343b93ea384faf3d877f000402100340.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/86bc5f56f8c9f0ce542561ae4f38fe60343b93ea384faf3d877f000402100340.jpg)

![86eda7cbf5e6d97fe30b5e7fa459d1e4f023f397edc39cae54580a484f09421a.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/86eda7cbf5e6d97fe30b5e7fa459d1e4f023f397edc39cae54580a484f09421a.jpg)

![8800e96cb447cdef467dcab694b23440ce3d9edbab7825084947aedb62908e50.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8800e96cb447cdef467dcab694b23440ce3d9edbab7825084947aedb62908e50.jpg)

![89b61b6f0e5638dd4f549ab07074facf3bf0b8a2f858e118caeec6aed349caaf.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/89b61b6f0e5638dd4f549ab07074facf3bf0b8a2f858e118caeec6aed349caaf.jpg)

![8ad742a8bdc1a6736b2d3cfdab384dfc53bfc708f20a368cf5f46d321662c3e3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8ad742a8bdc1a6736b2d3cfdab384dfc53bfc708f20a368cf5f46d321662c3e3.jpg)

![8c4cb4efefd3fd21b74e185d730b352403559a7512c4fd106efdb6bf46948ee2.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8c4cb4efefd3fd21b74e185d730b352403559a7512c4fd106efdb6bf46948ee2.jpg)

![8d7fa403ab125ff847d8786c39f6647847e997493a7a82c52628429771ac8bab.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8d7fa403ab125ff847d8786c39f6647847e997493a7a82c52628429771ac8bab.jpg)

![9100110906b68ee13dd0f4a53229fd1f10e23aa049fb6f59459c5d170393080d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9100110906b68ee13dd0f4a53229fd1f10e23aa049fb6f59459c5d170393080d.jpg)

![912da1602a20bfbe4a76861aa082756d44f883291d0afd93c94b46fd3c022c4c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/912da1602a20bfbe4a76861aa082756d44f883291d0afd93c94b46fd3c022c4c.jpg)

![93e4e918a97545d02573a04b48388bc5c46d131de244e19cc897cca6f2dbdb51.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/93e4e918a97545d02573a04b48388bc5c46d131de244e19cc897cca6f2dbdb51.jpg)

![94739d612a563cffdbde471b10315a520f1674bb8d8bbae545976e701d5eba03.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/94739d612a563cffdbde471b10315a520f1674bb8d8bbae545976e701d5eba03.jpg)

![9aaf08863584d40af857a214af0ca63630fa10ec52647f7b79c898551bb264c7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9aaf08863584d40af857a214af0ca63630fa10ec52647f7b79c898551bb264c7.jpg)

![9df7b6f3f37f8f58ee414689378b3717d2ff9055d9584996be268c0db41ff0a0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9df7b6f3f37f8f58ee414689378b3717d2ff9055d9584996be268c0db41ff0a0.jpg)

![9f71323ba88ff1cee9399c8f5eaf83a96260e1caeeb16abafec87c1702266074.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9f71323ba88ff1cee9399c8f5eaf83a96260e1caeeb16abafec87c1702266074.jpg)

![a03f97991a8cf139a26a24972648b18622935103b7192d0446f4911e7db1cd61.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a03f97991a8cf139a26a24972648b18622935103b7192d0446f4911e7db1cd61.jpg)

![a05f3b69f385be6496ba60c98b7059b43663dce3c6429e277f1265dc399f18ec.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a05f3b69f385be6496ba60c98b7059b43663dce3c6429e277f1265dc399f18ec.jpg)

![a090a426a5f07517a02fd7a0e4833a1b5e57731ace22df6de2d594fd4be64ae6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a090a426a5f07517a02fd7a0e4833a1b5e57731ace22df6de2d594fd4be64ae6.jpg)

![a1cd9b627fd52eaebb4ee60973c8d7485ae050b6a21c62f1bac5b02297f24025.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a1cd9b627fd52eaebb4ee60973c8d7485ae050b6a21c62f1bac5b02297f24025.jpg)

![a2f0773ea86e45da1b0ecfe1f88ed59aa5a7d60f3b72f9ce4b31bb7cf36c6a61.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a2f0773ea86e45da1b0ecfe1f88ed59aa5a7d60f3b72f9ce4b31bb7cf36c6a61.jpg)

![a982b685b6355b35333bcfe01c9e06ac4dfdae2e2399c919ab58292c0ecf47ba.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a982b685b6355b35333bcfe01c9e06ac4dfdae2e2399c919ab58292c0ecf47ba.jpg)

![ac5cc63ef95753a6cfb3f0a422a49df4466bb2b76b54a69964e8b27d89453ee1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ac5cc63ef95753a6cfb3f0a422a49df4466bb2b76b54a69964e8b27d89453ee1.jpg)

![ad469c4af3263f6bb1289a5f5975ff312400c9a5dcbe84381a12065ac88c029b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ad469c4af3263f6bb1289a5f5975ff312400c9a5dcbe84381a12065ac88c029b.jpg)

![ad90678846eee2761f9de7221e980972f143b29649375b7c24dcb1ded72b9611.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ad90678846eee2761f9de7221e980972f143b29649375b7c24dcb1ded72b9611.jpg)

![addde885099758428352ec46ade878af2272f618b2ddbd3cec06518b3491f8b5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/addde885099758428352ec46ade878af2272f618b2ddbd3cec06518b3491f8b5.jpg)

![ae4b8f1efd89e1c30ce8fdb1e8452e56b08efcd2b84862dc47b6615386fdcde1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ae4b8f1efd89e1c30ce8fdb1e8452e56b08efcd2b84862dc47b6615386fdcde1.jpg)

![b0b08632c624d25b786a247656c2b243c9227cb81eba62f3290be60861021912.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b0b08632c624d25b786a247656c2b243c9227cb81eba62f3290be60861021912.jpg)

![b5611283c658310d7e683786790216959a11c8b092c44eed371e7ef16b83344d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b5611283c658310d7e683786790216959a11c8b092c44eed371e7ef16b83344d.jpg)

![b6e9919780c9521801af9d468f6273b106d3347c901cf6fc6bf5b10a90b956d6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b6e9919780c9521801af9d468f6273b106d3347c901cf6fc6bf5b10a90b956d6.jpg)

![b6ea0b76e31bc218e82c43d796a040e45754b94d4ab829a872db70ac59990094.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b6ea0b76e31bc218e82c43d796a040e45754b94d4ab829a872db70ac59990094.jpg)

![b713eae540ad44d194f2440682e737430329de06559402389403617ed24027f0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b713eae540ad44d194f2440682e737430329de06559402389403617ed24027f0.jpg)

![b84928a116097f8e807629f8682fe0fc505a105965834e0606569973f2524344.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b84928a116097f8e807629f8682fe0fc505a105965834e0606569973f2524344.jpg)

![b8c80ffff49dc30d9e9053f27be0103e4b11f5cd52d6e12f45887fbaba626ef0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b8c80ffff49dc30d9e9053f27be0103e4b11f5cd52d6e12f45887fbaba626ef0.jpg)

![b9d5cce93ca4f696093c6792082a1239e4dba6270382bed8081b423ac452ae0c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b9d5cce93ca4f696093c6792082a1239e4dba6270382bed8081b423ac452ae0c.jpg)

![bfcae45abf5057a152428e1ee7bb629dee2d976c9a98b3f8638347f6a0d3f703.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/bfcae45abf5057a152428e1ee7bb629dee2d976c9a98b3f8638347f6a0d3f703.jpg)

![c054468320c6324f7cfabf6c652c428812ac638e57f20d2edf9d131b87113619.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c054468320c6324f7cfabf6c652c428812ac638e57f20d2edf9d131b87113619.jpg)

![c3edeb5d84f2bd60742dfa8b5ff5ed76e26550a21a9ddc6bf66cb2d726eeb9fa.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c3edeb5d84f2bd60742dfa8b5ff5ed76e26550a21a9ddc6bf66cb2d726eeb9fa.jpg)

![c51c00638a7fb92bbad53a382ed830c8f0f5e634cdcdfe87b509002b98570db6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c51c00638a7fb92bbad53a382ed830c8f0f5e634cdcdfe87b509002b98570db6.jpg)

![c5b85fdfe56cea9eef8df3a15b6220155d159070cb95e5f4daba4caca9adaf3e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c5b85fdfe56cea9eef8df3a15b6220155d159070cb95e5f4daba4caca9adaf3e.jpg)

![c74fb932744fff4dd9c679c69573050a053bdad19bf56d5e638f1103a27f9dc5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c74fb932744fff4dd9c679c69573050a053bdad19bf56d5e638f1103a27f9dc5.jpg)

![c847c74f0e3ab3208b5eb84c1334ea03594ade381e9d82f3f04fe180d4b8a8ea.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c847c74f0e3ab3208b5eb84c1334ea03594ade381e9d82f3f04fe180d4b8a8ea.jpg)

![ca9cbe0933fd7efc790b50215f4e3eaaa84792bcc60b5b3b9fbe84249ee8af79.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ca9cbe0933fd7efc790b50215f4e3eaaa84792bcc60b5b3b9fbe84249ee8af79.jpg)

![cc392434249763212745cf492b8699933629c4cfc6e0c4f65a7f150238e03340.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/cc392434249763212745cf492b8699933629c4cfc6e0c4f65a7f150238e03340.jpg)

![ce18318130b568b14a3f99d635a4bc5ba788fc287da2124f0ea08eb5e44cf2e3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ce18318130b568b14a3f99d635a4bc5ba788fc287da2124f0ea08eb5e44cf2e3.jpg)

![d0a5aa69c145ef3b61335103246378ee7bb11392962bdcbd4ea298a4e9c4819f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d0a5aa69c145ef3b61335103246378ee7bb11392962bdcbd4ea298a4e9c4819f.jpg)

![d0bc88296b7cbc6dca3f11cbb0be9184c38442253734176a0a7dc485fc600d55.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d0bc88296b7cbc6dca3f11cbb0be9184c38442253734176a0a7dc485fc600d55.jpg)

![d0fca59043609e04ccbc6e8498396f835e4c24f10ce198bcda3ae2e3c7a43aaf.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d0fca59043609e04ccbc6e8498396f835e4c24f10ce198bcda3ae2e3c7a43aaf.jpg)

![d21efdcb50460dba19bb6abe9e1bcd88533c02fa0a678cde0f788a5d3fcf9916.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d21efdcb50460dba19bb6abe9e1bcd88533c02fa0a678cde0f788a5d3fcf9916.jpg)

![d3b142c5755cd96a1aeefc1479886388435a79f4f23698d491c50e78da1d0008.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d3b142c5755cd96a1aeefc1479886388435a79f4f23698d491c50e78da1d0008.jpg)

![d56443afe920f538eae3920b7f1c6b03b72cdbb31478d7e4dc261204c22e406e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d56443afe920f538eae3920b7f1c6b03b72cdbb31478d7e4dc261204c22e406e.jpg)

![d6cf4f29d3e2b94ee36e046e9b9c9837e74c435a7d85e9204ef589028452d2ed.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d6cf4f29d3e2b94ee36e046e9b9c9837e74c435a7d85e9204ef589028452d2ed.jpg)

![d8383d70be6ce43b9d15ae4eecb7cdd04036972418273f3c4de185c88c7c268a.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d8383d70be6ce43b9d15ae4eecb7cdd04036972418273f3c4de185c88c7c268a.jpg)

![db20f5eb48bc00743d6101dbdc649842779792fa04aa8963fbffcc2794ea9008.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/db20f5eb48bc00743d6101dbdc649842779792fa04aa8963fbffcc2794ea9008.jpg)

![ddc46936e70ae93ae758683dceb0c2d860df5ab7f7ae660ca104cbfe303e421b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ddc46936e70ae93ae758683dceb0c2d860df5ab7f7ae660ca104cbfe303e421b.jpg)

![de2999eadd2acf9242cb567b930b92c89b72699cf3da149d47a2f29afb10c54b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/de2999eadd2acf9242cb567b930b92c89b72699cf3da149d47a2f29afb10c54b.jpg)

![de9b2e83c3171ed7c54064803d1e48e591e7c93b8fe400b0e1f4d7daad8fe683.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/de9b2e83c3171ed7c54064803d1e48e591e7c93b8fe400b0e1f4d7daad8fe683.jpg)

![deb5034ed5d1bc4f9bdd0da86c8881998785eb54073b8a19f2d2158fb338d4b1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/deb5034ed5d1bc4f9bdd0da86c8881998785eb54073b8a19f2d2158fb338d4b1.jpg)

![e123825a2d33d2a17057a004ff0fda487e8551ca65be0ed3a1863968bea6e07c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e123825a2d33d2a17057a004ff0fda487e8551ca65be0ed3a1863968bea6e07c.jpg)

![e7e29f9e02c29d8564332fc22b7f83300ab5f8e402789b99a47fa96bd6b1518f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e7e29f9e02c29d8564332fc22b7f83300ab5f8e402789b99a47fa96bd6b1518f.jpg)

![e980e1ed9328505dbda8d94b56d2c368411cdf0d32180755764d9d180b0eb0ea.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e980e1ed9328505dbda8d94b56d2c368411cdf0d32180755764d9d180b0eb0ea.jpg)

![e9d9a689661127fceebc03643e36bd218fe583db27c4cf0147812c92c43b32b5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e9d9a689661127fceebc03643e36bd218fe583db27c4cf0147812c92c43b32b5.jpg)

![ecdf9eb1a6336b7629aa409396be0e691740fc837510131c580a0d75bb794d40.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ecdf9eb1a6336b7629aa409396be0e691740fc837510131c580a0d75bb794d40.jpg)

![ee018925b52e76b0908b59794b7ffe940907d1a0cd38235b24c4adcdf936088c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ee018925b52e76b0908b59794b7ffe940907d1a0cd38235b24c4adcdf936088c.jpg)

![f3088319b6d44f2a8cafe076c403bf44e666e2c6fa590904b7db863baf7e2122.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f3088319b6d44f2a8cafe076c403bf44e666e2c6fa590904b7db863baf7e2122.jpg)

![f58044c3d7b1380a738a57a6e848c899410a21e1aa3e1c2c7ad2dab87b7ed03d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f58044c3d7b1380a738a57a6e848c899410a21e1aa3e1c2c7ad2dab87b7ed03d.jpg)

![f762a82227e5c0357cbcf0050b5426575f4df061d8a6d82039a3c17be87645ad.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f762a82227e5c0357cbcf0050b5426575f4df061d8a6d82039a3c17be87645ad.jpg)

![f7aa66bd69e01add2479abd6e93ab6c4f23db060a2fe36687b231bf43d4c7b92.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f7aa66bd69e01add2479abd6e93ab6c4f23db060a2fe36687b231bf43d4c7b92.jpg)

![f7afe24a4434846da6954cc59465260dc976a88cb4426e89fa7105d5cc0e5ed5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f7afe24a4434846da6954cc59465260dc976a88cb4426e89fa7105d5cc0e5ed5.jpg)

![f8d8bcbda005c1bb4ce71683cec949b1154db3ccca7d554ad666d41b8b320353.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f8d8bcbda005c1bb4ce71683cec949b1154db3ccca7d554ad666d41b8b320353.jpg)

![f90090287a739330f8b6a7e4e52788231c06e826c2eb4087a7814ca121f70df9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f90090287a739330f8b6a7e4e52788231c06e826c2eb4087a7814ca121f70df9.jpg)

![f9d459050b73872f2f1b37c4f016feff390c734dd10a09a6a798305ba4e6ff76.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f9d459050b73872f2f1b37c4f016feff390c734dd10a09a6a798305ba4e6ff76.jpg)

![fb1702c9bda1b6b2d85d3063e23a6be50e2caf6efc7745c3bfc2851b5d93fce7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/fb1702c9bda1b6b2d85d3063e23a6be50e2caf6efc7745c3bfc2851b5d93fce7.jpg)

![fc4d2009a31dd39008887a8f4dd7143f95a075b68b4fcbcb8be822689dd835b6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/fc4d2009a31dd39008887a8f4dd7143f95a075b68b4fcbcb8be822689dd835b6.jpg)

![fd897daa3e3cfa0b96154b165683064cf7f54bf9d4dad80c08c63f58c4c3bfab.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/fd897daa3e3cfa0b96154b165683064cf7f54bf9d4dad80c08c63f58c4c3bfab.jpg)

### Tables

![1763e899e1c5b64885da21194c950452738ede9d5e1a66387f3af9604b5040a6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/1763e899e1c5b64885da21194c950452738ede9d5e1a66387f3af9604b5040a6.jpg)

![223e61cd622921625424aaa8ba42c092ef22237547c973169f79855fe367af50.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/223e61cd622921625424aaa8ba42c092ef22237547c973169f79855fe367af50.jpg)

![34c3e9527fbd4cf19b34fff9551e08b569a0212a4b6c1b7abe6b8238a99584ef.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/34c3e9527fbd4cf19b34fff9551e08b569a0212a4b6c1b7abe6b8238a99584ef.jpg)

![68cc4ec3793e7a6cc653d14a3f4d5a969dbaaf9149e98fb5cd382325c225b23f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/68cc4ec3793e7a6cc653d14a3f4d5a969dbaaf9149e98fb5cd382325c225b23f.jpg)

![91c50ba0e5fef3e1000dbb5755b3794f6c8dda2e1f33929d306819026161baef.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/91c50ba0e5fef3e1000dbb5755b3794f6c8dda2e1f33929d306819026161baef.jpg)

![e2ddddf382b05742b629f02e6c61999ff441d49f128983779b3e07211e7df256.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/e2ddddf382b05742b629f02e6c61999ff441d49f128983779b3e07211e7df256.jpg)

## Progressive Compression with Universally Quantized Diffusion Models


### Images

![02193f1b031adf6155367104a67d54da589264d169cc2f9e91d1eb750630439f.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/02193f1b031adf6155367104a67d54da589264d169cc2f9e91d1eb750630439f.jpg)

![023c98bdc20e37f1b7ed28353676bb20e67081801f091f8bb608f513b610269b.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/023c98bdc20e37f1b7ed28353676bb20e67081801f091f8bb608f513b610269b.jpg)

![074353674b0cd6854162fe3d43a1a610b6f9515b46c51d123144e492fac552ea.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/074353674b0cd6854162fe3d43a1a610b6f9515b46c51d123144e492fac552ea.jpg)

![0d6318261afccae5ed7cccc55736dc7e591e5c1ef1fa04858a4a54860ba09b4b.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/0d6318261afccae5ed7cccc55736dc7e591e5c1ef1fa04858a4a54860ba09b4b.jpg)

![1508bb9aafce5f0b74820ed70395355e945f892c0be4e0d554be51f8058c8b12.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/1508bb9aafce5f0b74820ed70395355e945f892c0be4e0d554be51f8058c8b12.jpg)

![1bf230ef64e508c9eed7aaabb89761fc15b18d68cda0e8f07bc7a91877db1f85.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/1bf230ef64e508c9eed7aaabb89761fc15b18d68cda0e8f07bc7a91877db1f85.jpg)

![279bf76c481f5ad1181bdc02b78bba0f06135fc6ebe61a9f6ede67b168f63855.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/279bf76c481f5ad1181bdc02b78bba0f06135fc6ebe61a9f6ede67b168f63855.jpg)

![3cb508e1da0d8d4abbe48a3b1ed20ec5da6bd3f6de2b798d67819c37412ceec6.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/3cb508e1da0d8d4abbe48a3b1ed20ec5da6bd3f6de2b798d67819c37412ceec6.jpg)

![4b66aa06f081c8084f80aa229e9cf5d4b9868c18ec3b8aeae818573351901809.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/4b66aa06f081c8084f80aa229e9cf5d4b9868c18ec3b8aeae818573351901809.jpg)

![ebf217c37b7d2ab4922d22fbbe7177c6622f63399457d590743bd54ae9429871.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/ebf217c37b7d2ab4922d22fbbe7177c6622f63399457d590743bd54ae9429871.jpg)

## Training Language Models to Self-Correct via Reinforcement Learning


### Images

![0aeae14cbb110cdb9df17c4e28c66ddeebfd87aa2323e65c25cc26ae9a3168c9.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/0aeae14cbb110cdb9df17c4e28c66ddeebfd87aa2323e65c25cc26ae9a3168c9.jpg)

![4624ffc1fa4c18542031d73883a1b221eb6156f8ef1360bc89a8d10b4e32d55a.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/4624ffc1fa4c18542031d73883a1b221eb6156f8ef1360bc89a8d10b4e32d55a.jpg)

![489a513448410a1fed339e78b4cd5b6b977892e138dee984ce4b137353e9f4a1.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/489a513448410a1fed339e78b4cd5b6b977892e138dee984ce4b137353e9f4a1.jpg)

![4c57b00f6c35fc25038f08959904ea16d080327885feeb0e449e614f10ee142c.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/4c57b00f6c35fc25038f08959904ea16d080327885feeb0e449e614f10ee142c.jpg)

![64adacab4ad16cf173fb7cfb03b672d975ee88e0d71225a8814c8f2bb24541af.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/64adacab4ad16cf173fb7cfb03b672d975ee88e0d71225a8814c8f2bb24541af.jpg)

![6fcd00183463da0a358cda3fb25a0b71899ec5364a8c50db4e0a99c1591dcff8.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/6fcd00183463da0a358cda3fb25a0b71899ec5364a8c50db4e0a99c1591dcff8.jpg)

![7063fb0b5e617b4b1e4dd5fc0ee513e257cb9de7f59ca93a804eab1ef9aae7b6.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/7063fb0b5e617b4b1e4dd5fc0ee513e257cb9de7f59ca93a804eab1ef9aae7b6.jpg)

![800f3ae09c603d5b8a19a09d35388c64158d83ffbed37e098c1e8aee1696db98.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/800f3ae09c603d5b8a19a09d35388c64158d83ffbed37e098c1e8aee1696db98.jpg)

![82ef2b50b1facf6012f793a91b12c6eecca58cbeaffd647dd06347cbcf2c3d8c.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/82ef2b50b1facf6012f793a91b12c6eecca58cbeaffd647dd06347cbcf2c3d8c.jpg)

![bb4d561d4992913185a894052836df49be0325d7a7f1300c1c6f517744400153.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/bb4d561d4992913185a894052836df49be0325d7a7f1300c1c6f517744400153.jpg)

![f0bdc23e122e5aff0797564c0d893bd9d2c5054c6305b5d4351bd452a859fcf1.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/f0bdc23e122e5aff0797564c0d893bd9d2c5054c6305b5d4351bd452a859fcf1.jpg)

### Tables

![1284a74d21a8fb3c78ce0b54a07981119be5da1d3a565d17d2614a5c8d9e0687.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/1284a74d21a8fb3c78ce0b54a07981119be5da1d3a565d17d2614a5c8d9e0687.jpg)

![1fb6598a128cba680cfe0e3f8805f814a4923f50785af3e9ef271d69c4041a09.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/1fb6598a128cba680cfe0e3f8805f814a4923f50785af3e9ef271d69c4041a09.jpg)

![27a1b28b245cf5cd2b4b0751eb0f7aab70edbd828ed2198700ac11647977626e.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/27a1b28b245cf5cd2b4b0751eb0f7aab70edbd828ed2198700ac11647977626e.jpg)

![29fe97a1cf6ff53d748282090540676addc75365c0b1e48145a21ac7d9f568a3.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/29fe97a1cf6ff53d748282090540676addc75365c0b1e48145a21ac7d9f568a3.jpg)

![2db320b86a954ea433b128c81ec9328dba0f42707875cdbe550d42fdc5c59724.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/2db320b86a954ea433b128c81ec9328dba0f42707875cdbe550d42fdc5c59724.jpg)

![71e46951340cdf38f8207eb1fa65c9f9a4d20c8ead4ad8c151d58ac911af46dc.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/71e46951340cdf38f8207eb1fa65c9f9a4d20c8ead4ad8c151d58ac911af46dc.jpg)

![72c17a289b394b58e85bc8653e2fb7288a7a4aa2e3bb5225ff749069908d4975.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/72c17a289b394b58e85bc8653e2fb7288a7a4aa2e3bb5225ff749069908d4975.jpg)

![7e0fbffa441703fef269d053a7a51fd8c079f39f33bf99f17c9e5f1f8ecda087.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/7e0fbffa441703fef269d053a7a51fd8c079f39f33bf99f17c9e5f1f8ecda087.jpg)

![fb5656de444d1e0b33d62216873d5a49164d778a0572856d4d09dc7a4762690c.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/fb5656de444d1e0b33d62216873d5a49164d778a0572856d4d09dc7a4762690c.jpg)

## What should a neuron aim for? Designing local objective functions based on information theory


### Images

![37936f2b456c36c33256b135bbf6ab8193a981384903fa723c63c1a61e056677.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/37936f2b456c36c33256b135bbf6ab8193a981384903fa723c63c1a61e056677.jpg)

![670eb9820cdae0881dfba23d3ff84648cc256a62e19c0788e612dc2e2f532864.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/670eb9820cdae0881dfba23d3ff84648cc256a62e19c0788e612dc2e2f532864.jpg)

![71c0cd8cbc7494416425739984af1b81aa7906af475205b99069583e0d088154.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/71c0cd8cbc7494416425739984af1b81aa7906af475205b99069583e0d088154.jpg)

![8022d88722306b62da4f8c902c85d3a548ba4ed92ee715ffc03abf877c30ec56.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/8022d88722306b62da4f8c902c85d3a548ba4ed92ee715ffc03abf877c30ec56.jpg)

![a2274fddc204d7aaad8af67e9298d58014af70a5e8bf17a525ab1bc6138589ac.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/a2274fddc204d7aaad8af67e9298d58014af70a5e8bf17a525ab1bc6138589ac.jpg)

![b33f39ba253eb1c20c5956d9d8b55eea6af9d6fbcde1cad4c4c08467aada6c4d.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/b33f39ba253eb1c20c5956d9d8b55eea6af9d6fbcde1cad4c4c08467aada6c4d.jpg)

![c4a3fb3d583e9e03a449daedbb01aee5bc237c8caaa9d6fb57c33a13c443000d.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/c4a3fb3d583e9e03a449daedbb01aee5bc237c8caaa9d6fb57c33a13c443000d.jpg)

![c58f81d601c76252b17776fb2427ffeb038042cf223548402f02db15c892a674.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/c58f81d601c76252b17776fb2427ffeb038042cf223548402f02db15c892a674.jpg)

![ce4079bfc07d3dcd44255a6d69999d94d270c7a6b8a9dc6e113e2b2cd334a9a4.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/ce4079bfc07d3dcd44255a6d69999d94d270c7a6b8a9dc6e113e2b2cd334a9a4.jpg)

![dd6ec113dd3d125935398a7611cc044c935f560c9fc04a1b14bef918ee05e033.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/dd6ec113dd3d125935398a7611cc044c935f560c9fc04a1b14bef918ee05e033.jpg)

![fd93276eb6c31483b5a9a53bbd0022f23296dcb1787004e9d04a78b669ab18db.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/fd93276eb6c31483b5a9a53bbd0022f23296dcb1787004e9d04a78b669ab18db.jpg)

### Tables

![4da5f56c98b446181290c1c9874dfe942c03bec561eee0bed44a6fc025de4003.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/tables/4da5f56c98b446181290c1c9874dfe942c03bec561eee0bed44a6fc025de4003.jpg)

![5ce7207d73f87596fe3b5e0150a53e63e46bd5d02f01960beb540081ece4a7fd.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/tables/5ce7207d73f87596fe3b5e0150a53e63e46bd5d02f01960beb540081ece4a7fd.jpg)

![d22f9573efc96a455dd828b20ad160c9b2b18b55b0161ef24e04d99ab67a8786.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/tables/d22f9573efc96a455dd828b20ad160c9b2b18b55b0161ef24e04d99ab67a8786.jpg)

## Backtracking Improves Generation Safety


### Images

![2445f557107f12823e641b891588d02a0ef61474f938c7eb97d1f83c4fc9a25a.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/2445f557107f12823e641b891588d02a0ef61474f938c7eb97d1f83c4fc9a25a.jpg)

![45d1c6cb4a110c6db536ef58f257b522f29197b6ff3bf5fd11ebcef14028bc6a.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/45d1c6cb4a110c6db536ef58f257b522f29197b6ff3bf5fd11ebcef14028bc6a.jpg)

![913878472d751a23ed59ca831ca4224d743de2ce7c4e1251647873612c7465d6.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/913878472d751a23ed59ca831ca4224d743de2ce7c4e1251647873612c7465d6.jpg)

![d19ce3b915946fdd98cf886709829b64c948b1f5a2a452a58ee7b54d9cf5ad8e.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/d19ce3b915946fdd98cf886709829b64c948b1f5a2a452a58ee7b54d9cf5ad8e.jpg)

![fa165a4c03a41a0432f5a3e82f26f265104b8a003b3413a739eed4a0ed971f17.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/fa165a4c03a41a0432f5a3e82f26f265104b8a003b3413a739eed4a0ed971f17.jpg)

### Tables

![2590216ed4e8e0c039c1e3784b53f0b593b855a629368e30101762c9b9d2b372.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/2590216ed4e8e0c039c1e3784b53f0b593b855a629368e30101762c9b9d2b372.jpg)

![3e967c4725075df13ba442be6d8b8301f7dee0ad4425361dfd2ce9f30eef56f9.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/3e967c4725075df13ba442be6d8b8301f7dee0ad4425361dfd2ce9f30eef56f9.jpg)

![4d801a4a9fd11d4eee722125d827caf2a6de2739eb8b4857cdb2f3b71a2c7665.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/4d801a4a9fd11d4eee722125d827caf2a6de2739eb8b4857cdb2f3b71a2c7665.jpg)

![5dc0f1b4448be98f343e337e315fee8939402e0891f5bdae9ad431d0310fda41.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/5dc0f1b4448be98f343e337e315fee8939402e0891f5bdae9ad431d0310fda41.jpg)

![7c54bbb3fa5e7d2f89e7f705a319b213fafe4b0f412e21c9cbab3ac9316532b1.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/7c54bbb3fa5e7d2f89e7f705a319b213fafe4b0f412e21c9cbab3ac9316532b1.jpg)

![8b2c0b2d7e0ee9ce11a0a72cff028516de45633027b2f267e3477d573ca875a8.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/8b2c0b2d7e0ee9ce11a0a72cff028516de45633027b2f267e3477d573ca875a8.jpg)

![b3ec8b66e52418dce7e4bf1924fcdcb77d3d4a974066cfb0f1c49695f6c85050.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/b3ec8b66e52418dce7e4bf1924fcdcb77d3d4a974066cfb0f1c49695f6c85050.jpg)

![f960b90324618787bbc23235e722446a482cac586ddcd6b8c831d113dca99b2b.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/f960b90324618787bbc23235e722446a482cac586ddcd6b8c831d113dca99b2b.jpg)

## Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment


### Images

![5caebd593c76d75e380ed8fb2d31c3f8c1ce1a51a91db8362fe8b4ec7b3a67e4.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/5caebd593c76d75e380ed8fb2d31c3f8c1ce1a51a91db8362fe8b4ec7b3a67e4.jpg)

![672541a7f6652e32d1a05f4ccfb2c900ee95ef9259cf06a3cec255343dc46768.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/672541a7f6652e32d1a05f4ccfb2c900ee95ef9259cf06a3cec255343dc46768.jpg)

![749583c63d948046e4fa788bd22963fbd80ba97bf111069e45ec03f427ad524f.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/749583c63d948046e4fa788bd22963fbd80ba97bf111069e45ec03f427ad524f.jpg)

![bafe650fff391cc97dfc0c3d7cf602aeb7c6d6beffadebaee481c892dc47fe41.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/bafe650fff391cc97dfc0c3d7cf602aeb7c6d6beffadebaee481c892dc47fe41.jpg)

![e6016821480e999ef2060946795f70b61af52d438129997ec8f33e7355d64884.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/e6016821480e999ef2060946795f70b61af52d438129997ec8f33e7355d64884.jpg)

### Tables

![16ac9d79c3714cfb87ab0cb82e06672ba057b35b93dfb0a092a7c47a8ae44b0b.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/16ac9d79c3714cfb87ab0cb82e06672ba057b35b93dfb0a092a7c47a8ae44b0b.jpg)

![1e17367fc1255cd75abfa94c41aa678aacae19948e07be0f212a216e88e2ce1f.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/1e17367fc1255cd75abfa94c41aa678aacae19948e07be0f212a216e88e2ce1f.jpg)

![2ff2161276b2affd587e322095b74910bb3787d92673fd4f2ae66ff3ea3c317f.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/2ff2161276b2affd587e322095b74910bb3787d92673fd4f2ae66ff3ea3c317f.jpg)

![45524f7a5598cc51c2920560f43108640d2dc97759081a37b793c18ee218e86b.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/45524f7a5598cc51c2920560f43108640d2dc97759081a37b793c18ee218e86b.jpg)

![98015511aab3af65502bb04e0c209e86b60641741b61c5e78f4825481c399139.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/98015511aab3af65502bb04e0c209e86b60641741b61c5e78f4825481c399139.jpg)

![9b0e9eac22512cfb41365edc21a00a2e26e0cb3b304ff01581c68285913eb924.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/9b0e9eac22512cfb41365edc21a00a2e26e0cb3b304ff01581c68285913eb924.jpg)

![a828cafc6d4850d2d7909e6efdb810db72bd249fd664d161f9abe129d6533c18.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/a828cafc6d4850d2d7909e6efdb810db72bd249fd664d161f9abe129d6533c18.jpg)

![aa1559571563c2f5a33433459d2c80f955c3cd99d20524436d3e7d49d307be20.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/aa1559571563c2f5a33433459d2c80f955c3cd99d20524436d3e7d49d307be20.jpg)

![c8f0419b3c232d8189a651c41df66a0fa24b83684dd52ee246870fb39e94f810.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/c8f0419b3c232d8189a651c41df66a0fa24b83684dd52ee246870fb39e94f810.jpg)

![e8277a98207fd30bb5fd7531896071e4d37494a9026392f6328ff34999690403.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/e8277a98207fd30bb5fd7531896071e4d37494a9026392f6328ff34999690403.jpg)

![f772416d808f6d65038352f1e6fb7d2cf71ab3582d32ba63aa69dcbf2a316fb0.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/f772416d808f6d65038352f1e6fb7d2cf71ab3582d32ba63aa69dcbf2a316fb0.jpg)

![fe346b0cce1d86c4a326f5bce31cfe82dad003119bc59d5423e8582ff01662c4.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/fe346b0cce1d86c4a326f5bce31cfe82dad003119bc59d5423e8582ff01662c4.jpg)

## Global Convergence in Neural ODEs: Impact of Activation Functions


### Images

![1183715f61d1ef0c0a8ed7d0246a26ba8cdfdd2bfe7fd91a24f0123ee63f9e9e.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/1183715f61d1ef0c0a8ed7d0246a26ba8cdfdd2bfe7fd91a24f0123ee63f9e9e.jpg)

![145880ead73f6fff0f635963e32322d65865483078b5c340591f2ae0d7baabc4.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/145880ead73f6fff0f635963e32322d65865483078b5c340591f2ae0d7baabc4.jpg)

![1801d51ef4be3e511fb6d3b75fdadf18f5bc3193a0e347aad7c3ccb0dbf6921f.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/1801d51ef4be3e511fb6d3b75fdadf18f5bc3193a0e347aad7c3ccb0dbf6921f.jpg)

![2bf5e5ed7f1e5c5d02110aa2a5c6eb5fae6309820889515a7e34459823b6d14b.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/2bf5e5ed7f1e5c5d02110aa2a5c6eb5fae6309820889515a7e34459823b6d14b.jpg)

![33aec0a08b858478d39a7fe49a2d4e0a003bd8248aec0c1c8ae5a95142b3b930.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/33aec0a08b858478d39a7fe49a2d4e0a003bd8248aec0c1c8ae5a95142b3b930.jpg)

![437a0c999aa5601205626618f0610dfa7ba164c96bc994b97d0d56e321d3be55.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/437a0c999aa5601205626618f0610dfa7ba164c96bc994b97d0d56e321d3be55.jpg)

![55d3ebcf1f979b5bc8da5b8f0f61832d73270979d6cd0b646ca4b03313554e42.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/55d3ebcf1f979b5bc8da5b8f0f61832d73270979d6cd0b646ca4b03313554e42.jpg)

![64b08d72f8ec3ec13a63af1e715d2b173aa0a9ea0bc414ca8af7cf586882236f.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/64b08d72f8ec3ec13a63af1e715d2b173aa0a9ea0bc414ca8af7cf586882236f.jpg)

![ab61856a0d0793472ab157964724172b76b74c0bcb2170038ee0ae634ef89b15.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/ab61856a0d0793472ab157964724172b76b74c0bcb2170038ee0ae634ef89b15.jpg)

![b33dd2561267ec251ca44008faf582ad8874e3707b590f71263a7d767bf82a23.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/b33dd2561267ec251ca44008faf582ad8874e3707b590f71263a7d767bf82a23.jpg)

![bb61ddb1cb3564215431d3e0bc0ca674b12c6715ea4e9d48897541826bc174cc.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/bb61ddb1cb3564215431d3e0bc0ca674b12c6715ea4e9d48897541826bc174cc.jpg)

![c19f5247aaa17c0f7b667917cb40fbd0b40fbac5ad85c2e082977a2fb8fa4817.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/c19f5247aaa17c0f7b667917cb40fbd0b40fbac5ad85c2e082977a2fb8fa4817.jpg)

![eaf8e03024bc72d0d25e01eb1a953af0e010161f821052d24f2abfe85d464938.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/eaf8e03024bc72d0d25e01eb1a953af0e010161f821052d24f2abfe85d464938.jpg)

![f0e573dce18b94d5cec97da5db50d3a4b5c7764b8095a39e573ba5d3c3e68550.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/f0e573dce18b94d5cec97da5db50d3a4b5c7764b8095a39e573ba5d3c3e68550.jpg)

## Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces


### Images

![03210abf672c06ee754e1d8031b669c0a2eb450c8f606804a970623006721d0e.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/03210abf672c06ee754e1d8031b669c0a2eb450c8f606804a970623006721d0e.jpg)

![35d74592c55d9c7fed460cd452a355e7073f3fe28ebbfaefff2178afe0ba7874.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/35d74592c55d9c7fed460cd452a355e7073f3fe28ebbfaefff2178afe0ba7874.jpg)

![3f2dede7a2bf0763a4161a227067fc5c61d89335452ba2ba98f9c046e49df53a.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/3f2dede7a2bf0763a4161a227067fc5c61d89335452ba2ba98f9c046e49df53a.jpg)

![456cbb7c68e2135c596dac55c96d4f56aea3621f24b39bce95560b1941bba0d8.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/456cbb7c68e2135c596dac55c96d4f56aea3621f24b39bce95560b1941bba0d8.jpg)

![4c5f7363afabfa144f107901ff541714aabd53b7dd201d7b3721c945617d4cbd.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/4c5f7363afabfa144f107901ff541714aabd53b7dd201d7b3721c945617d4cbd.jpg)

![598d093867c29fba886c32131aa3a380b063cf85e30dd21700b178cf3f5f965d.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/598d093867c29fba886c32131aa3a380b063cf85e30dd21700b178cf3f5f965d.jpg)

![988dfd4d23bd4486a59d293350b29236b89c80bd7bf10b3960aaea6965adb9cf.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/988dfd4d23bd4486a59d293350b29236b89c80bd7bf10b3960aaea6965adb9cf.jpg)

![98e8965a308e9c886b3d9bf524444f78ab519c7c0da60a60b0951ac4de22fff9.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/98e8965a308e9c886b3d9bf524444f78ab519c7c0da60a60b0951ac4de22fff9.jpg)

![ce826d89c0987a02d6515bd2f29ada224974903d242704eb31efca71698b8825.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/ce826d89c0987a02d6515bd2f29ada224974903d242704eb31efca71698b8825.jpg)

## The Hidden Cost of Waiting for Accurate Predictions


### Images

![780743247ec9dffc45b27f485624d3e8e95b0e4a37e5193c39a1c00cdde3deef.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/images/780743247ec9dffc45b27f485624d3e8e95b0e4a37e5193c39a1c00cdde3deef.jpg)

![844e5bae84376681ca17d991659dccf7bdfcfc21a4c462cee5f6e7f1f039ed98.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/images/844e5bae84376681ca17d991659dccf7bdfcfc21a4c462cee5f6e7f1f039ed98.jpg)

![d5dd6018700a14406c3b38a59cf9a95ea9738f6678a9b317a2f65da48e7ddf49.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/images/d5dd6018700a14406c3b38a59cf9a95ea9738f6678a9b317a2f65da48e7ddf49.jpg)

### Tables

![943219d113fc99ea4c11d8cd17976a6b08424559bbd78f2626b5da52f0dea553.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/tables/943219d113fc99ea4c11d8cd17976a6b08424559bbd78f2626b5da52f0dea553.jpg)

## DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL


### Images

![016ff054581855709233076159fa584182428b004b4838cb8dac014642e2a511.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/016ff054581855709233076159fa584182428b004b4838cb8dac014642e2a511.jpg)

![2337acf869f2a789cbf645898ac4c4f352823289d36e5ad1cc440d2d289f5204.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/2337acf869f2a789cbf645898ac4c4f352823289d36e5ad1cc440d2d289f5204.jpg)

![2cd0f138393fab0c9e369fee6e6ebfe95895b714f829a0fb99e72e721b78669a.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/2cd0f138393fab0c9e369fee6e6ebfe95895b714f829a0fb99e72e721b78669a.jpg)

![48e9d920cfdff22e7cbb1e18f0504aa40c178101fe4d2236de79fac49d164c74.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/48e9d920cfdff22e7cbb1e18f0504aa40c178101fe4d2236de79fac49d164c74.jpg)

![5c3b24a486555f5b6ee48ef2bb7114231f9e276e9f9bdb60c5eceeade62e77b8.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/5c3b24a486555f5b6ee48ef2bb7114231f9e276e9f9bdb60c5eceeade62e77b8.jpg)

![6f1b6dd5da06d467538be618c6e98f40df467c8cc37afac8e278aacb785fa54b.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/6f1b6dd5da06d467538be618c6e98f40df467c8cc37afac8e278aacb785fa54b.jpg)

![a6ab5bf98f82c722cb24ee2620c9e6d04fd578d65c203d2933b0fd854af69550.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/a6ab5bf98f82c722cb24ee2620c9e6d04fd578d65c203d2933b0fd854af69550.jpg)

![b6d50978ec1e013ef78f22250a01033b50b70bc79de529535472397153313233.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/b6d50978ec1e013ef78f22250a01033b50b70bc79de529535472397153313233.jpg)

![c1a62d4a86e9b76531237df61660ccbe5fc6a5b503e019b2139982e3fec2d0b1.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/c1a62d4a86e9b76531237df61660ccbe5fc6a5b503e019b2139982e3fec2d0b1.jpg)

![c5b2fd1215e733a40c2dcc4f5814a558349f6703d189e6d80a4a679111b66e6d.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/c5b2fd1215e733a40c2dcc4f5814a558349f6703d189e6d80a4a679111b66e6d.jpg)

![cb2562d14ce12540976cc7f66c6a2ebdbcbc0459d6b32a85c3d9fabad70290eb.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/cb2562d14ce12540976cc7f66c6a2ebdbcbc0459d6b32a85c3d9fabad70290eb.jpg)

![e3cc494d65055ac29a939fc2363e9b4d094cc58a2a9cc84649f8a572bb1366c1.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/e3cc494d65055ac29a939fc2363e9b4d094cc58a2a9cc84649f8a572bb1366c1.jpg)

### Tables

![53e067feafbc6091f79d740caa371ebc76fb4cde29eca773e2918a7454a9ceb1.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/53e067feafbc6091f79d740caa371ebc76fb4cde29eca773e2918a7454a9ceb1.jpg)

![5d367277f549897554eebe85d6e47348d24ff8823865b266a0624dc941610f09.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/5d367277f549897554eebe85d6e47348d24ff8823865b266a0624dc941610f09.jpg)

![609e5515a8bf41cea49678f9e61463c9ff8f8f1648629b82f01ed2ff2b1da3f4.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/609e5515a8bf41cea49678f9e61463c9ff8f8f1648629b82f01ed2ff2b1da3f4.jpg)

![73641d4f0240c791b2712895d7569f445d013d6c7f54a3cd38395dce04f73558.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/73641d4f0240c791b2712895d7569f445d013d6c7f54a3cd38395dce04f73558.jpg)

![8a0d5fb2247dc8ed97c63a02177e1709087627c45c711ba88a6c29608098f5eb.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/8a0d5fb2247dc8ed97c63a02177e1709087627c45c711ba88a6c29608098f5eb.jpg)

![9afef1d23f06eeb9b61635bae0cb3228cf59500ccbdba1090d891cc1c493665f.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/9afef1d23f06eeb9b61635bae0cb3228cf59500ccbdba1090d891cc1c493665f.jpg)

![e90b3a054a948608d7a2c4640fb4500aa44c071603d11448a0a9a89d03a5baa4.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/e90b3a054a948608d7a2c4640fb4500aa44c071603d11448a0a9a89d03a5baa4.jpg)

![ed5ad9cbdcd47bd069c2e37546618795390ff3b1ec1079b73f7adedcefbcbc65.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/ed5ad9cbdcd47bd069c2e37546618795390ff3b1ec1079b73f7adedcefbcbc65.jpg)

## The Complexity of Two-Team Polymatrix Games with Independent Adversaries


### Images

![4e5a38f27eb21622427a720a76e7a0d41c33931fd136356276a8dcc1980b07ef.jpg](../iclr_results/187_The Complexity of Two-Team Polymatrix Games with Independent Adversaries/images/4e5a38f27eb21622427a720a76e7a0d41c33931fd136356276a8dcc1980b07ef.jpg)

![824a8e74da5858b8c0843917e6637a658af0fefe5ed636c799042c4ada79232f.jpg](../iclr_results/187_The Complexity of Two-Team Polymatrix Games with Independent Adversaries/images/824a8e74da5858b8c0843917e6637a658af0fefe5ed636c799042c4ada79232f.jpg)

## Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series


### Images

![305499386356c09e7aac1d8590d7794c5ca17a2fdd60576bc2c9f4c623068460.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/images/305499386356c09e7aac1d8590d7794c5ca17a2fdd60576bc2c9f4c623068460.jpg)

![3efaf89ae28ae52154c1e5148ab1a0809e39b28218da0005035caaec6de0f626.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/images/3efaf89ae28ae52154c1e5148ab1a0809e39b28218da0005035caaec6de0f626.jpg)

![a1b7ed94d61d8252cc21d05b51ff453c59ced5a23dae71e061d56c7ac0cd3bfb.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/images/a1b7ed94d61d8252cc21d05b51ff453c59ced5a23dae71e061d56c7ac0cd3bfb.jpg)

![ad67aa47fdd49e2e5b5e305aaa4f0e562f94c134465d24ffaad15f3a15bd4337.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/images/ad67aa47fdd49e2e5b5e305aaa4f0e562f94c134465d24ffaad15f3a15bd4337.jpg)

### Tables

![585c038be1c00983ef43d661013c5073be9ec02e5372c6027ad7fdb7acd01a57.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/tables/585c038be1c00983ef43d661013c5073be9ec02e5372c6027ad7fdb7acd01a57.jpg)

![d7223782aaeee4d4f6c8c5d1845d31eda1f31c0d85deb83faa063ae61e9f1acb.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/tables/d7223782aaeee4d4f6c8c5d1845d31eda1f31c0d85deb83faa063ae61e9f1acb.jpg)

![db5715ce633fb82e65a748cc39b8c721ded05ee26bc031b35415813d81d4fb6f.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/tables/db5715ce633fb82e65a748cc39b8c721ded05ee26bc031b35415813d81d4fb6f.jpg)

![dfdc92bdfd3799e2317ca6428f42324cd7584acba0cc8bc344e405d60f3eb4e4.jpg](../iclr_results/188_Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series/tables/dfdc92bdfd3799e2317ca6428f42324cd7584acba0cc8bc344e405d60f3eb4e4.jpg)

## MoDeGPT: Modular Decomposition for Large Language Model Compression


### Images

![076f948d38db0d957215f779b8b1de65709eee1daf2e7363bf16c8ae31c4fcdd.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/076f948d38db0d957215f779b8b1de65709eee1daf2e7363bf16c8ae31c4fcdd.jpg)

![339640512176ba5f7f3ad84c0ae5a27300224c99b385eb08e054918db7ed11eb.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/339640512176ba5f7f3ad84c0ae5a27300224c99b385eb08e054918db7ed11eb.jpg)

![3af5a02c7330240f92000e5f8e98e6355daca7ab7a0332ecef83b316ef4a79d6.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/3af5a02c7330240f92000e5f8e98e6355daca7ab7a0332ecef83b316ef4a79d6.jpg)

![480e205eede646d1adfe3a93b5ead231fb3904362e2cc9587e76793b839dfa93.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/480e205eede646d1adfe3a93b5ead231fb3904362e2cc9587e76793b839dfa93.jpg)

![7f1089bc60c5ef6eaa93c8cec4b0252403584c2e0254dd2300048178ad6d5680.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/7f1089bc60c5ef6eaa93c8cec4b0252403584c2e0254dd2300048178ad6d5680.jpg)

![8b8237af3e44ba4d2293afacccee7922c91d9b5c64d23f0c8b4663f0183bb850.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/8b8237af3e44ba4d2293afacccee7922c91d9b5c64d23f0c8b4663f0183bb850.jpg)

![973a3187a4c0360f87e8ca4241de0be67fb99412cab5949385d91d6abc6b1a8c.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/973a3187a4c0360f87e8ca4241de0be67fb99412cab5949385d91d6abc6b1a8c.jpg)

![9c8e93bbb2eed96ab1d79c34829e84b34cb3ba9257aee15c92a93cc1733b102f.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/9c8e93bbb2eed96ab1d79c34829e84b34cb3ba9257aee15c92a93cc1733b102f.jpg)

![ac1e95f6dc6c7bce353b0a1aa2d067be7a146a593c54f680b50eed2e32fa5761.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/ac1e95f6dc6c7bce353b0a1aa2d067be7a146a593c54f680b50eed2e32fa5761.jpg)

![b26c6c609decc2547cd15dafdbf52d4dc3cb6b04c3522bb5c4cf0566feca505e.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/b26c6c609decc2547cd15dafdbf52d4dc3cb6b04c3522bb5c4cf0566feca505e.jpg)

![b872fba21c591bbf16d46524372e9f80434366a03c7e387f1813200c68250e92.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/b872fba21c591bbf16d46524372e9f80434366a03c7e387f1813200c68250e92.jpg)

![f745e626eaa254163be4788a6b4a30f20d75f2ae7e9065d81121658d5cd279ad.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/images/f745e626eaa254163be4788a6b4a30f20d75f2ae7e9065d81121658d5cd279ad.jpg)

### Tables

![0365d9d5a553040d6fce98c7dc033d731cffb79e2d549102b7df91b1a0769228.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/0365d9d5a553040d6fce98c7dc033d731cffb79e2d549102b7df91b1a0769228.jpg)

![0453356d899e1f828273719b08f71a01bf25d76957add3912c3bae1cf0d74679.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/0453356d899e1f828273719b08f71a01bf25d76957add3912c3bae1cf0d74679.jpg)

![1233870dbd21499f38aee54c686d05ff8a576d9500bd724b19f0a97bff0d7d84.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/1233870dbd21499f38aee54c686d05ff8a576d9500bd724b19f0a97bff0d7d84.jpg)

![15ba1b05461f7738a1973cfd2005e643eff040e9bf860e8c1d4b1752c4017036.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/15ba1b05461f7738a1973cfd2005e643eff040e9bf860e8c1d4b1752c4017036.jpg)

![241d2efb2fc361b470f67da6e94eabd5fe10bc7b40910671f99901a6ad81d9fc.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/241d2efb2fc361b470f67da6e94eabd5fe10bc7b40910671f99901a6ad81d9fc.jpg)

![24f459a20ae98329a4786cfcfbbde6715bac7ff405c25a692ea15c7a7799af45.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/24f459a20ae98329a4786cfcfbbde6715bac7ff405c25a692ea15c7a7799af45.jpg)

![25f41f2558ed9909ec8348891c8fb0de461d4c6da384f07022c1ae47f80d6089.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/25f41f2558ed9909ec8348891c8fb0de461d4c6da384f07022c1ae47f80d6089.jpg)

![26ddbb3ac252b4793200e9e73241960d4501646b72e60ec34d2ad769cb5d41be.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/26ddbb3ac252b4793200e9e73241960d4501646b72e60ec34d2ad769cb5d41be.jpg)

![28323f1316002c51906ef593066fc6df7847665d51478979f1b065dfdf621882.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/28323f1316002c51906ef593066fc6df7847665d51478979f1b065dfdf621882.jpg)

![2873ccf6bbebee5a3605a23c38dc64d2808e5d34f8f3041874c321536edce71c.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/2873ccf6bbebee5a3605a23c38dc64d2808e5d34f8f3041874c321536edce71c.jpg)

![341b4f0860b8fd8ae2c5ae1e08d82f0731edeb452d7775e4b0a7e21aed30641b.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/341b4f0860b8fd8ae2c5ae1e08d82f0731edeb452d7775e4b0a7e21aed30641b.jpg)

![3a1d5276936524e1ab85411dba18061286261104e8be5b70fa124079f13985f3.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/3a1d5276936524e1ab85411dba18061286261104e8be5b70fa124079f13985f3.jpg)

![447a0ba02a178384d3c5524b908969f52e0e75eb3b51d8b14c8d7fdd54e668b1.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/447a0ba02a178384d3c5524b908969f52e0e75eb3b51d8b14c8d7fdd54e668b1.jpg)

![44aadb6556563e6cd50fec24d00aa7c1e00f3b571b197c5b052eb58d4e385b5f.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/44aadb6556563e6cd50fec24d00aa7c1e00f3b571b197c5b052eb58d4e385b5f.jpg)

![49c8a2194e5c87c901c385e0b20e1407755e3ce98e278777f22c6e4aa57a1dcf.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/49c8a2194e5c87c901c385e0b20e1407755e3ce98e278777f22c6e4aa57a1dcf.jpg)

![4c3f202a53cbc9f1f2eb5c738f6527a877085115f5accf2144b0ce263ae680ef.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/4c3f202a53cbc9f1f2eb5c738f6527a877085115f5accf2144b0ce263ae680ef.jpg)

![5bfc83b7197e17378ab8e8b5a5a974d1f0a4244a2c11bef4d65bed2dde95393b.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/5bfc83b7197e17378ab8e8b5a5a974d1f0a4244a2c11bef4d65bed2dde95393b.jpg)

![60362d3ce5b7f0a1f7f533dd0fb19e49809970e29f1ca148a029d8a1c4023f22.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/60362d3ce5b7f0a1f7f533dd0fb19e49809970e29f1ca148a029d8a1c4023f22.jpg)

![68998b02dfa4f72ccdf73643e76944b13803e72043b306055b1958bb730a2c8e.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/68998b02dfa4f72ccdf73643e76944b13803e72043b306055b1958bb730a2c8e.jpg)

![72814d44c6de56ea3ec69d0492efba04fda90bc9ac57ba4c1fb85172064104fb.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/72814d44c6de56ea3ec69d0492efba04fda90bc9ac57ba4c1fb85172064104fb.jpg)

![72a41da4fb6125db14cf0048461fb5f2020ba5d213d66c2c61dcf08414fbfc7f.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/72a41da4fb6125db14cf0048461fb5f2020ba5d213d66c2c61dcf08414fbfc7f.jpg)

![87734e495d8f2229707c9de56a67fb1b4daceaf703975daf5731e3b7735d1822.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/87734e495d8f2229707c9de56a67fb1b4daceaf703975daf5731e3b7735d1822.jpg)

![87efe5778f0f27e21f7ce6c5b9f293275d7cd4ab4e8f5c176f8094d5b1cf23b6.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/87efe5778f0f27e21f7ce6c5b9f293275d7cd4ab4e8f5c176f8094d5b1cf23b6.jpg)

![88d17a871cb46a27374fd9b54346372abe5b3a0c6a575588d36612f9ba3210ef.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/88d17a871cb46a27374fd9b54346372abe5b3a0c6a575588d36612f9ba3210ef.jpg)

![8f0e2326dc25d7a52eef29a30c7ef1b53c2b98711bfd8d8edec5407307fe7026.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/8f0e2326dc25d7a52eef29a30c7ef1b53c2b98711bfd8d8edec5407307fe7026.jpg)

![932d92a6725f5f5ef1f41fb28c2edfa67a0baf6a6a1df27c5156c578b0137f40.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/932d92a6725f5f5ef1f41fb28c2edfa67a0baf6a6a1df27c5156c578b0137f40.jpg)

![9c1d96a1dfde19ac0ac9ef1bed5d2060f4b2d48ddc42ac970843738ce95f3f8b.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/9c1d96a1dfde19ac0ac9ef1bed5d2060f4b2d48ddc42ac970843738ce95f3f8b.jpg)

![a87ad72f29bb372d22d749893f260ef301e34857d5ca4635f0e992bb779e6758.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/a87ad72f29bb372d22d749893f260ef301e34857d5ca4635f0e992bb779e6758.jpg)

![a95f302608e61a1df50e74a525645fcf9c8471c460f1a8a4eaad009bd9d707ca.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/a95f302608e61a1df50e74a525645fcf9c8471c460f1a8a4eaad009bd9d707ca.jpg)

![aa7a650afd55095c857eedecf9e2469ef276ac4bbca6b4159a7b51405666febe.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/aa7a650afd55095c857eedecf9e2469ef276ac4bbca6b4159a7b51405666febe.jpg)

![c23f4c135510b79b8fd82e0ae2cafc03fc8896b1c1d2dc723be2c25605d89fdb.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/c23f4c135510b79b8fd82e0ae2cafc03fc8896b1c1d2dc723be2c25605d89fdb.jpg)

![c3c0eec523921a6d639fd961a26b35b4f0219a5dceefc03321566c15a410818f.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/c3c0eec523921a6d639fd961a26b35b4f0219a5dceefc03321566c15a410818f.jpg)

![c7cb359d84a307cdfdcde27f0c1c72d82c8826743beb8c1b4d0fe0f072e9a8ea.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/c7cb359d84a307cdfdcde27f0c1c72d82c8826743beb8c1b4d0fe0f072e9a8ea.jpg)

![e0047ec6c29802bfaac86b0cf62089441f34cb1aebbadf0da27bccb80520cfae.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/e0047ec6c29802bfaac86b0cf62089441f34cb1aebbadf0da27bccb80520cfae.jpg)

![e24bd8985ba0741aea47320111a3f4c521d1cb6006abfb49ac051f4071540d86.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/e24bd8985ba0741aea47320111a3f4c521d1cb6006abfb49ac051f4071540d86.jpg)

![ee51b181db60dfb32c942966239c882cf1631c4d2e8d0f2503dd9d2778b4d043.jpg](../iclr_results/190_MoDeGPT_ Modular Decomposition for Large Language Model Compression/tables/ee51b181db60dfb32c942966239c882cf1631c4d2e8d0f2503dd9d2778b4d043.jpg)

## Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities


### Images

![035d8e3d9c8056541f874bd7716bf716ca7c9822a2ac44116a37c02c012a8f99.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/035d8e3d9c8056541f874bd7716bf716ca7c9822a2ac44116a37c02c012a8f99.jpg)

![21d59e1e9f5dc9ef06ecd1833fa98de06cae2281ac03a73941e4360bae0e9049.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/21d59e1e9f5dc9ef06ecd1833fa98de06cae2281ac03a73941e4360bae0e9049.jpg)

![39ecf7b5dae1b1a958a62452c1e7dac24893cefc11090396abf65fbeff3e9698.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/39ecf7b5dae1b1a958a62452c1e7dac24893cefc11090396abf65fbeff3e9698.jpg)

![3be9036e196433d9c41c19e34078c90472d97ecff38fefad7337d8fdc5ec7c4a.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/3be9036e196433d9c41c19e34078c90472d97ecff38fefad7337d8fdc5ec7c4a.jpg)

![3ca769c85c466ff436a65f642e5460fd7d02fb612df8917a74e4404efac894f7.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/3ca769c85c466ff436a65f642e5460fd7d02fb612df8917a74e4404efac894f7.jpg)

![503f5bba141b6823b6f93691d7c3e7004f106280f3bdbf117246ea48a00a9df8.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/503f5bba141b6823b6f93691d7c3e7004f106280f3bdbf117246ea48a00a9df8.jpg)

![52be6e36e47705bf7213658c2013afe3e62551b4f3e72ff725b4b0dcc2d63621.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/52be6e36e47705bf7213658c2013afe3e62551b4f3e72ff725b4b0dcc2d63621.jpg)

![53d2ba6fb6e9e294f4f0440a090c0c047e5bea16134918f0bb79f3d6042d0d8b.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/53d2ba6fb6e9e294f4f0440a090c0c047e5bea16134918f0bb79f3d6042d0d8b.jpg)

![8aaa0b1342a4950e9f037511e3a11a12648dd6c31ecb16caec01c0b9dea1e6fc.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/8aaa0b1342a4950e9f037511e3a11a12648dd6c31ecb16caec01c0b9dea1e6fc.jpg)

![8acd50b066818472d665307fc684e5183b8459355d482f89918105dae9454586.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/8acd50b066818472d665307fc684e5183b8459355d482f89918105dae9454586.jpg)

![8cfbaf9564ce629fde77ec5a6aed66a72ecb793e6ba45ab2cf966ca043282528.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/8cfbaf9564ce629fde77ec5a6aed66a72ecb793e6ba45ab2cf966ca043282528.jpg)

![9b26422d4a0a5152993c69522f87c66704b4d64d1810d920b67b865ac5aee1c5.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/9b26422d4a0a5152993c69522f87c66704b4d64d1810d920b67b865ac5aee1c5.jpg)

![a1678b006976efb03d802e5c97adadfb2f0ccc4570fee2d8d41f976c5287372b.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/a1678b006976efb03d802e5c97adadfb2f0ccc4570fee2d8d41f976c5287372b.jpg)

![aa5e3c05ca0a633ad6d8ca618e5cccff63ef7b3526307f6e2ae1c8e5bd2ed436.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/aa5e3c05ca0a633ad6d8ca618e5cccff63ef7b3526307f6e2ae1c8e5bd2ed436.jpg)

![c204de5f9b248be24695146691d98bf5097e5710a65a4e0dad3ca4c4621d9ef1.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/c204de5f9b248be24695146691d98bf5097e5710a65a4e0dad3ca4c4621d9ef1.jpg)

![db5e33abdf19f5597cc1d10d1fa0a4e7fd016f77fcd49c04d5e8c3e140d53988.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/db5e33abdf19f5597cc1d10d1fa0a4e7fd016f77fcd49c04d5e8c3e140d53988.jpg)

![e2ab5ff2bed46bc50c6b447c1a75c1fa5bd373370deec9125e319bf36bbdb319.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/e2ab5ff2bed46bc50c6b447c1a75c1fa5bd373370deec9125e319bf36bbdb319.jpg)

![ed28a4b117a70e7a010794396e81006360ff71795fdd7660d1a42eb0ab5ecde9.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/ed28a4b117a70e7a010794396e81006360ff71795fdd7660d1a42eb0ab5ecde9.jpg)

![ef0a8133acbb0b8eec48fd83f582a3033a683b9fd2ca97d2515d4b16a048b9a4.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/images/ef0a8133acbb0b8eec48fd83f582a3033a683b9fd2ca97d2515d4b16a048b9a4.jpg)

### Tables

![1eb82caf7fccce1c1c63a9928fa772c6350c2f674655b03f030b71c2348104ad.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/1eb82caf7fccce1c1c63a9928fa772c6350c2f674655b03f030b71c2348104ad.jpg)

![3be1d739bfe440c2ed43e9ff998dfd04bcf06a8d96e447446ab5bb0f6f713e41.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/3be1d739bfe440c2ed43e9ff998dfd04bcf06a8d96e447446ab5bb0f6f713e41.jpg)

![5411f0f5c3c92c2f5efcce32f1e670df1dede3c9b6fca2244a57422dd0193746.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/5411f0f5c3c92c2f5efcce32f1e670df1dede3c9b6fca2244a57422dd0193746.jpg)

![627e89f77362fbfec1cdc25d972fff0ab3122df300ef6fe962c246ea33718591.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/627e89f77362fbfec1cdc25d972fff0ab3122df300ef6fe962c246ea33718591.jpg)

![7637b30af9273c79fa6e5f8338e4a9b18489fd59e5490636f729db7abdf82566.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/7637b30af9273c79fa6e5f8338e4a9b18489fd59e5490636f729db7abdf82566.jpg)

![9aa27c27863440656c376daad74bb98016c8b771ce951dbbbd4eaa95cc778195.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/9aa27c27863440656c376daad74bb98016c8b771ce951dbbbd4eaa95cc778195.jpg)

![a32c5d159a33280b7d3d49fea1b02d1fb5893406f929a235cddda2f5140f9e6d.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/a32c5d159a33280b7d3d49fea1b02d1fb5893406f929a235cddda2f5140f9e6d.jpg)

![a5ee1cb9b64fca95f7441c9fa636aeeea1f7c676c6d603433c8714d30b02e1c4.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/a5ee1cb9b64fca95f7441c9fa636aeeea1f7c676c6d603433c8714d30b02e1c4.jpg)

![b275e4d36f0350b40a2bd5bebf1d75449e45af6970a54a7c3d4cf53302d3e9f6.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/b275e4d36f0350b40a2bd5bebf1d75449e45af6970a54a7c3d4cf53302d3e9f6.jpg)

![bc87a0c5ec4e19b95c157e1f63587fe32bf3908b69fc743a9c4fdadadadb9ed5.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/bc87a0c5ec4e19b95c157e1f63587fe32bf3908b69fc743a9c4fdadadadb9ed5.jpg)

![e4038bddf9dd055adbf8eb5b9c18c06abfed203d70f7bc9500eda6ea570c56bd.jpg](../iclr_results/191_Do Vision-Language Models Represent Space and How_ Evaluating Spatial Frame of Reference under Ambig/tables/e4038bddf9dd055adbf8eb5b9c18c06abfed203d70f7bc9500eda6ea570c56bd.jpg)

## Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects


### Images

![075fcb69778ab25b73fbb29cba6be4384d5240cd09d40302459d3ebf716263d8.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/075fcb69778ab25b73fbb29cba6be4384d5240cd09d40302459d3ebf716263d8.jpg)

![08f176bdb46350435613e7bad5361eb1b636dd5e6097b7c8435be20ba1bdde62.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/08f176bdb46350435613e7bad5361eb1b636dd5e6097b7c8435be20ba1bdde62.jpg)

![2adbccacecad16adc0264e024399150e2d692764207c5f3f52c1fb176650b0f1.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/2adbccacecad16adc0264e024399150e2d692764207c5f3f52c1fb176650b0f1.jpg)

![3e5bd0f51bf7a42517944510cc94bb84f65564df19530d8b11a8135fa1d92c1b.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/3e5bd0f51bf7a42517944510cc94bb84f65564df19530d8b11a8135fa1d92c1b.jpg)

![3f0b02ef457b00af7e46d2dc8b11802ecf43915ab145f6b22156a1e3ab849bbe.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/3f0b02ef457b00af7e46d2dc8b11802ecf43915ab145f6b22156a1e3ab849bbe.jpg)

![458014564a0f8b332c8fd8b319fe720536143ec9e2160ee33ab9d748e54e7613.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/458014564a0f8b332c8fd8b319fe720536143ec9e2160ee33ab9d748e54e7613.jpg)

![593803617fdb6c6040c3d58e4da3f2c93205eab229ebc09152af1b3573ba5789.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/593803617fdb6c6040c3d58e4da3f2c93205eab229ebc09152af1b3573ba5789.jpg)

![5a1ffc99d0eb7543b1d377e547bb0e32c55c3709cb752801e0489ec8c4d9fd96.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/5a1ffc99d0eb7543b1d377e547bb0e32c55c3709cb752801e0489ec8c4d9fd96.jpg)

![6a0e2ad5da0afbc3a8d822d2c4e531093e283d417e9dff771c36fbb0a34544f4.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/6a0e2ad5da0afbc3a8d822d2c4e531093e283d417e9dff771c36fbb0a34544f4.jpg)

![6b4c31157aa3a2c0fbab051a98d5e97723792d7bedd7723ce295199c6bbcbd5d.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/6b4c31157aa3a2c0fbab051a98d5e97723792d7bedd7723ce295199c6bbcbd5d.jpg)

![7520ae58d88f9a1c0f555f24c480d159fac2e867b7dc07641fcf32ec8f531136.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/7520ae58d88f9a1c0f555f24c480d159fac2e867b7dc07641fcf32ec8f531136.jpg)

![763126e34f6f6d40288f64b244e7850b8a00f946c5d6cebd5ed7351e4e90de67.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/763126e34f6f6d40288f64b244e7850b8a00f946c5d6cebd5ed7351e4e90de67.jpg)

![8aa960c87b3018257a7aa55fe33d19ff44746599ca7e8e2f695e94b4e81f280d.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/8aa960c87b3018257a7aa55fe33d19ff44746599ca7e8e2f695e94b4e81f280d.jpg)

![8e72c78424c5f03e5d4cba845a354315984b5dc54e5f34d24a48ec46ea0ed56a.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/8e72c78424c5f03e5d4cba845a354315984b5dc54e5f34d24a48ec46ea0ed56a.jpg)

![96431b795d59548f5b9fcbc907fbd7c85c9966989f0de97df4b800d01d04aa20.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/96431b795d59548f5b9fcbc907fbd7c85c9966989f0de97df4b800d01d04aa20.jpg)

![9c2142b1d6d4d56884ed76eac31470d0c7992a339a2d0f4f38e596556e20b4ff.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/9c2142b1d6d4d56884ed76eac31470d0c7992a339a2d0f4f38e596556e20b4ff.jpg)

![b83a12b25f0e8b7f9ab680d43cd886dfe3ba3751b418f791655c387f82197757.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/b83a12b25f0e8b7f9ab680d43cd886dfe3ba3751b418f791655c387f82197757.jpg)

![bb2d9508874fe79a3f27a49f232b2e251bce9616104296e0226de82c0d8e48f1.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/bb2d9508874fe79a3f27a49f232b2e251bce9616104296e0226de82c0d8e48f1.jpg)

![c7b12b0a57295f38a0ee95ffec5291c026c64c5e7ee3e00b8f2c7a4149d4beda.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/c7b12b0a57295f38a0ee95ffec5291c026c64c5e7ee3e00b8f2c7a4149d4beda.jpg)

![cee83605304313ef9e011c1d35e382239e4e42914e017153ad99d6bae014b70c.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/cee83605304313ef9e011c1d35e382239e4e42914e017153ad99d6bae014b70c.jpg)

![ceef9faa47cfe82993cb19a8abc5c198aba62af47d5b0df2f0b072cd6bb90370.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/ceef9faa47cfe82993cb19a8abc5c198aba62af47d5b0df2f0b072cd6bb90370.jpg)

![d017c400e396df43531ab7b9acf5e263a76c0d42d3a852564c65d533bb639940.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/d017c400e396df43531ab7b9acf5e263a76c0d42d3a852564c65d533bb639940.jpg)

![f0fc9e70f7282f1c911a8901ac7ac7f2a50d4480635c1686de36f96143b2a32a.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/f0fc9e70f7282f1c911a8901ac7ac7f2a50d4480635c1686de36f96143b2a32a.jpg)

![f8de026f7269a8a7f262ee3f19842adc844a8081e1481feb3341ecda4397047e.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/images/f8de026f7269a8a7f262ee3f19842adc844a8081e1481feb3341ecda4397047e.jpg)

### Tables

![205d21985dfea2059ed11aa98e1150eabb8ada34b73dee322216286a24c8b488.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/tables/205d21985dfea2059ed11aa98e1150eabb8ada34b73dee322216286a24c8b488.jpg)

![3f1f77a69793cff229ceb3017fcfde8fcc4803f79cd1ad86a2f1412ae061406a.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/tables/3f1f77a69793cff229ceb3017fcfde8fcc4803f79cd1ad86a2f1412ae061406a.jpg)

![58f6141f7652e1ebeb1fc17b67d31a84972fba94dac46d56a94d17c4af9a22f9.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/tables/58f6141f7652e1ebeb1fc17b67d31a84972fba94dac46d56a94d17c4af9a22f9.jpg)

![6b67157d6642a88a92164c4086c17483590135224c5b191f8812244a2cf25870.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/tables/6b67157d6642a88a92164c4086c17483590135224c5b191f8812244a2cf25870.jpg)

![f43e7b42bfee0a842ba46867b2592c1723dcee569b648772257e672b28bf68e6.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/tables/f43e7b42bfee0a842ba46867b2592c1723dcee569b648772257e672b28bf68e6.jpg)

![fe7a7bf18d475b9f845c84404999a769d9f8bf6f20096df4f6a0a69498fbc4fd.jpg](../iclr_results/192_Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects/tables/fe7a7bf18d475b9f845c84404999a769d9f8bf6f20096df4f6a0a69498fbc4fd.jpg)

## MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering


### Images

![113de8f7172c0798479571c5b8ff11f55f8a6cdbb8412c3de3b8fab8c7b61750.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/113de8f7172c0798479571c5b8ff11f55f8a6cdbb8412c3de3b8fab8c7b61750.jpg)

![2a596f4365c5b973641b57831fac35fa050edc8ed609a8c205c159f12bd534ef.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/2a596f4365c5b973641b57831fac35fa050edc8ed609a8c205c159f12bd534ef.jpg)

![35ea3926d7db527dfcebca563541f2cf9a98495d1dafa304e1523c30d127628f.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/35ea3926d7db527dfcebca563541f2cf9a98495d1dafa304e1523c30d127628f.jpg)

![74bbf95dd86b178b92320aacd8510d71cef589aa366742d31206ca2caa09e858.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/74bbf95dd86b178b92320aacd8510d71cef589aa366742d31206ca2caa09e858.jpg)

![a36f2d640a30621247bf69a85b47dfb80731a70536ec9baa2b13d4fd2728816c.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/a36f2d640a30621247bf69a85b47dfb80731a70536ec9baa2b13d4fd2728816c.jpg)

![cb64380fcd4cfce35d22852573195945b9790bd3d4c9b26fd30f69f1576bbc03.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/cb64380fcd4cfce35d22852573195945b9790bd3d4c9b26fd30f69f1576bbc03.jpg)

![cd23bd5753d620c2fe6c14742fec1b31da1931dc9336a89527ea5c2d702eeb0f.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/cd23bd5753d620c2fe6c14742fec1b31da1931dc9336a89527ea5c2d702eeb0f.jpg)

![e71f43e6bb95e75d232fbeff7e4bc006b8abdaadd848957ed4ef706271ce6c23.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/images/e71f43e6bb95e75d232fbeff7e4bc006b8abdaadd848957ed4ef706271ce6c23.jpg)

### Tables

![4234dd653565ccc15638699f95e882b71d96676dd136a69dc92a1cbfea583928.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/4234dd653565ccc15638699f95e882b71d96676dd136a69dc92a1cbfea583928.jpg)

![42fae37f26dddbdd33628e5609705a9fac29c9d3d012c129987e2cc21f4c2ad4.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/42fae37f26dddbdd33628e5609705a9fac29c9d3d012c129987e2cc21f4c2ad4.jpg)

![5326c2d86de9e612caddb783226a502e5b8b746ee377adfd6aebe8e0cf1b2861.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/5326c2d86de9e612caddb783226a502e5b8b746ee377adfd6aebe8e0cf1b2861.jpg)

![6aa6ed9819a07a96b05f4f33016865e788548cb1d99154a8d83285d1968946f0.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/6aa6ed9819a07a96b05f4f33016865e788548cb1d99154a8d83285d1968946f0.jpg)

![8f9e9638dd0bd1468da51332d86fa0b8b2121a3a819b542b150b13e01024a1f4.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/8f9e9638dd0bd1468da51332d86fa0b8b2121a3a819b542b150b13e01024a1f4.jpg)

![a2a3ac6e5bdc8ebcd791f790d5a0fbfab7c67e8dd6dcded8d8da34e13f76006b.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/a2a3ac6e5bdc8ebcd791f790d5a0fbfab7c67e8dd6dcded8d8da34e13f76006b.jpg)

![a2b042947cfaec87cf1d5195c3f17d51edceb23e7f5098141694da80d8e444ae.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/a2b042947cfaec87cf1d5195c3f17d51edceb23e7f5098141694da80d8e444ae.jpg)

![a955eb4a4ffc139cd8a794695cf81a56816937423750039f2b6bad08528686e0.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/a955eb4a4ffc139cd8a794695cf81a56816937423750039f2b6bad08528686e0.jpg)

![b4a159474db7db6a25f234bd3adb7cb7b4a7484ce71119ad3e796d4d0bb4fd52.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/b4a159474db7db6a25f234bd3adb7cb7b4a7484ce71119ad3e796d4d0bb4fd52.jpg)

![b4e81fff91ad8a8b5c868ee181f11025a106efb454706f3251ecb11b6082c851.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/b4e81fff91ad8a8b5c868ee181f11025a106efb454706f3251ecb11b6082c851.jpg)

![b58e59c70ee4db1bc80a0df1be5f93989a02a40a903a3ae47d5040aaa997549e.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/b58e59c70ee4db1bc80a0df1be5f93989a02a40a903a3ae47d5040aaa997549e.jpg)

![c4f310303903e89b650dd516d27366ebea61e9fdc9d0837a313f9ae109210c2f.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/c4f310303903e89b650dd516d27366ebea61e9fdc9d0837a313f9ae109210c2f.jpg)

![e4d9125a4c159a01e27c65af0c409efdab1a1484b3e74424182cd1bef79c456e.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/e4d9125a4c159a01e27c65af0c409efdab1a1484b3e74424182cd1bef79c456e.jpg)

![f636819df7570467fb31f1a35a3144b5ae1a3eda01931d04bf4dbef02b88b809.jpg](../iclr_results/193_MLE-bench_ Evaluating Machine Learning Agents on Machine Learning Engineering/tables/f636819df7570467fb31f1a35a3144b5ae1a3eda01931d04bf4dbef02b88b809.jpg)

## Safety Alignment Should be Made More Than Just a Few Tokens Deep


### Images

![4b800220cba658daf974a22cfe61008d50b895c3da4ef7308bb28d72ef6f0d68.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/images/4b800220cba658daf974a22cfe61008d50b895c3da4ef7308bb28d72ef6f0d68.jpg)

![76da006efc42a6b1dae5551122abd80c97b6b4667e97ca1bc5e5c9ec6fe7e9b7.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/images/76da006efc42a6b1dae5551122abd80c97b6b4667e97ca1bc5e5c9ec6fe7e9b7.jpg)

![a752cf94b9a071b6b3579c8c5f52f08ff3c1ffc075fc12767552c929e51fbd64.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/images/a752cf94b9a071b6b3579c8c5f52f08ff3c1ffc075fc12767552c929e51fbd64.jpg)

![c8d375bbbd04c393eac2856f828c23018f177f1732c9937d10d1ca7bc94624f1.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/images/c8d375bbbd04c393eac2856f828c23018f177f1732c9937d10d1ca7bc94624f1.jpg)

![fc5fe47dae4fdc141e6a62f745e5eb229ed2006fc1aa21b11e101de4207e08cf.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/images/fc5fe47dae4fdc141e6a62f745e5eb229ed2006fc1aa21b11e101de4207e08cf.jpg)

### Tables

![008ae47c21da533bb3c4da819adb5769e2b64943865bbf4686194b364c4a8d0f.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/008ae47c21da533bb3c4da819adb5769e2b64943865bbf4686194b364c4a8d0f.jpg)

![021e4d231bf996345f84bccb102ca6bea654b195adeb64ae1269c0fc1de30964.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/021e4d231bf996345f84bccb102ca6bea654b195adeb64ae1269c0fc1de30964.jpg)

![075b928526b8c7bcd8d6c6335348b53837c04281f6c1d5899e725f08735d76e2.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/075b928526b8c7bcd8d6c6335348b53837c04281f6c1d5899e725f08735d76e2.jpg)

![11242c98edd254c173173a8fcf34cdb4f8fa7bef28ce98f08da0a1f5815256d6.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/11242c98edd254c173173a8fcf34cdb4f8fa7bef28ce98f08da0a1f5815256d6.jpg)

![1777ae38040217bb772e4daeead2389ef748ff80be3a710af6a399289ea167ea.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/1777ae38040217bb772e4daeead2389ef748ff80be3a710af6a399289ea167ea.jpg)

![2677d0d1cbbd5d6cb4eed185073c4adfa40f1fbd5fc3b24a4d1b89e80a7ef36c.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/2677d0d1cbbd5d6cb4eed185073c4adfa40f1fbd5fc3b24a4d1b89e80a7ef36c.jpg)

![27ca532c216bc9a04a18810e42d082ccebae9b154dc9ba802f1b993d68c65769.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/27ca532c216bc9a04a18810e42d082ccebae9b154dc9ba802f1b993d68c65769.jpg)

![5672d4b273b83c2db2c6e908e839f40d626b01b1c813ec2d794376dd049a28f4.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/5672d4b273b83c2db2c6e908e839f40d626b01b1c813ec2d794376dd049a28f4.jpg)

![62945a7f55096ef0118ebdc4aeb3cdcbfb830f50b25660484ad21c0a49eb75a6.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/62945a7f55096ef0118ebdc4aeb3cdcbfb830f50b25660484ad21c0a49eb75a6.jpg)

![a0f446657228b702b2b0ddf0a9075e9c7c9f2b52a65247bcab3fc8f8cf4447ac.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/a0f446657228b702b2b0ddf0a9075e9c7c9f2b52a65247bcab3fc8f8cf4447ac.jpg)

![afb2f38a816e4b2baa1ee19505f5e5d8deb3ae850c30e3ac47437caf6b1b168a.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/afb2f38a816e4b2baa1ee19505f5e5d8deb3ae850c30e3ac47437caf6b1b168a.jpg)

![e9a6b5af4bcb99640eea55da3c9f8daeed12f6babd8c1d1987f3e3093573bd42.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/e9a6b5af4bcb99640eea55da3c9f8daeed12f6babd8c1d1987f3e3093573bd42.jpg)

![fc5245a4525737203787da49392de6f0211c41744dd65e7b6f54fffafef19e27.jpg](../iclr_results/194_Safety Alignment Should be Made More Than Just a Few Tokens Deep/tables/fc5245a4525737203787da49392de6f0211c41744dd65e7b6f54fffafef19e27.jpg)

## SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning


### Images

![24ea61501cba9b76bc1b8bb963ab7b0bc466f7a3db140509d005cecfc8476994.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/images/24ea61501cba9b76bc1b8bb963ab7b0bc466f7a3db140509d005cecfc8476994.jpg)

![3eefbd216c50d9f7392704dab2e0d71248639c3f50360140cfba7ff56f8b9fd6.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/images/3eefbd216c50d9f7392704dab2e0d71248639c3f50360140cfba7ff56f8b9fd6.jpg)

![4ac8662c11be378837cd6730c2f964dfaddc81db1af125e9c0e7d64ba92f3002.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/images/4ac8662c11be378837cd6730c2f964dfaddc81db1af125e9c0e7d64ba92f3002.jpg)

![54c7383ff887852315c06459abcdee358b68680598ec6a62db9caeb327d0a60f.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/images/54c7383ff887852315c06459abcdee358b68680598ec6a62db9caeb327d0a60f.jpg)

![715a437f872443d8345ea1977d6711d4e82ad056340bf114540e271255e51a00.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/images/715a437f872443d8345ea1977d6711d4e82ad056340bf114540e271255e51a00.jpg)

![75b5ef43c1fc7b2df41777eac4256226f9225b68af4900ddab0f094e4e9218cb.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/images/75b5ef43c1fc7b2df41777eac4256226f9225b68af4900ddab0f094e4e9218cb.jpg)

![8a5e1a2bd17c827016ee07aab28070afa18c80d8c42f97f51929453eb2507590.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/images/8a5e1a2bd17c827016ee07aab28070afa18c80d8c42f97f51929453eb2507590.jpg)

### Tables

![14ac2cf9c3e0a3c52221dab9c184711b75ba7913f77a01f9f9018af64a8694de.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/tables/14ac2cf9c3e0a3c52221dab9c184711b75ba7913f77a01f9f9018af64a8694de.jpg)

![2d21edeb9fab4100dc8c9d9a8d30d4cab21061dde839e2c4b458e22dac09382b.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/tables/2d21edeb9fab4100dc8c9d9a8d30d4cab21061dde839e2c4b458e22dac09382b.jpg)

![3e8aaaeac3f44ce15a1c57779305918e617f13490f2e47f55ae1302ac5b1df3e.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/tables/3e8aaaeac3f44ce15a1c57779305918e617f13490f2e47f55ae1302ac5b1df3e.jpg)

![44712f9dd949b444f6ca331ee24cafb9fbe269133f17d71ab0de809c999571ca.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/tables/44712f9dd949b444f6ca331ee24cafb9fbe269133f17d71ab0de809c999571ca.jpg)

![8ce5877883d23cc74272d8b3aedc75678fa94adde2169771591a67ff678ded88.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/tables/8ce5877883d23cc74272d8b3aedc75678fa94adde2169771591a67ff678ded88.jpg)

![da97092c8af025d91b948cebd9e9fdfa8e8af723ea5b06d7534ca02e69e4f876.jpg](../iclr_results/197_SD-LoRA_ Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning/tables/da97092c8af025d91b948cebd9e9fdfa8e8af723ea5b06d7534ca02e69e4f876.jpg)

## Prioritized Generative Replay


### Images

![1b340e6294f509ca3626e2a70ab3c780f052d499e347dbd978a5d3bec9f59d86.jpg](../iclr_results/199_Prioritized Generative Replay/images/1b340e6294f509ca3626e2a70ab3c780f052d499e347dbd978a5d3bec9f59d86.jpg)

![2615c6f9e73f7cf9938644bda62e4c45013f93fa970c9ff27f3bb46050e55a49.jpg](../iclr_results/199_Prioritized Generative Replay/images/2615c6f9e73f7cf9938644bda62e4c45013f93fa970c9ff27f3bb46050e55a49.jpg)

![5a7327157d3a62d15b92419f71157a50b2ca85b1adbcad280cb1e21351f9df6a.jpg](../iclr_results/199_Prioritized Generative Replay/images/5a7327157d3a62d15b92419f71157a50b2ca85b1adbcad280cb1e21351f9df6a.jpg)

![641f0794cc7fcfe3fd4a560ce108b47159ff7892947d6e66316ac1847de395e5.jpg](../iclr_results/199_Prioritized Generative Replay/images/641f0794cc7fcfe3fd4a560ce108b47159ff7892947d6e66316ac1847de395e5.jpg)

![64373d73783c60faa2bf25d8fd34ff4cc446f5802e6c7cff35d77c4e199e0e3e.jpg](../iclr_results/199_Prioritized Generative Replay/images/64373d73783c60faa2bf25d8fd34ff4cc446f5802e6c7cff35d77c4e199e0e3e.jpg)

![69574fa98cbf49ef8d547a7c328f55d50f5b44afb2ba00e39aa4aa6298c32916.jpg](../iclr_results/199_Prioritized Generative Replay/images/69574fa98cbf49ef8d547a7c328f55d50f5b44afb2ba00e39aa4aa6298c32916.jpg)

![aaff34f0c8e3fb17133b75ae45af7356837799cf1b6701e8efa79b05b7110f2c.jpg](../iclr_results/199_Prioritized Generative Replay/images/aaff34f0c8e3fb17133b75ae45af7356837799cf1b6701e8efa79b05b7110f2c.jpg)

![bac43a651208125ae07e36d9cdbc3e5ca9b261af40fea2793117c0d8318a0b1f.jpg](../iclr_results/199_Prioritized Generative Replay/images/bac43a651208125ae07e36d9cdbc3e5ca9b261af40fea2793117c0d8318a0b1f.jpg)

![c4956f5267aa3136573166c0be49dd0f702f94b58a2199bc12c889810843c013.jpg](../iclr_results/199_Prioritized Generative Replay/images/c4956f5267aa3136573166c0be49dd0f702f94b58a2199bc12c889810843c013.jpg)

### Tables

![1523091303972a4d2b79ebe9a7a0e332f5ea701e68ceee14906924ce154b341a.jpg](../iclr_results/199_Prioritized Generative Replay/tables/1523091303972a4d2b79ebe9a7a0e332f5ea701e68ceee14906924ce154b341a.jpg)

![4a80acbd5f05dfc40932425e3063038cb2b1992eafff9cdf54b53cdd8528274c.jpg](../iclr_results/199_Prioritized Generative Replay/tables/4a80acbd5f05dfc40932425e3063038cb2b1992eafff9cdf54b53cdd8528274c.jpg)

![62d330787b9d1e38c003f002fe9fb4b5c73a5d33cca56e50de99c82e4c759995.jpg](../iclr_results/199_Prioritized Generative Replay/tables/62d330787b9d1e38c003f002fe9fb4b5c73a5d33cca56e50de99c82e4c759995.jpg)

![67d717fb50c2e3766b2a8d91b9422373bec2fde60fe0f1cd30f32decadf31bb6.jpg](../iclr_results/199_Prioritized Generative Replay/tables/67d717fb50c2e3766b2a8d91b9422373bec2fde60fe0f1cd30f32decadf31bb6.jpg)

![68947766a1c519a0993d03b5cc79d65f3524d015478ac64430c01662af7dade1.jpg](../iclr_results/199_Prioritized Generative Replay/tables/68947766a1c519a0993d03b5cc79d65f3524d015478ac64430c01662af7dade1.jpg)

![7be92c4b01c8d543f7baa3107c8e62551fe40418e6043ab90dbcb1c661dabce1.jpg](../iclr_results/199_Prioritized Generative Replay/tables/7be92c4b01c8d543f7baa3107c8e62551fe40418e6043ab90dbcb1c661dabce1.jpg)

![957ba3239cd88292e6c451472eb773ef40f375bfc943471fdaa7f28263f2f297.jpg](../iclr_results/199_Prioritized Generative Replay/tables/957ba3239cd88292e6c451472eb773ef40f375bfc943471fdaa7f28263f2f297.jpg)

![b757e12e23b3e07600df835793277f12e9dce3222233a939dcbdcca46c34ad31.jpg](../iclr_results/199_Prioritized Generative Replay/tables/b757e12e23b3e07600df835793277f12e9dce3222233a939dcbdcca46c34ad31.jpg)

![efe6923a1db9a86a564f722a967f4210fe50d519f58a5b59eed0a60be7b69d46.jpg](../iclr_results/199_Prioritized Generative Replay/tables/efe6923a1db9a86a564f722a967f4210fe50d519f58a5b59eed0a60be7b69d46.jpg)

![fbd30bbe52b4970d0a0d46a9e9f80c0fa8da81cf5ecee31b3b47d5ca94d556d4.jpg](../iclr_results/199_Prioritized Generative Replay/tables/fbd30bbe52b4970d0a0d46a9e9f80c0fa8da81cf5ecee31b3b47d5ca94d556d4.jpg)

## A Probabilistic Perspective on Unlearning and Alignment for Large Language Models


### Images

![192430986084ea7db3552d7d41567eddba61458390f60824ccc05d55c4ff768d.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/images/192430986084ea7db3552d7d41567eddba61458390f60824ccc05d55c4ff768d.jpg)

![1b8ac8955049aed636ec299e368d2eb18a86bfeecf0b9a79c1af784968dea012.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/images/1b8ac8955049aed636ec299e368d2eb18a86bfeecf0b9a79c1af784968dea012.jpg)

![4c255b78e1bfecc651003bde44261a8be7f2f2f59081acd63bc4212ab627ab02.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/images/4c255b78e1bfecc651003bde44261a8be7f2f2f59081acd63bc4212ab627ab02.jpg)

![828c6ea84ff0cd46ec783575ace930fb303906d0d093ce547d5eb09b0ec5c543.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/images/828c6ea84ff0cd46ec783575ace930fb303906d0d093ce547d5eb09b0ec5c543.jpg)

![908851f01b16affc8175e5fb150f350789c93778ce9726c87d2f6aa9271c22c8.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/images/908851f01b16affc8175e5fb150f350789c93778ce9726c87d2f6aa9271c22c8.jpg)

![cfb485484c30ea9a5d04fbbd85549ba6dd0460e613e123c3a451f604d02e1e58.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/images/cfb485484c30ea9a5d04fbbd85549ba6dd0460e613e123c3a451f604d02e1e58.jpg)

### Tables

![1596a9283e4968da82316b9fb9bd50094111cb305de768b639d6a88ce026c01d.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/tables/1596a9283e4968da82316b9fb9bd50094111cb305de768b639d6a88ce026c01d.jpg)

![2afc992f41ea8371815bbde42ffefeef01043f1608d8315253c8b95d9f4b2e2a.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/tables/2afc992f41ea8371815bbde42ffefeef01043f1608d8315253c8b95d9f4b2e2a.jpg)

![96d0d0bc668ec631f967d8d7476099f8210185ba1e7d1643b91f8c0cdfd0c8a7.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/tables/96d0d0bc668ec631f967d8d7476099f8210185ba1e7d1643b91f8c0cdfd0c8a7.jpg)

![9eee52d4d62bc12e387d4ce1ed59388186cca978cd22e8e5df4eac50f139d876.jpg](../iclr_results/200_A Probabilistic Perspective on Unlearning and Alignment for Large Language Models/tables/9eee52d4d62bc12e387d4ce1ed59388186cca978cd22e8e5df4eac50f139d876.jpg)

## Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning


### Images

![055e8562312449c395de7995d925f3b3d8f45d739788d2aba83fb463ac5c54f1.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/055e8562312449c395de7995d925f3b3d8f45d739788d2aba83fb463ac5c54f1.jpg)

![0ff3c3d220a7d383c64adcb7a7b8f863cb2a0f662a1dbb84e25e12ed87c35a04.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/0ff3c3d220a7d383c64adcb7a7b8f863cb2a0f662a1dbb84e25e12ed87c35a04.jpg)

![2936218e52672cb65e208316c733f5a85ddfb226ce71ec96b7200db19a0001c7.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/2936218e52672cb65e208316c733f5a85ddfb226ce71ec96b7200db19a0001c7.jpg)

![88a6353d883eee296f58d73b5809f16f28485c47a02e20de0631121c34eec760.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/88a6353d883eee296f58d73b5809f16f28485c47a02e20de0631121c34eec760.jpg)

![a5f62906ac0896b07980f4445b682b8de29a4ab2e88670edaaf48d196560fbd6.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/a5f62906ac0896b07980f4445b682b8de29a4ab2e88670edaaf48d196560fbd6.jpg)

![c5c5b5aa91507553e2ddab585226e53573f0ce920afdc86ee089f10ccf94361b.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/c5c5b5aa91507553e2ddab585226e53573f0ce920afdc86ee089f10ccf94361b.jpg)

![d73f40371da2a31c603011f0601d439226758f9225b2ccad2787813caa1e50b5.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/d73f40371da2a31c603011f0601d439226758f9225b2ccad2787813caa1e50b5.jpg)

![ebfdae388f87bd9519fe22b656d4e7e38bd8114a0859243f79a0442ea2d0dced.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/images/ebfdae388f87bd9519fe22b656d4e7e38bd8114a0859243f79a0442ea2d0dced.jpg)

### Tables

![2ed0b435b705f9d51927c7e75e403b7cb9494c7ac807bc3ba381a81389fd2b8b.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/2ed0b435b705f9d51927c7e75e403b7cb9494c7ac807bc3ba381a81389fd2b8b.jpg)

![327b72c76e01304968f716e403565b8b1284121e79bcc9d7e0e9157570b324c7.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/327b72c76e01304968f716e403565b8b1284121e79bcc9d7e0e9157570b324c7.jpg)

![400c4858144248356920939f807f4976f4f38a3599551435335077fe00233f99.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/400c4858144248356920939f807f4976f4f38a3599551435335077fe00233f99.jpg)

![424e7e39315aabc72009c5d6834875cfcf21e0389abc3f6117ed5e7f4e8a03c5.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/424e7e39315aabc72009c5d6834875cfcf21e0389abc3f6117ed5e7f4e8a03c5.jpg)

![4da9c100c106db23c9855d9ee71707379b6b4cd2741f3bf3b24eed38159a9298.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/4da9c100c106db23c9855d9ee71707379b6b4cd2741f3bf3b24eed38159a9298.jpg)

![651f23622ff00a06c95702a81db43af0eda77fc27cf4975c45cc942ad7334fa0.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/651f23622ff00a06c95702a81db43af0eda77fc27cf4975c45cc942ad7334fa0.jpg)

![c063759aa091cbe180fcebbb03b648a06055b2dc00a506e6955e832c86477ba6.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/c063759aa091cbe180fcebbb03b648a06055b2dc00a506e6955e832c86477ba6.jpg)

![f1edceb239c0e6d4ab1986330b4c19af912138ac945de8cd581ecd43a5d66c8a.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/f1edceb239c0e6d4ab1986330b4c19af912138ac945de8cd581ecd43a5d66c8a.jpg)

![fa8c53b41386ceab35ca7c24493c23e4781284ee3ad9acc20374f9ca3f56d5f9.jpg](../iclr_results/202_Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning/tables/fa8c53b41386ceab35ca7c24493c23e4781284ee3ad9acc20374f9ca3f56d5f9.jpg)

## Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning


### Images

![0a768b5d305402cd78b4100512dc921ee148c3f1cc28245262b961d87f11bb1f.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/0a768b5d305402cd78b4100512dc921ee148c3f1cc28245262b961d87f11bb1f.jpg)

![2c8a1bbdd3e0d41b9c7d680360cbf52f84ae07777a407a3e8937ce4edbaaaa57.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/2c8a1bbdd3e0d41b9c7d680360cbf52f84ae07777a407a3e8937ce4edbaaaa57.jpg)

![2d493cac03b7cfd4d279d19c0e63f74d1355e5fcd0cd322ec8040c5bdcbdeb45.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/2d493cac03b7cfd4d279d19c0e63f74d1355e5fcd0cd322ec8040c5bdcbdeb45.jpg)

![2f478b201380511406db1e290f9daa834910e7c0db36e820251ab17889672eda.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/2f478b201380511406db1e290f9daa834910e7c0db36e820251ab17889672eda.jpg)

![3fd789dfcc16abb5c802f64be9672a3eb6883063d56a924a30a1c17bcc095c13.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/3fd789dfcc16abb5c802f64be9672a3eb6883063d56a924a30a1c17bcc095c13.jpg)

![6bdbc9bd59a76f341ef578ec9893693d8ef987cfc9b29900811155a10d0ad7a0.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/6bdbc9bd59a76f341ef578ec9893693d8ef987cfc9b29900811155a10d0ad7a0.jpg)

![704077b59a86e6d54a694be35229102ccd59c12ec03ffdba6e00cc5d6c4b365d.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/704077b59a86e6d54a694be35229102ccd59c12ec03ffdba6e00cc5d6c4b365d.jpg)

![70da7fb580312a07e751a7b3d1bc68685deb1a79229578920f54a5a258c6d2e4.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/70da7fb580312a07e751a7b3d1bc68685deb1a79229578920f54a5a258c6d2e4.jpg)

![78bf3dc339f916e3adea7e29ec0601e33c832646dfe20067c455ba8110c844de.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/78bf3dc339f916e3adea7e29ec0601e33c832646dfe20067c455ba8110c844de.jpg)

![8bd82e388d1095913e8a82c4e2c58213a2567c70d042e20ca2ebc20a5da83a9a.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/8bd82e388d1095913e8a82c4e2c58213a2567c70d042e20ca2ebc20a5da83a9a.jpg)

![978430283ac2e71a27421726954b1ad34c220479111865d91dabbd2e197eebe2.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/978430283ac2e71a27421726954b1ad34c220479111865d91dabbd2e197eebe2.jpg)

![aa77a167543f5647cd291b9e571ef32dd254b199b31387509ef7681b56617fce.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/aa77a167543f5647cd291b9e571ef32dd254b199b31387509ef7681b56617fce.jpg)

![ab4e947e3bdacaf40d432a4268ce4c79f53ffbd8cdc9a8e879a9e5b8349082d4.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/ab4e947e3bdacaf40d432a4268ce4c79f53ffbd8cdc9a8e879a9e5b8349082d4.jpg)

![b30ed5ad7334baa295ee526d76405ddd07273d33eab9fe382ea540957dc31f63.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/b30ed5ad7334baa295ee526d76405ddd07273d33eab9fe382ea540957dc31f63.jpg)

![b529f303b7e1b060a27d68d8740e690cec27cc4869979d65dc684b9250abe75a.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/b529f303b7e1b060a27d68d8740e690cec27cc4869979d65dc684b9250abe75a.jpg)

![bfeb6dce70eb51c2d7d4de1f1502beac3ddbfd699e3464f02af66b516ec7d5d5.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/bfeb6dce70eb51c2d7d4de1f1502beac3ddbfd699e3464f02af66b516ec7d5d5.jpg)

![d0fa1d08145e596f9e71f63ebb39959b3d4d35bd140fabc75832c923d84b1f90.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/d0fa1d08145e596f9e71f63ebb39959b3d4d35bd140fabc75832c923d84b1f90.jpg)

![df6c2584719204fb865211911db752dde8e36b5958b0a486cee9cb26563e7c0b.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/df6c2584719204fb865211911db752dde8e36b5958b0a486cee9cb26563e7c0b.jpg)

![f11847882f3ec4ca1e1a65ee88b626b98870c3ef064778dd4fc1c5d68ac619c7.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/images/f11847882f3ec4ca1e1a65ee88b626b98870c3ef064778dd4fc1c5d68ac619c7.jpg)

### Tables

![2220792cac31664e55a8f3a5f25f1aef33c0afa6f6659422f3aefd7e3fe7fc2a.jpg](../iclr_results/203_Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning/tables/2220792cac31664e55a8f3a5f25f1aef33c0afa6f6659422f3aefd7e3fe7fc2a.jpg)

## Compositional Entailment Learning for Hyperbolic Vision-Language Models

### Images

![12e8c8f43424f5827ef5a216a6417b6e01d0a234350bc7dd1a29532ed38d45f8.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/12e8c8f43424f5827ef5a216a6417b6e01d0a234350bc7dd1a29532ed38d45f8.jpg)

![2439f8023db7b99b91e11d3115e185e3bd284e17849b3f2bb16a0f49cf5f2b7f.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/2439f8023db7b99b91e11d3115e185e3bd284e17849b3f2bb16a0f49cf5f2b7f.jpg)

![44c41f760aac421c8c87aa9474e3bf5c9229e3a7e1f95cde1c507040e5e09958.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/44c41f760aac421c8c87aa9474e3bf5c9229e3a7e1f95cde1c507040e5e09958.jpg)

![4a26ff83fa5446ee08e82184b25d4e95bff692633bfdf4901b69b3892e5ff8fa.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/4a26ff83fa5446ee08e82184b25d4e95bff692633bfdf4901b69b3892e5ff8fa.jpg)

![7673681916a1226005df5e305c2cb18a98f8fff422199cea687c2b1b8111da73.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/7673681916a1226005df5e305c2cb18a98f8fff422199cea687c2b1b8111da73.jpg)

![860b55476b7769702719ddff30ce2bbac977efa0af4869cf12996a7d37dd765e.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/860b55476b7769702719ddff30ce2bbac977efa0af4869cf12996a7d37dd765e.jpg)

![863b3eb59a065a7dd2a90d6ca5ae2df733d264b7775c4864b903bc49d2955c31.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/863b3eb59a065a7dd2a90d6ca5ae2df733d264b7775c4864b903bc49d2955c31.jpg)

![8887fe74afe111c18e6ad0393c5ffe8eba48cdc2cd34f17e0676b212479f5950.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/8887fe74afe111c18e6ad0393c5ffe8eba48cdc2cd34f17e0676b212479f5950.jpg)

![9c0230e0d242b4e2588c4d95ae3e8eb619826d361b85dacb8e3e25f499ff89a5.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/9c0230e0d242b4e2588c4d95ae3e8eb619826d361b85dacb8e3e25f499ff89a5.jpg)

![ab2e803ecc849bff8811b7ca468cf56a1d1edfb54d11681500c04c472fe171f8.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/ab2e803ecc849bff8811b7ca468cf56a1d1edfb54d11681500c04c472fe171f8.jpg)

![b02d231207f047836451772184d83d643f54faa02886d8fc078706b83462af79.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/b02d231207f047836451772184d83d643f54faa02886d8fc078706b83462af79.jpg)

![b1a811acd3cae9c2d306896e63dada506d755203eb9dd0d1393ece186eb32301.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/b1a811acd3cae9c2d306896e63dada506d755203eb9dd0d1393ece186eb32301.jpg)

![c46a94ea23ff443cd042a7f82be31019d6f5875a48a213119f9002bca1f35208.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/c46a94ea23ff443cd042a7f82be31019d6f5875a48a213119f9002bca1f35208.jpg)

![f5008280b67dd3d440d4b6ced60b0f87f96a9817cb91d98c9042fbeea968d966.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/f5008280b67dd3d440d4b6ced60b0f87f96a9817cb91d98c9042fbeea968d966.jpg)

![f7d3bf160e0e2e82bb5fb3de7f7bbda7fd663880654126320c3410ee6b2ec435.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/f7d3bf160e0e2e82bb5fb3de7f7bbda7fd663880654126320c3410ee6b2ec435.jpg)

![feb75e79fc1134e3e9649f614ccc30546afcc93250d2c3081559f9e172a936ae.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/images/feb75e79fc1134e3e9649f614ccc30546afcc93250d2c3081559f9e172a936ae.jpg)

### Tables

![12bdc1e5606c49b402701916ac28965bb54951790c3dab099e4252dd90c3765c.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/12bdc1e5606c49b402701916ac28965bb54951790c3dab099e4252dd90c3765c.jpg)

![2542eb1220ae6b58e154bcb4ab358b16e1f033c0068d30613ce47902e711fdad.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/2542eb1220ae6b58e154bcb4ab358b16e1f033c0068d30613ce47902e711fdad.jpg)

![39c20595817f382ce4cd65cbcffb7cf99d524387e7514dc1a2328bc249af4007.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/39c20595817f382ce4cd65cbcffb7cf99d524387e7514dc1a2328bc249af4007.jpg)

![4cd365e133f290bf979a92441dbceca91b8bf328396d861b654ce32e3fe7ad83.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/4cd365e133f290bf979a92441dbceca91b8bf328396d861b654ce32e3fe7ad83.jpg)

![7f3abd855646d2535ca6b818676d9fecdb24c5ff52405adabd88f8b26f3cb358.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/7f3abd855646d2535ca6b818676d9fecdb24c5ff52405adabd88f8b26f3cb358.jpg)

![890dc03b00a6290703acc3763eda3645f74758d2fdb67942ebfefcdebe95bb01.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/890dc03b00a6290703acc3763eda3645f74758d2fdb67942ebfefcdebe95bb01.jpg)

![8e6cf3653ca0feacb155a5fa059366ed1cbbe8769f799b9d7a3094d87698c13f.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/8e6cf3653ca0feacb155a5fa059366ed1cbbe8769f799b9d7a3094d87698c13f.jpg)

![bbf08523582b6a0d0199a086823287c73b2a34aa6b413d4a94653fad3826d309.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/bbf08523582b6a0d0199a086823287c73b2a34aa6b413d4a94653fad3826d309.jpg)

![c2c8c03a92290bcd72beb78b5880dcce93160f2fe2515bfd2bee8c0b738f937f.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/c2c8c03a92290bcd72beb78b5880dcce93160f2fe2515bfd2bee8c0b738f937f.jpg)

![f0e09152c141ad191d6152ef98fb30482f5e4934818918ae5869121e3c3aca2f.jpg](../iclr_results/204_Compositional Entailment Learning for Hyperbolic Vision-Language Models/tables/f0e09152c141ad191d6152ef98fb30482f5e4934818918ae5869121e3c3aca2f.jpg)
