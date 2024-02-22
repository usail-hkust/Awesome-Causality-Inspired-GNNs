# Causality-Inspired-GNNs

This repository curated list of Causality-Inspired Graph Neural Network (CIGNN) works and relevant resources reviewed in our survey 'A Survey on Trustworthy Graph Neural Networks: From A Causal Perspective'. Recently, the integration of causal learning techniques into GNNs has demonstrated significant potential in mitigating trustworthiness issues. This is achieved by capturing the underlying data causality instead of relying on superficial correlations. In this survey, we comprehensively reviews the recent progress of CIGNNs within a novel taxonomy, aiming to distill the fundamental rationales behind these works in the lens of causality and spark further research on this promising direction. A preprint version can be found at [Link](http://arxiv.org/abs/2312.12477).

## Benchmark Datasets

### OOD
- H. Li, et al., “Out-of-distribution generalization on graphs: A survey,” CoRR, vol. abs/2202.07987, 2022. [Paper](https://arxiv.org/pdf/2202.07987)

- S. Gui, et al., “GOOD: A graph out-of-distribution benchmark,” in Adv. Neural Inf. Process. Syst. 35, 2022, pp. 2059–2073. [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/0dc91de822b71c66a7f54fa121d8cbb9-Paper-Datasets_and_Benchmarks.pdf), [Github](https://github.com/divelab/GOOD/)

- Y. Ji, et al., “Drugood: Out-of-distribution (OOD) dataset curator and benchmark for ai-aided drug discovery - A focus on affinity prediction problems with noise annotations,” CoRR, vol. abs/2201.09637, 2022. [Paper](https://arxiv.org/pdf/2201.09637), [Github](https://github.com/tencent-ailab/DrugOOD)

- Z. Wang, et al., “Towards out-of-distribution generalizable predictions of chemical kinetics properties,” CoRR, vol. abs/2310.03152, 2023. [Paper](https://arxiv.org/pdf/2310.03152)，[Github](https://github.com/zihao-wang/ReactionOOD)

### Fairness
- Y. Dong, et al., “Fairness in graph mining: A survey,” CoRR, vol. abs/2204.09888, 2022. [Paper](https://ieeexplore.ieee.org/iel7/69/4358933/10097603.pdf), [Github](https://github.com/yushundong/Graph-Mining-Fairness-Data)

### Explainability
- H. Yuan, et al., “Explainability in graph neural networks: A taxonomic survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 05, pp. 5782–5799, 2023. [Paper](https://ieeexplore.ieee.org/iel7/34/4359286/09875989.pdf)

- M. A. Prado-Romero, et al., “A survey on graph counterfactual explanations: Definitions, methods, evaluation, and research challenges,” ACM Comput. Surv., 2023, to be published.[Paper](https://dl.acm.org/doi/pdf/10.1145/3618105)

## Code and Packages
### CIGNNs
- S. Gui, et al., “GOOD: A graph out-of-distribution benchmark,” in Adv. Neural Inf. Process. Syst. 35, 2022, pp. 2059–2073. [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/0dc91de822b71c66a7f54fa121d8cbb9-Paper-Datasets_and_Benchmarks.pdf), [Github](https://github.com/divelab/GOOD/)

- M. A. Prado-Romero and G. Stilo, “GRETEL: graph counterfactual explanation evaluation framework,” in Proc. 31st ACM Int. Conf. Inf. Knowl. Manage., 2022, pp. 4389–4393. [Paper](https://dl.acm.org/doi/pdf/10.1145/3511808.3557608), [Github](https://github.com/MarioTheOne/GRETEL)

- M. A. Prado-Romero, et al., “Developing and evaluating graph counterfactual explanation with GRETEL,” in Proc. 16th ACM Int. Conf. Web Search and Data Mining, 2023, pp. 1180–1183. [Paper](https://dl.acm.org/doi/pdf/10.1145/3539597.3573026), [Github](https://github.com/MarioTheOne/GRETEL)

### Causal Learning
- P. Sheth, et al., “Causebox: A causal inference toolbox for benchmarking treatment effect estimators with machine learning methods,” in Proc. 30th ACM Int. Conf. Inf. Knowl. Manage., 2021, pp. 4789–4793. [Paper](https://dl.acm.org/doi/pdf/10.1145/3459637.3481974), [Github](https://github.com/paras2612/CauseBox)
- Y. Zheng, et al., “Causal-learn: Causal discovery in python,” CoRR, vol. abs/2307.16405, 2023. [Paper](https://arxiv.org/pdf/2307.16405), [Github](https://github.com/py-why/causal-learn)
- A. Bizeul, et al., "3DIdentBox: A Toolbox for Identifiability Benchmarking," Int. Conf. Learn. Representations, 2023. [Paper](https://www.cclear.cc/2023/AcceptedDatasets/bizeul23a.pdf), [Github](https://github.com/alicebizeul/3DIdentBox)

## Causality-Inspired GNN Works

### Causal Reasoning on Graphs

#### Group-level Causal Effect Estimation

##### Frontdoor Adjustment
- Y. Wu, et al., “Deconfounding to explanation evaluation in graph neural networks,” CoRR, vol. abs/2201.08802, 2022.

##### Instrumental Variable
- H. Gao, et al., “Robust causal graph representation learning against confounding effects,” CoRR, vol. abs/2208.08584, 2022.

##### Stable Learning
- S. Fan, et al., “Debiased graph neural networks with agnostic label selection bias,” CoRR, vol. abs/2201.07708, 2022.
- S. Fan, et al., “Generalizing graph neural networks on out-of-distribution graphs,” CoRR, vol. abs/2111.10657, 2021.
- H. Li, et al., “OOD-GNN: outof-distribution generalized graph neural network,” IEEE Trans. Knowl. Data Eng., vol. 35, no. 7, pp. 7328–7340, 2023.


#### Individual-level Causal Effect Estimation

##### Intervention
- F. Feng, et al., “Should graph convolution trust neighbors? A simple causal inference method,” in Proc. 44th Int. ACM SIGIR Conf. Res. Develops. Inf. Retrieval, 2021, p. 1208–1218.
- H. Wang, et al., “Causalbased supervision of attention in graph neural network: A better and simpler choice towards powerful attention,” in Proc. 32nd Int. Joint Conf. Artif. Intell., 2023, pp. 2315–2323.
- K. Zhang, et al., “Rumor detection with diverse counterfactual evidence,” in Proc. 29th ACM SIGKDD Conf. Knowl. Discov. and Data Mining, 2023, pp. 3321–3331.
- C. Agarwal, et al., “Towards a unified framework for fair and stable graph representation learning,” in Proc. 37th Conf. Uncertainty in Artif. Intell., vol. 161, 2021, pp. 2114–2124.
- X. Zhang, et al., “A multi-view confidencecalibrated framework for fair and stable graph representation learning,” in 2021 IEEE Int. Conf. Data Mining, 2021, pp. 1493–1498.
- W. Lin, et al., “Generative causal explanations for graph neural networks,” in Proc. 38th Int. Conf. Mach. Learn., vol. 139, 2021, pp. 6666–6679.
- M. Bajaj, et al., “Robust counterfactual explanations on graph neural networks,” in Adv. Neural Inf. Process. Syst. 34, 2021, pp. 5644– 5655.

##### Matching
- T. Zhao, et al., “Learning from counterfactual links for link prediction,” in Proc. 39th Int. Conf. Mach. Learn., vol. 162. PMLR, 2022, pp. 26 911–26 926.

##### Deep Generative Modeling
- J. Ma, et al., “Learning fair node representations with graph counterfactual fairness,” in Proc. 15th ACM Int. Conf. Web Search and Data Mining, 2022, pp. 695–703.

### Causal Discovery on Graphs
Coming soon.

### Causal Representation Learning on Graphs
Coming soon.

<!-- Repeat the above format for each relevant work -->

## Contributing

We appreciate your kind contributions and suggestions! If you know of any causality-inspired GNN works that are not listed, feel free to [open an issue](https://github.com/usail-hkust/Causality-Inspired-GNNs/issues). We will update them in our survey and repository.

## License

This project is licensed under the MIT License.

## References

If you find our work useful for your research, please consider citing

```bibtex
@article{jiang2023survey,
  title={Survey on Trustworthy Graph Neural Networks: From A Causal Perspective},
  author={Jiang, Wenzhao and Liu, Hao and Xiong, Hui},
  journal={arXiv preprint arXiv:2312.12477},
  year={2023}
}
```
