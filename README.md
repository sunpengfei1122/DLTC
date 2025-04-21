# README

This is the official code repository for the paper *"Delay Learning Based on Temporal Coding in Spiking Neural Networks"*.

**Paper Details:**

> - **Title:** Delay Learning Based on Temporal Coding in Spiking Neural Networks
> - **Authors:** Pengfei Sun, Jibin Wu, Malu Zhang, Paul Devos, Dick Botteldooren
> - **Journal:** Neural Networks
> - **Year:** 2024
> - **Pages:** 106678
> - **ISSN:** 0893-6080
> - **DOI:** [10.1016/j.neunet.2024.106678](https://doi.org/10.1016/j.neunet.2024.106678)
> - **URL:** [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0893608024006026)

## Abstract

Spiking Neural Networks (SNNs) offer great potential for mimicking the brain’s efficient information processing. While precise spike timing is known to be crucial for effective information encoding, current SNN research largely focuses on adjusting connection weights. This paper introduces Delay Learning based on Temporal Coding (DLTC), a novel approach that combines delay learning with temporal coding to optimize spike timing in SNNs. DLTC incorporates a learnable delay shift that assigns varying importance to different informational elements, alongside an adjustable threshold for regulating firing times. Tested in various vision and auditory classification tasks, DLTC consistently outperforms traditional weight-only SNNs, achieving significant improvements in accuracy and computational efficiency. 


## Requirements

```
- Tensorpack
```




## Installation

We use Tensorpack to accelerate the training process. Below is an example using the Fashion-MNIST dataset, where you can achieve an accuracy of 89.59% with a two fully-connected layer model. However, you can easily use TensorFlow if you are not familiar with Tensorpack, just replace the main training code.

**How to run:**
1. Install the Tensorpack package
2.  Run the example script 
```
- pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
- python examples/fmnist/DLTC.py
```


# Credits # 

The code for achieve the ttfs is based on (https://github.com/zbs881314/Temporal-Coded-Deep-SNN)
