# README #
This is the official code of the paper  'Delay learning based on temporal coding in Spiking Neural Networks'. 

@article{SUN2024106678,
title = {Delay learning based on temporal coding in Spiking Neural Networks},
journal = {Neural Networks},
pages = {106678},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106678},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024006026},
author = {Pengfei Sun and Jibin Wu and Malu Zhang and Paul Devos and Dick Botteldooren},
keywords = {Delay learning, Temporal coding, Spiking neural network, Supervised learning},
abstract = {Spiking Neural Networks (SNNs) hold great potential for mimicking the brain’s efficient processing of information. Although biological evidence suggests that precise spike timing is crucial for effective information encoding, contemporary SNN research mainly concentrates on adjusting connection weights. In this work, we introduce Delay Learning based on Temporal Coding (DLTC), an innovative approach that integrates delay learning with a temporal coding strategy to optimize spike timing in SNNs. DLTC utilizes a learnable delay shift, which assigns varying levels of importance to different informational elements. This is complemented by an adjustable threshold that regulates firing times, allowing for earlier or later neuron activation as needed. We have tested DLTC’s effectiveness in various contexts, including vision and auditory classification tasks, where it consistently outperformed traditional weight-only SNNs. The results indicate that DLTC achieves remarkable improvements in accuracy and computational efficiency, marking a step forward in advancing SNNs towards real-world applications. Our codes are accessible at https://github.com/sunpengfei1122/DLTC.}
}

We use Tensorpack to accelerate the training process. Below is an example using the Fashion-MNIST dataset, where you can achieve an accuracy of 89.59% with a two fully-connected layer model.

**How to run:**
- 1. Install the Tensorpack package from https://github.com/tensorpack/tensorpack
- 2. Run the example script:  python examples/fmnist/DLTC.py

# To do #
More examples will be added. 


