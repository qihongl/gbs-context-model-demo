# gbs-context-model

A conceptual replication of [1] with a feedforward neural network and a GRU. The feedforward network was tested on a context dependented classification task. 

## my conda env 
The env.txt was made with the following cmd:

$ conda list -e > env.txt

you can install my conda env with:

$ conda create -n <environment-name> --file env.txt

as you can see from env.txt, I'm using python 3.9, which is recommended. The key dependencies are torch, sklearn, numpy, pandas, scipy, matplotlib, seaborn. if you want to install the stuff on your own, it might be important to match the version of these packages.

## references 
[1] Hummos, A. (2022). Thalamus: a brain-inspired algorithm for biologically-plausible continual learning and disentangled representations. In arXiv [cs.AI]. arXiv. http://arxiv.org/abs/2205.11713 
