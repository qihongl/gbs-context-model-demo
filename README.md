# gbs-context-model

A implementation of act space gradient descent a feedforward neural network and a GRU. 
The feedforward network was tested on a context-dependent classification task. I haven't had enough time to move the tests for context-dependent GRU to here yet... 

## my conda env 
I'm using python 3.9, which is recommended. The key dependencies are torch, sklearn, numpy, pandas, scipy, matplotlib, seaborn. if you want to install the stuff on your own, it might be important to match the version of these packages.

## references 
[1] Hummos, A. (2022). Thalamus: a brain-inspired algorithm for biologically-plausible continual learning and disentangled representations. In arXiv [cs.AI]. arXiv. http://arxiv.org/abs/2205.11713 

[2] Giallanza, T., Campbell, D., Cohen, J. D., & Rogers, T. T. (2023). An Integrated Model of Semantics and Control. PsyArXiv. https://doi.org/10.31234/osf.io/jq7ta

[3] Rogers, T. T., & McClelland, J. L. (2004). Semantic cognition: A parallel distributed processing approach. MIT Press. https://psycnet.apa.org/fulltext/2004-18753-000.pdf 
