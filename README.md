# Machine Learning Models that Remember Too Much
This repo contains an example for attacks in the paper Machine Learning that Remember Too Much (https://arxiv.org/pdf/1709.07886.pdf). The example is based on CIFAR10 dataset.

### Train a malicious model
python train.py --attack ATTACK 

Available ATTACK are cap (Capacity abuse attack), cor (correlate value encoding attack) and sgn (sign encoding attack).

### Test attack quality 
python test_model --attack ATTACK
