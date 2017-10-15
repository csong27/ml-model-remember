# Machine Learning Models that Remember Too Much (in CCS 2017)
The code contains an example for CIFAR10 dataset.

### Train a malicious model
python train.py --attack ATTACK 

Available ATTACK are cap (Capacity abuse attack), cor (correlate value encoding attack) and sgn (sign encoding attack).

### Test attack quality 
python test_model --attack ATTACK
