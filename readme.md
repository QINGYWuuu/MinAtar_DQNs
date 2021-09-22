This project is the simple pytorch-based implementation of Q-Learning based methods.
The code is the modification of the official implementation of [MinAtar](https://github.com/kenjyoung/MinAtar).
## Methods

0. Basic Implementation 
   
   0.1 [DQN](http://www.nature.com/articles/nature14236)
  
1. For reducing estimation bias

   1.1. [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)

   1.2. [Averaged DQN, Ensemble DQN](https://arxiv.org/pdf/1611.01929.pdf)
    
   1.3. [Maxmin DQN](https://arxiv.org/pdf/2002.06487.pdf)
    
   1.4  [Ensemble Bootstrapping DQN](https://arxiv.org/pdf/2103.00445.pdf)

2. For improving exploration ability
  
   2.1. [Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)
    
   2.2. [Ensemble Voting DQN, UCB Exploration with Q-Ensembles](https://arxiv.org/pdf/1706.01502.pdf)
   
   2.3. [Sunrise DQN](https://arxiv.org/pdf/2007.04938.pdf)
    

## Environment

   1. [MinAtar](https://github.com/kenjyoung/MinAtar) is a miniaturized environment which includes 5 games of Atari 2600 (asterix, breakout, freeway, seaquest, space_invaders). [[Paper](https://arxiv.org/pdf/1903.03176)]

## citation

```
@misc{SimpleImplementation,
author = {Qingyuan, WU},
title = {A simple implementation of Q-learning based RL methods},
year = {2021},
publisher = {GitHub},
journal = {GitHub Repository},
howpublished = {\url{https://github.com/QINGYWuuu/The-Simple-Implementation-of-Q-Learning-Based-RL}}
}
```