This project is the pytorch-based implementation of DQN-based methods for MinAtar.

## Implemented Methods


### Rainbow
1. [x] [DQN](http://www.nature.com/articles/nature14236)
2. [x] [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
3. [x] [Dueling DQN](http://arxiv.org/abs/1511.06581)
4. [x] [Noisy DQN](https://arxiv.org/abs/1706.10295)
5. [x] [Multi-step DQN](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)
6. [x] [Piorized Experience Replay](http://arxiv.org/abs/1511.05952)
7. [x] [Distributional DQN](https://arxiv.org/abs/1707.06887)
8. [x] [Rainbow](https://arxiv.org/pdf/1710.02298.pdf)
### Ensemble DQN (in Multi-head Network style)
9. [x] [Averaged DQN, Ensemble DQN](https://arxiv.org/pdf/1611.01929.pdf) 
10. [x] [Maxmin DQN](https://arxiv.org/pdf/2002.06487.pdf)
11. [x] [Ensemble Bootstrapping DQN](https://arxiv.org/pdf/2103.00445.pdf)
12. [x] [Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)
13. [x] [Ensemble Voting DQN, UCB Exploration with Q-Ensembles](https://arxiv.org/pdf/1706.01502.pdf)
14. [x] [Sunrise DQN](https://arxiv.org/pdf/2007.04938.pdf)

## Implementation Structure
![DQN_Agent Structrue](pics/structure.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/10)
    


## Environment

[MinAtar](https://github.com/kenjyoung/MinAtar) is a miniaturized environment which includes 5 games of Atari 2600 (asterix, breakout, freeway, seaquest, space_invaders). [[Paper](https://arxiv.org/pdf/1903.03176)]

## Results
![Asterix](pics/asterix.png#pic_left)
todo

## Citation

```
@misc{MinAtar-torch-dqns,
author = {Qingyuan, WU},
title = {A pytorch-based implementation of DQN-based RL methods for MinAtar},
year = {2021},
publisher = {GitHub},
journal = {GitHub Repository},
howpublished = {\url{https://github.com/QINGYWuuu/The-Simple-Implementation-of-Q-Learning-Based-RL}}
}
```
