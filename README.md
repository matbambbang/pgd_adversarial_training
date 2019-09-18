# Pytorch Implementation of Adversarial Training with Projected Gradient Descent and Fast Gradient Sign Methods
This reopository is made by Byungjoo Kim, Korea University.

## Installation
1. Create your virtual environments ('conda' recommended).

```conda create -n ${envname} python=3.6```

Activate the virtual environment.
```source activate ssp```

2. Install require packages, this codes require `PyTorch 1.1`, `overrides`, `tqdm`.
For PyTorch, follow [this website](https://pytorch.org/)(note that you must install PyTorch>=1.1).

Make sure that this repository uses Tensorboard, you should install `tensorflow`.
In my machine, I used CUDA 9.0, so I installed `tensorflow 1.9.0`.

```
pip install tensorflow-gpu==1.9.0
pip install overrides
pip install https://github.com/bethgelab/foolbox/archive/master.zip
pip install tqdm
pip install tb-nightly
pip install future
```

3. Now, clone this repository.
```
git clone https://github.com/matbambbang/pgd_adversarial_training.git
```

This would make the repository named `pgd_adversarial_training`.
Move to this repository, and follow below descriptions.

## Description

This module is build for [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html).
For each task, I implemented different networks.
Please see `model/mnist.py` and `model/cifar10.py`.

## Training
**I would upload the sample trained model soon. The below descriptions are for running on scratch.**

For train the model, use `cifar10_train.py`.

```
python mnist_train.py --model ${model} --block ${block} --save ${save_path} --norm ${norm} --tbsize ${batch_size} --adv ${adv}
python cifar10_train.py --model ${model} --block ${block} --save ${save_path} --norm ${norm} --tbsize ${batch_size} --adv ${adv}
```

When train network while each groups are composed of  6 residual blocks,
```
python mnist_train.py --model res --block 6 --save res_mnist_6
python cifar10_train.py --model res --block 6 --save res_cifar_6
```

In the table below, we describe the arguments that you can control:

| args | Valid arguments |
|:----:|:---------------:|
| `model` | `res`(default), `wres`, `conv`, in MNIST, `wres` is not available |
| `block` | `6`(default), you can use any integer values. In CIFAR10, this argument only effects on `res` or `wres` |
| `epochs` | `200`(default) , in MNIST, 100 is default |
| `lr` | `0.1`(default), `0.001` or `0.0001` recommended when you use `adam` optimizer |
| `decay` | `0.0005`(default), not available when using `adam` optimizer |
| `opt` | `sgd`(default) or `adam`, when using GroupNorm, `adam` is recommended |
| `norm` | `b`(BatchNorm, default), `g` |
| `tbsize` | `128`(default), you can use any integer values |
| `adv` | `none`(default), for adversarial training, use `fgsm`, `pgd`, or `ball` |
| `save` | identify the folder name in this arguments, I recommend you the name should be the combination of `model`, `block` and `norm`, i.e., `res_6_b` |

After training your model, the models and logs are saved in `experiments/${save_path}`.

## Attack the trained model

After the model is trained, you can attack your model with attack algorithms.
This repository supports `l_inf` bounded attack, with using sign of gradients.

Valid attacks : `fgsm`, `pgd`.

```
python attack_test.py --model ${model} --eval ${eval} --attack ${attack} --eps ${eps} --load ${load} --norm ${norm}
```

| args | Valid arguments |
|:----:|:---------------:|
| `attack` | `fgsm`(default), `bim`, `mim`, `pgd`, `ball` |
| `eval` | `cifar10`(default), `mnist` |
| `eps` | `8.0`(default), any integer pixel intensity values. When the model is for MNIST classification, `eps` should |
| `load` | Identify the folder name which includes target model |

