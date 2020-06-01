---
typora-root-url: results\picture
---

### 1.实验部分

#### 1.1 目录结构

/src 源代码所在位置

​		/src/main.py 主体的训练，测试代码

​		/src/main_nlp.py 语言建模的训练，测试代码

​		/src/constants.py 常量

​		/src/plot.py 结果可视化相关代码

​		/src/algorithm 我们自己实现的RAdam和lookahead

​		/src/data 几种不同数据集的数据读取和预处理位置

​		/src/models 几种不同问题对应模型的读取位置

/results 实验结果所在位置

​		/results/text 实验结果的txt文档

​		/results/runs 实验结果的tensorboard可视化文件

​		/results/picture 实验图片结果

/data 实验数据所在位置

​		/data/cifar-10-batches-py  cifar10数据

​		/data/cifar-100-python  cifar100数据

​		/data/cnn_data  作业2的部分ImageNet图片数据

​		/data/ptb  作业3的语言建模数据

​		IMDB数据会自动下载

#### 1.2 实验环境

- ubuntu 18.04系统
- python 3.6.9
- pytorch 1.4.0
- numpy 1.18.4
- torchvision 0.5.0
- tensorboard 1.14.0
- keras 2.3.1 

#### 1.3 基本实验

我们对RAdam和Ranger(RAdam + lookahead)在cifar10，cifar100的效果进行了对比。两者RAdam的超参数和网络其他情况完全一致。

##### 超参数等

- 网络：Resnet18，从零开始训练
- 学习率：初始0.1
- 学习率衰减：在60,120,160epoch的时候下降至原来的0.2倍
- weight decay：5e-4
- lookahead步长k：5
- lookahead内部学习率alpha：0.8

##### 实验结果

![cifar10_base_loss](/cifar10_base_loss.png)

在cifar10下，比较两者训练时的损失函数，使用lookahead的Ranger算法明显比只有RAdam收敛的快。

![cifar10_base_acc](/cifar10_base_acc.png)

而对比两者收敛后的准确率，并没有明显变化。

![cifar100_base_loss](/cifar100_base_loss.png)

而在cifar100中，使用lookahead的收敛更加快速，效果更加明显。

![cifar100_base_acc](/cifar100_base_acc.png)

而且因为收敛效果好，准确率也提高了。

可以看出，lookahead可以对RAdam的收敛速度进行优化，而且不影响，甚至有利于提高准确率。

##### 实验复现方法

(注：以下实验均运行在src目录下)

cifar10：

Ranger(RAdam + lookahead):

```shell
python main.py
```

RAdam:

```shell
python main.py --lookahead 0
```

cifar100:

Ranger(RAdam + lookahead):

```shell
python main.py --dataset cifar100
```

RAdam:

```shell
python main.py --dataset cifar100 --lookahead 0
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

cifar10的train loss对比

Ranger(RAdam + lookahead):

```shell
tensorboard --logdir runs/000
```

cifar10的test accuracy对比

```shell
tensorboard --logdir runs/001
```

cifar100的train loss对比

```shell
tensorboard --logdir runs/100
```

cifar100的test accuracy对比

```shell
tensorboard --logdir runs/101
```



#### 1.4 参数鲁棒性实验

因为lookahead有两个超参数：内部步数k和内部学习率alpha，为了验证lookahead对RAdam优化的鲁棒性，我们对于几组不同的k和内部学习率alpha进行了实验。

##### 实验结果

![cifar10_param_loss](/../../../DL%20Optimizer%20Research/results/picture/cifar10_param_loss.png)

可以看出，在cifar10下，无论是哪组超参数，其优化速率都要快于没有lookahead。

![cifar10_param_acc](/../../../DL%20Optimizer%20Research/results/picture/cifar10_param_acc.png)

而无论是哪组超参数，都不会造成准确率的下降。

![](/../../../DL%20Optimizer%20Research/results/picture/cifar100_param_loss.png)

在cifar100下，结论也是如此，而且优化效果更加明显。

![cifar100_batch_acc](/../../../DL%20Optimizer%20Research/results/picture/cifar100_param_acc.png)

而且，几种超参数都能带来准确率的明显提升。

##### 实验复现方法

(注：以下实验均运行在src目录下)

cifar10：

Ranger（标准超参数 alpha = 0.8, k = 5)

```shell
python main.py
```

Ranger（alpha = 0.5)

```shell
python main.py --lookahead_lr 0.5
```

Ranger（k = 10)

```shell
python main.py --lookahead_steps 10
```

RAdam:

```shell
python main.py --lookahead 0
```

cifar100:

Ranger（标准超参数 alpha = 0.8, k = 5)

```shell
python main.py --dataset cifar100
```

Ranger（alpha = 0.5)

```shell
python main.py --lookahead_lr 0.5 --dataset cifar100
```

Ranger（k = 10)

```shell
python main.py --lookahead_steps 10 --dataset cifar100
```

RAdam:

```shell
python main.py --lookahead 0 --dataset cifar100
```

###### 可视化复现方法

（注：以下可视化均运行在results目录下）

cifar10的train loss对比

```shell
tensorboard --logdir runs/030
```

cifar10的test accuracy对比

```shell
tensorboard --logdir runs/031
```

cifar100的train loss对比

```shell
tensorboard --logdir runs/130
```

cifar100的test accuracy对比

```shell
tensorboard --logdir runs/131
```



#### 1.5 超参数修改实验

为了探究在什么条件下lookahead能有效优化RAdam，我们选取了不同的学习率，学习率衰减方式作为条件，对比RAdam与Ranger的性能。

##### 1.5.1 学习率变化

首先，我们控制其他不变，把学习率修改为0.01。在cifar10下，两者train loss变化如下：

![cifar10_001_loss](/cifar100_001_loss.png)

此时，lookahead并没有对RAdam进行明显的优化。

![cifar10_001_acc](/cifar10_001_acc.png)

而对比两者收敛后的准确率，并没有明显变化。

![cifar100_001_loss](/cifar100_001_loss.png)

而在cifar100下，lookahead对RAdam的优化就更明显一点

![cifar100_001_acc](/cifar100_001_acc.png)

而且，在准确率上，使用lookahead还能带来一些提高。

##### 实验复现方法

(注：以下实验均运行在src目录下)

cifar10：

Ranger(RAdam + lookahead):

```shell
python main.py --learning_rate 0.01
```

RAdam:

```shell
python main.py --learning_rate 0.01 --lookahead 0
```

cifar100:

Ranger(RAdam + lookahead):

```shell
python main.py --learning_rate 0.01 --dataset cifar100
```

RAdam:

```shell
python main.py --learning_rate 0.01 --dataset cifar100 --lookahead 0
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

cifar10的train loss对比

```shell
tensorboard --logdir runs/010
```

cifar10的test accuracy对比

```shell
tensorboard --logdir runs/011
```

cifar100的train loss对比

```shell
![cifar10_fast_loss](/cifar10_fast_loss.png)tensorboard --logdir runs/110
```

cifar100的test accuracy对比

```shell
tensorboard --logdir runs/111
```

##### 1.5.2 学习率衰减方式变化

之后，我们控制其他不变，把学习率衰减方式修改为快速衰减---在30,48,58 epoch的时候下降至原来的0.2倍。在cifar10下，两者train loss变化如下：

![cifar10_fast_loss](/cifar10_fast_loss.png)

此时，使用lookahead的效果反而还差一些

![cifar10_fast_acc](/../../../DL%20Optimizer%20Research/results/picture/cifar10_fast_acc.png)

即使两者准确率没有明显变化

![cifar100_fast_loss](/../../../DL%20Optimizer%20Research/results/picture/cifar100_fast_loss.png)

而在cifar100下，使用lookahead就仍旧能明显改进RAdam的性能。

![cifar100_fast_acc](/../../../DL%20Optimizer%20Research/results/picture/cifar100_fast_acc.png)

而且使用lookahead在准确率上也能带来一些提高。

##### 实验复现方法

(注：以下实验均运行在src目录下)

cifar10：

Ranger(RAdam + lookahead):

```shell
python main.py --lr_decay fast
```

RAdam:

```shell
python main.py --lr_decay fast --lookahead 0
```

cifar100:

Ranger(RAdam + lookahead):

```shell
python main.py --lr_decay fast --dataset cifar100
```

RAdam:

```shell
python main.py --lr_decay fast --dataset cifar100 --lookahead 0
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

cifar10的train loss对比

```shell
tensorboard --logdir runs/020
```

cifar10的test accuracy对比

```shell
tensorboard --logdir runs/021
```

cifar100的train loss对比

```shell
tensorboard --logdir runs/120
```

cifar100的test accuracy对比

```shell
tensorboard --logdir runs/121
```

##### 

#### 1.6 对比实验

既然Ranger优化器是lookahead和RAdam组合而成的，而且lookahead也可以组合SGD,Adam等其他优化器，因此，我们做了相应的对比实验。

##### 超参数等

- SGD学习率 0.1
- Adam学习率 0.001
- RAdam学习率 0.1
- 其余超参数同上，且都保持一致

![cifar10_ablation_loss](/cifar10_ablation_loss.png)

首先，对比cifar10下的train loss，可以看出，lookahead的确能优化Adam和RAdam，但是对SGD的优化并不明显。而且，在此问题下，SGD > Adam > RAdam。

![cifar10_ablation_acc](/cifar10_ablation_acc.png)

而对比此时的accuracy，差异并不明显，使用lookahead并不会降低准确率。

![cifar100_ablation_loss](/cifar100_ablation_loss.png)

而在cifar100下，lookahead能同时加速这三个优化算法，而且即使SGD算法本身效果并不好，使用lookahead后效果也非常好。

![cifar100_ablation_acc](/cifar100_ablation_acc.png)

此时，对于Adam，使用lookahead不会降低准确率，而其他两种算法的lookahead都能提升准确率。

##### 实验复现方法

(注：以下实验均运行在src目录下)

cifar10：

Ranger(RAdam + lookahead):

```shell
python main.py
```

RAdam:

```shell
python main.py --lookahead 0
```

Adam + lookahead:

```shell
python main.py --algorithm Adam
```

Adam:

```shell
python main.py --algorithm Adam --lookahead 0
```

SGD + lookahead:

```shell
python main.py --algorithm SGD
```

SGD:

```shell
python main.py --algorithm SGD --lookahead 0
```

cifar100:

Ranger(RAdam + lookahead):

```shell
python main.py --dataset cifar100
```

RAdam:

```shell
python main.py --lookahead 0 --dataset cifar100
```

Adam + lookahead:

```shell
python main.py --algorithm Adam --dataset cifar100
```

Adam:

```shell
python main.py --algorithm Adam --lookahead 0 --dataset cifar100
```

SGD + lookahead:

```shell
python main.py --algorithm SGD --dataset cifar100
```

SGD:

```shell
python main.py --algorithm SGD --lookahead 0 --dataset cifar100
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

cifar10的train loss对比

```shell
tensorboard --logdir runs/040
```

cifar10的test accuracy对比

```shell
tensorboard --logdir runs/041
```

cifar100的train loss对比

```shell
tensorboard --logdir runs/140
```

cifar100的test accuracy对比

```shell
tensorboard --logdir runs/141
```



#### 1.7 Batch与步数k的组合实验

//TODO:设计原因--这个让wxs好好说一下

我们选取了三组超参数：

- 步长5 + batch128
- 步长2 + batch320
- 步长10 + batch64

##### 实验结果

//TODO 解释

![cifar10_batch_loss](/cifar10_batch_loss.png)

![cifar10_batch_acc](/cifar10_batch_acc.png)

![cifar100_batch_loss](/cifar100_batch_loss.png)

![cifar100_batch_acc](/cifar100_batch_acc.png)

##### 实验复现方法

(注：以下实验均运行在src目录下)

cifar10：

步长5 + batch128

```shell
python main.py
```

步长2 + batch320

```shell 
python main.py --lookahead_steps 2 --batch_size 320
```

步长10 + batch64

```shell 
python main.py --lookahead_steps 10 --batch_size 64
```

cifar100:

步长5 + batch128

```shell
python main.py --dataset cifar100
```

步长2 + batch320

```shell 
python main.py --lookahead_steps 2 --batch_size 320 --dataset cifar100
```

步长10 + batch64

```shell 
python main.py --lookahead_steps 10 --batch_size 64 --dataset cifar100
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

cifar10的train loss对比

```shell
tensorboard --logdir runs/050
```

cifar10的test accuracy对比

```shell
tensorboard --logdir runs/051
```

cifar100的train loss对比

```shell
tensorboard --logdir runs/150
```

cifar100的test accuracy对比

```shell
tensorboard --logdir runs/151
```



#### 1.8 不同任务的实验

因为仅仅在Resnet18 + cifar10/100下验证是不够充分的，因此，我们选择了作业2,3的网络和数据集，以及用LSTM网络进行imdb情感分类的任务进行综合验证。

##### 1.8.1 作业2的CNN图片分类

##### 超参数

- 三层CNN网络
- 学习率0.0001，不进行衰减
- 内部学习率alpha = 0.8
- 内部步长k = 5

##### 实验结果

![hw2_loss](/hw2_loss.png)

此时，lookahead并没有有效优化RAdam

![hw2_acc](/hw2_acc.png)

而准确率两者差不多

##### 实验复现方法

(注：以下实验均运行在src目录下)

Ranger

```shell
python main.py --dataset self_cnn
```

RAdam

```shell 
python main.py --dataset self_cnn --lookahead 0
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

train loss对比

```shell
tensorboard --logdir runs/400
```

test accuracy对比

```shell
tensorboard --logdir runs/401
```

##### 1.8.2 作业3的自然语言建模

##### 超参数

- 2层单向GRU网络
- 学习率0.001，不进行衰减
- 内部学习率alpha = 0.8
- 内部步长k = 5

##### 实验结果

![nlp_train](/../../../DL%20Optimizer%20Research/results/picture/nlp_train.png)

此时，lookahead并没有有效优化RAdam

![hw2_acc](/../../../DL%20Optimizer%20Research/results/picture/nlp_test.png)

而结果两者差不多

##### 实验复现方法

(注：以下实验均运行在src目录下)

Ranger

```shell
python main_nlp.py
```

RAdam

```shell 
python main_nlp.py --lookahead 0
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

train loss对比

```shell
tensorboard --logdir runs/200
```

test accuracy对比

```shell
tensorboard --logdir runs/201
```



##### 1.8.3 LSTM进行IMDB数据集的情感分类

##### 超参数

- 2层双向LSTM网络
- 学习率0.1，不进行衰减
- 内部学习率alpha = 0.8
- 内部步长k = 5

##### 实验结果

![nlp_train](/imdb_loss.png)

此时，lookahead并没有有效优化RAdam

![hw2_acc](/imdb_acc.png)

而结果两者差不多

##### 实验复现方法

(注：以下实验均运行在src目录下)

Ranger

```shell
python main.py --dataset imdb
```

RAdam

```shell 
python main.py --lookahead 0 --dataset imdb
```

##### 可视化复现方法

（注：以下可视化均运行在results目录下）

train loss对比

```shell
tensorboard --logdir runs/300
```

test accuracy对比

```shell
tensorboard --logdir runs/301
```



##### 参考代码

lookahead代码：https://github.com/michaelrzhang/lookahead
cifar10/100：https://github.com/uoguelph-mlrg/Cutout/blob/master/train.py
IMDB情感分类： https://github.com/Cong-Huang/Pytorch-imdb-classification

