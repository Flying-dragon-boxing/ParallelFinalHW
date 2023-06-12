# 并行程序设计 期末大作业

## 概述

程序采用了`MPI`和`OpenMP`的并行技术。设置了处理分布函数的类`dist`，存储点信息的库`mesh`，存储能量函数的库`venergy`。对于每个工作的进程都存储了`venergy`和所有点的`mesh`的一个副本。程序按照计算得到的矩阵元（仅计算上三角）划分进程，在每个进程里，积分的循环用多线程加速。

## 在Bohrium上构建与运行

在Bohrium的平台的容器`c32_m128_cpu`上运行`ubuntu:22.04-py3.10`镜像。需要使用软件包管理器`apt`安装各依赖。
    
```bash
apt install cmake gcc gfortran
apt install libopenmpi-dev libomp-dev
apt install libblas-dev liblapack-dev liblapacke-dev
apt install libscalapack-openmpi-dev
```


## 程序中使用的优化

* **根据分布函数的截断半径，缩小积分区域**
分布函数截断半径远小于积分的范围。我们可以把积分范围缩小为以被选中两点中一个点为中心，长宽高都是截断半径的一个正方体。选中的点和这个正方体外任意一点的距离都大于分布函数的截断半径，被积函数在这些点上的值为0。于是可以只对正方体内部积分。这大大减少了计算量。

* **增加缓存，减少频繁计算**
对于可用内存比较大的情况（具体来说，是可用内存可以至少存储两份`venergy`），循环会存储一份选中的第一个点在格点上的分布，以减少重复计算。

* 动态分配MPI进程和OpenMP线程
在分配了过多进程而内存不足以存储所有`venergy`的情况下，程序会自动增加线程数以充分利用计算资源。
