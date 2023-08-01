## Distributed Framework Optimization - Data Hack Summit 2023

Deep Learning Frameworks form the baseline over which millions of models (LLMs, multimodals , auto regressive) are being compiled and built on.
Many of these frameworks require sophisticated optimization to make models train and infer faster in constrained hardware chips. The intrinsic kernels which form a part of these Frameworks (such as Pytorch) leverage profound adaptive features to help break perf- benchmarks in supercomputing and federated deep learning . This is a glimpse of different sub-kernel, intermediate framework and superficial model optimization techniques which help people run large models such as GPTs on constrained environments and clusters.


![image1](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/fdfc9d05-edca-4794-bb4f-c55575fea960)

### Pytorch

Most of the session would revolve around different model optimizations strategies and how the Pytorch framework can make training and finetuning efficient. This would involve features such as aten Graph Capture ,Lowering , Composite Graph Compilation ( by Inductor) followed by device specific IR which the device compiler can optimize further for model performance.

![image](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/c675f3dc-2c0e-45c3-bed5-1de82ef76d55)


### Distributed Pytorch

To extend different parallelisms over a dedicated set of hardware device combinations (CPU-GPU,GPU-GPU,multi XPU,multi TPU ,MPS) the distributed backend of pytorch comes into picture. It enables scale up and out of sharded models , data and parameters to efficiently distribute gradients, checkpoints, activations across different devices.

![image](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/67381a06-3f1c-4ca2-b5d4-5cdc738dd556)

### Data ,Model and Pipeline Parallelism

In data parallel training, the dataset is split into several shards, each shard is allocated to a device. This is equivalent to parallelize the training process along the batch dimension.

![image](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/cabb12c4-0497-4186-b3da-56b697e29e2b)

Model Parallelism involves sharding model blocks (not separate tensor lists) across devices in a uniform manner.

![image](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/eccc28c2-1c5d-44a1-92bd-2e080eaae557)

Pipeline parallelism splits  the model layer by layer into several chunks, each chunk is given to a device. The caveat here includes a single optimizer.step forces forward (increasing pipeline stages) and backward (decreasing pipeline stages) in an interleaved manner .

![image](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/a237d352-bf80-4d03-8ad6-df2537110daa)


### Deepspeed Zero

ZeRO leverages the aggregate computation and memory resources of data parallelism to reduce the memory and compute requirements of each device (GPU) used for model training. ZeRO reduces the memory consumption of each GPU by partitioning the various model training states (weights, gradients, and optimizer states) across the available devices (GPUs and CPUs) in the distributed training hardware.

![image](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/155f415b-1690-477b-9ba2-d675dae45a37)


### Triton Compiler

Triton is a deep learning compiler created specifically to abstract IR code and optimize kernels which would otherwise be difficult to optimiz in cuda. 

![image](https://github.com/abhilash1910/Framework-Optimization/assets/30946547/fc557aa1-5792-49d8-9f2b-f354a5d925c4)




