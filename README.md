# [FCOS](https://arxiv.org/abs/1904.01355) with [VoVNet](https://arxiv.org/abs/1904.09730)(CVPRW'19) Efficient Backbone Networks

This repository contains [FCOS](https://arxiv.org/abs/1904.01355)(ICCV'19) with [VoVNet](https://arxiv.org/abs/1904.09730) (CVPRW'19) efficient backbone networks. This code based on pytorch imeplementation of [FCOS](https://github.com/tianzhi0549/FCOS) 

## Highlights

- Memory efficient 
- Better performance, especially for *small* object
- Faster speed



## Comparison with ResNet backbones

- same hyperparameters
- same training protocols ( max epoch, learning rate schedule, etc)
- 8 x TITAN Xp GPU
- pytorch1.1
- CUDA v9
- cuDNN v7.2


| Backbone  | Multi-scale training | Inference time (ms) | Box AP (AP/APs/APm/APl)  | DOWNLOAD |
|----------|:---------------:|:-------------------:|:------------------------:| :---:|
| R-50-FPN-1x       | No           | 84                  | 37.5/21.3/40.3/49.5      | - |
 **V-39**-FPN-1x       | No           |**82**                 | 37.7/**22.4**/41.8/48.4      | [link](https://dl.dropbox.com/s/8n0wyypfggliplw/FCOS-V-39-FPN-1x.pth?dl=1)|
 ||
| R-101-FPN-2x       | Yes           | 104                  | 41.3/25.0/45.5/53.0      | -                          
| **V-57**-FPN-2x        |Yes           | **91**                  | 41.6/**25.9**/45.6/53.1      |  [link](https://dl.dropbox.com/s/f1posfwebb2ynnp/FCOS-V-57-MS-FPN-2x.pth?dl=1)|
||
| R-101-32x8d-FPN-2x        | Yes           |171                   | 42.5/26.0/46.1/54.2      | -  |
| **V-93**-FPN-2x        | Yes           | **113**                  |   42.1/**26.2**/46.0/53.9    | [link](https://dl.dropbox.com/s/2v7go4lenvvjd1s/FCOS-V-93-MS-FPN-2x.pth?dl=1)|



## ImageNet pretrained weight

- [VoVNet-39](https://dl.dropbox.com/s/s7f4vyfybyc9qpr/vovnet39_statedict_norm.pth?dl=1)
- [VoVNet-57](https://dl.dropbox.com/s/b826phjle6kbamu/vovnet57_statedict_norm.pth?dl=1)
- [VoVNet-75](https://dl.dropbox.com/s/ve1h1ol2ge7yfta/vovnet75_statedict_norm.pth.tar?dl=1)
- [VoVNet-93](https://dl.dropbox.com/s/qtly316zv1isn0t/vovnet93_statedict_norm.pth.tar?dl=1)


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions which is orginate from [FCOS](https://github.com/tianzhi0549/FCOS#installation)



## Training
Follow [the instructions](https://github.com/tianzhi0549/FCOS#training)

For example,

```bash
# specify the number of GPU you can use.
export NGPUS=8 
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/vovnet/fcos_V_39_FPN_1x.yaml" 
```

## Evaluation
Follow [the instructions](https://github.com/tianzhi0549/FCOS#inference)

First of all, you have to download the weight file you want to inference.

For examaple,
##### multi-gpu evaluation & test batch size 16,

```bash
wget https://dl.dropbox.com/s/8n0wyypfggliplw/FCOS-V-39-FPN-1x.pth?dl=1
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/vovnet/fcos_V_39_FPN_1x.yaml" TEST.IMS_PER_BATCH 16 MODEL.WEIGHT FCOS-V-39-FPN-1x.pth
```

##### single-gpu evaluation & test batch size 1,

```bash
wget https://dl.dropbox.com/s/8n0wyypfggliplw/FCOS-V-39-FPN-1x.pth?dl=1
CUDA_VISIBLE_DEVICES=0
python tools/test_net.py --config-file "configs/vovnet/e2e_faster_rcnn_V_39_FPN_2x.yaml" TEST.IMS_PER_BATCH 1 MODEL.WEIGHT FCOS-V-39-FPN-1x.pth
```

## Related projects

- [VoVNet-Classification](https://github.com/vov-net/VoVNet-Detectron)
- [VoVNet-Detectron](https://github.com/vov-net/VoVNet-Classification)
