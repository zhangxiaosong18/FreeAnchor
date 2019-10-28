## FreeAnchor

The Code for ["FreeAnchor: Learning to Match Anchors for Visual Object Detection"](https://arxiv.org/abs/1909.02466).

This repo is based on maskrcnn-benchmark, and FreeAnchor has also been implemented in mmdetection\[[link](https://github.com/yhcao6/mmdetection/tree/free-anchor-ret/configs/free_anchor)\], thanks [@yhcao6](https://github.com/yhcao6).

![architecture](architecture.png)

#### New performance on COCO
We added multi-scale testing support and updated experiments. The previous version is in [this branch](https://github.com/zhangxiaosong18/FreeAnchor/tree/previous). 

| Backbone        | Iteration | Training scales | Multi-scale<br>testing | AP<br>(minival) | AP<br>(test-dev) | model link |
| :-------------------: | :-------: | :-------------: | :--------------: | :-------------: | :--------------: | :--------: |
| ResNet-50-FPN         | 90k       | 800             | N                | 38.7            | 38.7             |            |
| ResNet-101-FPN        | 90k       | 800             | N                | 40.5            | 40.9             |            |
| ResNet-101-FPN        | 180k      | [640, 800]      | N                | 42.7            | 43.1             |            |
| ResNet-101-FPN        | 180k      | [480, 960]      | N                | 43.2            | 43.9             |            |
| ResNet-101-FPN        | 180k      | [480, 960]      | Y                | -               | 45.2             |            |
| ResNeXt-64x4d-101-FPN | 180k      | [640, 800]      | N                | 44.5            | 44.9             |            |
| ResNeXt-64x4d-101-FPN | 180k      | [480, 960]      | N                | 45.6            | 46.0             |            |
| ResNeXt-64x4d-101-FPN | 180k      | [480, 960]      | Y                | -               | 47.3             |            |

## Installation 
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Usage
You will need to download the COCO dataset and configure your own paths to the datasets.

For that, all you need to do is to modify `maskrcnn_benchmark/config/paths_catalog.py` to point to the location where your dataset is stored.

#### Config Files
We provide four configuration files in the configs directory.

| Config File                               | Backbone                | Iteration | Training scales |
| :---------------------------------------: | :---------------------: | :-------: | :-------------: |
| configs/free_anchor_R-50-FPN_1x.yaml      | ResNet-50-FPN           | 90k       | 800             | 
| configs/free_anchor_R-101-FPN_1x.yaml     | ResNet-101-FPN          | 90k       | 800             |
| configs/free_anchor_R-101-FPN_j2x.yaml    | ResNet-101-FPN          | 180k      | [640, 800]      |
| configs/free_anchor_X-101-FPN_j2x.yaml    | ResNeXt-64x4d-101-FPN   | 180k      | [640, 800]      |
| configs/free_anchor_R-101-FPN_e2x.yaml    | ResNet-101-FPN          | 180k      | [480, 960]      |
| configs/free_anchor_X-101-FPN_e2x.yaml    | ResNeXt-64x4d-101-FPN   | 180k      | [480, 960]      |


