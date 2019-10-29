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
| ResNet-101-FPN        | 180k      | [480, 960]      | Y                | 44.7            | 45.2             |            |
| ResNeXt-64x4d-101-FPN | 180k      | [640, 800]      | N                | 44.5            | 44.9             |            |
| ResNeXt-64x4d-101-FPN | 180k      | [480, 960]      | N                | 45.6            | 46.0             |            |
| ResNeXt-64x4d-101-FPN | 180k      | [480, 960]      | Y                | 46.8            | 47.3             |            |

**Notes:**

- We use 8 GPUs with 2 image / GPU. 
- In multi-scale testing, we use image scales in {480, 640, 800, 960, 1120, 1280} and max_size are 1.666&times; than scales. 


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

#### Training with 8 GPUs

```bash
cd path_to_free_anchor
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "path/to/config/file.yaml"
```

#### Test on COCO test-dev

```bash
cd path_to_free_anchor
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT "path/to/.pth file" DATASETS.TEST "('coco_test-dev',)"
```

#### Multi-scale testing

```bash
cd path_to_free_anchor
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/multi_scale_test.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT "path/to/.pth file" DATASETS.TEST "('coco_test-dev',)"
```

#### Evaluate NMS Recall

```bash
cd path_to_free_anchor
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/eval_NR.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT "path/to/.pth file"
```
## Citations
Please consider citing our paper in your publications if the project helps your research.
```
@inproceedings{zhang2019freeanchor,
  title   =  {{FreeAnchor}: Learning to Match Anchors for Visual Object Detection},
  author  =  {Zhang, Xiaosong and Wan, Fang and Liu, Chang and Ji, Rongrong and Ye, Qixiang},
  booktitle =  {Neural Information Processing Systems},
  year    =  {2019}
}
```

