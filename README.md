## FreeAnchor

The Code for "FreeAnchor: Learning to Match Anchors for Visual Object Detection", available at [https://arxiv.org/abs/](https://arxiv.org/abs/). 

## Installation 
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Perform Training on COCO dataset
You will need to download the COCO dataset and configure your own paths to the datasets.

For that, all you need to do is to modify `maskrcnn_benchmark/config/paths_catalog.py` to point to the location where your dataset is stored.

#### Config Files
We provide four configuration files in the configs directory.

| Backbone | Iter. | Setting | Config File |  
| :-----: | :---: | :---: | :----------: |
| ResNet-50-FPN    |   90k |   std.  | configs/free_anchor_R-50-FPN_1x.yaml      | 
| ResNet-101-FPN   |   90k |   std.  | configs/free_anchor_R-101-FPN_1x.yaml     | 
| ResNet-101-FPN   |  135k |   std.  | configs/free_anchor_R-101-FPN_1.5x.yaml   | 
| ResNeXt-101-FPN  |  135k |   dev.  | configs/free_anchor_X-101-FPN_j1.5x.yaml  | 


#### 4 GPU (32GB memory) Training

```bash
cd path_to_free_anchor
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "path/to/config/file.yaml"
```

#### 8 GPU (>10GB memory) Training

```bash
cd path_to_free_anchor
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "path/to/config/file.yaml"
```

#### Test on MS-COCO test-dev

```bash
cd path_to_free_anchor
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT "path/to/.pth file" DATASETS.TEST "('coco_test-dev',)"
```

#### Evaluate NMS Recall

```bash
cd path_to_free_anchor
python  -m torch.distributed.launch --nproc_per_node=$NGPUS tools/eval_NR.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT "path/to/.pth file"
```
