# Working with scale: 2nd place solution to Product Detection in Densely Packed Scenes

## Introduction
This repository contains code for the 2nd place solution of the detection challenge which is held within CVPR 2020 Retail-Vision workshop.
For more information see my [report](https://arxiv.org/abs/2006.07825). For all the experiments [MMDetection v1](https://github.com/open-mmlab/mmdetection/tree/v1.0.0) was used.

## Dataset
The dataset has been originally announced by [Eran Goldman et. al](https://arxiv.org/abs/1904.00853). 
In order to obtain the dataset for research purpose, please concat the authors.

## Getting started

For evaluation purpose please clone pycocotools, change the parameter `maxDets` to 300 [here](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L427) and then install locally.

#### 1. Convert SKU110k csv format to COCO-like json

```
python sku110k_scripts/sku110k_to_coco.py --args
```

#### 2. Convert a full frame COCO-like dataset to a tiled one

```
python sku110k_scripts/split_on_tiles.py --args
```

#### 3. Training with mmdet

```
./tools/dist_train configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_1x_4tiles_test_half_res.py 2
```

#### 4. Testing with mmdet

```
./tools/dist_test configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_1x_4tiles_test_half_res.py workdir/faster_rcnn_r50_fpn_anchor_1x_4tiles/latest.pth 2 --eval bbox
```

#### 5. Create a dummy json file for the leaderboard-test 

```
python sku110k_scripts/lb_test_to_coco.py --args
```

#### 6. Inferencing with mmdet

```
./tools/dist_test configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_1x_4tiles_test_half_res.py workdir/faster_rcnn_r50_fpn_anchor_1x_4tiles/latest.pth 2 --format_only --options "jsonfile_prefix=./submit"
```

#### 7. Convert json output back to SKU110k csv format

```
python sku110k_scripts/json_out_to_submit.py --args
```

## Experiments

### 1. Initial experiments

| Config                                                                | Backbone   | Lr schd | Base lr  | imgs_p_gpu | img_scale  | anchor_sc  | mAP   | AP@0.5 | AP@0.75 | AR    | Tr.mAP | Tr.AP@0.5 | Tr.AP@0.75 | Tr.AR  |
|-------------------------------------------------------------------------------|:----------:|:-------:|:--------:|:----------:|:----------:|:----------:|:-----:|:------:|:-------:|:-----:|:------:|:---------:|:----------:|:------:|
| [RetinaNet-r50-fpn](configs/sku110k/sku110k_retinanet_r50_fpn_1x.py)          | r50        | 1x      | 0.001    | 2          |(1333, 800) | 4 (octave) | 0.463 | 0.751  | 0.532   | 0.512 | 0.467  | 0.752     | 0.535      | 0.516  |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_1333_1x.py) | r50        | 1x      | 0.005    | 2          |(1333, 800) | [8]        | 0.523 | 0.850  | 0.592   | 0.582 | 0.537  | 0.862     | 0.612      | 0.594  |

### 2. Non-dense anchoring

| Config                                                                           | Backbone   | Lr schd | Base lr  | imgs_p_gpu | img_scale  | anchor_sc  | 4tiles | mAP   | AP@0.5 | AP@0.75  | AR     | Tr.mAP | Tr.AP@0.5 | Tr.AP@0.75 | Tr.AR  |
|------------------------------------------------------------------------------------------|:----------:|:-------:|:--------:|:----------:|:----------:|:----------:|:------:|:-----:|:------:|:--------:|:------:|:------:|:---------:|:----------:|:------:|
| [GA-RetinaNet-r50-fpn](configs/sku110k/sku110k_ga_retinanet_r50_fpn_1x.py)               | r50        | 1x      | 0.001    | 2          |(816, 1088) | 4 (octave) | ☐      | 0.523 | 0.870  | 0.579    | 0.583  | 0.532  | 0.881     | 0.590      | 0.591  |
| [GA-RetinaNet-x101-32x4d-fpn](configs/sku110k/sku110k_ga_retinanet_x101_32x4d_fpn_1x.py) | x101-32x4d | 1x      | 0.001    | 2          |(816, 1088) | 4 (octave) | ☐      | 0.537 | 0.882  | 0.602    | 0.598  | 0.552  | 0.896     | 0.623      | 0.610  |
| [RepPoints-moment-r50-fpn](configs/sku110k/sku110k_reppoints_moment_r50_fpn_1x.py)       | r50        | 1x      | 0.02     | 6          |(816, 1088) | 4 (base)   | ☐      | 0.505 | 0.815  | 0.578    | 0.562  | 0.519  | 0.820     | 0.601      | 0.574  |

### 3. Comparison of different anchor scales for Faster-RCNN

| Config                                                                    | Backbone   | Lr schd | Base lr  | imgs_p_gpu | img_scale  | anchor_sc  | mAP   | AP@0.5 | AP@0.75 | AR    | Tr.mAP | Tr.AP@0.5 | Tr.AP@0.75 | Tr.AR  |
|-----------------------------------------------------------------------------------|:----------:|:-------:|:--------:|:----------:|:----------:|:----------:|:-----:|:------:|:-------:|:-----:|:------:|:---------:|:----------:|:------:|
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_1x.py)          | r50        | 1x      | 0.005    | 2          |(816, 1088) | [8]        | 0.522 | 0.850  | 0.591   | 0.577 | 0.534  | 0.862     | 0.611      | 0.590  |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_1x.py )  | r50        | 1x      | 0.005    | 2          |(816, 1088) | [4]        | 0.551 | 0.912  | 0.614   | 0.613 | 0.567  | 0.926     | 0.636      | 0.629  |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_3_1x.py) | r50        | 1x      | 0.005    | 2          |(816, 1088) | [3]        | 0.549 | 0.911  | 0.611   | 0.614 |        |           |            |        |

### 4. Comparison of different anchor scales for RetinaNet

| Config                                                                      | Backbone   | Lr schd | Base lr  | imgs_p_gpu | img_scale  | anchor_sc  | mAP    | AP@0.5 | AP@0.75 | AR     | Tr.mAP | Tr.AP@0.5 | Tr.AP@0.75 | Tr.AR  |
|-------------------------------------------------------------------------------------|:----------:|:-------:|:--------:|:----------:|:----------:|:----------:|:------:|:------:|:-------:|:------:|:------:|:---------:|:----------:|:------:|
| [RetinaNet-r50-fpn](configs/sku110k/sku110k_retinanet_r50_fpn_1x.py)                | r50        | 1x      | 0.001    | 2          |(1333, 800) | 4 (octave) | 0.463  | 0.751  | 0.532   | 0.512  | 0.467  | 0.752     | 0.535      | 0.516  |
| [RetinaNet-r50-fpn](configs/sku110k/sku110k_retinanet_r50_fpn_1x_tuned_acnhors2.py) | r50        | 1x      | 0.001    | 2          |(1333, 800) | 3 (octave) | 0.508  | 0.849  | 0.564   | 0.569  | 0.513  | 0.853     | 0.574      | 0.574  |

### 5. Bells and whistles testing

| Config                                                                             | Backbone   | Lr schd | Base lr  | imgs_p_gpu | img_scale                           | anchor_sc  | 4tiles | s-nms test| extra augs | traintime flip | testtime  flip | mAP   | AP@0.5| AP@0.75| AR     |
|--------------------------------------------------------------------------------------------|:----------:|:-------:|:--------:|:----------:|:-----------------------------------:|:----------:|:------:|:---------:|:----------:|:--------------:|:--------------:|:-----:|:-----:|:------:|:------:|
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_multiscale_1x.py) | r50        | 1x      | 0.005    | 2          |(752, 1024), (816, 1088), (880, 1152)| [4]        | ☐      | ☐         | ☐          | ✓              | ☐              | 0.552 | 0.912 | 0.615  | 0.616  |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_augs_1x.py)       | r50        | 1x      | 0.005    | 2          |(816, 1088)                          | [4]        | ☐      | ☐         | ✓          | ☐              | ☐              | 0.548 | 0.911 | 0.608  | 0.612  |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_augs_flip_2x.py)  | r50        | 2x      | 0.005    | 2          |(816, 1088)                          | [4]        | ☐      | ☐         | ✓          | ✓              | ☐              | 0.540 | 0.906 | 0.596  | 0.606  |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_augs_flip_2x.py)  | r50        | 2x      | 0.005    | 2          |(816, 1088)                          | [4]        | ☐      | ☐         | ✓          | ✓              | ✓              | 0.510 | 0.888 | 0.543  | 0.584  |

### 6. Cascade-RCNN comparison

| Config                                                                                                        | Backbone   | Lr schd | Base lr  | imgs_p_gpu | img_scale  | anchor_sc  | 4tiles | s-nms test| mAP   | AP@0.5| AP@0.75| AR    | Tr.mAP | Tr.AP@0.5 | Tr.AP@0.75 | Tr.AR  |
|---------------------------------------------------------------------------------------------------------------|:----------:|:-------:|:--------:|:----------:|:----------:|:----------:|:------:|:---------:|:-----:|:-----:|:------:|:-----:|:------:|:---------:|:----------:|:------:|
| [Cascade-RCNN-r50-fpn](configs/sku110k/sku110k_cascade_rcnn_r50_fpn_1x.py)                                    | r50        | 1x      | 0.005    | 2          |(816, 1088) | [8]        | ☐      | ☐         | 0.525 | 0.840 | 0.604  | 0.582 | 0.542  | 0.862     | 0.647      | 0.596  |
| [Cascade-RCNN-r50-fpn](configs/sku110k/sku110k_cascade_rcnn_r50_fpn_anchor_1x.py)                             | r50        | 1x      | 0.005    | 2          |(816, 1088) | [4]        | ☐      | ☐         | 0.553 | 0.902 | 0.626  | 0.615 | 0.574  | 0.926     | 0.653      | 0.634  |
| [Cascade-RCNN-r50-fpn](configs/sku110k/sku110k_cascade_rcnn_r50_fpn_anchor_1x_soft_nms_test.py)               | r50        | 1x      | 0.005    | 2          |(816, 1088) | [4]        | ☐      | ✓         | 0.556 | 0.900 | 0.632  | 0.622 | 0.577  | 0.925     | 0.659      | 0.642  |
| [Cascade-RCNN-x101-32x4d-fpn](configs/sku110k/sku110k_cascade_rcnn_x101_32x4d_fpn_anchor_1x)                  | x101-32x4d | 1x      | 0.005    | 2          |(768, 1024) | [4]        | ☐      | ☐         | 0.556 | 0.903 | 0.629  | 0.617 | 0.583  | 0.929     | 0.665      | 0.640  |
| [Cascade-RCNN-x101-32x4d-fpn](configs/sku110k/sku110k_cascade_rcnn_x101_32x4d_fpn_anchor_1x_soft_nms_test.py) | x101-32x4d | 1x      | 0.005    | 2          |(768, 1024) | [4]        | ☐      | ✓         | 0.560 | 0.902 | 0.635  | 0.623 | 0.585  | 0.929     | 0.672      | 0.647  |

### 7. Tiling strategies

| Config                                                                                                            | Backbone   | Lr schd | Base lr  | imgs_p_gpu | img_scale   | anchor_sc  | 4tiles | s-nms test| mAP    | AP@0.5| AP@0.75| AR    |
|-------------------------------------------------------------------------------------------------------------------|:----------:|:-------:|:--------:|:----------:|:-----------:|:----------:|:------:|:---------:|:------:|:-----:|:------:|:-----:|
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_1x_4tiles.py) (w/o merging)                     | r50        | 1x      | 0.005    | 2          |(816, 1088)  | [8]        | ✓      | ☐         | 0.561  | 0.912 | 0.632  | 0.628 |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_1x_4tiles.py) (w/o merging)              | r50        | 1x      | 0.005    | 2          |(816, 1088)  | [4]        | ✓      | ☐         | 0.566  | 0.928 | 0.636  | 0.636 |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_1x_4tiles.py) (merged)                   | r50        | 1x      | 0.005    | 2          |(816, 1088)  | [4]        | ✓      | ☐         | 0.547  | 0.894 | 0.615  | 0.611 |
| [Faster-RCNN-r50-fpn](configs/sku110k/sku110k_faster_rcnn_r50_fpn_anchor_1x_4tiles_test_half_res.py) (full frame) | r50        | 1x      | 0.005    | 2          |(816, 1088)  | [4]        | ✓      | ✓         | 0.577  | 0.928 | 0.659  | 0.654 |


## Citation

Feel free to cite my report if you use any of the results for benchmarking in your work.

```
@misc{kozlov2020working,
    title={Working with scale: 2nd place solution to Product Detection in Densely Packed Scenes [Technical Report]},
    author={Artem Kozlov},
    year={2020},
    eprint={2006.07825},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
