# YOLO for Uniform Directed Object detection

This project is an implementation of the paper [YUDO: YOLO for Uniform Directed Object Detection](https://arxiv.org/abs/2308.04542). The codebase is an adaptation of the popular [YOLOv7](https://github.com/WongKinYiu/yolov7) model, used for detection of directed objects with uniform dimensions.

## Requirements
```bash
docker build --rm --no-cache -t yudo:version_1 -f Dockerfile .
```

```bash
docker run --gpus device=0 --rm --shm-size=1G -ti -v {YOUR CODE PATH}:/yudo --name yudo yudo:version_1
```

Install in addition:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Prepare annotations
The dataset used in this project is obtained from [Honeybee Segmentation and Tracking Datasets](https://groups.oist.jp/bptu/honeybee-tracking-dataset). An image cropping and adapting labels' format to the yolo format can be done using the `gen_yolo_anns.py` script.

## Run the training
A training command example:
```bash
python train.py \
--epochs 200 \
--workers 4 \
--device 0 \
--batch-size 2 \
--data data/data.yaml \
--img-size 512 512 \
--cfg cfg/training/yolov7-tiny.yaml \
--weights 'yolov7-tiny.pt' \
--name model_001 \
--hyp data/hyp.scratch.yaml \
--image-weights \
--exist-ok \
--adam
```