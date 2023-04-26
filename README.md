# YOLO for Uniform Directed Object detection

## Docker build
```bash
docker build --rm --no-cache -t yudo:version_1 -f Dockerfile .
```

## Docker run
```bash
docker run --gpus device=0 --rm --shm-size=1G -ti -v {YOUR CODE PATH}:/yudo --name yudo yudo:version_1
```

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