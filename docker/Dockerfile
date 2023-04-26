FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'