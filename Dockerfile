## Pull from existing image
FROM nvcr.io/nvidia/pytorch:22.07-py3
## FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
## FROM python:3.8


RUN mkdir /myhome/
RUN chmod -R 777 /myhome

COPY ./src/requirements.txt .

## Install Python packages in Docker image
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

COPY ./src /myhome
RUN chmod -R 777 /myhome

## RUN cd /myhome/

## Execute the inference command 
CMD ["/myhome/run_inference.py"]
ENTRYPOINT ["python3"]
