FROM nvcr.io/nvidia/tritonserver:21.09-py3

RUN pip3 install numpy pillow torchvision easyocr

COPY init.py init.py
RUN python3 init.py

COPY . .

CMD ["tritonserver", "--model-repository=/models"]
