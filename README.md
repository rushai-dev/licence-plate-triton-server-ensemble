# licence-plate-triton-server-ensemble
Triton backend that enables pre-processing, post-processing and other logic to be implemented in Python. In the repository, I use tech stack including YOLOv8, ONNX, EasyOCR, Triton Inference Server, CV2, Docker, and K8S. All of which we deploy on k80 and use CUDA 11.4

# inference

```
docker run --gpus=all --rm --shm-size=10Gb -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models rushai/licence-plate-triton-server-ensemble:21.09-py3 tritonserver --model-repository=/models
```
