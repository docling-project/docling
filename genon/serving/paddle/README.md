https://github.com/PaddlePaddle/PaddleX
https://github.com/PaddlePaddle/Serving?tab=readme-ov-file
https://www.paddleocr.ai/main/en/version3.x/deployment/serving.html
https://www.paddlepaddle.org.cn/documentation/docs/en/install/docker/linux-docker_en.html?utm_source=chatgpt.com

## paddlepaldde image list
```shell

## GPU

docker pull paddlepaddle/paddle:3.2.0-gpu-cuda11.8-cudnn8.9

docker pull paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5

docker pull paddlepaddle/paddle:3.2.0-gpu-cuda12.9-cudnn9.9


##CPU

docker pull paddlepaddle/paddle:3.2.0

```

## 1. Install PaddlePaddle
```shell
# CPU 版本
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU 版本，需显卡驱动程序版本 ≥550.54.14（Linux）或 ≥550.54.14（Windows）
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
## 2. Install PaddleX

```shell
pip install "paddlex[base]"
```

## 3. Install serving dependencies

```shell
paddlex --install serving
```

## 4. Start serving

```shell
paddlex --serve --pipeline {PaddleX pipeline registration name or pipeline configuration file path} [{other command-line options}]
paddlex --serve --pipeline OCR

Name	Description
--pipeline	PaddleX pipeline registration name or pipeline configuration file path.
--device	Deployment device for the pipeline. By default, a GPU will be used if available; otherwise, a CPU will be used."
--host	Hostname or IP address to which the server is bound. Defaults to 0.0.0.0.
--port	Port number on which the server listens. Defaults to 8080.
--use_hpip	If specified, uses high-performance inference. Refer to the High-Performance Inference documentation for more information.
--hpi_config	High-performance inference configuration. Refer to the High-Performance Inference documentation for more information.
```

### Docker images
https://github.com/PaddlePaddle/Serving/blob/v0.9.0/doc/Docker_Images_EN.md


### VLM

```shell
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```