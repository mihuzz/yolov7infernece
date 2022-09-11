# YOLOv7 on Triton Inference Server

Instructions to deploy YOLOv7 as TensorRT engine to [Triton Inference Server](https://github.com/NVIDIA/triton-inference-server).

Triton Inference Server takes care of model deployment with many out-of-the-box benefits, like a GRPC and HTTP interface, automatic scheduling on multiple GPUs, shared memory (even on GPU), dynamic server-side batching, health metrics and memory resource management.

There are no additional dependencies needed to run this deployment, except a working docker daemon with GPU support.

## Export TensorRT

See https://github.com/WongKinYiu/yolov7#export for more info.

```bash
# Pytorch Yolov7 -> ONNX with grid, EfficientNMS plugin and dynamic batch size
python export.py --weights ./yolov7.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
# ONNX -> TensorRT with trtexec and docker
docker run -it --rm --gpus=all nvcr.io/nvidia/tensorrt:22.07-py3
# Copy onnx -> container: docker cp yolov7.onnx <container-id>:/workspace/
# Export with FP16 precision, min batch 1, opt batch 8 and max batch 8
./tensorrt/bin/trtexec --onnx=yolov7.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --fp16 --workspace=4096 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache
# Test engine
./tensorrt/bin/trtexec --loadEngine=yolov7-fp16-1x8x8.engine
# Copy engine -> host: docker cp <container-id>:/workspace/yolov7-fp16-1x8x8.engine .
```

Example output of test with RTX 3070.

```
[09/11/2022-22:54:32] [I] === Performance summary ===
[09/11/2022-22:54:32] [I] Throughput: 1004.27 qps
[09/11/2022-22:54:32] [I] Latency: min = 1.17474 ms, max = 4.4541 ms, mean = 1.20348 ms, median = 1.1897 ms, percentile(99%) = 1.50061 ms
[09/11/2022-22:54:32] [I] Enqueue Time: min = 0.303711 ms, max = 4.79843 ms, mean = 0.336177 ms, median = 0.337891 ms, percentile(99%) = 0.387817 ms
[09/11/2022-22:54:32] [I] H2D Latency: min = 0.198151 ms, max = 0.228271 ms, mean = 0.201239 ms, median = 0.200623 ms, percentile(99%) = 0.213379 ms
[09/11/2022-22:54:32] [I] GPU Compute Time: min = 0.965637 ms, max = 4.24448 ms, mean = 0.993252 ms, median = 0.97998 ms, percentile(99%) = 1.29016 ms
[09/11/2022-22:54:32] [I] D2H Latency: min = 0.00549316 ms, max = 0.0202637 ms, mean = 0.00898109 ms, median = 0.00891113 ms, percentile(99%) = 0.0114746 ms
[09/11/2022-22:54:32] [I] Total Host Walltime: 3.0022 s
[09/11/2022-22:54:32] [I] Total GPU Compute Time: 2.99465 s
[09/11/2022-22:54:32] [W] * GPU compute time is unstable, with coefficient of variance = 12.3891%.
[09/11/2022-22:54:32] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[09/11/2022-22:54:32] [I] Explanations of the performance metrics are printed in the verbose logs.
[09/11/2022-22:54:32] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8401] # ./tensorrt/bin/trtexec --loadEngine=yolov7-fp16-1x8x8.engine
```

## Model Repository

See [Triton Model Repository Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#model-repository) for more info.

```bash
# Create folder structure
mkdir -p triton-deploy/models/yolov7/1/
touch triton-deploy/models/yolov7/config.pbtxt
# Place model
mv yolov7-fp16-1x8x8.engine triton-deploy/models/yolov7/1/model.plan
```

## Model Configuration

See [Triton Model Configuration Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#model-configuration) for more info.

Minimal configuration for `triton-deploy/models/yolov7/config.pbtxt`:

```
name: "yolov7"
platform: "tensorrt_plan"
max_batch_size: 8
dynamic_batching { }
```

Example repository:

```bash
$ tree triton-deploy/
triton-deploy/
└── models
    └── yolov7
        ├── 1
        │   └── model.plan
        └── config.pbtxt

3 directories, 2 files
```

## Start Triton Inference Server

```
docker run --gpus all --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/triton-deploy/models:/models nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose 1
```

In the log you should see:

```
+--------+---------+--------+
| Model  | Version | Status |
+--------+---------+--------+
| yolov7 | 1       | READY  |
+--------+---------+--------+
```

## Performance with Model Analyzer

See [Triton Model Analyzer Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_analyzer.md#model-analyzer) for more info.

Performance numbers @ RTX 3070 + 11th Gen Intel® Core™ i9-11900KF @ 3.50GHz × 16 

Throughput for 16 clients with batch size 1 is the same as for a single thread running the engine at 16 batch size locally thanks to Triton [Dynamic Batching Strategy](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher). Result without dynamic batching (disable in model configuration) considerably worse:


## How to run model in your code

Example client can be found in client.py. It can run dummy input, images and videos.

```bash
pip3 install tritonclient[all] opencv-python
python3 client.py image /path/to/your/data

if you want to use webcam or other devices
python3 client.py video_device 0 -o /way/to/save/data.mp4
```



```
$ python3 client.py --help
usage: client.py [-h] [-m MODEL] [--width WIDTH] [--height HEIGHT] [-u URL] [-o OUT] [-f FPS] [-i] [-v] [-t CLIENT_TIMEOUT] [-s] [-r ROOT_CERTIFICATES] [-p PRIVATE_KEY] [-x CERTIFICATE_CHAIN] {dummy,image,video,video_device} [input]

positional arguments:
  {dummy,image,video,video_device}
                        Run mode. 'dummy' will send an emtpy buffer to the server to test if inference works. 'image' will process an image. 'video' will process a video. 'video_device' will process a video from device.
  input                 Input file to load from in image or video mode

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Inference model name, default yolov7
  --width WIDTH         Inference model input width, default 640
  --height HEIGHT       Inference model input height, default 640
  -u URL, --url URL     Inference server URL, default localhost:8001
  -o OUT, --out OUT     Write output into file instead of displaying it
  -f FPS, --fps FPS     Video output fps, default 24.0 FPS
  -i, --model-info      Print model status, configuration and statistics
  -v, --verbose         Enable verbose client output
  -t CLIENT_TIMEOUT, --client-timeout CLIENT_TIMEOUT
                        Client timeout in seconds, default no timeout
  -s, --ssl             Enable SSL encrypted channel to the server
  -r ROOT_CERTIFICATES, --root-certificates ROOT_CERTIFICATES
                        File holding PEM-encoded root certificates, default none
  -p PRIVATE_KEY, --private-key PRIVATE_KEY
                        File holding PEM-encoded private key, default is none
  -x CERTIFICATE_CHAIN, --certificate-chain CERTIFICATE_CHAIN
                        File holding PEM-encoded certicate chain default is none

```
