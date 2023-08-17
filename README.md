## 配置 opencv ， onnx runtime

#### 1. 配置

```
# opencv 配置参考 https://blog.csdn.net/qq_50972633/article/details/130444196
# onnx runtime 配置参考 https://blog.csdn.net/likesomething1/article/details/125643598
```

#### 2. 版本

cmake 3.10.2; gcc 8.4.0; opencv 4.1.0; onnx runtime 1.10.0

## 使用

#### 1. 文件夹结构

```
xxx/
    prj-opencv-cpp/
        data/
            face_data/
                CarUser1/
                    .jpg（应使用待完成的注册功能新建文件夹并采集对齐后的人脸图像）
            videos/
                存储测试视频
            results/
                存储测试结果
        cpp/
        CMakeLists.txt
        demo.cpp
    models/
        onnx/
            .onnx（权重文件）
```

#### 2. 测试

```
cd xxx/prj-opencv-cpp
mkdir build
cd build && cmake .. && make
./demo ../data/videos/video-1.mp4 video-1-result
```