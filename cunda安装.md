

#### docker内安装 cuda驱动
```
docker run --gpus all --privileged --name test_tpu -v /home/nyy:/home/nyy -dit sophgo/tpuc_dev:v3.3
cat /etc/os-release
cuda历史版本
https://developer.nvidia.com/cuda-toolkit-archive
```

#### cuda历史版本
```
https://developer.nvidia.com/cuda-toolkit-archive
```
#### pythorch 历史版本
```
https://download.pytorch.org/whl/torch_stable.html
```

#### cuda安装 
```
sudo sh sh ./cuda_11.7.1_515.65.01_linux.run
```

#### 稍等片刻，会提示接受 EULA 协议。输入 accept 接受协议。
```
┌──────────────────────────────────────────────────────────────────────────────┐
│  End User License Agreement                                                  │
│  --------------------------                                                  │
│                                                                              │
│  NVIDIA Software License Agreement and CUDA Supplement to                    │
│  Software License Agreement. Last updated: October 8, 2021                   │
│                                                                              │
│  The CUDA Toolkit End User License Agreement applies to the                  │
│  NVIDIA CUDA Toolkit, the NVIDIA CUDA Samples, the NVIDIA                    │
│  Display Driver, NVIDIA Nsight tools (Visual Studio Edition),                │
│  and the associated documentation on CUDA APIs, programming                  │
│  model and development tools. If you do not agree with the                   │
│  terms and conditions of the license agreement, then do not                  │
│  download or use the software.                                               │
│                                                                              │
│  Last updated: October 8, 2021.                                              │
│                                                                              │
│                                                                              │
│  Preface                                                                     │
│  -------                                                                     │
│                                                                              │
│──────────────────────────────────────────────────────────────────────────────│
│ Do you accept the above EULA? (accept/decline/quit):                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

```
#### 输入accept,同意协议后，按照提示进行安装，选择自定义安装，只选择 CUDA Toolkit 和相关库。
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 520.61.05                                                           │
│ + [X] CUDA Toolkit 12.1                                                      │
│   [X] CUDA Demo Suite 12.1                                                   │
│   [X] CUDA Documentation 12.1                                                │
│ - [ ] Kernel Objects                                                         │
│      [ ] nvidia-fs                                                           │
│   Options                                                                    │
│   Install                                                                  │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 安装完成后，输出如下：
```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-12.1/

Please make sure that
 -   PATH includes /usr/local/cuda-12.1/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.1/lib64, or, add /usr/local/cuda-12.1/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.1/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 520.00 is required for CUDA 12.1 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log

```

#### 配置环境变量
```
vim ~/.bashrc
export PATH=/usr/local/cuda/bin:/usr/local/cuda-11.7.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda

echo 'export PATH=/usr/local/cuda/bin:/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
```

### 安装系统级cudnn
```

```

#### 解压安装包
tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.9.0.131_cuda11-archive
#### 将解压后的文件复制到cuda安装路径
sudo cp -r include/* /usr/local/cuda/include
sudo cp -r lib/* /usr/local/cuda/lib64
#### 也可以将其复制到 /usr/include/ 和 /usr/lib/x86_64-linux-gnu/
#### 修改文件权限
sudo chmod a+r /usr/local/cuda/include/cudnn*.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
#### 添加环境变量
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
#### 刷新ldconfig
sudo ldconfig
source ~/.bashrc

### 安装系统级 cudnn