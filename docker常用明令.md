
### 导入导出镜像
```
sudo docker commit 466177937b1c myolotrain_web:v1 # 保存为镜像images
sudo docker save -o myolotrain_web_v1.tar myolotrain_web:v1 # 镜像保存为压缩文件
sudo docker load -i /devel/zj_docker/vediogpu_v1.tar

docker update --shm-size 64g 7e6 
docker update --memory 32g 7e6 

docker save -o sophgo_tpuc_dev_v3.3.tar.gz sophgo/tpuc_dev:v3.3
docker save -o rknn_model_v1.tar.gz rknn_model:v1
docker save -o cvat_ui_v2.39.0.tar.gz cvat/ui:v2.39.0

docker save -o stream_dev:0.3.tar.gz stream_dev:0.3
docker save -o nova-build:v2.2.0.tar.gz nova-build:v2.2.0
docker save -o ubuntu:22.04.tar.gz ubuntu:22.04 
docker save -o arm64v8/ubuntu:22.04.tar.gz arm64v8/ubuntu:22.04
```

### 使用dockerfile创建镜像

```
docker build  --platform=linux/arm64 -t arm64v8/ubuntu:20.04 .
docker save -o arm64v8_ubuntu_20.04.tar arm64v8/ubuntu:20.04
docker save -o ubuntu_20.04.tar ubuntu:20.04

docker build -t cvat/ui:v2.39.4 -f Dockerfile.ui .
```

### 批量删除docker

```
# 停止所有名称包含 "cvat_" 的容器
docker stop $(docker ps -a --format '{{.Names}}' | grep "cvat_")

# 强制删除（如果容器正在运行）
docker rm -f $(docker ps -a --format '{{.Names}}' | grep "cvat_")

#删除所有名称包含 cvat_ 的镜像
docker rmi -f $(docker images --format '{{.Repository}}:{{.Tag}}' | grep "cvat")
```



## docker 常见错误

#### Client.Timeout 

Error response from daemon: Get "https://registry-1.docker.io/v2/": context deadline exceeded (Client.Timeout exceeded while awaiting headers)

```
sudo vim /etc/docker/daemon.json
sudo systemctl restart docker

{
    "registry-mirrors": [
        "https://docker.1ms.run",
        "https://docker.xuanyuan.me",
		"https://docker.1panel.live",
		"https://hub.rat.dev",
		"https://docker.actima.top",
		"https://atomhub.openatom.cn",
		"https://docker.m.daocloud.io",
		"https://docker.nastool.de",
		"https://dockerpull.org",
		"https://registry.dockermirror.com",
		"https://docker.m.daocloud.io",
		"https://docker.aityp.com",
		"https://dockerhub.xisoul.cn",
		"https://docker.imgdb.de",
		"https://hub.littlediary.cn",
		"https://docker.unsee.tech",
		"https://hub.crdz.gq",
		"https://hub.firefly.store",
		"https://docker.kejilion.pro",
		"https://dhub.kubesre.xyz",
		"https://hub.xdark.top",
		"https://docker.udayun.com"
    ],
    "default-runtime": "nvidia",
    "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

