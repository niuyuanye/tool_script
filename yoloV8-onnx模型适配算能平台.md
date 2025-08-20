
## bm1688 yolov8 模型转换及量化 环境 
```
docker run --privileged --name sophon_npu -v /home/nyy:/home/nyy -dit sophgo/tpuc_dev:v3.3

docker run --privileged --name sophon_npu_model -v /home/nyy:/home/nyy -dit sophgo/tpuc_dev:v3.3
docker run --privileged --name sophon_npu_model3 -v /home/nyy:/home/nyy -dit sophgo/tpuc_dev:v3.3
docker run --privileged --network host --name sophon_npu_mode2 -v /var/run/docker.sock:/var/run/docker.sock -v /home/nyy:/home/nyy -dit sophgo/tpuc_dev:v3.3

pip install tpu_mlir[onnx] -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install tpu_mlir[torch] -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple/
apt install net-tools
```

### bm1688 模型转换、量化
```
model_transform.py \
        --model_name yolov8s \
        --model_def ./smokephone_v8s_15000.onnx \
        --input_shapes [[1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir smokephone_v8s_15000_1b.mlir

model_deploy.py \
            --mlir smokephone_v8s_15000_1b.mlir \
            --quantize F16 \
            --chip bm1688 \
            --model smokephone_v8s_15000_fp16_1b_2core.bmodel \
            --num_core 2

model_transform.py \
        --model_name yolov11s \
        --model_def ./firesmoke_v11s.onnx \
        --input_shapes [[1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir firesmoke_v11s_1b.mlir

model_deploy.py \
            --mlir firesmoke_v11s_1b.mlir \
            --quantize F16 \
            --chip bm1688 \
            --model firesmoke_v11s_fp16_1b_2core.bmodel \
            --num_core 2



model_deploy \
    --mlir yolov8s.mlir \
    --quantize F16 \
    --processor bm1688 \
    --num_core 2 \
    --test_input yolov8s_in_f32.npz \
    --test_reference yolov8s_top_outputs.npz \
    --model yolov8s_1688_f16.bmodel

model_deploy \
    --mlir yolov8s.mlir \
    --quantize F16 \
    --processor bm1688 \
    --test_input yolov8s_in_f32.npz \
    --test_reference yolov8s_top_outputs.npz \
    --model yolov8s_1688_f16.bmodel
    
    
    model_transform.py \
        --model_name yolov8s \
        --model_def ./best.onnx \
        --input_shapes [[1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir ./best_1b.mlir
        
   model_deploy.py \
            --mlir best_1b.mlir \
            --quantize F16 \
            --chip bm1688 \
            --model best_fp16_1b_2core.bmodel \
            --num_core 2
```

## bm1688 yolov8 测试编译环境 docker
```
docker run --privileged --name sophon_build_nyy -v /home/nyy:/home/nyy -dit stream_dev:0.2
docker run --privileged --name sophon_build_nyy -v /home/niuyuanye:/home/niuyuanye -dit stream_dev:0.2
docker run --privileged --name sophon_build_nyy1 -v /home/nyy:/home/nyy -dit stream_dev:0.3
```
### cpp demo 测试
git 下载 sophon-demo示例 https://github.com/sophgo/sophon-demo.git
编译soc版本；主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包
```
cd /sophon-demo-0.3.0/sample/YOLOv8_plus_det

cd cpp/yolov8_bmcv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make -j33
```
### 板卡测试问题
```
端口9000冲突
lsof -i :9000
systemctl stop ruijing
systemctl disable ruijing

docker 启动失败：
在cubeai-app中的 
cp daemon.json /etc/docker/daemon.json
sync
```

#### 板卡测试
把编译的 yolov8_bmcv.soc文件下载到bm1688中， 图片测试实例如下：
```
sudo -i
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/bm1688-cubeai/rules/common/


先通过web网页配置1个任务，视频源，算法任选；
手动docker rm -f 删除web启动的docker容器
taskmaster修改：
手动启动bm1688-taskmaster会创建/task/config/sys.yaml
修改/task/config/sys.yaml->task->url为真实要测试的视频流地址
修改/task/config/sys.yaml->algorithm->name为待调试的算法名称
algorithm修改
先手动启动一次bm1688-algorithm
创建/algorithm/models/目录，将你的算法模型文件拷贝到此目录
修改/algorithm/config/sys.yaml->model->paths: 为真实模型绝对路径
修改/algorithm/config/sys.yaml->algorithm->name为真实算法名称
修改/algorithm/config/sys.yaml->algorithm->classname为真实算法类别
再次重新启动taskmaster和algorithm，即可通过web任务预览查看效果

```
#### 测试图片
把编译的 yolov8_bmcv.soc文件下载到bm1688中， 图片测试实例如下：
```
sudo ./yolov8_bmcv.soc --input=../../datasets/Helmet_test --bmodel=../../models/BM1688/Helmet_v8s_10000_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./yolov8_bmcv.soc --input=../../datasets/Helmet_test --bmodel=../../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./yolov8_bmcv.soc --input=../../datasets/Helmet_test --bmodel=../../models/BM1688/Helmet_v8s_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./yolov8_bmcv.soc --input=../../datasets/Helmet_test --bmodel=../../models/BM1688/Helmet_v11s_10000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 



sudo ./yolov8_bmcv.soc --input=../../datasets/test --bmodel=../../models/BM1688/yolov8s_fp32_1b.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/coco.names 

sudo ./yolov8_bmcv.soc --input=../../datasets/test --bmodel=../../models/BM1688/yolov8s_fp16_b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/coco.names 

sudo ./yolov8_bmcv.soc --input=../../datasets/test_car_person_1080P.mp4 --bmodel=../../models/BM1688/yolov8s_fp16_b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/coco.names

sudo ./demo.soc --input=../../datasets/Helmet_test --bmodel=../../models/BM1688/Helmet_v11s_10000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./demo.soc --input=/home/admin/data/Helmet/ --output=/home/admin/data/Helmet_result/ --bmodel=../../models/BM1688/Helmet_v11s_10000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.45 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./yolov8_bmcv.soc --input=/home/admin/data/Helmet/ --bmodel=../../models/BM1688/Helmet_v11s_10000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.45 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./yolov8_bmcv.soc --input=/home/admin/data/Helmet/ --bmodel=../../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 


sudo ./yolov8_bmcv.soc --input=../../datasets/Helmet_test --bmodel=../../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./yolov8_bmcv.soc --input=./test --bmodel=./Helmet_v8s_parm_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=./helmet.names 

sudo ./yolov8_bmcv.soc --input=../../datasets/Helmet --bmodel=../../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 


sudo ./yolov8_bmcv.soc --input=../../datasets/Helmet --bmodel=../../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../../datasets/Helmet.names 

sudo ./yolov8_bmcv.soc --input=/home/admin/data/Helmet --bmodel=../../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=/home/admin/data/Helmet.names 

sudo ./yolov8_bmcv.soc --input=/home/admin/sophon-demo-0.3.1/sample/YOLOv8_plus_det/datasets/Helmet --bmodel=../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../datasets/Helmet.names 

sudo ./demo.soc --input=/home/admin/test/datasets/Helmet --output=/home/admin/test/datasets/Helmet_result/ --bmodel=../models/BM1688/Helmet_v11s_10000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.45 --nms_thresh=0.5 --classnames=../datasets/Helmet.names 

sudo ./demo.soc --input=/home/admin/test/datasets/Helmet --output=/home/admin/test/datasets/Helmet_result/ --bmodel=../models/BM1688/Helmet_v8s_parm_fp32_1b_2core.bmodel --dev_id=0 --conf_thresh=0.45 --nms_thresh=0.5 --classnames=../datasets/Helmet.names 


sudo ./demo.soc --input=/home/admin/test/datasets/car --output=/home/admin/test/datasets/car_result/ --bmodel=../models/BM1688/person_car_v8s_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.45 --nms_thresh=0.5 --classnames=../datasets/person_car.names 

sudo ./demo.soc --input=/home/admin/test/datasets/firesmoke --output=/home/admin/test/datasets/firesmoke_result/ --bmodel=../models/BM1688/firesmoke_v8s_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.45 --nms_thresh=0.5 --classnames=../datasets/firesmoke.names 


sudo ./demo.soc --input=/home/admin/test/datasets/smokephone --output=/home/admin/test/datasets/smokephone_result/ --bmodel=../models/BM1688/smokephone_v8s_7500_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.5 --classnames=../datasets/smokephone.names 

sudo ./demo.soc --input=/home/admin/test/datasets/smokephone --output=/home/admin/test/datasets/smokephone_result/ --bmodel=../models/BM1688/smokephone_v8s_15000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.5 --classnames=../datasets/smokephone.names 

sudo ./demo_per.soc --input=/home/admin/test/datasets/smokephone --output=/home/admin/test/datasets/smokephone_result/ --is_smoke=1 --bmodel=../models/BM1688/smokephone_v8s_15000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.5 --classnames=../datasets/smokephone.names 



sudo ./demo_per.soc --input=./test --output=/home/admin/test/datasets/smokephone_result/ --is_smoke=0 --bmodel=./smokephone_v8s_15000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.5 --classnames=.smokephone.names 


sudo ./demo.soc --input=/home/admin/test/datasets/smokephone --output=/home/admin/test/datasets/smokephone_result/ --is_smoke=1 --bmodel=../models/BM1688/smokephone_v8s_15000_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../datasets/smokephone.names

sudo ./demo.soc --input=/home/admin/test/datasets/test --output=/home/admin/test/datasets/test_result/ --is_smoke=1 --bmodel=../models/BM1688/smokephone_v8s_15000_fp16_1b_2core.bmodel --bmodel_per=../models/BM1688/yolov8s_fp16_b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../datasets/smokephone.names

sudo ./demo.soc --input=/home/admin/test/datasets/smokephone --output=/home/admin/test/datasets/smokephone_result/ --is_smoke=1 --bmodel=../models/BM1688/smokephone_v8s_15000_fp16_1b_2core.bmodel --bmodel_per=../models/BM1688/yolov8s_fp16_b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../datasets/smokephone.names


sudo ./demo.soc --input=/home/admin/test/datasets/test --output=/home/admin/test/datasets/test_result/ --is_smoke=1 --bmodel=../models/BM1688/smokephone_v8s_15000_fp16_1b_2core.bmodel --bmodel_per=../models/BM1688/yolov8s_fp16_b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../datasets/smokephone.names


sudo ./demo.soc --input=/home/admin/test/datasets/test --output=/home/admin/test/datasets/test_result/ --is_smoke=1 --bmodel=../models/BM1688/smokephone_v8s_7500_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=../datasets/smokephone.names



docker run --privileged --name sophon_npu -v /home/nyy:/home/nyy -dit sophgo/tpuc_dev:v3.3

sudo docker run --privileged --name v8 -v /home/admin:/home/admin -dit sophon1688-smokedetect:cubeai-algorithm-v0.1
sudo docker run --privileged --name v8 -v /home/admin:/home/admin -dit sophon1688-smokedetect:cubeai-algorithm-v0.1

sudo docker run --privileged --name sophon_build -v /home/admin:/home/nyy -dit stream_dev:0.2


ffmpeg -r 30 -i " ./input.mp4" -r 1 "./Helmet/frame_%04d.jpg"

ffmpeg -r 30 -i "./constructionSite_265.mp4" -f image2 -strftime 1 ./Helmet/Helmet_%s.jpg


ffmpeg -r 25 -i "./traffic_flow_264.mp4" -f image2 -strftime 1 ./car/car_%s.jpg
ffmpeg -r 25 -i "./sh_street_01_264.mp4" -f image2 -strftime 1 ./car/car_sh_%s.jpg
ffmpeg -r 30 -i "./people_flow_264.mp4" -f image2 -strftime 1 ./car/people_%s.jpg
ffmpeg -r 30 -i "./person_264.mp4" -f image2 -strftime 1 ./car/person_%s.jpg

ffmpeg -r 20 -i "./fireAccident_265.mp4" -f image2 -strftime 1 ./firesmoke/firedmoke_%s.jpg
ffmpeg -r 20 -i "./FireWarn_265.mp4" -f image2 -strftime 1 ./firesmoke/firesmoke_%s.jpg


ffmpeg -r 20 -i "./movie-smoke_264.mp4" -f image2 -strftime 1 ./smokephone/smoke_%s.jpg
ffmpeg -r 20 -i "./phone_264.mp4" -f image2 -strftime 1 ./smokephone/phone_%s.jpg


sudo ./demo.soc --input=/home/admin/test_demo/datasets/smokephone/test --output=/home/admin/test/datasets/smokephone_result/ --is_smoke=0 --bmodel=./smokephone_v8s_15000_fp16_1b_2core.bmodel --bmodel_per=./person_car_v8s_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=./smokephone.names

sudo ./demo.soc --input=/home/admin/images/ --is_smoke=0 --bmodel=./person_car_v8s_fp16_1b_2core.bmodel --bmodel_per=./person_car_v8s_fp16_1b_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=./personcar.names

./demo.soc --input=./datasets/images/ --is_smoke=0 --bmodel=./yolov8s-personcar_f16_2core.bmodel --bmodel_per=./yolov8s-personcar_f16_2core.bmodel --dev_id=0 --conf_thresh=0.25 --nms_thresh=0.7 --classnames=./personcar.names

```

