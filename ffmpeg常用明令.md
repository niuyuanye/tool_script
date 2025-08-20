
### 导入导出镜像
```
ffmpeg -f concat -safe 0 -i "input0.mp4|input1.mp4|input2.mp4" -c copy output.mp4



ffmpeg -f concat -safe 0 -i "fi085.mp4|fi086.mp4|fi087.mp4|fi088.mp4|fi089.mp4" -c copy fi_5.mp4


# 1. 创建文件列表 filelist.txt
file 'fi085.mp4'
file 'fi086.mp4'
file 'fi087.mp4'
file 'fi088.mp4'

# 2. 执行合并命令
ffmpeg -f concat -safe 0 -i filelist.txt -c copy fi_4.mp4

```

### 导入导出镜像

```
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 23 -c:a aac -b:a 128k output.mp4
ffmpeg -i fight_room.mp4 -c:v libx264 -crf 23 -c:a copy fight_2_h264.mp4
ffmpeg -i fight_room.mp4 -c:v libx264 -crf 23 fight_2_h264.mp4
ffmpeg -i fight_room.mp4 -c:v libx264 -preset slow -crf 23 -r 30 output.mp4

ffmpeg -i fight_room.mp4 -codec:v copy fight_2_h264.mp4

ffmpeg -i fight_2_h264.mp4 -map 0:v -c:v copy video.mp4
ffmpeg -i video.mp4 -c:v libx264 -crf 23 fight_room.mp4
```



\```bash

\# 导出所有帧为图片序列

ffmpeg -i fight_2_h264.mp4 frame_%05d.png

\# 从图片重建视频

ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 output.mp4

\```



### 导入导出镜像

```
ffmpeg -r 30 -i " ./fight_2_h264.mp4" -r 1 "./Helmet/frame_%04d.jpg"

ffmpeg -r 30 -i "./constructionSite_265.mp4" -f image2 -strftime 1 ./Helmet/Helmet_%s.jpg


ffmpeg -r 25 -i "./traffic_flow_264.mp4" -f image2 -strftime 1 ./car/car_%s.jpg
ffmpeg -r 25 -i "./sh_street_01_264.mp4" -f image2 -strftime 1 ./car/car_sh_%s.jpg
ffmpeg -r 30 -i "./people_flow_264.mp4" -f image2 -strftime 1 ./car/people_%s.jpg
ffmpeg -r 30 -i "./person_264.mp4" -f image2 -strftime 1 ./car/person_%s.jpg

ffmpeg -r 20 -i "./fireAccident_265.mp4" -f image2 -strftime 1 ./firesmoke/firedmoke_%s.jpg
ffmpeg -r 20 -i "./FireWarn_265.mp4" -f image2 -strftime 1 ./firesmoke/firesmoke_%s.jpg

ffmpeg -r 20 -i "./movie-smoke_264.mp4" -f image2 -strftime 1 ./smokephone/smoke_%s.jpg
ffmpeg -r 20 -i "./phone_264.mp4" -f image2 -strftime 1 ./smokephone/phone_%s.jpg4
```



