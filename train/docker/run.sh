#!/bin/sh

dirname=$(pwd | xargs dirname)
dataset="/share/private/27th/hirotaka_saito/dataset/"
logs="/share/private/27th/hirotaka_saito/logs/"

docker run -it \
  --privileged \
  --gpus all \
  -p 15900:5900 \
  --rm \
  --mount type=bind,source=$dirname,target=/root/badgr \
  --mount type=bind,source=$dataset,target=/root/dataset \
  --mount type=bind,source=$logs,target=/root/logs \
   --net host \
   --shm-size=10000m \
  badgr
  bash
