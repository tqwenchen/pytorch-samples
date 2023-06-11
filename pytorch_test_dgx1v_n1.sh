export LD_LIBRARY_PATH=/usr/local/openmpi_nccl/lib:/usr/local/cuda/lib64
export NCCL_DEBUG=WARN
export LC_TIME=en_US.UTF-8
export NCCL_RINGS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0"

for i in {0..7}; do (nohup python -u ./imagenet/main.py -j 0 -a resnet50 --print-freq 1 --batch-size 32 --world-size 16 --dist-url 'tcp://10.23.136.227:20000' ~/data/tiny-imagenet-200 --epochs 1 --dist-backend 'nccl' --rank $i &); done
