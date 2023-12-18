# python -u core/datasets.py --name raft-kitti \
#     --stage MPI-Flow --validation kitti \
#     --restore_ckpt checkpoints/raft-sintel.pth \
#     --gpus 0 --num_steps 50000 --batch_size 5 \
#     --lr 0.0001 --image_size 256 384 --wdecay 0.00001 \
#     --gamma=0.85 --mixed_precision

# python -u train.py --name raft-kitti \
#     --stage MPI-Flow --validation kitti \
#     --restore_ckpt checkpoints/raft-things.pth \
#     --gpus 0 --num_steps 50000 --batch_size 4 \
#     --lr 0.0001 --image_size 256 384 --wdecay 0.00001 \
#     --gamma=0.85 --mixed_precision

python -u train.py --name raft-kitti \
    --stage MPI-Flow --validation kitti \
    --restore_ckpt checkpoints/raft-things.pth \
    --gpus 0 --num_steps 50000 --batch_size 4 \
    --lr 0.0001 --image_size 384 512 --wdecay 0.00001 \
    --gamma=0.85 --mixed_precision
