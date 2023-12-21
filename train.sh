python -m torch.distributed.launch --nproc_per_node=4 train.py --name ours1221 \
    --stage MPI-Flow --validation kitti \
    --restore_ckpt checkpoints/raft-things.pth \
    --num_steps 50000 --batch_size 4 \
    --lr 0.0001 --image_size 512 768 --wdecay 0.00001 \
    --gamma=0.85 --mixed_precision --debug
