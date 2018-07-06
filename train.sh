CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset pascal_voc --net res101 \
                   --bs 1 --nw 4 \
                   --lr 4e-3 --lr_decay_step 8 \
                   --cuda