CUDA_VISIBLE_DEVICES=0,1  python3 main_segformer.py --data_root "/home/student/workspace_Yufei/CityScapes/leftImg8bit/"\
                        --dataset "cityscapes" \
                        --model "segformer_b0"\
                        --output_stride 8 \
                        --batch_size 12 \
                        --crop_size_h 512 \
                        --crop_size_w 1024 \
                        --gpu_id 0,1 \
                        --lr 0.00006 \
                        --weight_decay 0.01 \
                        > segformer_B0_cityscapese.out
