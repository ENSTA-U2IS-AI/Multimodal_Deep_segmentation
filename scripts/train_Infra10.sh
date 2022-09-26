python main.py --model deeplabv3plus_mobilenet --dataset infra10 --loss_type cross_entropy --enable_vis --vis_port 8097 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root /home/soumik/workspace_hard1/workspaceremi/DataRemi/INFRA10  

#python main.py --model deeplabv3plus_resnet101 --dataset cityscapes --loss_type focal_loss --enable_vis --vis_port 8097 --gpu_id 0  --lr 0.001  --crop_size 768 --batch_size 16 --output_stride 16 --data_root /home/soumik/workspace_hard1/workspaceremi/DataRemi/archives_cityscapes
