CUDA_VISIBLE_DEVICES=0 python scripts/LIS.py --batch_size 8 \
                                             --config configs/stable-diffusion/v1-finetune_COCO.yaml \
                                             --ckpt freestyle-sd-v1-4-coco.ckpt \
                                             --dataset COCO \
                                             --outdir outputs/COCO_LIS \
                                             --txt_file /home/xuehan/dataset/COCO-Stuff/COCO_val.txt \
                                             --data_root /home/xuehan/dataset/COCO-Stuff \
                                             --plms 