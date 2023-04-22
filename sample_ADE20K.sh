CUDA_VISIBLE_DEVICES=0 python scripts/LIS.py --batch_size 8 \
                                             --config configs/stable-diffusion/v1-finetune_ADE20K.yaml \
                                             --ckpt freestyle-sd-v1-4-ade20k.ckpt \
                                             --dataset ADE20K \
                                             --outdir outputs/ADE20K_LIS \
                                             --txt_file /home/xuehan/dataset/ADEChallengeData2016/ADE20K_val.txt \
                                             --data_root /home/xuehan/dataset/ADEChallengeData2016 \
                                             --plms 