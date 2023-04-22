CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_ADE20K.yaml \
                                      -t \
                                      --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
                                      -n exp_ADE20K \
                                      --gpus 0, \
                                      --data_root /home/xuehan/dataset/ADEChallengeData2016 \
                                      --train_txt_file /home/xuehan/dataset/ADEChallengeData2016/ADE20K_train.txt \
                                      --val_txt_file /home/xuehan/dataset/ADEChallengeData2016/ADE20K_val.txt