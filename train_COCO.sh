CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_COCO.yaml \
                                      -t \
                                      --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
                                      -n exp_COCO \
                                      --gpus 0, \
                                      --data_root /home/xuehan/dataset/COCO-Stuff \
                                      --train_txt_file /home/xuehan/dataset/COCO-Stuff/COCO_train.txt \
                                      --val_txt_file /home/xuehan/dataset/COCO-Stuff/COCO_val.txt