CUDA_VISIBLE_DEVICES=0 python scripts/FLIS.py --config configs/stable-diffusion/v1-inference_FLIS.yaml \
                                              --ckpt freestyle-sd-v1-4-coco.ckpt \
                                              --json examples/layout_flower.json \
                                              --outdir outputs/FLIS \
                                              --plms 
