import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import json
from transformers import CLIPTokenizer


def cal_num_tokens(tokenizer, text):
    batch_encoding = tokenizer(text, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"][0].numpy().tolist() 
    begin = tokens.index(49406)
    end = tokens.index(49407)
    num_tokens = end-begin-1
    return num_tokens

def load_data(file_path, tokenizer):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)

    label = Image.open(data_dict['layout_path'])
    label = label.resize((512, 512), resample=Image.NEAREST)
    label = np.array(label).astype(np.float32)
    label = torch.from_numpy(label).unsqueeze(0)

    text_label_mapping = data_dict['text_label_mapping']
    prompt = ''
    class_ids = []
    for text in text_label_mapping.keys():
        for item in text.split(' '):
            prompt += item
            prompt += ' '
            num_tokens = cal_num_tokens(tokenizer, item)
            for _ in range(num_tokens):
                class_ids.append(text_label_mapping[text])
    prompt = prompt[:-1]
    class_ids = torch.Tensor(class_ids).unsqueeze(0)

    img_name = data_dict['layout_path'].split('/')[-1]

    return label, class_ids, prompt, img_name


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/layout2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference_FLIS.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="examples/data.json",
        help="path to json file of data",
    ) 

    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    label, class_ids, prompt, img_name = load_data(opt.json, tokenizer)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                label = label.to(device)
                class_ids = class_ids.to(device)
                c = model.get_learned_conditioning(prompt)
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(1 * [""])
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 label=label,
                                                 class_ids=class_ids,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=start_code)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                x_sample = x_samples_ddim[0]
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img_name = img_name.split('.')[0]
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outpath, f"{img_name}-{prompt}.jpg"))

if __name__ == "__main__":
    main()
