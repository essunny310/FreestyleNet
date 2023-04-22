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
from ldm.data.COCO import COCO_dict
from ldm.data.ADE20K import ADE20K_dict

from torch.utils.data import DataLoader, Dataset


class COCOVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'stuffthingmaps_trainval2017/val2017', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(182)
        text = ''
        for i in range(len(class_ids)):
            text += COCO_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

class ADE20KVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'annotations/validation', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(151)
        text = ''
        for i in range(len(class_ids)):
            text += ADE20K_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

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
        default=8,
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
        default="configs/stable-diffusion/v1-finetune_COCO.yaml",
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
        "--data_root", 
        type=str, 
        required=True, 
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="path to txt file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="which dataset to evaluate",
        choices=["COCO", "ADE20K"],
        default="COCO"
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

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size
    if opt.dataset == "COCO":
        val_dataset = COCOVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "ADE20K":
        val_dataset = ADE20KVal(data_root=opt.data_root, txt_file=opt.txt_file)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data in val_dataloader:
                    label = data["label"].to(device)
                    class_ids = data["class_ids"].to(device)
                    text = data["caption"]
                    c = model.get_learned_conditioning(text)
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     label=label,
                                                     class_ids=class_ids,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for i in range(len(x_samples_ddim)):
                        x_sample = x_samples_ddim[i]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img_name = data["img_name"][i]
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, f"{img_name}.jpg"))

if __name__ == "__main__":
    main()
