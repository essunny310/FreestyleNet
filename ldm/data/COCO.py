import os
from re import L
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

# Some words may differ from the class names defined in COCO-Stuff to minimize ambiguity
COCO_dict = {
'0':'person',
'1':'bicycle',
'2':'car',
'3':'motorcycle',
'4':'airplane',
'5':'bus',
'6':'train',
'7':'truck',
'8':'boat',
'9':'traffic light',
'10':'fire hydrant',
'11':'street sign',
'12':'stop sign',
'13':'parking meter',
'14':'bench',
'15':'bird',
'16':'cat',
'17':'dog',
'18':'horse',
'19':'sheep',
'20':'cow',
'21':'elephant',
'22':'bear',
'23':'zebra',
'24':'giraffe',
'25':'hat',
'26':'backpack',
'27':'umbrella',
'28':'shoe',
'29':'eye glasses',
'30':'handbag',
'31':'tie',
'32':'suitcase',
'33':'frisbee',
'34':'skis',
'35':'snowboard',
'36':'sports ball',
'37':'kite',
'38':'baseball bat',
'39':'baseball glove',
'40':'skateboard',
'41':'surfboard',
'42':'tennis racket',
'43':'bottle',
'44':'plate',
'45':'wine glass',
'46':'cup',
'47':'fork',
'48':'knife',
'49':'spoon',
'50':'bowl',
'51':'banana',
'52':'apple',
'53':'sandwich',
'54':'orange',
'55':'broccoli',
'56':'carrot',
'57':'hot dog',
'58':'pizza',
'59':'donut',
'60':'cake',
'61':'chair',
'62':'couch',
'63':'potted plant',
'64':'bed',
'65':'mirror',
'66':'dining table',
'67':'window',
'68':'desk',
'69':'toilet',
'70':'door',
'71':'tv',
'72':'laptop',
'73':'mouse',
'74':'remote',
'75':'keyboard',
'76':'cell phone',
'77':'microwave',
'78':'oven',
'79':'toaster',
'80':'sink',
'81':'refrigerator',
'82':'blender',
'83':'book',
'84':'clock',
'85':'vase',
'86':'scissors',
'87':'teddy bear',
'88':'hair drier',
'89':'toothbrush',
'90':'hair brush',
'91':'banner',
'92':'blanket',
'93':'branch',
'94':'bridge',
'95':'building',
'96':'bush',
'97':'cabinet',
'98':'cage',
'99':'cardboard',
'100':'carpet',
'101':'ceiling',
'102':'tile ceiling',
'103':'cloth',
'104':'clothes',
'105':'clouds',
'106':'counter',
'107':'cupboard',
'108':'curtain',
'109':'desk',
'110':'dirt',
'111':'door',
'112':'fence',
'113':'marble floor',
'114':'floor',
'115':'stone floor',
'116':'tile floor',
'117':'wood floor',
'118':'flower',
'119':'fog',
'120':'food',
'121':'fruit',
'122':'furniture',
'123':'grass',
'124':'gravel',
'125':'ground',
'126':'hill',
'127':'house',
'128':'leaves',
'129':'light',
'130':'mat',
'131':'metal',
'132':'mirror',
'133':'moss',
'134':'mountain',
'135':'mud',
'136':'napkin',
'137':'net',
'138':'paper',
'139':'pavement',
'140':'pillow',
'141':'plant',
'142':'plastic',
'143':'platform',
'144':'playingfield',
'145':'railing',
'146':'railroad',
'147':'river',
'148':'road',
'149':'rock',
'150':'roof',
'151':'rug',
'152':'salad',
'153':'sand',
'154':'sea',
'155':'shelf',
'156':'sky',
'157':'skyscraper',
'158':'snow',
'159':'solid',
'160':'stairs',
'161':'stone',
'162':'straw',
'163':'structural',
'164':'table',
'165':'tent',
'166':'textile',
'167':'towel',
'168':'tree',
'169':'vegetable',
'170':'brick wall',
'171':'concrete wall',
'172':'wall',
'173':'panel wall',
'174':'stone wall',
'175':'tile wall',
'176':'wood wall',
'177':'water',
'178':'waterdrops',
'179':'blind window',
'180':'window',
'181':'wood'
}


class COCOBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
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
        self.flip_p = flip_p

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path = self.image_paths[i]
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        path_ = self.image_paths[i][:-4]
        if 'train' in path_:
            path2 = os.path.join(self.data_root, 'stuffthingmaps_trainval2017/train2017', path_.split('/')[-1] + '.png')
        else:
            path2 = os.path.join(self.data_root, 'stuffthingmaps_trainval2017/val2017', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)

        flip = random.random() < self.flip_p
        if self.size is not None:
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        if flip:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            pil_image2 = pil_image2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image = np.array(pil_image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
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


class COCOTrain(COCOBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class COCOValidation(COCOBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
