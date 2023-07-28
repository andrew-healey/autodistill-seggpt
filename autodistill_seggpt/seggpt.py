import numpy as np
import supervision as sv
from supervision.dataset.core import DetectionDataset
from supervision import Detections

from autodistill.core import Ontology
from autodistill.detection import DetectionOntology,DetectionBaseModel

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union


import torch
from PIL import Image
import cv2
from torch.nn import functional as F

# SegGPT repo files
from seggpt.seggpt_engine import run_one_image
from seggpt.seggpt_inference import prepare_model
from .few_shot_ontology import FewShotOntology

from math import inf

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

ckpt_path = 'seggpt_vit_large.pth'
device = torch.device("cuda")
model = "seggpt_vit_large_patch16_input896x448"

res, hres = 448, 448

# my home-brewed SegGPT utils
from . import colors
from .postprocessing import quantize, quantized_to_bitmasks, bitmasks_to_detections

use_colorings = True

def show_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)

    print(f"Total: {t//(10**9)}, Reserved: {r//(10**9)}, Allocated: {a//(10**9)}, Free: {(t-a)//(10**9)}")

class SegGPT(DetectionBaseModel):

    ontology: FewShotOntology
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, ontology: FewShotOntology):
        self.ontology = ontology
        self.model = prepare_model(ckpt_path, model, colors.seg_type).to(self.DEVICE)
        # print("Model loaded.")

    def preprocess(self, img:np.ndarray)->np.ndarray:
        img = cv2.resize(img, dsize=(res, hres))
        img = img / 255.
        return img
    def imagenet_preprocess(self, img:np.ndarray)->np.ndarray:
        img = img - imagenet_mean
        img = img / imagenet_std
        return img
    
    # convert an img + detections into an img + mask.
    # note: all the detections map to the same base-model class in the Ontology--they can be treated as the same class.
    def prepare_ref_img(self,img:np.ndarray,detections:Detections):
        og_img = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocess(img)
        img = self.imagenet_preprocess(img)

        # draw masks onto image
        mask = np.zeros_like(og_img)
        colors.reset_colors()
        for detection in detections:
            curr_rgb = colors.next_rgb()
            det_box,det_mask,*_ = detection
            mask[det_mask] = curr_rgb
        
        mask = self.preprocess(mask)
        mask = self.imagenet_preprocess(mask)

        return img, mask
    
    # Convert a list of reference images into a SegGPT-friendly batch.
    def prepare_ref_imgs(self,refs:List[Tuple[np.ndarray,Detections]]):
        imgs, masks = [], []
        min_area = inf
        for ref_img, detections in refs:
            img, mask = self.prepare_ref_img(ref_img,detections)

            img_min_area = detections.area.min()
            min_area = min(min_area,img_min_area)

            imgs.append(img)
            masks.append(mask)
        imgs = np.stack(imgs, axis=0)
        masks = np.stack(masks, axis=0)
        return imgs,masks,min_area


    @torch.no_grad()
    def predict(self,input:Union[str,np.ndarray], confidence:int = 0.5) -> sv.Detections:
        detections = []
        for keyId,ref_imgs in enumerate(self.ontology.rich_prompts()):
            ref_imgs,ref_masks,min_ref_area = self.prepare_ref_imgs(ref_imgs)

            if type(input) == str:
                if input in self.ontology.ref_dataset.images:
                    image = Image.fromarray(self.ref_dataset.images[input])
                image = Image.open(input).convert("RGB")
            else:
                image = Image.fromarray(input)

            size = image.size
            input_image = np.array(image)

            img = self.preprocess(input_image)
            img = self.imagenet_preprocess(img) # shape (H,W,C)

            # convert ref_imgs from (N,H,W,C) to (N,2H,W,C)
            img_repeated = np.repeat(img[np.newaxis,...],len(ref_imgs),axis=0)

            # SegGPT uses this weird format--it needs images/masks to be in format (N,2H,W,C)--where the first H rows are the reference image, and the next H rows are the input image.
            img = np.concatenate((ref_imgs,img_repeated),axis=1)
            mask = np.concatenate((ref_masks,ref_masks),axis=1)
            # show_usage()

            for i in range(len(img)):
                cv2.imwrite(f"debug/img_{i}.png",(img[i]*imagenet_std+imagenet_mean)*255)
                cv2.imwrite(f"debug/mask_{i}.png",(mask[i]*imagenet_std+imagenet_mean)*255)

            torch.manual_seed(2)
            output = run_one_image(img, mask, self.model, self.DEVICE)
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2), 
                size=[size[1], size[0]], 
                mode='nearest',
            ).permute(0, 2, 3, 1)[0].numpy()

            quant_output = quantize(output)
            
            if use_colorings:
                to_bitmask_output = quant_output
                to_bitmask_palette = colors.palette
            else:
                to_bitmask_output = quant_output.sum(axis=-1) > 10
                to_bitmask_output = to_bitmask_output[...,None] * 255
                to_bitmask_palette = np.asarray([[255,255,255]])

            bitmasks = quantized_to_bitmasks(to_bitmask_output,to_bitmask_palette)
            new_detections = bitmasks_to_detections(bitmasks, keyId)

            if len(new_detections) > 0:
                detections.append(new_detections)

        # filter <100px detections
        detections = Detections.merge(detections)

        detections = detections[detections.area > min_ref_area * 0.75]
        if len(detections) > 0:
            detections = detections[has_polygons(detections.mask)]

        return detections

from supervision.dataset.utils import approximate_mask_with_polygons
def has_polygons(masks:np.ndarray) -> np.ndarray:
    n,h,w = masks.shape
    
    ret = np.zeros((n,),dtype=bool)
    for i in range(n):
        mask = masks[i]
        polygons = approximate_mask_with_polygons(mask)
        if len(polygons) > 0:
            ret[i] = True

    return ret