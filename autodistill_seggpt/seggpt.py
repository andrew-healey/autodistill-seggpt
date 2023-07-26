import numpy as np
import supervision as sv
from supervision.dataset.core import DetectionDataset
from supervision import Detections

from autodistill.core import Ontology
from autodistill.detection import DetectionOntology,DetectionBaseModel

from dataclasses import dataclass
from typing import Dict, List, Tuple



@dataclass
class FewShotOntology(Ontology):
    def __init__(self,
                 ref_dataset:DetectionDataset,

                 # each tuple in the list has form:
                 # ( (training_class_name, [reference_image_ids]), output_class_name )]))
                 # i.e. ( ("1-climbing-holds",["demo-holds-1.jpg","demo-holds-2.jpg"]), "climbing-hold" )
                 ontology: List[Tuple[
                        Tuple[str,List[str]],
                        str
                     ]]
        ):
        self.ref_dataset = ref_dataset
        self.ontology = ontology
        rich_ontology = self.enrich_ontology(ontology)
        self.rich_ontology = rich_ontology
    
    def prompts(self)->List[Tuple[str,List[str]]]:
        return [key for key,val in self.ontology]
    def rich_prompts(self)->List[List[Tuple[np.ndarray,Detections]]]:
        return [key for key,val in self.rich_ontology]
    def classes(self)->List[str]:
        return [val for key,val in self.ontology]
    def rich_prompt_to_class(self,rich_prompt:List[Tuple[np.ndarray,Detections]])->str:
        for key,val in self.rich_ontology:
            if key == rich_prompt:
                return val
        raise Exception("No class found for prompt.")

    # using lists-of-pairs instead of dicts:
    def enrich_ontology(self, ontology: List[Tuple[
                        Tuple[str,List[str]],
                        str
                        ]]
        )->List[Tuple[List[Tuple[np.ndarray,Detections]],str]]:

        rich_ontology = []

        for basic_key,val in ontology:
            cls_name, ref_img_names = basic_key

            cls_names = [f"{i}-{cls_name}" for i,cls_name in enumerate(self.ref_dataset.classes)]
            cls_id = cls_names.index(cls_name)

            new_key = []
            for ref_img_name in ref_img_names:
                detections = self.ref_dataset.annotations[ref_img_name]
                detections = detections[detections.class_id==cls_id]
                image = self.ref_dataset.images[ref_img_name]
                new_key.append((image,detections))
            rich_ontology.append((new_key,val))
        return rich_ontology

import torch
from PIL import Image
import cv2
from torch.nn import functional as F

# SegGPT repo files
from seggpt.seggpt_engine import run_one_image
from seggpt.seggpt_inference import prepare_model

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

ckpt_path = 'seggpt_vit_large.pth'
device = torch.device("cuda")
model = "seggpt_vit_large_patch16_input896x448"
seg_type = "instance"

res, hres = 448, 448

# my home-brewed SegGPT utils
from . import colors
from .postprocessing import quantize, quantized_to_bitmasks, bitmasks_to_detections


use_colorings = True

class SegGPT(DetectionBaseModel):

    ontology: FewShotOntology
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, ontology: FewShotOntology):
        self.ontology = ontology
        self.model = prepare_model(ckpt_path, model, seg_type).to(self.DEVICE)
        print("Model loaded.")

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
            # mask += det_mask[...,None] * curr_rgb[None,None,:]
            mask[det_mask] = curr_rgb
        
        cv2.imwrite("debug/mask_demo.png",mask)
        
        mask = self.preprocess(mask)
        mask = self.imagenet_preprocess(mask)

        print("img",img.shape,img.sum(),img.std())
        print("mask",mask.shape,mask.sum(),mask.std())
        
        return img, mask
    
    # Convert a list of reference images into a SegGPT-friendly batch.
    def prepare_ref_imgs(self,refs:List[Tuple[np.ndarray,Detections]]):
        imgs, masks = [], []
        for ref_img, detections in refs:
            img, mask = self.prepare_ref_img(ref_img,detections)
            imgs.append(img)
            masks.append(mask)
        imgs = np.stack(imgs, axis=0)
        masks = np.stack(masks, axis=0)
        return imgs,masks


    def predict(self,input:str, confidence:int = 0.5) -> sv.Detections:
        detections = []
        for keyId,ref_imgs in enumerate(self.ontology.rich_prompts()):
            ref_imgs,ref_masks = self.prepare_ref_imgs(ref_imgs)

            image = Image.open(input).convert("RGB")
            size = image.size
            input_image = np.array(image)

            img = self.preprocess(input_image)
            img = self.imagenet_preprocess(img) # shape (H,W,C)

            # convert ref_imgs from (N,H,W,C) to (N,2H,W,C)
            img_repeated = np.repeat(img[np.newaxis,...],len(ref_imgs),axis=0)

            # SegGPT uses this weird format--it needs images/masks to be in format (N,2H,W,C)--where the first H rows are the reference image, and the next H rows are the input image.
            img = np.concatenate((ref_imgs,img_repeated),axis=1)
            mask = np.concatenate((ref_masks,ref_masks),axis=1)

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
            detections.append(bitmasks_to_detections(bitmasks, keyId))

        # filter <100px detections
        detections = Detections.merge(detections)
        detections = detections[detections.area > 100]
        return detections

    # def predict(self,input:str, confidence:int = 0.5)-> sv.Detections:

    #     image = Image.open(input).convert("RGB")
    #     input_image = np.array(image)
    #     size = image.size
    #     image = np.array(image.resize((res, hres))) / 255.

    #     image_batch, target_batch = [], []
    #     for img2, tgt2 in zip(img2_paths, tgt2_paths):
    #         img2 = cv2.resize(img2, dsize=(res, hres))
    #         img2 /= 255.

    #         tgt2 = Image.open(tgt2_path).convert("RGB")
    #         tgt2 = tgt2.resize((res, hres), Image.NEAREST)
    #         tgt2 = np.array(tgt2) / 255.

    #         tgt = tgt2  # tgt is not available
    #         tgt = np.concatenate((tgt2, tgt), axis=0)
    #         img = np.concatenate((img2, image), axis=0)
        
    #         assert img.shape == (2*res, res, 3), f'{img.shape}'
    #         # normalize by ImageNet mean and std
    #         img = img - imagenet_mean
    #         img = img / imagenet_std

    #         assert tgt.shape == (2*res, res, 3), f'{img.shape}'
    #         # normalize by ImageNet mean and std
    #         tgt = tgt - imagenet_mean
    #         tgt = tgt / imagenet_std

    #         image_batch.append(img)
    #         target_batch.append(tgt)

    #     img = np.stack(image_batch, axis=0)
    #     tgt = np.stack(target_batch, axis=0)
    #     """### Run SegGPT on the image"""
    #     # make random mask reproducible (comment out to make it change)
    #     torch.manual_seed(2)
    #     output = run_one_image(img, tgt, model, device)
    #     output = F.interpolate(
    #         output[None, ...].permute(0, 3, 1, 2), 
    #         size=[size[1], size[0]], 
    #         mode='nearest',
    #     ).permute(0, 2, 3, 1)[0].numpy()
    #     output = Image.fromarray(output.astype(np.uint8))
    #     output.save(out_path)
