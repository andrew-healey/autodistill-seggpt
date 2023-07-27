# Given a dataset of a few images, find a good ontology for it.
# An ontology has k example images for each class.
# Try a bunch of random k-example-image-sets for each class, then pick the one which generalizes best to the rest of the images.
# Then combine those best-performing k-example-image-sets into a single ontology.
import supervision as sv
from supervision.dataset.core import DetectionDataset
from autodistill.detection import DetectionBaseModel
from .few_shot_ontology import FewShotOntology

from .metrics import iou

from typing import List,Type

from random import sample

import numpy as np

from tqdm import tqdm

import math

# TODO: make multiple eval metrics. A metric could fit the interface get_score(gt_dataset,pred_dataset)->float,str.
# The float is the score, the str is a human-readable description of the score (plus some extra metadata like mAP-large, etc.)
# Metrics could include: mask AP, mask IoU, box AP, etc.

def find_best_examples(
        ref_dataset:DetectionDataset,
        model_class:Type[DetectionBaseModel],
        num_examples:int=2,
        num_trials:int=10
):
    # create few-shot ontologies for each class.
    cls_names = [f"{i}-{cls_name}" for i,cls_name in enumerate(ref_dataset.classes)]

    best_examples = {}

    for i,cls in enumerate(cls_names):
        # get best example set for this class
        examples_scores = []

        positive_examples = []
        gt_detections = {}

        for img_name in ref_dataset.images:
            detections = ref_dataset.annotations[img_name]
            detections = detections[detections.class_id==np.array([i])]
            detections.class_id = np.zeros_like(detections.class_id) # we use only one class for each ontology

            gt_detections[img_name] = detections

            if len(detections)>0:
                positive_examples.append(img_name)
        
        if len(positive_examples)==0:
            best_examples[cls] = []
            continue
        
        num_combos = choose(len(positive_examples),num_examples)

        num_iters = min(num_combos,num_trials)
        combo_hashes = range(num_combos)
        sub_combo_hashes = sample(combo_hashes,num_iters)

        print(f"Finding best examples for class {cls}.")

        combo_pbar = tqdm(sub_combo_hashes)
        max_score = -math.inf

        for combo_hash in combo_pbar:
            image_choices = combo_hash_to_choices(combo_hash,positive_examples,num_examples) 

            onto_tuples = [(
                (cls,image_choices),
                cls
            )]

            ontology = FewShotOntology(ref_dataset,onto_tuples)

            model = model_class(ontology) # model must take only an Ontology as a parameter

            pred_detections = {}

            # inference on dataset
            for img_name in ref_dataset.images:
                img = ref_dataset.images[img_name]
                detections = model.predict(img)
                pred_detections[img_name] = detections
            
            gt_dataset = DetectionDataset(classes=[cls],images=ref_dataset.images,annotations=gt_detections)
            pred_dataset = DetectionDataset(classes=[cls],images=ref_dataset.images,annotations=pred_detections)

            score = iou(gt_dataset,pred_dataset).tolist()

            examples_scores.append((image_choices,score))

            max_score = max(max_score,score)
            combo_pbar.set_description(f"Best IoU: {round(max_score,2)}")
        
        my_best_examples,best_score = max(examples_scores,key=lambda x:x[1])
        best_examples[cls] = my_best_examples
    return best_examples


def combo_hash_to_choices(fact:int,candidates:List[any],num_choices:int)->List[any]:
    chosen = []
    curr_fact = fact
    remaining_candidates = candidates
    for i in range(num_choices,0,-1):
        which_remaining_candidate = curr_fact%i
        chosen.append(remaining_candidates.pop(which_remaining_candidate))
        curr_fact = curr_fact//i
    return chosen

import math
def choose(n,k):
    return int(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))


