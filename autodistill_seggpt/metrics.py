from supervision import DetectionDataset,Detections
import numpy as np

eps = 1e-6

def iou(gt_dataset:DetectionDataset,pred_dataset:DetectionDataset)->float:
    # assert gt_dataset.classes == pred_dataset.classes, f"gt classes: {gt_dataset.classes}, pred classes: {pred_dataset.classes}"
    assert gt_dataset.images.keys() == pred_dataset.images.keys()

    running_intersection = 0
    running_union = 0

    for img_name in gt_dataset.images:
        gt_detections = gt_dataset.annotations[img_name]
        pred_detections = pred_dataset.annotations[img_name]

        img = gt_dataset.images[img_name]

        gt_mask = get_combined_mask(img,gt_detections)
        pred_mask = get_combined_mask(img,pred_detections)

        intersection = np.sum(gt_mask*pred_mask)
        union = np.sum(np.logical_or(gt_mask,pred_mask))

        running_intersection += intersection
        running_union += union
    
    return running_intersection/(running_union+eps)


def get_combined_mask(img:np.ndarray,detections:Detections)->np.ndarray:
    mask = np.zeros(img.shape[:2],dtype=np.uint8)

    for detection in detections:
        det_box,det_mask,*_ = detection
        mask[det_mask.astype(bool)] = 1

    mask = np.clip(mask,0,1)

    return mask