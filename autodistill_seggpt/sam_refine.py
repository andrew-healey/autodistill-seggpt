import numpy as np
import supervision as sv
import cv2

from segment_anything import SamPredictor

sam_res = (256,256)

def refine_detections(img:np.ndarray,detections:sv.Detections,predictor:SamPredictor):
    # TODO use each detection mask (or bbox) as a prompt for SAM

    predictor.set_image(img)

    new_detections = detections.copy()

    for detection in detections:
        det_box,det_mask,*_ = detection
        resized_mask = cv2.resize(det_mask,sam_res)
        masks_np,iou_predictions,low_res_masks = predictor.predict(mask_input=det_mask,boxes=det_box[None,:])

        print(masks_np.shape,iou_predictions)
        raise 1

        new_detections.append()

