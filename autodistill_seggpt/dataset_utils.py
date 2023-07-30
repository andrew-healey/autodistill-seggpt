from supervision import DetectionDataset
from autodistill.detection import DetectionBaseModel

# label dataset

def label_dataset(dataset:DetectionDataset,model:DetectionBaseModel)->DetectionDataset:
    if len(dataset.images)==0:
        # copy dataset
        return DetectionDataset(
            classes=[*dataset.classes],
            images={},
            annotations={},
        )

    # check if any images have masks
    dataset_has_masks = any(dataset.annotations[image_name].mask is not None for image_name in dataset.images)

    # get ontology of model--this determines pred_dataset.classes
    pred_classes = model.ontology.classes()

    # now label all images in dataset
    pred_annotations = {}

    for img_name,img in dataset.images.items():
        detections = model.predict(img)
        pred_annotations[img_name] = detections
    
    pred_dataset = DetectionDataset(
        classes=pred_classes,
        images=dataset.images,
        annotations=pred_annotations,
    )

    return pred_dataset

from random import sample

def shrink_dataset_to_size(dataset:DetectionDataset,max_imgs:int=15)->DetectionDataset:
    imgs = list(dataset.images.keys())

    if len(imgs) <= max_imgs: return dataset
    
    imgs = sample(imgs,max_imgs)

    new_images = {img_name:dataset.images[img_name] for img_name in imgs}
    new_annotations = {img_name:dataset.annotations[img_name] for img_name in imgs}

    return DetectionDataset(
        classes=dataset.classes,
        images=new_images,
        annotations=new_annotations
    )
