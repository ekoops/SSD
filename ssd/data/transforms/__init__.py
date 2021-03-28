from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
from torchvision import transforms as tf
import inspect

def build_transforms(cfg, phase):
    if phase == "train":
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
        return Compose(transform)
    elif phase == "style":
        style_size = cfg.ADAIN.INPUT.STYLE_SIZE
        crop = cfg.ADAIN.INPUT.STYLE_CROP
        transform = [tf.ToTensor()]
        if style_size != 0:
            transform.append(tf.Resize(style_size))
        if crop:
            transform.append(tf.CenterCrop(style_size))
        transform = tf.Compose(transform)
        print("<<<<<<<<<<<<<<<<<<<<<<")
        print(inspect.signature(transform))
        print("<<<<<<<<<<<<<<<<<<<<<<")
        return lambda image, boxes, labels: (
            transform(image),
            boxes,
            labels
        )
    elif phase == "test":
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
        return Compose(transform)
    else:
        raise RuntimeError("You shouldn't be here")


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
