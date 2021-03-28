from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
from torchvision import transforms as tf


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
    elif phase == "style":
        style_size = cfg.ADAIN.INPUT.STYLE_SIZE
        crop = cfg.ADAIN.INPUT.STYLE_CROP
        transform = []
        if style_size != 0:
            transform.append(tf.Resize(style_size))
        if crop:
            transform.append(tf.CenterCrop(style_size))
    elif phase == "test":
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    else:
        raise RuntimeError("You shouldn't be here")
    if phase != "style":
        transform = Compose(transform)
    else:
        transform = tf.Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
