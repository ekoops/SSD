from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
from torchvision import transforms


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
        style_size = cfg.ADAIN.INPUT.STYLE.SIZE
        crop = cfg.ADAIN.INPUT.STYLE.CROP
        transform = []
        if style_size != 0:
            transform.append(transforms.Resize(style_size))
        if crop:
            transform.append(transforms.CenterCrop(style_size))
    elif phase == "test":
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    else:
        raise RuntimeError("You shouldn't be here")
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
