from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
import transforms as ctf
from torchvision import transforms as tf
import inspect

def build_transforms(cfg, phase):
    if phase == "train":
        transform = [
            ctf.ConvertFromInts(),
            ctf.PhotometricDistort(),
            ctf.Expand(cfg.INPUT.PIXEL_MEAN),
            ctf.RandomSampleCrop(),
            ctf.RandomMirror(),
            ctf.ToPercentCoords(),
            ctf.Resize(cfg.INPUT.IMAGE_SIZE),
            ctf.SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ctf.ToTensor(),
        ]
        return ctf.Compose(transform)
    elif phase == "style":
        style_size = cfg.ADAIN.INPUT.STYLE_SIZE
        crop = cfg.ADAIN.INPUT.STYLE_CROP
        transform = []
        if style_size != 0:
            transform.append(tf.Resize(style_size))
        if crop:
            transform.append(tf.CenterCrop(style_size))
        transform.append(tf.ToTensor())
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
            ctf.Resize(cfg.INPUT.IMAGE_SIZE),
            ctf.SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ctf.ToTensor()
        ]
        return ctf.Compose(transform)
    else:
        raise RuntimeError("You shouldn't be here")


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
