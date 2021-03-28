import numpy as np
import torch
from torch.nn.functional import interpolate
import os
import sys


class AdainLoader():
    def __init__(self, cfg, content_loader, style_loader):
        self.original_size = cfg.INPUT.IMAGE_SIZE
        self.content_loader = content_loader
        self.style_loader = style_loader
        self.alpha = cfg.ADAIN.MODEL.ALPHA
        self.transfer_ratio = cfg.ADAIN.LOADER.TRANSFER_RATIO
        self.device = torch.device(cfg.MODEL.DEVICE)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.insert(0, cfg.ADAIN.IMPL_FOLDER)

        from AdaIN.test import get_net, style_transfer, coral
        vgg_path = cfg.ADAIN.MODEL.VGG
        decoder_path = cfg.ADAIN.MODEL.DECODER
        # 1) Import get_net function in order to
        # prepare the AdaINNet.
        self.adain_net = get_net(decoder_path=decoder_path, vgg_path=vgg_path)
        # 2) Import style_transfer function in order to
        # apply the style transferring
        self.style_transfer = style_transfer
        # 3) Import coral function in order to
        # preserve color
        self.preserve_color = cfg.ADAIN.INPUT.PRESERVE_COLOR
        self.coral = coral if self.preserve_color else None

        sys.path.pop(0)
        #os.chdir(dir_path)

    def __iter__(self):
        for (train_batch, train_targets, train_indexes), (style_batch, _, _) in zip(self.content_loader, self.style_loader):
            train_batch = self.style_batch(
                content_batch=train_batch,
                style_batch=style_batch,
            )
            yield train_batch, train_targets, train_indexes

    def __len__(self):
        return self.content_loader.__len__()

    def style_batch(self, content_batch, style_batch):
        for idx, (content_image, style_image) in enumerate(zip(content_batch, style_batch)):
            if np.random.rand() <= self.transfer_ratio:
                if self.preserve_color:
                    style_image = self.coral(style_image, content_image)
                style_image = style_image.to(self.device).unsqueeze(0)
                content_image = content_image.to(self.device).unsqueeze(0)
                with torch.no_grad():
                    output = self.style_transfer(
                        vgg=self.adain_net.vgg,
                        decoder=self.adain_net.decoder,
                        content=content_image,
                        style=style_image,
                        alpha=self.alpha
                    ).cpu()
                    content_batch[idx] = interpolate(
                        output,
                        (self.original_size, self.original_size)
                    )
        return content_batch
