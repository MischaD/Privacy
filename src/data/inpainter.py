import torch
from abc import ABC, abstractmethod
from einops import repeat
import numpy as np
from numpy.random import Generator, PCG64
import os
from utils import make_exp_config, viz_array, repeat_channels, safe_viz_array
from PIL import Image
import torchvision.transforms as T
from pietorch import blend


_INPAINTERS = {}


def register_inpainter(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _INPAINTERS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _INPAINTERS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_inpainter(config):
    return _INPAINTERS[config.af_inpainter_name](config=config)


class Inpainter(ABC):
    def __init__(self, config):
        self.config = config  # config.saf

    @abstractmethod
    def __call__(self, src, target):
        # B x C x H x W
        mask = 0
        return target, mask

    # @abstractmethod
    # def get_occlusion_augmentation(self, src, target):
    #    return


@register_inpainter(name="circle")
class CircleInpainter(Inpainter):
    def __init__(self, config):
        self.config = config  # config.saf
        self.saf_config = config.data.saf
        self.rng = Generator(PCG64(seed=self.config.seed))

    def __call__(self, src, target):
        c, h, w = target.size()
        mask = self.create_circular_mask(h, w)
        target = torch.clone(target)
        target[0, mask] = 0.5
        target[1, mask] = 0.5
        target[2, mask] = 0.5
        return target, repeat(mask, "h w -> 3 h w")

    def create_circular_mask(self, h, w, center=None):
        if (
            hasattr(self.saf_config, "circle_deterministic_center")
            and self.saf_config.circle_deterministic_center is not None
        ):
            center = self.saf_config.circle_deterministic_center
        else:
            center = (self.rng.integers(h), self.rng.integers(w))

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= self.saf_config.radius
        return torch.tensor(mask)

    # def get_occlusion_augmentation(self, src, target):


@register_inpainter(name="pii")
class PIIInpainter(Inpainter):
    def __init__(self, config):
        self.config = config
        self.saf_config = config.data.saf
        self.rng = Generator(PCG64(seed=self.saf_config.seed))
        self.transforms = T.Compose(
            [
                T.PILToTensor(),
                T.Resize(config.data.image_size),
                T.CenterCrop(config.data.image_size),
            ]
        )
        self.source_mask = None
        self.source_image = None
        self.load_source_image()

    def __call__(self, src, target):
        c, h, w = target.size()
        mask = self.source_mask
        target = torch.clone(target)
        assert (
            target.min() >= 0 and target.max() <= 1
        ), f"It appears there is a problem with the normalization. Min:{target.min()}, max:{target.max()}"

        mask = self.source_mask[0].to(torch.float32)
        corner = torch.tensor([0, 0])

        result = blend(
            target, self.source_image, mask, corner, True, channels_dim=0
        ).clip(0, 1)
        # mask = self.create_circular_mask(center=corner.numpy() + self.saf_config.radius)
        return result, repeat_channels(mask)

    def load_source_image(self):
        base_path = os.path.join(
            os.path.dirname(self.config.data_dir), self.saf_config.pii_source_image
        )
        self.source_image = self.transforms(Image.open(base_path)) / 255.0
        if len(self.source_image.size()) == 2 or self.source_image.size()[0] == 1:
            self.source_image = repeat(self.source_image, "1 h w -> 3 h w")
        # source image shift
        channels, height, width = self.source_image.shape
        shift_width, shift_height = tuple(self.config.data.saf.pii_source_image_shift)

        # shift source image to sample from a different location
        if shift_width != 0 or shift_height != 0:
            shifted_img = torch.zeros_like(self.source_image)
            if shift_width < 0:
                shift_width = abs(shift_width)
                s_sx = shift_width
                s_ex = width
                t_sx = 0
                t_ex = width - shift_width  # negative shift_width
            else:
                s_sx = 0
                s_ex = width - shift_width
                t_sx = shift_width
                t_ex = width

            if shift_height < 0:
                shift_height = abs(shift_height)
                s_sy = shift_height
                s_ey = height
                t_sy = 0
                t_ey = height - shift_height  # negative shift_width
            else:
                s_sy = 0
                s_ey = height - shift_height
                t_sy = shift_height
                t_ey = height
            shifted_img[:, t_sy:t_ey, t_sx:t_ex] = self.source_image[
                :, s_sy:s_ey, s_sx:s_ex
            ]
            self.source_image = shifted_img

        self.source_mask = self.create_circular_mask(
            center=self.saf_config.pii_source_center
        )
        safe_viz_array(
            self.source_image, os.path.join(self.config.log_dir, "source_img.png")
        )
        safe_viz_array(
            self.source_mask, os.path.join(self.config.log_dir, "source_mask.png")
        )

    def create_circular_mask(self, center):
        h, w = tuple(self.source_image.size()[-2:])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= self.config.data.saf.radius
        mask = torch.tensor(mask)
        mask = repeat(mask, "h w -> 3 h w")
        return mask


@register_inpainter(name="identity")
class IdentityInpainter(Inpainter):
    def __init__(self, config):
        self.config = config
        self.saf_config = config.data.saf

    def __call__(self, src, target):
        return target, torch.zeros_like(target[0])
