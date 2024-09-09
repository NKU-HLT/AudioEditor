import abc

import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List, Dict, Optional
import matplotlib.pyplot as plt

from prompt2prompt import ptp_utils

LOW_RESOURCE = True
NUM_DDIM_STEPS = 100

class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, token_indices: List[int], alpha: float, method: str, cross_retain_steps: float, n: int, iter_each_step: int, max_step_to_erase: int,
                 lambda_retain=1, lambda_erase=-.5, lambda_self_retain=1, lambda_self_erase=-.5):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.baseline = True
        # for suppression content
        self.ddim_inv = False
        self.token_indices = token_indices
        self.uncond = True
        self.alpha = alpha
        self.method = method  # default: 'soft-weight'
        self.i = None
        self.cross_retain_steps = cross_retain_steps * NUM_DDIM_STEPS
        self.n = n
        self.text_embeddings_erase = None
        self.iter_each_step = iter_each_step
        self.MAX_STEP_TO_ERASE = max_step_to_erase
        # lambds of loss
        self.lambda_retain = lambda_retain
        self.lambda_erase = lambda_erase
        self.lambda_self_retain = lambda_self_retain
        self.lambda_self_erase = lambda_self_erase


def aggregate_attention(prompts, attention_store: AttentionStore, res: List[int], from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res[0] * res[1]
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res[0], res[1], item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_cross_attention(stable, prompts, attention_store: AttentionStore, res: List[int], from_where: List[str], select: int = 0, image_size: List[int]=[1024, 256], num_rows: int = 1, font_scale=2, thickness=4, cmap_name="plasma", save_name="null-text+ptp"):
    # tokens = stable.tokenizer_list[0].encode(prompts[select])
    # decoder = stable.tokenizer_list[0].decode                    # auffusion-full
    tokens = stable.tokenizer.encode(prompts[select])
    decoder = stable.tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []

    cmap = plt.get_cmap(cmap_name)
    cmap_r = cmap.reversed()

    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)

        image = cmap(np.array(image)[:,:,0])[:, :, :3]
        image = (image - image.min()) / (image.max() - image.min())
        image = Image.fromarray(np.uint8(image*255))
        image = np.array(image.resize(image_size))

        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])), font_scale=font_scale, thickness=thickness)
        images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0), num_rows=num_rows, save_name=save_name)