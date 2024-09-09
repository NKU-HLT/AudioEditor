import os
import ast
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
from diffusers.utils.import_utils import is_xformers_available

from diffusers import StableDiffusionPipeline, DDIMScheduler
from auffusion.auffusion_pipeline import AuffusionPipeline

from null_text_inversion.null_text_inversion import NullInversion
from utils.converter import load_wav, mel_spectrogram, normalize_spectrogram, denormalize_spectrogram, Generator, get_mel_spectrogram_from_audio
from utils.utils import pad_spec, image_add_color, torch_to_pil, normalize, denormalize


from prompt2prompt.attn_control import AttentionStore, show_cross_attention
from suppresseot.run_and_display import run_and_display

### We will release the code for the main methods proposed in our paper 
### after its acceptance immediately!

def main():
    gen_audio()
    edit_audio()

    
if __name__=="__main__":
    main()