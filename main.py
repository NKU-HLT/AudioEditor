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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_version", type=str, default='auffusion-full-no-adapter', help='version of stable diffusion model.')
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inference.")
    parser.add_argument("--output_dir", type=str, default="./audio_examples/output_audios", help="The output directory where the model predictions will be written.")
    parser.add_argument('--prompt', type=str, default='After a gunshot, there was a burst of dog barking', help='prompt for generated or real audio')
    parser.add_argument('--audio_path', type=str, default='audio_examples/input_audios/After a gunshot, there was a burst of dog barking.wav', help='audio path')

    parser.add_argument("--enable_xformers", default=True, action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="The scale of guidance.")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of inference steps to perform.")
    
    parser.add_argument("--width", type=int, default=1024, help="Width of the spec.")
    parser.add_argument("--height", type=int, default=256, help="Height of the spec.") 
    parser.add_argument("--sample_rate", type=int, default=16000, help="The sample rate of audio.")
    parser.add_argument("--duration", type=int, default=10, help="The duration(s) of audio.")
    
    
    parser.add_argument("--input_dir", type=str, default="./audio_examples/input_audios", help="The input directory where put the audios to be edited.")
    parser.add_argument('--inversion', type=str, default='NT', help='NT (Null-text), NPI (Negative-prompt-inversion).')
    
    parser.add_argument('--token_indices', type=ast.literal_eval, default='[[1,]]', help='index of without words.')
    parser.add_argument('--cross_retain_steps', type=ast.literal_eval, default='[.2,]', help='perform the "wo" punish when step >= cross_wo_steps')
    parser.add_argument('--alpha', type=ast.literal_eval, default='[1.,]', help="punishment ratio")
    parser.add_argument('--iter_each_step', type=int, default=5, help="the number of iteration for each step to update text embedding")
    parser.add_argument('--max_step_to_erase', type=int, default=20, help='erase/suppress max step of diffusion model')
    parser.add_argument('--method', type=str, default='soft-weight', help='soft-weight, alpha, beta, delete, weight')
    
    parser.add_argument('--lambda_retain', type=float, default=1., help='lambda for cross attention retain loss')
    parser.add_argument('--lambda_erase', type=float, default=-.5, help='lambda for cross attention erase loss')
    parser.add_argument('--lambda_self_retain', type=float, default=1., help='lambda for self attention retain loss')
    parser.add_argument('--lambda_self_erase', type=float, default=-.5, help='lambda for self attention erase loss')
    
    args = parser.parse_args()
    return args

def load_model(sd_version, device):
    if sd_version == "auffusion-full-no-adapter":
        pretrained_model_name_or_path = "ckpt/auffusion-full-no-adapter"
        ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
    elif  sd_version == sd_version == "sd_1_4":
        pretrained_model_name_or_path = "/home/jiayuhang/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, scheduler=scheduler).to(device)
    else:
        raise ValueError('Unsupported stable diffusion version')
    return ldm_stable


def load_1024_256(audio_path):
    audio, sampling_rate = load_wav(audio_path)
    audio, spec = get_mel_spectrogram_from_audio(audio)
    norm_spec = normalize_spectrogram(spec)
    norm_spec = pad_spec(norm_spec, 1024)
    norm_spec = norm_spec.permute(1, 2, 0).cpu().numpy() # (256, 1024, 3) # reshape
    return norm_spec


def store_1024_256(output_spec, sample_rate, audio_path):
    ### vocoder
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    vocoder = Generator.from_pretrained("ckpt/auffusion-full-no-adapter", subfolder="vocoder")
    vocoder = vocoder.to(device=device, dtype=torch.float16)
    
    norm_spec = output_spec[:, :1000, :]
    norm_spec = torch.from_numpy(norm_spec).permute(2, 0, 1).to(device=device)  # reshape torch.Size([3, 256, 1000])
    denorm_spec = denormalize_spectrogram(norm_spec)
    with torch.autocast("cuda"):
        denorm_spec_audio = vocoder.inference(denorm_spec)
    write(audio_path, sample_rate, denorm_spec_audio.squeeze())
    print(f"Successfully save {audio_path}")

    
def show_spec(output_spec, audio_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    norm_spec = torch.from_numpy(output_spec).permute(2, 0, 1).to(device=device)
    raw_image = image_add_color(torch_to_pil(norm_spec[:,:,:1000]))
    raw_image.save(audio_path)
    
    
def edit_audio():
    args = parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = load_model(args.sd_version, device)
    ldm_stable.set_progress_bar_config(disable=True)
    if is_xformers_available() and args.enable_xformers:
        ldm_stable.enable_xformers_memory_efficient_attention()
    
    orig_prompt = args.prompt
    input_audio_path = args.audio_path
    input_audio = load_1024_256(input_audio_path)
    os.makedirs(os.path.join(args.output_dir, orig_prompt), exist_ok=True)

    # Null-Text Inversion
    null_inversion = NullInversion(ldm_stable)
    with torch.autocast("cuda"):
        (audio_gt, audio_rec), x_t, uncond_embeddings = null_inversion.invert(input_audio, orig_prompt, inversion=args.inversion, \
                                                                                num_inner_steps=args.iter_each_step, verbose=True)
        
    show_spec(audio_gt, f"audio_examples/output_audios/{orig_prompt}/audio_gt.png")
    show_spec(audio_rec, f"audio_examples/output_audios/{orig_prompt}/audio_rec.png")
    store_1024_256(audio_gt, args.sample_rate, f"audio_examples/output_audios/{orig_prompt}/audio_gt.wav")
    store_1024_256(audio_rec, args.sample_rate, f"audio_examples/output_audios/{orig_prompt}/audio_rec.wav")
    
    #SuppressEOT
    for token_indices in args.token_indices:
        for cross_retain_steps in args.cross_retain_steps:
            for alpha in args.alpha:
                controller = AttentionStore(token_indices, \
                                            alpha, \
                                            args.method, \
                                            cross_retain_steps, \
                                            len(ldm_stable.tokenizer.encode((orig_prompt))), \
                                            args.iter_each_step, \
                                            args.max_step_to_erase, \
                                            lambda_retain=args.lambda_retain, \
                                            lambda_erase=args.lambda_erase, \
                                            lambda_self_retain=args.lambda_self_retain, \
                                            lambda_self_erase=args.lambda_self_erase)
                
                with torch.autocast("cuda"):
                    audio_edit, x_t = run_and_display(ldm_stable, \
                                                    [orig_prompt], \
                                                    controller, \
                                                    latent=x_t, \
                                                    uncond_embeddings=uncond_embeddings, \
                                                    args=args)
                
                
                show_spec(audio_edit[1], f"audio_examples/output_audios/{orig_prompt}/audio_eot_edit.png")
                store_1024_256(audio_edit[1], args.sample_rate, f"audio_examples/output_audios/{orig_prompt}/audio_eot_edit.wav")
                

def gen_audio():
    args = parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = load_model(args.sd_version, device)
    ldm_stable.set_progress_bar_config(disable=True)
    if is_xformers_available() and args.enable_xformers:
        ldm_stable.enable_xformers_memory_efficient_attention()
        
    generator = torch.Generator(device=device).manual_seed(args.seed)
    text_prompt = args.edit_prompt
    audio_path = os.path.join(args.input_dir, text_prompt + '.wav')

    with torch.autocast("cuda"):
        output = ldm_stable(prompt = text_prompt, \
                            num_inference_steps = args.num_inference_steps, \
                            guidance_scale = args.guidance_scale, \
                            generator = generator, \
                            width = args.width, \
                            height = args.height,
                            output_type="pt") 

    if args.sd_version == "auffusion":
        audio_length = args.sample_rate * args.duration
        audio = output.audios[0][:audio_length]
        write(audio_path, args.sample_rate, audio)
    elif args.sd_version == "auffusion-full-no-adapter":
        output_spec = output.images[0].permute(1, 2, 0).cpu().numpy()
        store_1024_256(output_spec, args.sample_rate, audio_path)
    else:
        raise ValueError('Unsupported stable diffusion version')
    

def main():
    # gen_audio()
    edit_audio()

    
if __name__=="__main__":
    main()