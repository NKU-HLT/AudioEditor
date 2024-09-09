import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, List

from prompt2prompt import ptp_utils
from prompt2prompt.attn_control import aggregate_attention
from suppresseot import wo_utils
from suppresseot.attn_loss import AttnLoss


LOW_RESOURCE = True

def update_context(context: torch.Tensor, loss: torch.Tensor, scale: int, factor: float) -> torch.Tensor:
    """
    Update the text embeddings according to the attention loss.

    :param context: text embeddings to be updated
    :param loss: ours loss
    :param factor: factor for update text embeddings.
    :return:
    """
    grad_cond = torch.autograd.grad(outputs=loss.requires_grad_(True), inputs=[context], retain_graph=False)[0]
    context = context - (scale * factor) * grad_cond
    return context

@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=100,
        return_type='image'
):
    ptp_utils.register_attention_control(model, controller)
    height = 256
    width = 1024
    batch_size = len(prompt)

    # text_input = model.tokenizer_list[0]( # auffusion-full
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        # max_length=model.tokenizer_list[0].model_max_length, # auffusion-full
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # text_embeddings = model.text_encoder_list[0](text_input.input_ids.to(model.device))[0] # auffusion-full
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        # uncond_input = model.tokenizer_list[0]([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt") # auffusion-full
        uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        # uncond_embeddings_ = model.text_encoder_list[0](uncond_input.input_ids.to(model.device))[0] # auffusion-full
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        scale = 20
    else:
        uncond_embeddings_ = None
        scale = 5

    latent, _ = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)

    _latent, _latent_erase = latent.clone().to(model.device), latent.clone().to(model.device)
    latents = torch.cat([_latent, _latent_erase])

    attn_loss_func = AttnLoss(model.device, 'cosine', controller.n, controller.token_indices,
                              controller.lambda_retain, controller.lambda_erase, controller.lambda_self_retain, controller.lambda_self_erase)

    model.scheduler.set_timesteps(num_inference_steps)
    # text embedding for erasing
    controller.text_embeddings_erase = text_embeddings.clone()

    scale_range = np.linspace(1., .1, len(model.scheduler.timesteps))
    pbar = tqdm(model.scheduler.timesteps[-start_time:], desc='Suppress EOT', ncols=100, colour="red")
    for i, t in enumerate(pbar):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            if LOW_RESOURCE:
                context = (uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings)
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
            if LOW_RESOURCE:
                context = (uncond_embeddings_, text_embeddings)
        controller.i = i

        # conditional branch: erase content for text embeddings
        if controller.i >= controller.cross_retain_steps:
            controller.text_embeddings_erase = \
                wo_utils.woword_eot_context(text_embeddings.clone(), controller.token_indices, controller.alpha,
                                            controller.method, controller.n)

        controller.baseline = True
        if controller.MAX_STEP_TO_ERASE > controller.i >= controller.cross_retain_steps and not (controller.text_embeddings_erase == text_embeddings).all() and \
                (attn_loss_func.lambda_retain or attn_loss_func.lambda_erase or attn_loss_func.lambda_self_retain or attn_loss_func.lambda_self_erase):
            controller.uncond = False
            controller.cur_att_layer = 32  # w=1, skip unconditional branch
            controller.attention_store = {}
            noise_prediction_text = model.unet(_latent, t, encoder_hidden_states=text_embeddings)["sample"]
            attention_maps = aggregate_attention(controller, 16, ["up", "down"], is_cross=True)
            self_attention_maps = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=False)

            del noise_prediction_text
            # update controller.text_embeddings_erase for some timestep
            iter = controller.iter_each_step
            while iter > 0:
                with torch.enable_grad():
                    controller.cur_att_layer = 32  # w=1, skip unconditional branch
                    controller.attention_store = {}
                    # conditional branch
                    text_embeddings_erase = controller.text_embeddings_erase.clone().detach().requires_grad_(True)
                    # forward pass of conditional branch with text_embeddings_erase
                    noise_prediction_text = model.unet(_latent_erase, t, encoder_hidden_states=text_embeddings_erase)["sample"]
                    model.unet.zero_grad()
                    attention_maps_erase = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=True)
                    self_attention_maps_erase = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=False)

                    # attention loss
                    loss = attn_loss_func(attention_maps, attention_maps_erase, self_attention_maps, self_attention_maps_erase)
                    if loss != .0:
                        pbar.set_postfix({'loss': loss if isinstance(loss, float) else loss.item()})
                        text_embeddings_erase = update_context(context=text_embeddings_erase, loss=loss,
                                                               scale=scale, factor=np.sqrt(scale_range[i]))
                    del noise_prediction_text
                    torch.cuda.empty_cache()
                    controller.text_embeddings_erase = text_embeddings_erase.clone().detach().requires_grad_(False)
                iter -= 1

        # "uncond_embeddings_ is None" for real images, "uncond_embeddings_ is not None" for generated images.
        context_erase = (uncond_embeddings[i].expand(*text_embeddings.shape), controller.text_embeddings_erase) \
            if uncond_embeddings_ is None else (uncond_embeddings_, controller.text_embeddings_erase)
        controller.attention_store = {}
        controller.baseline = False
        contexts = [torch.cat([context[0], context_erase[0]]), torch.cat([context[1], context_erase[1]])]
        latents = ptp_utils.diffusion_step(model, controller, latents, contexts, t, guidance_scale, low_resource=LOW_RESOURCE)
        _latent, _latent_erase = latents
        _latent, _latent_erase = _latent.unsqueeze(0), _latent_erase.unsqueeze(0)

    if return_type == 'image':
        latents = 1 / 0.18215 * latents.detach()
        _image = model.vae.decode(latents)['sample']
        _image = (_image / 2 + 0.5).clamp(0, 1)
        null_edit = _image.cpu().permute(0, 2, 3, 1).numpy()[0]
        eot_edit = _image.cpu().permute(0, 2, 3, 1).numpy()[1]
        image = np.stack((null_edit, eot_edit), axis=0)
    else:
        image = latents
    return image, latent

def run_and_display(ldm_stable, prompts, controller, latent=None, generator=None, uncond_embeddings=None, args=None):
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                                        generator=generator, uncond_embeddings=uncond_embeddings)
    return images, x_t