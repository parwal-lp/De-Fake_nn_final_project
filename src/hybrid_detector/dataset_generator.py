import os
import io
import warnings
import json
import subprocess

from IPython.display import display
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

import pandas as pd

def SD_generation(captions_file, image_save_dir, start_row=None, n_row=None):

    df = pd.read_csv(captions_file, skiprows=start_row, nrows=n_row)

    os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
    os.environ['STABILITY_KEY'] = 'sk-ycUshgL0LUVYID9ITyDBcKrcHpdIsov0k8fDA6OVfGIrXMqU'

    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'], # API Key reference.
        verbose=True, # Print debug messages.
        engine="stable-diffusion-v1-5", # Set the engine to use for generation.
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 (<- incompatible with CLIP Guidance) stable-inpainting-v1-0 stable-inpainting-512-v2-0 
    )

    for index, row in df.iterrows():
        text_prompt = df.iloc[index]['caption']
        image_id = df.iloc[index]['img_id']
        #try to generate the image with the given prompt
        try:
            answers = stability_api.generate(
                prompt=text_prompt,
                seed=123126, # Note: Seeded CLIP Guided generations will attempt to stay near its original generation. 
                            # However unlike non-clip guided inference, there's no way to guarantee a deterministic result, even with the same seed.
                steps=30, # Step Count defaults to 30 if not specified here.
                cfg_scale=7.0, # Influences how strongly your generation is guided to match your prompt. Setting this value higher increases the strength in which it tries to match your prompt. Defaults to 7.0 if not specified.
                width=512, # Generation width, defaults to 512 if not included.
                height=512, # Generation height, defaults to 512 if not included.
                sampler=generation.SAMPLER_K_DPMPP_2S_ANCESTRAL, # Choose which sampler we want to denoise our generation with. Defaults to k_dpmpp_2s_ancestral. CLIP Guidance only supports ancestral samplers.
                                                                # (Available Samplers: ddim, k_euler_ancestral, k_dpm_2_ancestral, k_dpmpp_2s_ancestral)
                guidance_preset=generation.GUIDANCE_PRESET_FAST_BLUE # Enables CLIP Guidance. 
                                                                # (Available Presets: _NONE, _FAST_BLUE, _FAST_GREEN)
            )
            # Set up our warning to print to the console if the adult content classifier is tripped. If adult content classifier is not tripped, display generated image.
            for resp in answers:
                for artifact in resp.artifacts:
                    #if the prompt doesnt pass the filter check, generate a completely black image
                    if artifact.finish_reason == generation.FILTER:
                        img = Image.new('RGB', (256, 256), color='black')
                        img.save(f'{image_save_dir}/fake_{image_id}_invalid_prompt.jpg', 'JPEG') #the black image is assigned a recognizable name
                    if artifact.type == generation.ARTIFACT_IMAGE: #if the generation went ok, then save the fake generated image
                        img = Image.open(io.BytesIO(artifact.binary))
                        img.save(f'{image_save_dir}/fake_{image_id}.jpg', 'JPEG')
                        
        except Exception as e:
            #if the error occurred, then print the prompt that was detected as invalid
            print(f"Your request activated the API's safety filters and could not be processed.Please modify the prompt and try again.\nCurrent prompt (detected invalid)s: {text_prompt}")


def LD_generation(captions_file, start_row=None, n_row=None):
    ld_dir = "/home/parwal/Documents/GitHub/latent-diffusion"

    if start_row!=None: n_row+=1 #se non inizio dalla prima riga, devo comunque leggere la riga 0, che contiene gli header, quindi dovro leggere una riga in piu

    df = pd.read_csv(captions_file, skiprows=[i for i in range(1,start_row)], nrows=n_row)

    os.chdir(ld_dir)

    prompt_dictionary = {}

    for index, row in df.iterrows():
        text_prompt = df.iloc[index]['caption']
        image_id = str(df.iloc[index]['img_id'])
        prompt_dictionary[image_id] = text_prompt

    stringed_prompt_dict = json.dumps(prompt_dictionary)

    print(stringed_prompt_dict)

    command = [
        "python3",
        "scripts/txt2img_batch.py",
        "--prompt",
        stringed_prompt_dict,
        "--ddim_eta",
        "0.0",
        "--n_samples",
        "1",
        "--n_iter",
        "1",
        "--scale",
        "10.0",
        "--ddim_steps",
        "100"
    ]
    #print(f"start generating images")
    subprocess.run(command)
    print("-----------------------------------------------------")

    print(f"All images generated successfully. You can find them at latent-diffusion/outputs/txt2img-samples")

    proj_dir = "../De-Fake_nn_final_project"
    os.chdir(proj_dir)


def GLIDE_generation(captions_file, image_save_dir, start_row=None, n_row=None):
    if start_row!=None: n_row+=1
    df = pd.read_csv(captions_file, skiprows=[i for i in range(1,start_row)], nrows=n_row)

    for index, row in df.iterrows():
        prompt = df.iloc[index]['caption']
        image_id = df.iloc[index]['img_id']

        #-------- GENERATE IMAGE WITH BASE MODEL ---------------------------------

        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
            [], options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model.del_cache()

        # ----- UPSCALE IMAGE WITH UPSAMPLER MODEL --------------------------------

        tokens = model_up.tokenizer.encode(prompt)
        tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
            tokens, options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model_up.del_cache()
        up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
        up_samples = diffusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model_up.del_cache()

        # ----- SAVE THE FINAL OUTPUT IMAGE -------------------------------------

        scaled = ((up_samples + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([up_samples.shape[2], -1, 3])
        img = Image.fromarray(reshaped.numpy())
        img.save(f'data/GLIDE+MSCOCO/images/fake_{image_id}.jpg', 'JPEG')