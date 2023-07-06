import os
import io
import warnings

from IPython.display import display
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

import pandas as pd

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-ZNLjRdeC97wjt82n9GxvlHSPylVkqU476vLyOYUJgYhuKJOM'

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-v1-5", # Set the engine to use for generation.
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 (<- incompatible with CLIP Guidance) stable-inpainting-v1-0 stable-inpainting-512-v2-0 
)

captions_file = 'data/MSCOCO/mscoco_captions.csv'
df = pd.read_csv(captions_file)

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
                    img.save(f'data/SD+MSCOCO/images/fake_{image_id}_invalid_prompt.jpg', 'JPEG') #the black image is assigned a recognizable name
                if artifact.type == generation.ARTIFACT_IMAGE: #if the generation went ok, then save the fake generated image
                    img = Image.open(io.BytesIO(artifact.binary))
                    img.save(f'data/SD+MSCOCO/images/fake_{image_id}.jpg', 'JPEG')
                    
    except Exception as e:
        #if the error occurred, then print the prompt that was detected as invalid
        print(f"Your request activated the API's safety filters and could not be processed.Please modify the prompt and try again.\nCurrent prompt (detected invalid)s: {text_prompt}")