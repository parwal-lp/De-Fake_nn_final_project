import json
import subprocess
import pandas as pd
import os

captions_file = 'data/MSCOCO_for_LD/mscoco_captions.csv'

ld_dir = "/home/parwal/Documents/GitHub/latent-diffusion"

df = pd.read_csv(captions_file)

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