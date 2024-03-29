
from rudalle import get_rudalle_model, get_tokenizer, get_vae
import torch
from model.functions import generate
import os 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device)
vae = get_vae().to("cuda")

model_path = os.path.join(f'../../VA-design-generator/checkpoints/lookingglass_dalle_90000.pt')
model.load_state_dict(torch.load(model_path))

prompts = ['Fashion 16th century of style High Fashion, Theatrical, Baroque, Figure',
'Fashion 17th century of style High Fashion, Theatrical, Figure, by Versace',
'Fashion 18th century of style High Fashion, Theatrical, Figure, by Balenciaga',
'Fashion 18th century of style Rococo, Theatrical, Figure, by Versace',
'Fashion 18th century of style Rococo, Art Nouveau, Figure, by IZNIK',
'Fashion 19th century of style Rococo, Art Nouveau, Figure, by IZNIK',
'Fashion 20th century of style High Fashion, Theatrical, Cocktail, Figure, by Versace',
'Fashion 20th century of style High Fashion, Art Deco, Dress Figure, by Yoruba Women',
'Fashion 20th century of style High Fashion, Art Deco, Dress Figure, by Yves Saint Laurent',
'Fashion 20th century of style High Fashion, Art Deco, Dress Figure, by Versace',
'Fashion 20th century of style High Fashion, East Asian, Dress Figure, by Versace',

'Fashion of style East Asian, Art Deco, Figure']
for prompt in prompts:
    filepath = f'output/{prompt}'
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    generate(vae, model, prompt, confidence = 'Low', variability = 'Ultra-High', rurealesrgan_multiplier='x8', output_filepath=filepath, image_amount = 10)
