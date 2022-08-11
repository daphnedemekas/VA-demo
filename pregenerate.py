
from rudalle import get_rudalle_model, get_tokenizer, get_vae
import torch
from model.functions import generate
import os 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device)
vae = get_vae().to("cuda")

model_path = os.path.join(f'../../VA-design-generator/checkpoints/lookingglass_dalle_90000.pt')
model.load_state_dict(torch.load(model_path))


def construct_prompt(collection, century, style, artist, subject, material):
    prompt = ''
    prompt += f'{collection}, '
    prompt += f'{century}, '
    prompt += f'of style {style}, '
    if artist == 'Rie Lucie':
        artist = 'Lucie Rie'
    if artist == 'Iznik':
        artist = 'IZNIK'
    prompt += f'by {artist}, '
    prompt += f'{subject}, '
    prompt += f'{material}'

    return prompt

os.mkdir("pregenerated/")

for collection in ["Ceramics", "Fashion", "Furniture & Metalwork"]:
    for century in ["16th century", "18th century", "20th century"]:
        for subject in ["Teapot", "Tile", "Figure", "Costume", "Dress", "Shoes","Candlestick", "Clock", "Chair"]:
            for style in ["Arts & Crafts", "Ottoman", "Modernist", "East Asian", "High Fashion", "Theatrical", "Art Deco", "Baroque", "Rococo"]:
                for artist in ['Lucie Rie', 'Johann Kandler', 'Iznik', 'Versace', 'Vivienne Westwood', 'Yoruba Women','Ashbee Robert','Hester Bateman', 'Joseph Wilmore']:
                    for material in ['Earthenware','Glass', 'Marble', 'Cotton', 'Silk',  'Leather', 'Wood', 'Metal', 'Silver']:
                        prompt = construct_prompt(collection, century, style, artist, subject, material)
                        filepath = f'pregenerated/{prompt}'
                        if not os.path.exists(filepath):
                            os.mkdir(filepath)
                        prompt += '.jpg'
                        generate(vae, model, prompt, confidence = 'Low', variability = 'Ultra-High', rurealesrgan_multiplier='x8', output_filepath=filepath, image_amount = 1)
