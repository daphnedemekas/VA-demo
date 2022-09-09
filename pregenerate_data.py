import numpy as np
import os
from PIL import Image

def get_closest_training_images_by_clip(prompt, directory):
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    from tqdm import tqdm
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    filenames = os.listdir(directory)[::3]
    score = 0
    for i, f in enumerate(tqdm(filenames)):
        image = Image.open(f'{directory}/{f}')
        inputs = processor(text=[prompt + '.jpg'], images = image, return_tensors = 'pt', padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        s = logits_per_image.item()
        if s > score:
            score = s
            index = i
    return filenames[index]


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

os.mkdir("pregenerated_data/")

for collection in ["Ceramics", "Fashion", "Furniture & Metalwork"]:
    for century in ["16th century", "18th century", "20th century"]:
        for subject in ["Teapot", "Tile", "Figure", "Costume", "Dress", "Shoes","Candlestick", "Clock", "Chair"]:
            for style in ["Arts & Crafts", "Ottoman", "Modernist", "East Asian", "High Fashion", "Theatrical", "Art Deco", "Baroque", "Rococo"]:
                for artist in ['Lucie Rie', 'Johann Kandler', 'Iznik', 'Versace', 'Vivienne Westwood', 'Yoruba Women','Ashbee Robert','Hester Bateman', 'Joseph Wilmore']:
                    for material in ['Earthenware','Glass', 'Marble', 'Cotton', 'Silk',  'Leather', 'Wood', 'Metal', 'Silver']:
                        prompt = construct_prompt(collection, century, style, artist, subject, material)
                        filepath = f'pregenerated_data/{prompt}'
                        if not os.path.exists(filepath):
                            os.mkdir(filepath)
                        prompt += '.jpg'

                        if collection == 'Ceramics':

                            img_filename = get_closest_training_images_by_clip(artist, prompt, '../../VA-design-generator/images-labelled/images-labelled/ceramics')
                            img = Image.open(f'../../VA-design-generator/images-labelled/images-labelled/ceramics/{img_filename}')
                            img.save(filepath/{img_filename})
                        elif collection == 'Fashion':
                            img_filename = get_closest_training_images_by_clip(artist, prompt, '../../VA-design-generator/images-labelled/images-labelled/fashion')
                            img = Image.open(f'../../VA-design-generator/images-labelled/images-labelled/fashion/{img_filename}')
                            img.save(filepath/{img_filename})

                        elif collection.value == 'Furniture & Metalwork':
                            img_filename = get_closest_training_images_by_clip(artist, prompt, '../../VA-design-generator/images-labelled/images-labelled/furniture')
                            img = Image.open(f'../../VA-design-generator/images-labelled/images-labelled/furniture/{img_filename}')
                            img.save(filepath/{img_filename})
