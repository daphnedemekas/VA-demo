from rudalle.pipelines import (
    generate_images,
    super_resolution,
)
from rudalle import get_tokenizer, get_realesrgan
import numpy as np
from datetime import datetime
import torch
import os
tokenizer = get_tokenizer()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_params(
    input_text, prompt_text="",confidence="Ultra-Low", variability="Ultra-High", rurealesrgan_multiplier="x1"
):
    from translatepy import Translator
    ts = Translator()

    """Parameters
    Confidence is how closely the AI will attempt to match the input images.
    Higher confidence, the more the AI can go "off the rails".
    This variable is also called "top_p". Think of confidence as the "conceptual similarity" control. Default for prior versions is Low.

    #Variability affects the potential number of options that the AI can choose from for each "token", or each 8x8 chunk of pixels that it generates.
    # Higher variability, higher amount. This variable is also called "top_k". Think of variability as the "stylistic similarity" control. Default for prior versions is "High".

    """
    if confidence == "Ultra-High":
        generation_p = 0.8
    elif confidence == "High":
        generation_p = 0.9
    elif confidence == "Medium":
        generation_p = 0.99
    elif confidence == "Low":
        generation_p = 0.999
    elif confidence == "Ultra-Low":
        generation_p = 0.9999

    if variability == "Ultra-High":
        generation_k = 8192
    elif variability == "High":
        generation_k = 2048
    elif variability == "Medium":
        generation_k = 512
    elif variability == "Low":
        generation_k = 128
    elif variability == "Ultra-Low":
        generation_k = 32

    # @markdown If you'd like to prompt the AI with a different text input than what your images are captioned with, you can do so here. Leave blank to use your `input_text`. Text is automatically translated to Russian.
    if prompt_text == "":
        prompt_text = input_text
    input_lang = ts.language(prompt_text).result.alpha2
    if input_lang != "ru":
        prompt_text = ts.translate(input_text, "ru").result

    # Uses RealESRGAN to upscale your images at the end. That's it! Set to x1 to disable. Not recommended to be combined w/ Stretchsizing.
    realesrgan = None
    if rurealesrgan_multiplier != "x1":
        realesrgan = get_realesrgan(rurealesrgan_multiplier, device=device)


    return generation_p, generation_k, prompt_text, realesrgan
def generate_images_amt(vae, model, images_num, realesrgan, generation_p, generation_k, prompt_text):
    _pil_images, _scores = generate_images(
        prompt_text,
        tokenizer,
        model,
        vae,
        top_k=generation_k,
        images_num=images_num,
        top_p=generation_p,
    )

    if realesrgan:
        _pil_images = super_resolution(_pil_images, realesrgan)
    return _pil_images, _scores

def save_pil_images(
    pil_images,
    scores,
    save_output=True,
    output_filepath="output",
):
    filenames = []
    for k in range(len(pil_images)):
        score = f"{scores[k]}.png"
        if save_output:
            current_time = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            output_name = f"{current_time}_{k}_score:{score}.png"
            f = os.path.join(output_filepath, output_name)
            pil_images[k].save(f)
            filenames.append(f)
    return filenames
    

def filter_by_top(num_imgs, pil_images, scores):
    if num_imgs > len(scores):
        num_imgs = len(scores)-1
    top_indices = np.argpartition(scores, -num_imgs)[-num_imgs:]
    print(top_indices)
    top_imgs = [pil_images[j] for j in top_indices]
    return top_imgs


def generate(
    vae,
    model,
    input_text,
    confidence,
    variability,
    rurealesrgan_multiplier,
    output_filepath,
    num_filtered = None,
    image_amount = 9):
    (
        generation_p,
        generation_k,
        prompt_text,
        realesrgan, #super resolution
    ) = load_params(input_text=input_text, confidence=confidence, variability=variability, rurealesrgan_multiplier=rurealesrgan_multiplier)
    pil_images = []
    scores = []
    pil_images, scores = generate_images_amt(vae, model,
        image_amount, realesrgan, generation_p, generation_k, prompt_text
    )
    if num_filtered is not None:
        pil_images = filter_by_top(num_filtered, pil_images, scores)

    filenames = save_pil_images(pil_images, scores, output_filepath = output_filepath)
    return filenames

def get_closest_training_images_by_text(prompt, directory):
    from sklearn import feature_extraction
    from PIL import Image
    from tqdm import tqdm
    filenames = os.listdir(directory)
    vect = feature_extraction.text.TfidfVectorizer(min_df=1, stop_words = 'english')
    # text_features = model.encode_text(text)
    filenames.append(prompt + '.jpg') 
    tfidf = vect.fit_transform(filenames)
    similarity = (tfidf * tfidf.T).toarray()
    np.fill_diagonal(similarity, np.nan) 

    result_idx = np.nanargmax(similarity[-1])
   
    return filenames[result_idx]



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
