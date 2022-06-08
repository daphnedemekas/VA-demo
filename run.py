
from rudalle import get_rudalle_model, get_tokenizer, get_vae
import torch
from model.functions import generate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device)
vae = get_vae().to("cuda")

generate(vae, model, prompt, confidence = confidence, variability = variability, rurealesrgan_multiplier=rurealesrgan_multiplier, output_filepath=directory_name, filter_by_top = 9, image_amount = 20)
