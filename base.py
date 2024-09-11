import json
import clip
import open_clip
import torch
import torch.nn as nn
import tqdm

from tqdm import tqdm
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


print(open_clip.list_pretrained())

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')#open clip model
#tokenizer = open_clip.get_tokenizer('ViT-B-32')#open clip model
device = torch.device("cuda")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
#model = torch.load('trained.pt', weights_only=False) #uncomment to get updated model
model.eval()
model.to(device)

img_path = "data/Images/pic8.jpg"
image = preprocess(Image.open(img_path)).unsqueeze(0).cuda(device=device)
text = clip.tokenize(["an oven", "lofra"]).cuda(device=device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

print("Label probs:", text_probs)