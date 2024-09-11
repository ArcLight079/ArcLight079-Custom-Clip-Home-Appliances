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

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#tokenizer = open_clip.get_tokenizer('ViT-B-32')
device = torch.device("cuda")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.to(device)
class image_title_dataset():
    def __init__(self, list_image_path,list_txt,transform):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  =  clip.tokenize(list_txt)
        self.transform=transform

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        image = self.transform(image)
        return image, title
list_image_path=['data/Images/pic7.jpg','data/Images/pic8.jpg']
list_txt=['lofra','lofra']


transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset=image_title_dataset(list_image_path,list_txt,transform=transforms)

trainDataLoader=DataLoader(dataset,shuffle=True, batch_size=15)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

num_epochs = 30
for epoch in range(num_epochs):
    pbar = tqdm(trainDataLoader, total=len(trainDataLoader))
    for batch in pbar:
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)
        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
torch.save(model, 'trained')