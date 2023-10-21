# Load a checkpoint and find to which category each unit is the most sensitive to (mean activation to a category > 3SD from the mean activation to all other categories)

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
from collections import defaultdict
import clean_cornets    #custom networks based on the CORnet family from di carlo lab


# Load the checkpoint
def load_checkpoint(save_path):
    net = clean_cornets.CORNet_Z_biased_words()
    ckpt_data = torch.load(f'{save_path}/save_lit_bias_z_79_full_nomir.pth.tar', map_location=torch.device('cpu'))
    net.load_state_dict(ckpt_data['state_dict'])
    net.eval()
    return net

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load all images from a given directory so they can be later all processed as a batch
def load_images_from_folder(folder):
    images = []
    files = (p.resolve() for p in Path(folder).glob("*") if p.suffix in {".jpg", ".JPEG"})
    for filename in files:
        img = Image.open(filename)
        if 'L' in img.getbands():
            img = img.convert('RGB') # convert grayscale images to RGB
        img = transform(img)
        images.append(img)
    return images

acts = defaultdict(list) # dictionary to store activations for each category
net = load_checkpoint('/project/3011213.01/Origins-of-VWFA/save')
categories = ['bodies', 'faces', 'houses', 'tools', 'words']
layers = ['v1', 'v2', 'v4', 'it', 'h', 'out']

for cat in categories:
    imgs = load_images_from_folder(f'/project/3011213.01/Origins-of-VWFA/cat_eval/{cat}')
    batch_imgs = torch.stack(imgs)
    batch_imgs = Variable(batch_imgs)

    # Get activations for all images; process images as one batch
    with torch.no_grad():
        ret = net(batch_imgs) # returns a tuple of 6 tensors, one for each layer
        for lx, layer in enumerate(layers):
            h = ret[lx]
            acts[layer].append(h.detach().numpy())


for layer in layers:
    acts_layer = np.stack(acts[layer]) # stack activations for all categories
    # Find units that are most sensitive to a given category
    mean_acts = np.mean(acts_layer, axis=1)
    std_acts = np.std(acts_layer, axis=1)
    thr = (mean_acts + 3 * std_acts) # threshold for each unit: mean activation + 3SD

    # Find units that are most sensitive to a given category
    for cat in np.arange(len(categories)):
        print(f'{layer} {categories[cat]}: {np.sum(mean_acts[cat,:] > thr[np.arange(len(categories)) != cat,:].max(axis=0))}')


