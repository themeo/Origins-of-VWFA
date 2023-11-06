# Load a checkpoint and find to which category each unit is the most sensitive to (mean activation to a category > 3SD from the mean activation to all other categories)

import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
from collections import defaultdict
import clean_cornets    #custom networks based on the CORnet family from di carlo lab


# Load the checkpoint
def load_checkpoint(save_path, model_choice='z', mode='lit_bias'):
    if mode == 'lit_bias':
        if model_choice == 'z':
            net = clean_cornets.CORNet_Z_biased_words()
        elif model_choice == 's':
            net = clean_cornets.CORnet_S_biased_words()
    elif mode == 'lit_no_bias':
        if model_choice == 'z':
            net = clean_cornets.CORNet_Z_nonbiased_words()
        elif model_choice == 's':
            net = clean_cornets.CORNet_S_nonbiased_words()
    ckpt_data = torch.load(f'{save_path}/save_{mode}_{model_choice}_79_full_nomir.pth.tar', map_location=torch.device('cpu'))
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
def load_images(files):
    images = []
    for filename in files:
        img = Image.open(filename)
        if 'L' in img.getbands():
            img = img.convert('RGB') # convert grayscale images to RGB
        img = transform(img)
        images.append(img)
    images = torch.stack(images)
    # images = Variable(images)
    return images

mode = 'lit_bias' #  'lit_no_bias', 'lit_bias'
model_choice = 's'
net = load_checkpoint('/project/3011213.01/Origins-of-VWFA/save', mode=mode, model_choice=model_choice)
layers = ['v1', 'v2', 'v4', 'it', 'h', 'out']

acts = defaultdict(list) # dictionary to store activations for each category
category_dir = {}
if 0:
    categories = ['bodies', 'faces', 'houses', 'tools', 'words']
    for cat in categories:
        category_dir[cat] = Path(f'/project/3011213.01/Origins-of-VWFA/cat_eval/{cat}')
else:
    categories = ['imgs', 'words']
    category_dir['imgs'] = Path('/project/3011213.01/imagenet/ILSVRC/Data/CLS-LOC/val')
    category_dir['words'] = Path('/project/3011213.01/Origins-of-VWFA/wordsets/test_acts')


for cat in categories:
    filenames = list(p.resolve() for p in category_dir[cat].glob("**/*") if p.suffix in {".jpg", ".JPEG"})
    filenames = random.sample(filenames, min(1000, len(filenames)))
    batch_imgs = load_images(filenames)

    # Get activations for all images; process images as one batch
    with torch.no_grad():
        ret = net(batch_imgs) # returns a tuple of 6 tensors, one for each layer
        for lx, layer in enumerate(layers):
            h = ret[lx]
            acts[layer].append(h.detach().numpy())


# Find units that are most sensitive to a given category, in each layer
selective_units = {}
for layer in layers:
    acts_layer = np.stack(acts[layer]) # stack activations for all categories
    mean_acts = np.mean(acts_layer, axis=1)
    std_acts = np.std(acts_layer, axis=1)
    thr = (mean_acts + 3 * std_acts) # threshold for each unit: mean activation + 3SD

    # Find units that are most sensitive to a given category
    for cat in np.arange(len(categories)):
        if categories[cat] != 'words':
            continue

        # Compare with the max threshold activation in all other categories
        selectivity_filter = mean_acts[cat,:] > thr[np.arange(len(categories)) != cat,:].max(axis=0)
        print(f'{layer} {categories[cat]}: {np.sum(selectivity_filter)}')
        selective_units[layer] = np.where(selectivity_filter)[0]  # indices of selective units

        if layer == 'h' and categories[cat] == 'words':
            print(selective_units[layer])

# Save the indices of word-sensitive units to an .npy file
np.save(f'/project/3011213.01/Origins-of-VWFA/word_sensitive_units_{mode}_{model_choice}.npy', selective_units)
