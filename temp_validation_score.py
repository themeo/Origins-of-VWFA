
import subprocess, shlex, shutil, io, os
import torch, torchvision, glob, tqdm, scipy, gc, time, re
from time import strftime, localtime
from datetime import timedelta
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils import data
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

import clean_cornets    #custom networks based on the CORnet family from di carlo lab
##import analysis     #custom module for analysing models
#import plots        #custom module for plotting
import ds2          #custom module for datasets

import argparse 
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_choice', default='s',
                    help='z for cornet Z,  s for cornet S')
parser.add_argument('--img_path', default='/project/3011213.01/imagenet/ILSVRC/Data/CLS-LOC',
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--wrd_path', default='wordsets2',
                    help='path to word folder that contains train and val folders')
parser.add_argument('--save_path', default='save/',
                    help='path for saving ')
parser.add_argument('--output_path', default='activations/',
                    help='path for storing activations')
parser.add_argument('--restore_file', default=None,
                    help='name of file from which to restore model (ought to be located in save path, e.g. as save/cornet_z_epoch25.pth.tar)')
parser.add_argument('--img_classes', default=1000,
                    help='number of image classes')
parser.add_argument('--wrd_classes', default=1000,
                    help='number of word classes')
parser.add_argument('--num_train_items', default=1300,
                    help='number of training items in each category')
parser.add_argument('--num_val_items', default=50,
                    help='number of validation items in each category')
parser.add_argument('--num_workers', default=20,
                    help='number of workers to load batches in parallel')
parser.add_argument('--mode', default='pre',
                    help='pre for pre-schooler mode, lit for literate mode')
parser.add_argument('--max_epochs_pre', default=50, type=int,
                    help='number of epochs to run as pre-schooler - training on images only')
parser.add_argument('--max_epochs_lit', default=30, type=int,
                    help='number of epochs to run as literate - training on images and words')
parser.add_argument('--batch_size', default=1200, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.01, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_schedule', default='StepLR')
parser.add_argument('--step_size', default=20, type=int,
                    help='after how many epoch learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')

FLAGS, _ = parser.parse_known_args()
main_dir = '/project/3011213.01/Origins-of-VWFA/'


# useful
def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def AccLogit(out, label):
    # out and labels are tensors
    out, label = out.cpu(), label.cpu()
    out, label = np.argmax(out.detach().numpy(), axis=1), np.argmax(label.numpy(), axis=1)
    score = 100*np.mean(out==label)
    #print ('score', score)
    return score
    
def Acc(out, label, Print=0):
    # out and labels are tensors
    out, label = out.cpu(), label.cpu()
    out, label = np.argmax(out.detach().numpy(), axis=1), label.numpy()
    score = 100*np.mean(out==label)
    # print ('out', out)
    # print ('label', label)
    # print ('')
    return score


def train(mode = FLAGS.mode, model_choice=FLAGS.model_choice, restore_path=None, save_path=FLAGS.save_path, plot=0, show=0):
    start_time = time.time()
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #device = "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"Using {device} for training")

    if mode == 'pre':
        print('building pre-schooler model')

        # Datasets and Generators
        # train_imgset = ds2.ImageDataset(data_path=FLAGS.img_path, folder='train')
        # training_gen = data.DataLoader(train_imgset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
        # del train_imgset
        gc.collect()

        print('loading validation set')
                
        val_imgset = ds2.ImageDataset(data_path=FLAGS.img_path, folder='val')
        validation_gen = data.DataLoader(val_imgset, batch_size=FLAGS.num_val_items, shuffle=False, num_workers=FLAGS.num_workers)
        
        # variables, labels, prints, and titles for plots
        cat_scores = np.zeros((FLAGS.max_epochs_pre, FLAGS.img_classes))
 
        trainloss, valloss = [], []
        max_epochs = FLAGS.max_epochs_pre
        shift_epoch = 0
        print_save = 'saving pre-schooler model'

        checkpoint_files = glob.glob(f'{main_dir}/{FLAGS.save_path}/save_{mode}_{model_choice}_*_full_nomir.pth.tar')
        epochs = [int(re.search(r'_(\d+)_full_nomir.pth.tar', file).group(1)) for file in checkpoint_files]
        start_epoch = max(epochs) if len(epochs) else -1
        if start_epoch >= 0:
            print('Last-trained epoch:', start_epoch)

            # Load checkpoint data
            ckpt_data = torch.load(f'{main_dir}/{FLAGS.save_path}save_{mode}_{model_choice}_{start_epoch}_full_nomir.pth.tar')
            assert start_epoch == ckpt_data['epoch']

            for k in list(ckpt_data['state_dict'].keys()):
                if 'module' in k:
                    ckpt_data['state_dict'][k.replace('module.', '')] = ckpt_data['state_dict'].pop(k)


        print(f'loading pre-schooler model {model_choice}')
        if model_choice == 'z':
            net = clean_cornets.CORnet_Z_tweak(out_img=FLAGS.img_classes)
        elif model_choice == 's':
            net = clean_cornets.CORnet_S_tweak(out_img=FLAGS.img_classes)
        
        if ckpt_data:        
            net.load_state_dict(ckpt_data['state_dict'])

        print('pre-schooler model has been built')


    if 'lit' in mode:
        print ('building literate model')
        # Datasets and Generators
        print ('loading datasets')
        train_wrdset = ds2.WordDataset(data_path=FLAGS.wrd_path, folder='train')
        train_img2set = ds2.Image2Dataset(data_path=FLAGS.img_path, folder='train')
        train_set = torch.utils.data.ConcatDataset((train_img2set,train_wrdset))
        training_gen = data.DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
        
        val_wrdset = ds2.WordDataset(data_path=FLAGS.wrd_path, folder='val')
        val_img2set = ds2.Image2Dataset(data_path=FLAGS.img_path, folder='val')
        val_set = torch.utils.data.ConcatDataset((val_img2set, val_wrdset))
        validation_gen = data.DataLoader(val_set, batch_size=FLAGS.num_val_items, shuffle=False, num_workers=FLAGS.num_workers)
        
        # variables, labels, prints, and titles for plots
        print('loading variables')
        classes = FLAGS.img_classes + FLAGS.wrd_classes
        max_epochs = FLAGS.max_epochs_lit
        cat_scores = np.zeros((FLAGS.max_epochs_pre + FLAGS.max_epochs_lit, classes))
        cat_scores_pre = np.load(FLAGS.save_path + 'cat_scores_pre_z_full_nomir.npy')
        cat_scores[:FLAGS.max_epochs_pre, :-FLAGS.wrd_classes] = np.copy(cat_scores_pre[:FLAGS.max_epochs_pre])
        print ('np.shape(cat_scores)',np.shape(cat_scores))        
        trainloss, valloss = np.load(FLAGS.save_path + 'trainloss_pre_z_full_nomir.npy').tolist(), np.load(FLAGS.save_path + 'valloss_pre_z_full_nomir.npy').tolist()
        lim = len(trainloss)

        shift_epoch = FLAGS.max_epochs_pre

        # Find latest checkpoint file
        checkpoint_files = glob.glob(f'{main_dir}/{FLAGS.save_path}/save_{mode}_{model_choice}_*_full_nomir.pth.tar')
        if not len(checkpoint_files):
            checkpoint_files = glob.glob(f'{main_dir}/{FLAGS.save_path}/save_pre_{model_choice}_*_full_nomir.pth.tar')
            last_checkpoint_mode = 'pre'
        else:
            last_checkpoint_mode = mode

        epochs = [int(re.search(r'_(\d+)_full_nomir.pth.tar', file).group(1)) for file in checkpoint_files]
        start_epoch = max(epochs) if len(epochs) else -1
        print(f'Last-trained epoch: {last_checkpoint_mode}_{start_epoch}')

        
        # Model
        if model_choice == 'z':
            net_pre = clean_cornets.CORnet_Z_tweak
            if mode == 'lit_bias':
                net = clean_cornets.CORNet_Z_biased_words
            elif mode == 'lit_no_bias':
                net = clean_cornets.CORNet_Z_nonbiased_words

        elif model_choice == 's':
            net_pre = clean_cornets.CORnet_S_tweak
            if mode == 'lit_bias':
                net = clean_cornets.CORNet_S_biased_words
            elif mode == 'lit_no_bias':
                net = clean_cornets.CORNet_S_nonbiased_words

        if last_checkpoint_mode == 'pre':
            print(f'loading pre-schooler model {model_choice}')
            net_pre = net_pre(out_img=FLAGS.img_classes)
            ckpt_data = torch.load(f'{FLAGS.save_path}/save_pre_{model_choice}_{start_epoch}_full_nomir.pth.tar')
            assert start_epoch == ckpt_data['epoch']
            net_pre.load_state_dict(ckpt_data['state_dict'])
            net = net(net_pre)
        else:
            print(f'loading literate model {model_choice}')
            net = net()
            ckpt_data = torch.load(f'{FLAGS.save_path}/save_{last_checkpoint_mode}_{model_choice}_{start_epoch}_full_nomir.pth.tar')
            assert start_epoch == ckpt_data['epoch']
            net.load_state_dict(ckpt_data['state_dict'])

    #use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    
    #transfer model to device
    net.to(device)
        
#    if device != "cpu":
#        net.cuda()
    

    exec_time = secondsToStr(time.time() - start_time)
    print ('execution time so far: ',exec_time)
    
    # Build loss function, model and optimizer.
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
            
    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    
    """
    train
    """
    
    # Validation
    net.eval()
    with torch.set_grad_enabled(False):
        cat_index = 0
        for local_batch_val, local_labels_val in validation_gen:

            # Transfer to GPU
            local_batch_val, local_labels_val = local_batch_val.to(device), local_labels_val.to(device)

            # Model computations
            
            v1, v2, v4, it, h, pred_val = net(local_batch_val)
            
            # Per category acc
#                print '-->ground truth label:',local_labels_val
#                print '-->predicted label:',pred_val.numpy()
            
            scores = Acc(pred_val, local_labels_val)
            print (f'category {cat_index} accuracy scores: {scores}')
            cat_scores[shift_epoch + start_epoch, cat_index] = scores
            # exec_time = secondsToStr(time.time() - start_time)
            # print ('execution time so far: ',exec_time)
            cat_index += 1
            
            # Compute loss.
            loss_val = criterion(pred_val, local_labels_val)
            valloss += [loss_val.item()]

    print(np.mean(cat_scores[shift_epoch + start_epoch, :]))
            
    end_time = time.time()
    exec_time = secondsToStr(end_time - start_time)
    print ('execution time: ',exec_time)
    


#    """
#    plot results
#    """
#    if plot:
#        
#        plots.plot_confus(confus, epoch, labels=labels, title=title_confus, show=show)
#        
#        if mode == 'pre':     
#            plots.plot_acc(cat_scores, show=show)
#            
#        if mode == 'lit':
#            plots.plot_acc2(cat_scores, FLAGS.max_epochs_pre, show=show)
#            plots.plot_acc3_img_vs_word(cat_scores, FLAGS.max_epochs_pre, mode=FLAGS.mode, show=show)
#            #plots.plot_results(net_pre, net, init_wrd, listloss, valloss, lim=lim, show=show)
#            
#        if mode == 'illit':
#            plots.plot_acc3_img_vs_word(cat_scores, FLAGS.max_epochs_pre, mode=FLAGS.mode, show=show)
    
    return #net, cat_scores, trainloss#, valloss


def main():
    train()

if __name__ == "__main__":
    main()
