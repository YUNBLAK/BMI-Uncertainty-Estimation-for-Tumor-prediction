import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import time
import cv2
import pandas as pd
import tifffile as tiff
import argparse
from torch.optim import lr_scheduler
import copy
import torch.nn.parallel
import torch.optim as optim
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, average_precision_score, confusion_matrix
import sys
import torch.backends.cudnn as cudnn
import time
import matplotlib.pyplot as plt
from PIL import ImageFile
import openslide
ImageFile.LOAD_TRUNCATED_IMAGES = True

def saveTruthMaskImage(filename, new_size, output_folder):
    name = "../dataset/" + filename + ".png"
    original_image = Image.open(name)
    resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
    saved_name = output_folder + "/" + filename + "_truth.png"
    resized_image.save(saved_name)

def saveWSI(filename, new_size, output_folder):
    name = "../dataset/" + filename + ".svs"
    slide = openslide.OpenSlide(name)
    thumbnail = slide.get_thumbnail(new_size)
    resized_image = thumbnail.resize(new_size, Image.Resampling.LANCZOS)
    saved_name = output_folder + "/" + filename + "_WSI.png"
    resized_image.save(saved_name)


APS = 350;
PS = 224
TileFolder = sys.argv[1] + '/';
# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])


BatchSize = 96;

heat_map_out = sys.argv[3];


print("CHECKING")
print(sys.argv[1].split('/')[2][:-4])

globalfilename = sys.argv[1].split('/')[2]

old_model = sys.argv[4]

def mean_std(type = 'none'):
    if type == 'vahadane':
        mean = [0.8372, 0.6853, 0.8400]
        std = [0.1135, 0.1595, 0.0922]
    elif type == 'macenko':
        mean = [0.8196, 0.6938, 0.8131]
        std = [0.1417, 0.1707, 0.1129]
    elif type == 'reinhard':
        mean = [0.8364, 0.6738, 0.8475]
        std = [0.1315, 0.1559, 0.1084]
    elif type == 'macenkoMatlab':
        mean = [0.7805, 0.6230, 0.7068]
        std = [0.1241, 0.1590, 0.1202]
    else:
        mean = [0.7238, 0.5716, 0.6779]
        std = [0.1120, 0.1459, 0.1089]
    return mean, std

type = 'none'
mu, sigma = mean_std(type)

#mu = [0.5, 0.5, 0.5]
#sigma = [0.5, 0.5, 0.5]

torch.cuda.set_device(0)
device = torch.device(f"cuda:{0}")

data_aug = transforms.Compose([
    transforms.Resize(PS),
    transforms.ToTensor(),
    transforms.Normalize(mu, sigma)])

def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


def softmax_np(x):
    x = x - np.max(x, 1, keepdims=True)
    x = np.exp(x) / (np.sum(np.exp(x), 1, keepdims=True))
    return x

def iterate_minibatches(inputs, augs, targets):
    if inputs.shape[0] <= BatchSize:
        yield inputs, augs, targets;
        return;

    start_idx = 0;
    for start_idx in range(0, len(inputs) - BatchSize + 1, BatchSize):
        excerpt = slice(start_idx, start_idx + BatchSize);
        yield inputs[excerpt], augs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - BatchSize:
        excerpt = slice(start_idx + BatchSize, len(inputs));
        yield inputs[excerpt], augs[excerpt], targets[excerpt];


def load_data(todo_list, rind):
    X = torch.zeros(size=(BatchSize*40, 3, PS, PS));
    inds = np.zeros(shape=(BatchSize*40,), dtype=np.int32);
    coor = np.zeros(shape=(200000, 2), dtype=np.int32);

    normalized = False  # change this to true if dont have images normalized and normalize on the fly
    parts = 4
    if normalized:
        parts = 4

    xind = 0;
    lind = 0;
    cind = 0;
    for fn in todo_list:
        lind += 1;
        full_fn = TileFolder + '/' + fn;
        if not os.path.isfile(full_fn):
            continue;
        if (len(fn.split('_')) != parts) or ('.png' not in fn):
            continue;

        try:
            x_off = float(fn.split('_')[0]);
            y_off = float(fn.split('_')[1]);
            svs_pw = float(fn.split('_')[2]);
            png_pw = float(fn.split('_')[3].split('.png')[0]);
        except:
            print('error reading image')
            continue

        png = np.array(Image.open(full_fn).convert('RGB'));
        for x in range(0, png.shape[1], APS):
            if x + APS > png.shape[1]:
                continue;
            for y in range(0, png.shape[0], APS):
                if y + APS > png.shape[0]:
                    continue;

                if (whiteness(png[y:y+APS, x:x+APS, :]) >= 12):
                    a = png[y:y + APS, x:x + APS, :]
                    a = Image.fromarray(a.astype('uint8'), 'RGB')
                    a = data_aug(a)
                    X[xind, :, :, :] = a
                    inds[xind] = rind
                    xind += 1

                coor[cind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                coor[cind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);

                cind += 1;
                rind += 1;
                if rind % 100 == 0: print('Processed: ', rind)
        if xind >= BatchSize:
            break;

    X = X[0:xind];
    inds = inds[0:xind];
    coor = coor[0:cind];

    return todo_list[lind:], X, inds, coor, rind;


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;


def val_fn_epoch_on_disk(classn, val_fn):
    all_or = np.zeros(shape=(500000, classn), dtype=np.float32);
    all_inds = np.zeros(shape=(500000,), dtype=np.int32);
    all_coor = np.zeros(shape=(500000, 2), dtype=np.int32);
    rind = 0;
    n1 = 0;
    n2 = 0;
    n3 = 0;
    todo_list = os.listdir(TileFolder);
    # todo_list = todo_list[0: 30]
    processed = 0
    total = len(todo_list)
    start = time.time()
    coor_c = 0
    while len(todo_list) > 0:
        todo_list, inputs, inds, coor, rind = load_data(todo_list, rind);
        coor_c += len(coor)

        #if len(inputs) == 0:
        #    print('len of inputs is 0"')
        #    break;
        if inputs.size(0) < 2:
            print('len of inputs if less than 2')
        else:
            processed = total - len(todo_list)
            print('Processed: {}/{} \t Time Remaining: {}mins'.format(processed, total, (time.time() - start)/60*(total/processed - 1)))
            with torch.no_grad():
                inputs = Variable(inputs.to(device))
                output = val_fn(inputs)
            output = output.data.cpu().numpy()
            output = softmax_np(output)[:, 1]
            all_or[n1:n1+len(output)] = output.reshape(-1,1)
            n1 += len(output)
            all_inds[n2:n2+len(inds)] = inds;
            n2 += len(inds);

        all_coor[n3:n3+len(coor)] = coor;
        n3 += len(coor);

    all_or = all_or[:n1];
    all_inds = all_inds[:n2];
    all_coor = all_coor[:n3];
    return all_or, all_inds, all_coor;

# def confusion_matrix(Or, Tr, thres):
#     tpos = np.sum((Or>=thres) * (Tr==1));
#     tneg = np.sum((Or< thres) * (Tr==0));
#     fpos = np.sum((Or>=thres) * (Tr==0));
#     fneg = np.sum((Or< thres) * (Tr==1));
#     return tpos, tneg, fpos, fneg;

def auc_roc(Pr, Tr):
    fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0);
    return auc(fpr, tpr);

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=[0])
        cudnn.benchmark = True
        return model
def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model

def probability_getter(coor, predictions, patch_size=APS, threshold=0.6):
    max_x, max_y = np.max(coor, axis=0)
    image_size = (max_x + patch_size, max_y + patch_size)
    mask = Image.new('L', image_size, 0) 
    for idx, (x, y) in enumerate(coor):
        color = int(predictions[idx] * 255)
        patch = Image.new('L', (patch_size, patch_size), color)
        mask.paste(patch, (x, y))  # PIL은 (x, y)가 아닌 (y, x) 순서로 좌표를 받음
    return mask

def create_binary_mask_from_predictions(coor, predictions, patch_size=APS, threshold=0.6):

    max_x, max_y = np.max(coor, axis=0)
    image_size = (max_x + patch_size, max_y + patch_size)
    mask = Image.new('L', image_size, 0) 

    for idx, (x, y) in enumerate(coor):
        if predictions[idx] >= threshold:

            color = 255
        else:
            color = 0
        patch = Image.new('L', (patch_size, patch_size), color)

        mask.paste(patch, (x, y)) 
    return mask

def create_mask_from_predictions(coor, predictions, patch_size=APS):
    max_x, max_y = np.max(coor, axis=0)
    image_size = (max_x + patch_size, max_y + patch_size)
    cmap = plt.cm.jet
    mask = Image.new('RGB', image_size, (0, 0, 0))
    for idx, (x, y) in enumerate(coor):
        # Apply colormap directly to the prediction value
        rgba_color = cmap(predictions[idx])
        # Convert RGBA to RGB (ignore the alpha channel)
        rgb_color = tuple((np.array(rgba_color[:3]) * 255).astype(int))
        # Create a patch with the RGB color
        patch = Image.new('RGB', (patch_size, patch_size), rgb_color)
        # Paste the patch on the mask
        mask.paste(patch, (x, y))  # Note: PIL uses (y, x) order for coordinates

    return mask

def resize_and_save_image(image, new_size, save_path):
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    resized_image.save(save_path)
    return resized_image

def load_ground_truth(filename, new_size):
    name = "../dataset/" + filename + ".png"
    truth_image = cv2.imread(name, 0) 
    resized_truth = cv2.resize(truth_image, new_size, interpolation=cv2.INTER_AREA) 
    _, truth_data = cv2.threshold(resized_truth, 127, 255, cv2.THRESH_BINARY)
    return truth_data

def max_pooling(image, kernel_size):

    image_array = np.asarray(image)

    if image_array.ndim == 3:  
        image_padded = np.pad(image_array, [(kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)], mode='constant')
    else:  
        image_padded = np.pad(image_array, [(kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)], mode='constant')
    
    pooled_image = np.zeros_like(image_array)
    

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if image_array.ndim == 3:  
                pooled_image[i, j] = np.max(image_padded[i:i+kernel_size, j:j+kernel_size, :], axis=(0, 1))
            else:  
                pooled_image[i, j] = np.max(image_padded[i:i+kernel_size, j:j+kernel_size])

    pooled_image_pil = Image.fromarray(pooled_image)
    return pooled_image_pil, pooled_image

def calculate_ece(y_true, y_pred_probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.where((y_pred_probs >= bin_lower) & (y_pred_probs < bin_upper))[0]
        bin_true = y_true[in_bin]
        bin_pred = y_pred_probs[in_bin]

        bin_accuracy = np.mean(bin_true) if len(bin_true) > 0 else 0
        bin_confidence = np.mean(bin_pred) if len(bin_pred) > 0 else 0

        ece += np.abs(bin_accuracy - bin_confidence) * len(in_bin) / len(y_pred_probs)
    return ece

def calculate_nll_brier(y_true, y_pred_prob):
    y_pred_prob = np.clip(y_pred_prob, 1e-10, 1 - 1e-10)
    brier = 0

    y_pred_prob = torch.tensor(y_pred_prob, dtype=torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.float32)
    nll = F.binary_cross_entropy(y_pred_prob, y_true)
    return nll.item(), brier

def preprocess_ground_truth(ground_truth):
    ground_truth = np.where(ground_truth > 128, 1, 0)
    return ground_truth


def calculate_evaluation_metrics(predictions, ground_truth):

    ground_truth = preprocess_ground_truth(ground_truth)

    if len(np.unique(ground_truth)) > 1:
        auc_roc = roc_auc_score(ground_truth, predictions)
    else:
        auc_roc = np.nan 

    f1 = f1_score(ground_truth, predictions > 0.6)
    brier = brier_score_loss(ground_truth, predictions)
    aupr = average_precision_score(-1 * ground_truth + 1, -1 * predictions)
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions > 0.6).ravel()
    nll, b2 = calculate_nll_brier(ground_truth, predictions)
    ece = calculate_ece(ground_truth, predictions)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print("Brier:", b2)
    confidence = predictions
    correctness = (predictions > 0.6) == ground_truth

    sorted_indices = np.argsort(-confidence)
    sorted_correctness = correctness[sorted_indices]
    risk = np.cumsum(1 - sorted_correctness) / np.arange(1, len(correctness) + 1)
    coverage = np.linspace(1 / len(confidence), 1.0, len(confidence))
    aurc = np.trapz(risk, coverage)

    optimal_risk = risk[-1]
    optimal_risk_area = optimal_risk + (1 - optimal_risk) * np.log(1 - optimal_risk) if optimal_risk > 0 else 0
    eaurc = aurc - optimal_risk_area

    return {
        "F1 Score": f1,
        "AUC-ROC": auc_roc,
        "Brier Score": brier,
        "FPR" : fpr,
        "AUPR": aupr,
        "ECE" : ece,
        "NLL" : nll,
        "AURC": aurc,
        "EAURC": eaurc
    }

name = "../dataset/" + globalfilename
slide = openslide.OpenSlide(name)
num_patches_x = slide.dimensions[0] // 350
num_patches_y = slide.dimensions[1] // 350

print(num_patches_x , num_patches_y)
globalfilename = globalfilename[:-4]

try:
    saveWSI(globalfilename, 
            new_size= (num_patches_x*65, num_patches_y*65), 
            output_folder = "../prediction_masks"
            )
except:
    print("NO FILE")

print('start predicting...')
start = time.time()

print("| Load pretrained at  %s..." % old_model)
checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
model = checkpoint['model']
model = unparallelize_model(model)
model.to(device)
model.train(False)
best_auc = checkpoint['auc']
print('previous best AUC: \t%.4f'% best_auc)
print('=============================================')

Or, inds, coor = val_fn_epoch_on_disk(1, model);
Or_all = np.zeros(shape=(coor.shape[0],), dtype=np.float32);
Or_all[inds] = Or[:, 0];

print('len of all coor: ', coor.shape)
print('shape of Or: ', Or.shape)
print('shape of inds: ', inds.shape)

print()
print()
print()
print("HELLO WORLD")
print(TileFolder)
print()
print()
print()

namename = TileFolder[:-5].split("/")[2] + "_" + heat_map_out


print("")
print("--------------------------------------------------")
print("")
print("")
print("")
print(TileFolder[:-5])
print('../prediction_values/' + namename)
print("")
print("")
print("")
print("--------------------------------------------------")
print("")
fid = open('../prediction_values/' + namename, 'w');
for idx in range(0, Or_all.shape[0]):
    fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or_all[idx]))


print('Elapsed Time: ', (time.time() - start)/60.0)
print('DONE!');

print()
print("START MASKING")
mask = create_mask_from_predictions(coor, Or_all)
imgname = "../prediction_masks/" + globalfilename +  "_MASK_TIFF"


imgname = "../prediction_masks/" + globalfilename +  "_MASK" + ".png"
new_size = (num_patches_x * 65, num_patches_y*65) 


print()
print("SAVE PROB HEAT MAP")
mask = resize_and_save_image(mask, new_size, imgname)

print()
print("CREAT BIN MASK MAP")
mask2 = create_binary_mask_from_predictions(coor, Or_all)
imgname  = "../prediction_masks/" + globalfilename +  "_MASK_BIN" + ".png"


print()
print("SAVE BIN MASK MAP")
mask2 = resize_and_save_image(mask2, new_size, imgname)
