import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from pathlib import Path
from PIL import Image
import os
from torchvision import transforms
import torchvision
import random 
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import  Dataset,DataLoader, WeightedRandomSampler
from torchvision.models import vgg19
from torch import nn
from sty import fg,rs,bg
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinarySpecificity, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
import torchmetrics
from sklearn.metrics import roc_curve, auc, brier_score_loss
import sys
sys.path.append('../')
from dataset_class import *

vgg_weight = torchvision.models.VGG19_Weights.DEFAULT
vgg_transformer_orin = vgg_weight.transforms()
vgg_transformer = torchvision.transforms.Compose([
    vgg_transformer_orin,
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(15, translate  = (0,0.1)),
])
images = list(Path('../images/').glob('*.png'))
spec_metric = []
cutoff_metric = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# For loop here
for loop in range(1,201):
    scaler = MinMaxScaler()
    X_train, X_remain= train_test_split(images,test_size=0.4)
    X_val, X_test= train_test_split(X_remain,test_size=0.5)

    train_data = imagecsvdataset(X_train,'../feature_complete.xlsx',['BMI','Neck'], vgg_transformer)
    feature = [item[1] for item in train_data]
    scaler.fit(np.concatenate(feature))
    

    train_data = imagecsvdataset(X_train,'../feature_complete.xlsx',['BMI','Neck'], vgg_transformer, scaler)
    test_data = imagecsvdataset(X_test,'../feature_complete.xlsx',['BMI','Neck'], vgg_transformer_orin, scaler)
    val_data = imagecsvdataset(X_val,'../feature_complete.xlsx',['BMI','Neck'], vgg_transformer_orin, scaler)

    targets = train_data.targets()
    class_sample_count = np.unique(targets, return_counts=True)[1]
    class_weight = 1/ (class_sample_count/ len(train_data) )
    class_sample_weight = class_weight[targets]
    sampler =  WeightedRandomSampler(class_sample_weight, num_samples=len(train_data), replacement=True)

    train_dataset = DataLoader(train_data, batch_size=32, sampler=sampler)
    test_dataset = DataLoader(test_data, batch_size=32, shuffle=False)
    val_dataset = DataLoader(val_data, batch_size=32, shuffle=False)

    vgg = vgg19(weights=vgg_weight).to(device)

    vgg.classifier = nn.Sequential( # overwrite classifer aka output layer
        nn.Linear(in_features=25088, out_features=4096, bias = True),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=4096, out_features=4096, bias = True),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=4096, out_features=1, bias=True)# change the last layer to our output outcome
    ).to(device)

    pt = '../bestaccmicro.pt'
    dic = torch.load(pt, map_location=torch.device(device))
    vgg.load_state_dict(dic, strict=False)
    vgg.classifier[-1] = nn.Linear(in_features=4096, out_features=1000, bias=True)

    for params in vgg.parameters():
        params.requires_grad = False


    model = combinemodel(vgg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.BCEWithLogitsLoss().to(device)
    acc1 = torchmetrics.Accuracy('multiclass', num_classes=2, average='macro' ).to(device)
    acc2 = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro' ).to(device)
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)
    f1score = BinaryF1Score().to(device)
    spec = BinarySpecificity().to(device)

    result =combined_fit_loop(100, train_dataset, val_dataset,model,optimizer,loss, acc1,acc2,recall, precision, f1score, spec, device, 20)
    torch.save(model.state_dict(), 'combinemodel.pt')

    y_proba_list = []
    for pt in ['combinedbestacc.pt','combinedbestacc2.pt', 'combinedbestf1.pt', 'combinemodel.pt']:
        y_probas = []
        model.load_state_dict(torch.load(pt, map_location=torch.device(device)))# change here
        model.eval()
        with torch.inference_mode():
            for image,s, y in test_dataset:
                image,s,y = image.to(device), s.type(torch.float32).to(device), y.to(device)
                y_logit = model(image, s)
                y_proba = torch.sigmoid(y_logit)
                y_probas.append(y_proba)
            y_proba_list.append(torch.cat(y_probas))
        
    y =  torch.tensor(test_data.targets()).unsqueeze(dim=1)
    y_preds = []
    for y_proba in y_proba_list:
        fpr,tpr, thres = roc_curve(y.to('cpu'), y_proba.to('cpu'))
        thres = thres[[i for i,x in enumerate(fpr) if x>=0.5][0]]
        y_preds.append([1 if i>thres else 0 for i in y_proba]) 


    name=['bestacc','bestacc2', 'bestf1', 'combinemodel']

    y = y.squeeze().to(device)
    for i,(y_pred, y_proba) in enumerate(zip(y_preds,y_proba_list)):
        y_pred = torch.tensor(y_pred).to(device)
        y_proba = y_proba.to(device)
        accmacro = acc1(y_pred, y).to(device)
        accmicro = acc2(y_pred, y).to(device)
        recallscore = recall(y_pred,y).to(device)
        precisionscore = precision(y_pred, y).to(device)
        f1scoreresult  = f1score(y_pred, y).to(device)
        specscore = spec(y_pred, y).to(device)
        fpr, tpr, thes = roc_curve(y.to('cpu'), y_proba.to('cpu'))
        auc_score = auc(fpr,tpr)
        brier_score = brier_score_loss(y.to('cpu'), y_proba.to('cpu'))
        spec_metric.append([name[i],loop,accmacro.item(), accmicro.item(), recallscore.item(), precisionscore.item(), f1scoreresult.item(), specscore.item(),auc_score.item(), brier_score.item()])

    for i,y_proba in enumerate(y_proba_list):
        y_proba = y_proba.to(device)
        y_pred = torch.round(y_proba).squeeze()
        accmacro = acc1(y_pred, y).to(device)
        accmicro = acc2(y_pred, y).to(device)
        recallscore = recall(y_pred,y).to(device)
        precisionscore = precision(y_pred, y).to(device)
        f1scoreresult  = f1score(y_pred, y).to(device)
        specscore = spec(y_pred, y).to(device)
        fpr, tpr, thes = roc_curve(y.to('cpu'), y_proba.to('cpu'))
        auc_score = auc(fpr,tpr)
        brier_score = brier_score_loss(y.to('cpu'), y_proba.to('cpu'))
        cutoff_metric.append([name[i],loop,accmacro.item(), accmicro.item(), recallscore.item(), precisionscore.item(), f1scoreresult.item(), specscore.item(),auc_score.item(), brier_score.item()])


result_spec = pd.DataFrame(spec_metric, columns=['name','loop','acc macro','acc micro', 'recall', 'precision', 'f1score','spec', 'auc','brier'])
result_cutoff = pd.DataFrame(spec_metric, columns=['name','loop','acc macro','acc micro', 'recall', 'precision', 'f1score','spec', 'auc','brier'])
result_cutoff.to_excel('result_cutoff.xlsx')
result_spec.to_excel('result_spec.xlsx')
