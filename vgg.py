import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sty import fg,rs,bg
import os 
import random
from PIL import Image
import torchvision
from pathlib import Path
from tqdm import tqdm
# import optuna
import seaborn as sns
from torch import nn
from torchinfo import summary
# from skorch import NeuralNetClassifier
# from sklearn.model_selection import GridSearchCV
from torchmetrics import Accuracy
from torchmetrics.classification import BinarySpecificity, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
from torchvision.models import vgg19
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from dataset_class import *
from torch.utils.data import DataLoader, WeightedRandomSampler


vgg_weight = torchvision.models.VGG19_Weights.DEFAULT
vgg_transformer_orin = vgg_weight.transforms()
vgg_transformer = torchvision.transforms.Compose([
    vgg_transformer_orin,
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(15, translate  = (0,0.1)),
    # torchvision.transforms.ToTensor()
])
vgg_transformer
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

spec_metric = []
cut_metric = []
images = list(Path('../images/').glob('*.png'))
#loop
for epoch in range(1,201):

    X_train, X_remain= train_test_split(images,test_size=0.4)
    X_val, X_test= train_test_split(X_remain,test_size=0.5)

    train_data = imagecustom(X_train,'../feature_complete.xlsx', vgg_transformer)
    test_data = imagecustom(X_test,'../feature_complete.xlsx', vgg_transformer_orin)
    val_data = imagecustom(X_val,'../feature_complete.xlsx', vgg_transformer_orin)

    targets = train_data.targets()
    class_sample_count = np.unique(targets, return_counts=True)[1]
    class_weight = 1/ (class_sample_count/ len(train_data) )
    class_sample_weight = class_weight[targets]
    sampler =  WeightedRandomSampler(class_sample_weight, num_samples=len(train_data), replacement=True)

    train_dataset = DataLoader(train_data, batch_size=32, sampler=sampler)
    val_dataset = DataLoader(val_data, batch_size=32, shuffle = False)
    test_dataset = DataLoader(test_data, batch_size=32, shuffle=False)

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

    for param in list(vgg.features.parameters()):
        param.requires_grad = False

    Vggmodel = vgg.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    acc_fn = Accuracy('multiclass', num_classes=2,average='macro').to(device)
    acc_fn2 = Accuracy('multiclass', num_classes=2,average='micro').to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1score = BinaryF1Score().to(device)
    optimizer = torch.optim.Adam(vgg.parameters(), lr = 5e-5)
    result = vgg_fit_loop(Vggmodel, 100, train_dataset, val_dataset, optimizer, loss_fn, acc_fn,acc_fn2,recall,precision, f1score, device, 20)

    torch.save(Vggmodel.state_dict(), 'vgg.pt')

    y_problist = []
    for pt in ['vgg.pt','vggbestacc.pt','vggbestf1.pt', 'vggbestaccmicro.pt']:
        Vggmodel.load_state_dict(torch.load(pt,map_location=torch.device(device)))
        Vggmodel.eval()
        y_probas = []
        with torch.inference_mode():
            for X, y in test_dataset:
                X,y = X.to(device),y.to(device)
                y_logits = Vggmodel(X)
                y_proba = torch.sigmoid(y_logits)
                y_probas.append(y_proba)
            y_problist.append(torch.cat(y_probas))

    y = torch.tensor(test_data.targets()).unsqueeze(1)
    name = ['Vgg19 model full epoch', 'vgg19 best epoch', ' vgg19 best f1', 'best micro']
    y_predlist = []
    for proba in y_problist:
        fpr, tpr, threshold = roc_curve(y, proba.squeeze().to('cpu') )
        thres = threshold[[i for i,n in enumerate(fpr) if n>=0.5][0]]
        y_predlist.append(torch.tensor([1 if i>thres else 0  for i in proba]))

    spec_fn = BinarySpecificity()
    acc_fntest = Accuracy('multiclass', num_classes=2, average='macro')
    acc_fntest2 = Accuracy('multiclass', num_classes=2, average='micro')
    for i,(y_probas,y_pred) in enumerate(zip(y_problist,y_predlist)):
        y_proba,y = y_probas.squeeze(),y.squeeze() 
        test_recall = recall(y_pred, y).to('cpu')
        test_precision = precision(y_pred, y).to('cpu')
        test_f1score = f1score(y_pred, y).to('cpu')
        test_acc = acc_fntest(y_pred, y).to('cpu')
        test_acc2 = acc_fntest2(y_pred, y).to('cpu')
        test_spec = spec_fn(y_pred, y).to('cpu')
        brier = brier_score_loss(y, y_pred)
        fpr, tpr, threshold = roc_curve(y, y_proba.to('cpu'))
        auc_score = auc(fpr, tpr)
        spec_metric.append([name[i],epoch,test_acc.item(),test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(), test_spec.item(), auc_score, brier])

    for i,y_probas in enumerate(y_problist):
        y_proba,y = y_probas.squeeze(),y.squeeze()
        y_pred = torch.round(y_proba).to('cpu') 
        test_recall = recall(y_pred, y).to('cpu')
        test_precision = precision(y_pred, y).to('cpu')
        test_f1score = f1score(y_pred, y).to('cpu')
        test_acc = acc_fntest(y_pred, y).to('cpu')
        test_acc2 = acc_fntest2(y_pred, y).to('cpu')
        test_spec = spec_fn(y_pred, y).to('cpu')
        brier = brier_score_loss(y, y_pred)
        fpr, tpr, threshold = roc_curve(y, y_proba.to('cpu'))
        auc_score = auc(fpr, tpr)
        cut_metric.append([name[i],epoch,test_acc.item(),test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(), test_spec.item(), auc_score, brier])



spec_vgg = pd.DataFrame(spec_metric, columns = ['name','loop','acc macro','acc micro', 'recall', 'precision', 'f1score','spec', 'auc','brier'])
cut_vgg = pd.DataFrame(cut_metric, columns = ['name','loop','acc macro','acc micro', 'recall', 'precision', 'f1score','spec', 'auc','brier'])
spec_vgg.to_excel('spec_vgg.xlsx')
cut_vgg.to_excel('cut_vgg.xlsx')