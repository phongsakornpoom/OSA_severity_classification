import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sty import fg,rs,bg
import os 
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
# import optuna
import seaborn as sns
import sys
from torch import nn
from torchinfo import summary
# from sklearn.model_selection import GridSearchCV
from torchmetrics.classification import BinaryRecall, BinaryF1Score, BinaryPrecision, BinarySpecificity
from torchmetrics import Accuracy,Precision, Recall, F1Score
from sklearn.metrics import  brier_score_loss, roc_curve, auc
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# from dataset_class import *
sys.path.append('../')
from dataset_class import *




transformer = transforms.Compose( #work similar to image generator
    [transforms.Resize((256,256)), # resize the image 
    transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomAffine(degrees=15, translate=(0,0.1)), # random flip augmenting
    transforms.ToTensor()]
    # transforms.Normalize(0,255)]# change to tensor
)

test_transformer = transforms.Compose(
    [transforms.Resize((256,256)),
        transforms.ToTensor()]
        #  transforms.Lambda(lambda x : x/255) ]
)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
spec_metric = []
cut_metric = []

class CNN_model(torch.nn.Module):
    def __init__(self, hidden, output):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,hidden,kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden, hidden*2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(hidden*2, hidden*3, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden*3, hidden*2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden*2*62*62, output)
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.classifer(x)

images = list(Path('../images/').glob('*.png'))

for epoch in range(1,201):
    X_train, X_remain= train_test_split(images,test_size=0.4)
    X_val, X_test= train_test_split(X_remain,test_size=0.5)
    train_data = imagecustom(X_train,'../feature_complete.xlsx', transformer)
    test_data = imagecustom(X_test,'../feature_complete.xlsx', test_transformer)
    val_data = imagecustom(X_val,'../feature_complete.xlsx', test_transformer)

    targets = train_data.targets()
    class_sample_count = np.unique(targets, return_counts=True)[1]
    class_weight = 1/ (class_sample_count/ len(train_data) )
    class_sample_weight = class_weight[targets]
    sampler =  WeightedRandomSampler(class_sample_weight, num_samples=len(train_data), replacement=True)

    train_dataset = DataLoader(train_data, batch_size=32, sampler=sampler)
    val_dataset = DataLoader(val_data, batch_size=32, shuffle = True)
    test_dataset = DataLoader(test_data, batch_size=32, shuffle=False)

    Cnn = CNN_model(16, 1).to(device)#change the output layer
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    acc_fn = Accuracy('multiclass', num_classes=2,average='macro').to(device)
    acc_fn2 = Accuracy('multiclass', num_classes=2,average='micro').to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1score = BinaryF1Score().to(device)
    optimizer = torch.optim.Adam(Cnn.parameters(), lr = 1e-6)

    result = fit_loop(Cnn, 1, train_dataset, val_dataset, optimizer, loss_fn, acc_fn,acc_fn2,recall,precision, f1score, device, 20)

    torch.save(Cnn.state_dict(), 'cnnbinary.pt')

    y_problist = []
    for pt in ['cnnbinary.pt','bestacccnn.pt','bestf1cnn.pt', 'bestaccmicrocnn.pt']:
        Cnn.load_state_dict(torch.load(pt,map_location=torch.device(device)))
        Cnn.eval()
        y_probas = []
        with torch.inference_mode():
            for X, y in test_dataset:
                X,y = X.to(device),y.to(device)
                y_logits = Cnn(X)
                y_proba = torch.sigmoid(y_logits)
                y_probas.append(y_proba)
            y_problist.append(torch.cat(y_probas))
    y = torch.tensor(test_data.targets()).unsqueeze(1)
    name = ['CNN model full epoch', 'Cnn best macro epoch', ' Cnn best f1', 'Cnn best micro']
    y_predlist = []
    for proba in y_problist:
        fpr, tpr, threshold = roc_curve(y, proba.squeeze().to('cpu') )
        thres = threshold[[i for i,n in enumerate(fpr) if n>=0.5][0]]
        y_predlist.append(torch.tensor([1 if i>thres else 0  for i in proba]))

    spec_fn = BinarySpecificity()
    acc_fntest = Accuracy('multiclass', num_classes=2, average='macro')
    acc_fntest2 = Accuracy('multiclass', num_classes=2, average='micro')

    for i,(y_probas, y_pred) in enumerate(zip(y_problist,y_predlist)):
        y_proba,y = y_probas.squeeze(),y.squeeze() 
        test_recall = recall(y_pred, y).to('cpu')
        test_precision = precision(y_pred, y).to('cpu')
        test_f1score = f1score(y_pred, y).to('cpu')
        test_acc = acc_fntest(y_pred, y).to('cpu')
        test_acc2 = acc_fntest2(y_pred, y).to('cpu')
        test_spec = spec_fn(y_pred, y).to('cpu')
        brier = brier_score_loss(y, y_pred)
        fpr, tpr, threshold = roc_curve(y,y_proba.to('cpu'))
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
        fpr, tpr, threshold = roc_curve(y,y_proba.to('cpu'))
        auc_score = auc(fpr, tpr)
        cut_metric.append([name[i],epoch,test_acc.item(),test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(), test_spec.item(), auc_score,brier])




spec_result = pd.DataFrame(spec_metric, columns = ['name','loop','acc macro','acc micro', 'recall', 'precision', 'f1score','spec', 'auc','brier'])
cut_result = pd.DataFrame(cut_metric, columns = ['name','loop','acc macro','acc micro', 'recall', 'precision', 'f1score','spec', 'auc','brier'])
cut_result.to_excel('cut_cnn.xlsx')
spec_result.to_excel('spec_cnn.xlsx')