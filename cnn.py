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
from sklearn.metrics import  brier_score_loss, roc_curve, auc, matthews_corrcoef, confusion_matrix, precision_recall_curve
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
recall_metric = []

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
    
torch.cuda.manual_seed(424242)
torch.manual_seed(424242)
np.random.seed(424242)
random.seed(424242)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    val_dataset = DataLoader(val_data, batch_size=32, shuffle = False)
    test_dataset = DataLoader(test_data, batch_size=32, shuffle=False)

    Cnn = CNN_model(16, 1).to(device)#change the output layer
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    acc_fn = Accuracy('multiclass', num_classes=2,average='macro').to(device)
    acc_fn2 = Accuracy('multiclass', num_classes=2,average='micro').to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1score = BinaryF1Score().to(device)
    optimizer = torch.optim.Adam(Cnn.parameters(), lr = 1e-6)
    print('loop nnumber:', epoch)
    result = fit_loop(epoch,Cnn, 100, train_dataset, val_dataset, optimizer, loss_fn, acc_fn,acc_fn2,recall,precision, f1score, device, 20)#change number of epoch here

    # torch.save(Cnn.state_dict(), 'cnnbinary.pt')

    y_problist = []
    for pt in [str(epoch)+'bestf1cnn.pt', str(epoch)+'bestaccmicrocnn.pt']:
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
    name = [ ' Cnn best f1', 'Cnn best micro']
    y_pred_spec = []
    for proba in y_problist:
        fpr, tpr, threshold = roc_curve(y, proba.squeeze().to('cpu') )
        thres = threshold[[i for i,n in enumerate(fpr) if n>=0.5][0]]
        y_pred_spec.append(torch.tensor([1 if i>thres else 0  for i in proba]))


    y_pred_recall = []
    for proba in y_problist:
        fpr, tpr, threshold = roc_curve(y, proba.squeeze().to('cpu') )
        thres = threshold[[i for i,n in enumerate(tpr) if n>=0.8][0]]
        y_pred_recall.append(torch.tensor([1 if i>thres else 0  for i in proba]))

    spec_fn = BinarySpecificity()
    acc_fntest = Accuracy('multiclass', num_classes=2, average='macro')
    acc_fntest2 = Accuracy('multiclass', num_classes=2, average='micro')

    for i,(y_probas, y_pred) in enumerate(zip(y_problist,y_pred_spec)):
        y_proba,y = y_probas.squeeze(),y.squeeze() 
        test_recall = recall(y_pred, y).to('cpu')
        test_precision = precision(y_pred, y).to('cpu')
        test_f1score = f1score(y_pred, y).to('cpu')
        # test_acc = acc_fntest(y_pred, y).to('cpu')
        test_acc2 = acc_fntest2(y_pred, y).to('cpu')
        fpr, tpr, thes = roc_curve(y.to('cpu'), y_proba.to('cpu'))
        np.save('roc'+str(epoch)+'.npy',np.concatenate([fpr,tpr]))
        matthewscore = matthews_corrcoef(y.to('cpu'), y_pred.to('cpu'))
        test_spec = spec_fn(y_pred, y).to('cpu')
        confusion = confusion_matrix(y.to('cpu'), y_pred.to('cpu'))
        np.save(name[i]+'_confusionspec_'+str(epoch),confusion)
        brier = brier_score_loss(y.to('cpu'), y_pred.to('cpu'))
        fpr, tpr, threshold = roc_curve(y.to('cpu'),y_proba.to('cpu'))
        precision_curve,recall_curve, threshold = precision_recall_curve(y.to('cpu'), y_proba.to('cpu'))
        np.save('prcurve'+str(epoch)+'.npy',np.concatenate([precision_curve,recall_curve]))
        auc_score = auc(fpr, tpr)
        spec_metric.append([name[i],epoch,test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(),matthewscore.item() ,test_spec.item(), auc_score, brier])

    for i,(y_probas, y_pred) in enumerate(zip(y_problist,y_pred_recall)):
        y_proba,y = y_probas.squeeze(),y.squeeze() 
        test_recall = recall(y_pred, y).to('cpu')
        test_precision = precision(y_pred, y).to('cpu')
        test_f1score = f1score(y_pred, y).to('cpu')
        # test_acc = acc_fntest(y_pred, y).to('cpu')
        test_acc2 = acc_fntest2(y_pred, y).to('cpu')
        matthewscore = matthews_corrcoef(y.to('cpu'), y_pred.to('cpu'))
        test_spec = spec_fn(y_pred, y).to('cpu')
        brier = brier_score_loss(y.to('cpu'), y_pred.to('cpu'))
        fpr, tpr, threshold = roc_curve(y.to('cpu'),y_proba.to('cpu'))
        auc_score = auc(fpr, tpr)
        recall_metric.append([name[i],epoch,test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(),matthewscore.item() ,test_spec.item(), auc_score, brier])


    # for i,y_probas in enumerate(y_problist):
    #     y_proba,y = y_probas.squeeze(),y.squeeze()
    #     y_pred = torch.round(y_proba).to('cpu') 
    #     test_recall = recall(y_pred, y).to('cpu')
    #     test_precision = precision(y_pred, y).to('cpu')
    #     test_f1score = f1score(y_pred, y).to('cpu')
    #     test_acc = acc_fntest(y_pred, y).to('cpu')
    #     test_acc2 = acc_fntest2(y_pred, y).to('cpu')
    #     test_spec = spec_fn(y_pred, y).to('cpu')
    #     brier = brier_score_loss(y, y_pred)
    #     fpr, tpr, threshold = roc_curve(y,y_proba.to('cpu'))
    #     auc_score = auc(fpr, tpr)
    #     cut_metric.append([name[i],epoch,test_acc.item(),test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(), test_spec.item(), auc_score,brier])




spec_result = pd.DataFrame(spec_metric, columns = ['name','loop','acc micro', 'recall', 'precision', 'f1score','MCC','spec', 'auc','brier'])
recall_result = pd.DataFrame(recall_metric, columns = ['name','loop','acc micro', 'recall', 'precision', 'f1score','MCC','spec', 'auc','brier'])
recall_result.to_excel('recall_cnn.xlsx')
spec_result.to_excel('spec_cnn.xlsx')