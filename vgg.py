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
from sklearn.metrics import roc_curve, auc, brier_score_loss, matthews_corrcoef,confusion_matrix,precision_recall_curve
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
torch.cuda.manual_seed(424242)
torch.manual_seed(424242)
np.random.seed(424242)
random.seed(424242)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
spec_metric = []
recall_metric = []
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
    result = vgg_fit_loop(epoch,Vggmodel, 100, train_dataset, val_dataset, optimizer, loss_fn, acc_fn,acc_fn2,recall,precision, f1score, device, 20)

    # torch.save(Vggmodel.state_dict(), 'vgg.pt')
    # df = pd.DataFrame(result)
    # df.to_csv('VGG_train_loss.csv')
    y_problist = []
    for pt in [str(epoch)+'vggbestf1.pt', str(epoch)+'vggbestaccmicro.pt']:
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
    name = ['vgg19 best f1', 'vgg best acc']
    y_predspec = []
    for proba in y_problist:
        fpr, tpr, threshold = roc_curve(y, proba.squeeze().to('cpu') )
        thres = threshold[[i for i,n in enumerate(fpr) if n>=0.5][0]]
        y_predspec.append(torch.tensor([1 if i>thres else 0  for i in proba]))
    y_predrecall = []
    for proba in y_problist:
        fpr, tpr, threshold = roc_curve(y, proba.squeeze().to('cpu') )
        thres = threshold[[i for i,n in enumerate(tpr) if n>=0.8][0]]
        y_predrecall.append(torch.tensor([1 if i>thres else 0  for i in proba]))

    spec_fn = BinarySpecificity()
    acc_fntest = Accuracy('multiclass', num_classes=2, average='macro')
    acc_fntest2 = Accuracy('multiclass', num_classes=2, average='micro')
    for i,(y_probas,y_pred) in enumerate(zip(y_problist,y_predspec)):
        y_proba,y = y_probas.squeeze(),y.squeeze() 
        test_recall = recall(y_pred, y).to('cpu')
        test_precision = precision(y_pred, y).to('cpu')
        test_f1score = f1score(y_pred, y).to('cpu')
        # test_acc = acc_fntest(y_pred, y).to('cpu')
        test_acc2 = acc_fntest2(y_pred, y).to('cpu')
        test_spec = spec_fn(y_pred, y).to('cpu')
        brier = brier_score_loss(y.to('cpu'), y_proba.to('cpu'))
        matthewscore = matthews_corrcoef(y.to('cpu'), y_pred.to('cpu'))
        fpr, tpr, threshold = roc_curve(y.to('cpu'), y_proba.to('cpu'))
        np.save('roc'+str(epoch)+'.npy',np.concatenate([fpr,tpr]))
        confusion = confusion_matrix(y, y_pred)
        np.save(name[i]+'_confusionspec_'+str(epoch),confusion)
        precision_curve,recall_curve, threshold = precision_recall_curve(y.to('cpu'), y_proba.to('cpu'))
        np.save('prcurve'+str(epoch)+'.npy',np.concatenate([precision_curve,recall_curve]))
        auc_score = auc(fpr, tpr)
        spec_metric.append([name[i],epoch,test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(), matthewscore.item(),test_spec.item(), auc_score, brier])

    for i,(y_probas,y_pred) in enumerate(zip(y_problist,y_predrecall)):
        y_proba,y = y_probas.squeeze(),y.squeeze() 
        test_recall = recall(y_pred, y).to('cpu')
        test_precision = precision(y_pred, y).to('cpu')
        test_f1score = f1score(y_pred, y).to('cpu')
        # test_acc = acc_fntest(y_pred, y).to('cpu')
        test_acc2 = acc_fntest2(y_pred, y).to('cpu')
        matthewscore = matthews_corrcoef(y.to('cpu'), y_pred.to('cpu'))
        test_spec = spec_fn(y_pred, y).to('cpu')
        brier = brier_score_loss(y.to('cpu'), y_proba.to('cpu'))
        fpr, tpr, threshold = roc_curve(y, y_proba.to('cpu'))
        confusion = confusion_matrix(y, y_pred)
        np.save(name[i]+'_confusionrecall_'+str(epoch),confusion)
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
    #     fpr, tpr, threshold = roc_curve(y, y_proba.to('cpu'))
    #     auc_score = auc(fpr, tpr)
    #     cut_metric.append([name[i],epoch,test_acc.item(),test_acc2.item(), test_recall.item(), test_precision.item(), test_f1score.item(), test_spec.item(), auc_score, brier])



spec_vgg = pd.DataFrame(spec_metric, columns = ['name','loop','acc micro', 'recall', 'precision', 'f1score','MCC','spec', 'auc','brier'])
recall_vgg = pd.DataFrame(recall_metric, columns = ['name','loop','acc micro', 'recall', 'precision', 'f1score','MCC','spec', 'auc','brier'])
spec_vgg.to_excel('spec_vgg.xlsx')
recall_vgg.to_excel('recall_vgg.xlsx')

