import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from sty import fg,rs,bg
from torch import nn
from tqdm import tqdm

class imagecsvdataset(Dataset):
    def __init__(self, image_path, csv,target_columns, transform = None, scaler=None):
        self.image_path = image_path
        self.csv = pd.read_excel(csv)
        self.target_columns = target_columns
        self.transform = transform
        self.scaler = scaler

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        image_path = self.image_path[index]
        image_filename = os.path.basename(image_path)
        hn = int(image_filename[:image_filename.find('.')])
        data_row = self.csv[self.csv['HN']==hn]
        target_value = data_row['treatment'].values
        structure_data = data_row[self.target_columns].values
        if self.scaler:
            structure_data = self.scaler.transform(data_row[self.target_columns].values)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, structure_data, target_value
    
    def targets(self):
        target = []

        for index in range(len(self)):
            a,b,y = self.__getitem__(index)
            target.append(y.item())
        return target
    

class imagecustom(Dataset):

    def __init__(self, image_path, csv, transform=None):
        super().__init__()
        self.image_path = image_path
        self.csv = pd.read_excel(csv)
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        image_path = self.image_path[index]
        image_name = os.path.basename(image_path)
        hn = int(image_name[:image_name.find('.')])
        target_value =  self.csv[self.csv['HN']==hn]['treatment'].values
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, target_value

    def targets(self):

        target = []
        for index in range(len(self)):
            x,y = self.__getitem__(index)
            target.append(y.item())

        return target

class combinemodel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1002, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=1)
        )
    def forward(self, image, structure):
        output = self.model(image)
        merge = torch.cat((output, structure.view(structure.shape[0],2)), dim=1)
        final_output = self.classifier(merge)
        return final_output


def combined_train_loop(model, dataset , loss_fn, optimizer, acc1, acc2, recall, precision, f1score, spec, device):
    train_loss = 0
    y_list, y_pred = [],[]
    for X,s,y in dataset:
        X,s,y = X.to(device),s.type(torch.float32).to(device), y.to(device)
        y_logit = model(X, s)
        loss = loss_fn(y_logit, y.type(torch.float32))
        train_loss+=loss
        y_list.append(y)
        y_pred.append(torch.round(torch.sigmoid(y_logit)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss/=len(dataset)
    y_list = torch.cat(y_list)
    y_pred = torch.cat(y_pred)
    train_acc = acc1(y_pred, y_list)
    train_acc2 = acc2(y_pred, y_list)
    train_recall = recall(y_pred, y_list)
    train_precision = precision(y_pred, y_list)
    train_f1 = f1score(y_pred, y_list)
    train_spec = spec(y_pred, y_list)
    return train_loss.item(), train_acc.item(), train_acc2.item(), train_recall.item(),train_precision.item(), train_f1.item(), train_spec.item()

def combined_val_loop(model, dataset, loss_fn, acc1, acc2, recall, precision, f1score, spec, device):
    val_loss = 0
    y_list, y_pred = [],[]
    for X,s,y in dataset:
        X,s,y = X.to(device), s.type(torch.float32).to(device),y.to(device)
        y_logit = model(X,s)
        loss = loss_fn(y_logit, y.type(torch.float32))
        val_loss+=loss
        y_list.append(y)
        y_pred.append(torch.round(torch.sigmoid(y_logit)))
    val_loss/=len(dataset)
    y_list = torch.cat(y_list)
    y_pred = torch.cat(y_pred)
    val_acc = acc1(y_pred, y_list)
    val_acc2 = acc2(y_pred, y_list)
    val_recall = recall(y_pred, y_list)
    val_precision = precision(y_pred, y_list)
    val_f1 = f1score(y_pred, y_list)
    val_spec = spec(y_pred, y_list)
    return val_loss.item(), val_acc.item(), val_acc2.item(), val_recall.item(),val_precision.item(), val_f1.item(), val_spec.item()


def combined_fit_loop(loop,epochs, train_data, val_data, model,optmizer, loss_fn, acc1, acc2, recall, precision, f1score, spec,device,patience):
    result = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_acc2': [], 'val_acc2': [], 'train_recall': [], 'val_recall': [], 'train_precision': [], 'val_precision': [],'train_f1':[], 'val_f1': [], 'train_spec':[], 'val_spec': []}
    best_valacc = 0
    best_valacc2 = 0
    cur_patience = patience
    best_valf1 = 0
    for epoch in range(epochs):
        train_loss, train_acc, train_acc2, train_recall, train_precision, train_f1, train_spec = combined_train_loop(model,train_data, loss_fn, optmizer, acc1, acc2, recall, precision, f1score, spec,device)
        val_loss, val_acc, val_acc2, val_recall, val_precision, val_f1, val_spec  = combined_val_loop(model, val_data,loss_fn, acc1,acc2, recall, precision, f1score, spec,device)
        print(f' Epoch {epoch+1} train loss : {fg.yellow + str(train_loss)+ fg.rs}  train acc macro : {fg.cyan + str(train_acc) + fg.rs} train acc micro: {fg.cyan + str(train_acc2) + fg.rs} train recall : {fg.cyan + str(train_recall) + fg.rs} train precision : {fg.cyan + str(train_precision) + fg.rs} train f1score : {fg.cyan + str(train_f1) + fg.rs} \n Epoch {epoch+1} val loss : {fg.yellow + str(val_loss)+ fg.rs}  val acc : {fg.cyan + str(val_acc) + fg.rs} val acc micro : {fg.cyan + str(val_acc2) + fg.rs} val recall : {fg.cyan + str(val_recall) + fg.rs} val precision : {fg.cyan + str(val_precision) + fg.rs} val f1 : {fg.cyan + str(val_f1) + fg.rs}')
        result['train_loss'].append(train_loss)
        result['val_loss'].append(val_loss)
        result['train_acc'].append(train_acc)
        result['val_acc'].append(val_acc)
        result['train_acc2'].append(train_acc2)
        result['val_acc2'].append(val_acc2)
        result['train_recall'].append(train_recall)
        result['val_recall'].append(val_recall)
        result['train_precision'].append(train_precision)
        result['val_precision'].append(val_precision)
        result['train_f1'].append(train_f1)
        result['val_f1'].append(val_f1)
        result['train_spec'].append(train_spec)
        result['val_spec'].append(val_spec)
        #save best model
        if best_valacc2<val_acc2:
            best_valacc2=val_acc2
            torch.save(model.state_dict(), str(loop)+'combinedbestacc.pt')
        if best_valf1<val_f1:
            best_valf1 = val_f1
            torch.save(model.state_dict(), str(loop)+'combinedbestf1.pt')
        if epoch==0:
            pred_loss = val_loss
        else:
            if val_loss<pred_loss:
                print(f'loop no {loop} epoch {fg.yellow + str(epoch + 1) + fg.rs} : val loss has improved from {str(pred_loss)} to {bg.li_green+ str(val_loss)+ bg.rs}') 
                cur_patience = patience
            elif val_loss > pred_loss:
                print(f'loop no {loop} epoch {fg.yellow + str(epoch + 1) + fg.rs} :val loss has worsened from {str(pred_loss)} to {bg.red+ str(val_loss)+ bg.rs}')
                cur_patience-=1
                if cur_patience ==0:
                    print('Early stop activated')
                    break
            elif val_loss == pred_loss:
                print('the loss didnt change from last epoch')
        pred_loss = val_loss
    return result



def vgg_train_step(model, train_set , optimizer, loss_fn, acc_fn, acc_fn2, recall, precision, f1score,device):
    train_loss, train_acc,train_acc2,train_recall, train_precision, train_f1 = 0,0,0,0,0,0
    y_preds = []
    y_list = []
    for X,y in train_set:
        model.train()
        X,y = X.to(device), y.to(device)
        y_logit = model(X).squeeze() # flatten the model 
        loss = loss_fn(y_logit, y.float())# input as logit since loss is bcewithlogitloss and y to float
        y_list.append(y)
        y_preds.append(torch.round(torch.sigmoid(y_logit)))# convert for metric calculation
        train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss/=len(train_set)
    y_preds = torch.cat(y_preds)
    y_list = torch.cat(y_list)
    train_acc = acc_fn(y_preds, y_list)
    train_acc2 = acc_fn2(y_preds,y_list)
    train_recall = recall(y_preds,y_list)
    train_precision = precision(y_preds,y_list)
    train_f1 = f1score(y_preds, y_list)
    #print(f'train loss : {fg.yellow + str(train_loss.item())+ fg.rs}  train acc : {fg.cyan + str(train_acc.item()) + fg.rs}')
    return train_loss.item(), train_acc.item(),train_acc2.item(), train_recall.item(), train_precision.item(),train_f1.item()

def vgg_val_step(model, val_set, loss_fn, acc_fn,acc_fn2, recall, precision, f1score, device):
    val_loss, val_acc, val_acc2,val_recall,val_precision,val_f1= 0,0,0,0,0,0
    y_preds = []
    y_list = []
    model.eval()
    with torch.inference_mode():
        for X,y in val_set:
            X,y = X.to(device), y.to(device)
            y_logit = model(X).squeeze()
            y_preds.append(torch.round(torch.sigmoid(y_logit)))
            y_list.append(y)
            val_loss += loss_fn(y_logit, y.float())
        val_loss/=len(val_set)
        y_preds = torch.cat(y_preds)
        y_list = torch.cat(y_list)
        val_acc = acc_fn(y_preds, y_list)
        val_acc2 = acc_fn2(y_preds,y_list)
        val_recall = recall(y_preds,y_list)
        val_precision = precision(y_preds,y_list)
        val_f1 = f1score(y_preds, y_list)
        #print(f'val loss : {fg.yellow + str(val_loss.item())+ fg.rs}  val acc : {fg.cyan + str(val_acc.item()) + fg.rs}')
        return val_loss.item(), val_acc.item(), val_acc2.item(), val_recall.item(), val_precision.item(),val_f1.item()
    


def vgg_fit_loop(loop,model, epoch, train_set, val_set, optimizer, loss_fn, acc_fn,acc_fn2,recall, precision, f1score, device, patience):
    result = {'trainacc':[], 'trainloss': [],'train_acc2':[], 'train_precision':[],'train_recall':[], 'train_f1score':[],'valoss': [], 'valacc':[], 'valacc2': [], 'valrecall': [], 'valprecision': [], 'valf1score': []}
    cur_patience = patience
    best_valacc = 0
    best_valaccmicro = 0
    best_valf1 = 0
    for i in tqdm(range(epoch)):
        train_loss, train_acc,train_acc2, train_recall, train_precision, train_f1 = train_step(model, train_set, optimizer, loss_fn, acc_fn,acc_fn2, recall, precision, f1score, device)
        val_loss, val_acc, val_acc2, val_recall, val_precision, val_f1 = val_step(model, val_set, loss_fn,acc_fn,acc_fn2, recall, precision, f1score, device)
        print(f' Epoch {i+1} train loss : {fg.yellow + str(train_loss)+ fg.rs}  train acc macro : {fg.cyan + str(train_acc) + fg.rs} train acc micro: {fg.cyan + str(train_acc2) + fg.rs} train recall : {fg.cyan + str(train_recall) + fg.rs} train precision : {fg.cyan + str(train_precision) + fg.rs} train f1score : {fg.cyan + str(train_f1) + fg.rs} \n Epoch {i+1} val loss : {fg.yellow + str(val_loss)+ fg.rs}  val acc : {fg.cyan + str(val_acc) + fg.rs} val acc micro : {fg.cyan + str(val_acc2) + fg.rs} val recall : {fg.cyan + str(val_recall) + fg.rs} val precision : {fg.cyan + str(val_precision) + fg.rs} val f1 : {fg.cyan + str(val_f1) + fg.rs}')
        result['valacc'].append(val_acc)
        result['trainacc'].append(train_acc)
        result['trainloss'].append(train_loss)
        result['valoss'].append(val_loss)
        result['train_acc2'].append(train_acc2)
        result['valacc2'].append(val_acc2)
        result['train_recall'].append(train_recall)
        result['valrecall'].append(val_recall)
        result['train_precision'].append(train_precision)
        result['valprecision'].append(val_precision)
        result['valf1score'].append(val_f1)
        result['train_f1score'].append(train_f1)
        if val_f1>best_valf1:
            best_valf1=val_f1
            torch.save(model.state_dict(), str(loop)+'vggbestf1.pt')
        if val_acc2>best_valaccmicro:
            best_valaccmicro=val_acc2
            torch.save(model.state_dict(), str(loop)+'vggbestaccmicro.pt')
        if i==0:
                pred_loss = val_loss
        else:
            if val_loss<pred_loss:
                print(f'loop no {loop} epoch {fg.yellow + str(i + 1) + fg.rs} : val loss has improved from {str(pred_loss)} to {bg.li_green+ str(val_loss)+ bg.rs}') 
                cur_patience = patience
            elif val_loss > pred_loss:
                print(f'loop no {loop} epoch {fg.yellow + str(i + 1) + fg.rs} :val loss has worsened from {str(pred_loss)} to {bg.red+ str(val_loss)+ bg.rs}')
                cur_patience-=1
                if cur_patience ==0:
                    print('Early stop activated')
                    break
            elif val_loss == pred_loss:
                print('the loss didnt change from last epoch')
        pred_loss = val_loss
    return result

def train_step(model, train_set , optimizer, loss_fn, acc_fn, acc_fn2, recall, precision, f1score,device):
    train_loss, train_acc,train_acc2,train_recall, train_precision, train_f1 = 0,0,0,0,0,0
    y_preds = []
    y_list = []
    for X,y in train_set:
        model.train()
        X,y = X.to(device), y.to(device)
        y_logit = model(X).squeeze() # flatten the model 
        loss = loss_fn(y_logit, y.float().squeeze())# input as logit since loss is bcewithlogitloss and y to float
        y_list.append(y)
        y_preds.append(torch.round(torch.sigmoid(y_logit)))# convert for metric calculation
        train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss/=len(train_set)
    y_preds = torch.cat(y_preds)
    y_list = torch.cat(y_list).squeeze()
    train_acc = acc_fn(y_preds, y_list)
    train_acc2 = acc_fn2(y_preds,y_list)
    train_recall = recall(y_preds,y_list)
    train_precision = precision(y_preds,y_list)
    train_f1 = f1score(y_preds, y_list)
    #print(f'train loss : {fg.yellow + str(train_loss.item())+ fg.rs}  train acc : {fg.cyan + str(train_acc.item()) + fg.rs}')
    return train_loss.item(), train_acc.item(),train_acc2.item(), train_recall.item(), train_precision.item(),train_f1.item()

def val_step(model, val_set, loss_fn, acc_fn,acc_fn2, recall, precision, f1score, device):
    val_loss, val_acc, val_acc2,val_recall,val_precision,val_f1= 0,0,0,0,0,0
    y_preds = []
    y_list = []
    model.eval()
    with torch.inference_mode():
        for X,y in val_set:
            X,y = X.to(device), y.to(device)
            y_logit = model(X).squeeze()
            y_preds.append(torch.round(torch.sigmoid(y_logit)))
            y_list.append(y)
            val_loss += loss_fn(y_logit, y.float().squeeze())
        val_loss/=len(val_set)
        y_preds = torch.cat(y_preds)
        y_list = torch.cat(y_list).squeeze()
        val_acc = acc_fn(y_preds, y_list)
        val_acc2 = acc_fn2(y_preds,y_list)
        val_recall = recall(y_preds,y_list)
        val_precision = precision(y_preds,y_list)
        val_f1 = f1score(y_preds, y_list)
        #print(f'val loss : {fg.yellow + str(val_loss.item())+ fg.rs}  val acc : {fg.cyan + str(val_acc.item()) + fg.rs}')
        return val_loss.item(), val_acc.item(), val_acc2.item(), val_recall.item(), val_precision.item(),val_f1.item()
    


def fit_loop(loop,model, epoch, train_set, val_set, optimizer, loss_fn, acc_fn,acc_fn2,recall, precision, f1score, device, patience):
    result = {'trainacc':[], 'trainloss': [],'train_acc2':[], 'train_precision':[],'train_recall':[], 'train_f1score':[],'valoss': [], 'valacc':[], 'valacc2': [], 'valrecall': [], 'valprecision': [], 'valf1score': []}
    cur_patience = patience
    best_acc = 0
    best_accmicro = 0
    best_f1 = 0
    for i in tqdm(range(epoch)):
        train_loss, train_acc,train_acc2, train_recall, train_precision, train_f1 = train_step(model, train_set, optimizer, loss_fn, acc_fn,acc_fn2, recall, precision, f1score, device)
        val_loss, val_acc, val_acc2, val_recall, val_precision, val_f1 = val_step(model, val_set, loss_fn,acc_fn,acc_fn2, recall, precision, f1score, device)
        print(f' Epoch {i+1} train loss : {fg.yellow + str(train_loss)+ fg.rs}  train acc macro : {fg.cyan + str(train_acc) + fg.rs} train acc micro: {fg.cyan + str(train_acc2) + fg.rs} train recall : {fg.cyan + str(train_recall) + fg.rs} train precision : {fg.cyan + str(train_precision) + fg.rs} train f1score : {fg.cyan + str(train_f1) + fg.rs} \n Epoch {i+1} val loss : {fg.yellow + str(val_loss)+ fg.rs}  val acc : {fg.cyan + str(val_acc) + fg.rs} val acc micro : {fg.cyan + str(val_acc2) + fg.rs} val recall : {fg.cyan + str(val_recall) + fg.rs} val precision : {fg.cyan + str(val_precision) + fg.rs} val f1 : {fg.cyan + str(val_f1) + fg.rs}')
        result['valacc'].append(val_acc)
        result['trainacc'].append(train_acc)
        result['trainloss'].append(train_loss)
        result['valoss'].append(val_loss)
        result['train_acc2'].append(train_acc2)
        result['valacc2'].append(val_acc2)
        result['train_recall'].append(train_recall)
        result['valrecall'].append(val_recall)
        result['train_precision'].append(train_precision)
        result['valprecision'].append(val_precision)
        result['valf1score'].append(val_f1)
        result['train_f1score'].append(train_f1)
        if val_acc2>best_accmicro:
            best_accmicro = val_acc2
            torch.save(model.state_dict(), str(loop)+'bestaccmicrocnn.pt')
        if val_f1>best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), str(loop)+'bestf1cnn.pt')
        if i==0:
            pred_loss = val_loss
        else:
            if val_loss<pred_loss:
                print(f'loop {loop} epoch {fg.yellow + str(i + 1) + fg.rs} : val loss has improved from {str(pred_loss)} to {bg.li_green+ str(val_loss)+ bg.rs}') 
                cur_patience = patience
            elif val_loss > pred_loss:
                print(f'loop {loop} epoch {fg.yellow + str(i + 1) + fg.rs} :val loss has worsened from {str(pred_loss)} to {bg.red+ str(val_loss)+ bg.rs}')
                cur_patience-=1
                if cur_patience ==0:
                    print('Early stop activated')
                    break
            elif val_loss == pred_loss:
                print('the loss didnt change from last epoch')
        pred_loss = val_loss
    return result

