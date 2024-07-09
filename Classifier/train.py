# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import sys, os
import numpy as np
from tqdm import tqdm

from .model import MLP
from buffer import Buffer

# Hyper Parameters
batch_size = 128
reply_size = 128
learning_rate = 1e-3
epochs = 100
alpha = 0.5
beta = 0.5


def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)  # 输出的output为概率分布
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)


# predict traffic data's label
# def predict(data_loader, model, device, alpha=0.5):
#     preds = []
#     for i, data in enumerate(data_loader):

#         # Forward + Backward + Optimize
#         feats = data.to(dtype=torch.float32)

#         if device is not None:
#             torch.cuda.set_device(device)
#             feats = feats.cuda()

#         logits = model(feats)
#         outputs = F.softmax(logits, dim=1)
#         preds.append((outputs[:, 1] > alpha).detach().cpu().numpy())

#     return np.concatenate(preds, axis=0)


# # Train the Model
# def train(train_loader, model, optimizer, device):
#     train_total = 0
#     train_correct = 0

#     for i, data_labels in enumerate(train_loader):  # 每循环一次，返回一个batch数据

#         feats = data_labels[:, :-1].to(dtype=torch.float32)
#         labels = data_labels[:, -1].to(dtype=int)
#         if device is not None:
#             torch.cuda.set_device(device)
#             feats = feats.cuda()
#             labels = labels.cuda()

#         logits = model(feats)
#         prec = accuracy(logits, labels)
#         train_total += 1
#         train_correct += prec

#         loss = F.cross_entropy(logits, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     train_acc = float(train_correct) / float(train_total)

#     return train_acc  # 没啥用


# Pretrain the Model
def pretrain(train_dataset, batch_size, model_dir, buffer, cuda_device):
    cuda_device = int(cuda_device)
    device = int(cuda_device) if cuda_device != 'None' else None
    if device != None:
        torch.cuda.set_device(device)

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print('building model...')
    mlp = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp.to_cuda(device)
        mlp = mlp.cuda()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    # 开始训练
    mlp.train()
    for epoch in tqdm(range(epochs)):
        # train_total = 0
        # train_correct = 0

        for i, data_labels in enumerate(train_loader):  # 每循环一次，返回一个batch数据

            feats = data_labels[:, :-1].to(dtype=torch.float32)
            labels = data_labels[:, -1].to(dtype=int)
            if device != None:
                torch.cuda.set_device(device)
                feats = feats.cuda()
                labels = labels.cuda()

            logits = mlp(feats)  # 未经激活函数的logits
            # 在最后一轮epoch中，将数据存入buffer
            if epoch == epochs - 1:
                buffer.add_data(examples=feats, labels=labels, logits=logits.data)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss}")

        # train_acc = float(train_correct) / float(train_total)
    
    print('saving model...')
    mlp = mlp.cpu()
    mlp.to_cpu()
    torch.save(mlp, os.path.join(model_dir, 'Detection_Model.pkl'))



# 一次增量更新训练，往buffer里存放了一次样本数据
def derpp(train_dataset, batch_size, reply_size, model_dir, buffer, alpha, beta, cuda_device): 
    cuda_device = int(cuda_device)
    device = int(cuda_device) if cuda_device != 'None' else None
    if device is not None:
        torch.cuda.set_device(device)

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    total_batch=len(train_loader)
    print(total_batch)

    print('loading model...')
    mlp = torch.load(os.path.join(model_dir, 'Detection_Model.pkl'))
    if device is not None:
        mlp.to_cuda(device)
        mlp = mlp.cuda()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    all_feats=torch.zeros((3, 3))
    all_labels=torch.zeros((3, 3))
    all_logits=torch.zeros((3, 3))
    # 开始训练
    mlp.train()
    for epoch in tqdm(range(epochs)):

        for batch_idx, data_labels in enumerate(train_loader):  # 每循环一次，返回一个batch数据
            feats = data_labels[:, :-1].to(dtype=torch.float32)
            labels = data_labels[:, -1].to(dtype=int)

            if device != None:
                torch.cuda.set_device(device)
                feats = feats.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()

            logits = mlp(feats)  # logits
            # print(torch.sum(logits))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            tot_loss = loss.item()
            if not buffer.is_empty():  # TODO 如果getdata取的很少
                buf_inputs, _, buf_logits = buffer.get_data(size=reply_size, transform=None, device=cuda_device)

                buf_mlp_logits = mlp(buf_inputs)
                loss_mse = alpha* F.mse_loss(buf_mlp_logits, buf_logits)

                loss_mse.backward()
                tot_loss += loss_mse.item()

                buf_inputs, buf_labels, _ = buffer.get_data(size=reply_size, transform=None, device=cuda_device)
                buf_inputs=buf_inputs.cuda()
                buf_labels=buf_labels.cuda()

                buf_mlp_logits = mlp(buf_inputs)
                loss_ce = beta* F.cross_entropy(buf_mlp_logits, buf_labels)
                loss_ce.backward()
                tot_loss += loss_ce.item()
            optimizer.step()  # 根据所有loss的总梯度更新参数

            if epoch == epochs - 1:
                if batch_idx==0 :
                    all_feats=feats
                    all_labels=labels
                    all_logits=logits
                else: 
                    all_feats=torch.cat((all_feats,feats),dim=0)
                    all_labels=torch.cat((all_labels,labels),dim=0)
                    all_logits=torch.cat((all_logits,logits),dim=0)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {tot_loss}")

    buffer.add_data(examples=all_feats, labels=all_labels, logits=all_logits.data)      
    
    mlp = mlp.cpu()
    mlp.to_cpu()
    torch.save(mlp, os.path.join(model_dir, 'New_Detection_Model.pkl'))


def main(feat_dir, model_dir, INCRE, buffer, cuda_device):
    # get the origin training set
    be = np.load(os.path.join(feat_dir, 'be.npy'))[:, :32]  # TODO 注意，无法区分预训练数据集和增量数据集
    ma = np.load(os.path.join(feat_dir, 'ma.npy'))[:, :32]

    print("benign data :",be.shape,"malicious data :", ma.shape)

    train_data = np.concatenate([be, ma], axis=0)
    train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    if INCRE is False:
        print("pretrain")
        pretrain(train_dataset, batch_size, model_dir, buffer, cuda_device)  # TODO 这样调用可以吗？会重复创建buffer吗？
    else:
        print("derpp")
        derpp(train_dataset, batch_size, reply_size, model_dir, buffer, alpha, beta, cuda_device)  # TODO 注意，需要手动更换数据集
