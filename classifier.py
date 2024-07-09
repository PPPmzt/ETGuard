# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
import numpy as np
from tqdm import tqdm

from Classifier.model import MLP
# from Classifier.loss import loss_coteaching

import derpp

# Hyper Parameters
batch_size = 128
learning_rate = 1e-3
epochs = 100


def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)


# Train derpp
# def train_derpp(train_loader, epoch, model, optimizer, device):
#     model = derpp(MLP, loss, args, None)
#     model.observe(inputs, labels, not_aug_inputs, epoch=None)
#
#     for epoch in range(epoch):
#         model.train()
#         total_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(model.device), labels.to(model.device)
#
#             # 获取未增强的输入（用于数据增强的反向传播）
#             not_aug_inputs = inputs  # 这里假设未增强的输入与增强后的输入相同
#
#             # 观察数据
#             loss = model.observe(inputs, labels, not_aug_inputs, epoch=epoch)
#             total_loss += loss
#
#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
#
#     print("Training completed.")


# Train the Model
def train(train_loader, epoch, model, optimizer, device):
    train_total = 0
    train_correct = 0

    for i, data_labels in enumerate(train_loader):  # 每循环一次，返回一个batch数据

        feats = data_labels[:, :-1].to(dtype=torch.float32)
        labels = data_labels[:, -1].to(dtype=int)
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
            labels = labels.cuda()

        logits = model(feats)
        prec = accuracy(logits, labels)
        train_total += 1
        train_correct += prec

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)

    return train_acc


# predict unknown traffic data's label
def predict(test_loader, model, device, alpha=0.5):
    preds = []
    for i, data in enumerate(test_loader):

        # Forward + Backward + Optimize
        feats = data.to(dtype=torch.float32)

        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()

        logits = model(feats)
        outputs = F.softmax(logits, dim=1)
        preds.append((outputs[:, 1] > alpha).detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


def main(feat_dir, model_dir, result_dir, cuda_device):
    cuda_device = int(cuda_device)
    # get the origin training set
    be = np.load(os.path.join(feat_dir, 'be.npy'))[:, :32]
    ma = np.load(os.path.join(feat_dir, 'ma.npy'))[:, :32]

    print(be.shape, ma.shape)

    train_data = np.concatenate([be, ma], axis=0)
    train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    test_data_label = np.load(os.path.join(feat_dir, 'test.npy'))
    test_data = test_data_label[:, :32]
    test_label = test_data_label[:, -1]

    device = int(cuda_device) if cuda_device != 'None' else None

    if device != None:
        torch.cuda.set_device(device)

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Define models
    print('building model...')
    mlp = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp.to_cuda(device)
        mlp = mlp.cuda()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    # 开始训练
    epoch = 0
    mlp.train()
    for epoch in tqdm(range(epochs)):
        print(epoch)
        train(train_loader, epoch, mlp, optimizer, device)

    # 开始评估 evaluate
    mlp.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    preds = predict(test_loader, mlp, device)
    np.save(os.path.join(result_dir, 'prediction.npy'), preds)

    # 开始 metric 评估
    scores = np.zeros((2, 2))
    for label, pred in zip(test_label, preds):
        scores[int(label), int(pred)] += 1
    TP = scores[1, 1]
    FP = scores[0, 1]
    TN = scores[0, 0]
    FN = scores[1, 0]

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1score = 2 * Recall * Precision / (Recall + Precision)
    print(Recall, Precision, F1score)

    with open('data/result/detection_result.txt', 'w') as fp:
        fp.write('Testing data: Benign/Malicious = %d/%d\n' % ((TN + FP), (TP + FN)))
        fp.write('Recall: %.2f, Precision: %.2f, F1: %.2f\n' % (Recall, Precision, F1score))
        fp.write('Acc: %.2f\n' % (Accuracy))

    mlp = mlp.cpu()
    mlp.to_cpu()
    torch.save(mlp, os.path.join(model_dir, 'Detection_Model.pkl'))
