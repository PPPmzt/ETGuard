# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
import numpy as np
from tqdm import tqdm

# Hyper Parameters
batch_size = 128


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
    test_data_label = np.load(os.path.join(feat_dir, 'testSet.npy'))
    # print("1",test_data_label)
    test_data = test_data_label[:, :32]
    test_label = test_data_label[:, -1]
    # print("2",test_data,"3",test_label)
    cuda_device = int(cuda_device)
    device = int(cuda_device) if cuda_device != 'None' else None
    if device != None:
        torch.cuda.set_device(device)

    # Loading models
    print('loading model...')
    mlp = torch.load(os.path.join(model_dir, 'Detection_Model.pkl'))
    if device != None:
        mlp.to_cuda(device)
        mlp = mlp.cuda()

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
    # print(Recall, Precision, F1score)
    # print("Recall: {Recall:.4f}, Precision: {Precision:.4f}, F1score: {F1score:.4f}".format(Recall=Recall, Precision=Precision, F1score=F1score))
    print(f"Recall: {Recall:.4f}, Precision: {Precision:.4f}, F1score: {F1score:.4f}")


    with open('data/result/detection_result.txt', 'w') as fp:
        fp.write('Testing data: Benign/Malicious = %d/%d\n' % ((TN + FP), (TP + FN)))
        fp.write('Recall: %.2f, Precision: %.2f, F1: %.2f\n' % (Recall, Precision, F1score))
        fp.write('Acc: %.2f\n' % (Accuracy))
