# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import sys,os
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
sys.path.append("../")
# from models.utils.continual_model import ContinualModel
# from utils.args import add_rehearsal_args, ArgumentParser
import AE
from buffer import Buffer
from backbone import MLP
import copy
# Hyper Parameters
batch_size = 128
reply_size = 128
learning_rate = 1e-3
epochs = 100
alpha = 0.5
beta = 0.5
NAME = 'etguard'
class ETGuard(nn.Module):

    # COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    #
    # @staticmethod
    # def get_parser() -> ArgumentParser:
    #     parser = ArgumentParser(description='Gradient based sample selection for online continual learning')
    #     add_rehearsal_args(parser)
    #     parser.add_argument('--batch_num', type=int, default=1,
    #                         help='Number of batches extracted from the buffer.')
    #     parser.add_argument('--gss_minibatch_size', type=int, default=None,
    #                         help='The batch size of the gradient comparison.')
    #     return parser

    def __init__(self,cuda_device):
        super(ETGuard, self).__init__()
        print('model name:{}'.format(NAME))
        device = int(cuda_device) if cuda_device != 'None' else None
        if device != None:
            torch.cuda.set_device(device)
        self.backbone = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)

    def forward(self, inputs):
        return self.backbone(inputs)
    def pretrain(self,train_dataset, batch_size, model_dir, buffer, cuda_device):
        cuda_device = int(cuda_device)
        device = int(cuda_device) if cuda_device != 'None' else None
        if device != None:
            torch.cuda.set_device(device)

        # Data Loader (Input Pipeline)
        print('loading dataset...')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if device != None:
            self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # 开始训练
        self.train()
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

                logits = self.forward(feats)  # 未经激活函数的logits
                # 在最后一轮epoch中，将数据存入buffer
                if epoch == epochs - 1:
                    buffer.add_data(examples=feats, labels=labels, logits=logits.data)

                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss}")

            # train_acc = float(train_correct) / float(train_total)

        print('saving model...')
        self.to('cpu')
        torch.save(self, os.path.join(model_dir, f'{NAME}_0.pkl'))
        print(len(buffer))

    def pre_func(self, test_loader, device, alpha=0.5):
        preds = []
        for i, data in enumerate(test_loader):

            # Forward + Backward + Optimize
            feats = data.to(dtype=torch.float32)

            if device != None:
                torch.cuda.set_device(device)
                feats = feats.cuda()

            logits = self.forward(feats)
            outputs = F.softmax(logits, dim=1)
            # print(outputs)
            preds.append((outputs[:, 1] > alpha).detach().cpu().numpy())

        return np.concatenate(preds, axis=0)
    def predict_part(self, feat_dir, result_dir, cuda_device):
        test_data_label = np.load(os.path.join(feat_dir, 'test_part.npy'))
        # print("1", test_data_label)
        test_data = test_data_label[:, :32]
        test_label = test_data_label[:, -1]
        # print("2", test_data, "3", test_label)
        cuda_device = int(cuda_device)
        device = int(cuda_device) if cuda_device != 'None' else None
        if device != None:
            torch.cuda.set_device(device)

        # Loading models
        print('loading model...')
        if device != None:
            self.to(device)

        # 开始评估 evaluate
        self.eval()
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        preds = self.pre_func(test_loader, device)
        np.save(os.path.join(result_dir, f'{NAME}_testPart_{INCRE_ID}.npy'), preds)

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

        with open(f'results/{NAME}_testPart_{INCRE_ID}.txt', 'w') as fp:
            fp.write('model:pre\n')
            fp.write('Testing data: Benign/Malicious = %d/%d\n' % ((TN + FP), (TP + FN)))
            fp.write('Recall: %.4f, Precision: %.4f, F1: %.4f\n' % (Recall, Precision, F1score))
            fp.write('Acc: %.4f\n' % (Accuracy))

    def predict_set(self, feat_dir, result_dir, cuda_device):
        test_data_label = np.load(os.path.join(feat_dir, 'testSet.npy'))
        # print("1", test_data_label)
        test_data = test_data_label[:, :32]
        test_label = test_data_label[:, -1]
        # print("2", test_data, "3", test_label)
        cuda_device = int(cuda_device)
        device = int(cuda_device) if cuda_device != 'None' else None
        if device != None:
            torch.cuda.set_device(device)

        # Loading models
        print('loading model...')
        if device != None:
            self.to(device)

        # 开始评估 evaluate
        self.eval()
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        preds = self.pre_func(test_loader, device)
        np.save(os.path.join(result_dir, f'{NAME}_testSet_{INCRE_ID}.npy'), preds)

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

        with open(f'results/{NAME}_testSet_{INCRE_ID}.txt', 'w') as fp:
            fp.write('model:pre\n')
            fp.write('Testing data: Benign/Malicious = %d/%d\n' % ((TN + FP), (TP + FN)))
            fp.write('Recall: %.4f, Precision: %.4f, F1: %.4f\n' % (Recall, Precision, F1score))
            fp.write('Acc: %.4f\n' % (Accuracy))

    def continual_learn(self, train_dataset, batch_size, model_dir, buffer, cuda_device):
        """
        etguard trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        cuda_device = int(cuda_device)
        device = int(cuda_device) if cuda_device != 'None' else None
        if device is not None:
            torch.cuda.set_device(device)

        # Data Loader (Input Pipeline)
        print('loading dataset...')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        total_batch = len(train_loader)
        print(total_batch)

        print('loading model...')
        if device is not None:
            self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        all_feats = torch.zeros((3, 3))
        all_labels = torch.zeros((3, 3))
        all_logits = torch.zeros((3, 3))
        # 开始训练
        self.train()
        for epoch in tqdm(range(epochs)):

            for batch_idx, data_labels in enumerate(train_loader):  # 每循环一次，返回一个batch数据
                feats = data_labels[:, :-1].to(dtype=torch.float32)
                labels = data_labels[:, -1].to(dtype=int)

                if device != None:
                    torch.cuda.set_device(device)
                    feats = feats.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                logits = self.forward(feats)  # logits
                # print(torch.sum(logits))
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                tot_loss = loss.item()
                if not buffer.is_empty():  # TODO 如果getdata取的很少
                    buf_inputs, _, buf_logits = buffer.get_data(size=batch_size, transform=None, device=cuda_device)

                    buf_mlp_logits = self.forward(buf_inputs)
                    loss_mse = F.mse_loss(buf_mlp_logits, buf_logits)
                    gama = 1 + (1 / (1 + np.exp(-loss_mse.item()*10)))
                    # print("gama:", gama, "loss_mse:", loss_mse)
                    loss_mse.backward()
                    tot_loss += alpha * loss_mse.item()

                    buf_inputs, buf_labels, _ = buffer.get_data(size=int(gama*batch_size), transform=None, device=cuda_device)
                    buf_inputs = buf_inputs.cuda()
                    buf_labels = buf_labels.cuda()

                    buf_mlp_logits = self.forward(buf_inputs)
                    loss_ce = F.cross_entropy(buf_mlp_logits, buf_labels)
                    loss_ce.backward()
                    tot_loss += beta * loss_ce.item()

                optimizer.step()  # 根据所有loss的总梯度更新参数

                if epoch == epochs - 1:
                    if batch_idx == 0:
                        all_feats = feats
                        all_labels = labels
                        all_logits = logits
                    else:
                        all_feats = torch.cat((all_feats, feats), dim=0)
                        all_labels = torch.cat((all_labels, labels), dim=0)
                        all_logits = torch.cat((all_logits, logits), dim=0)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {tot_loss}")
        buffer.add_data(examples=all_feats, labels=all_labels, logits=all_logits.data)

        self.to('cpu')
        torch.save(self, os.path.join(model_dir, f'{NAME}_{INCRE_ID}.pkl'))

if __name__ == "__main__":
    buffer_size = 200
    buffer = Buffer(buffer_size)  # 初始化buffer
    for INCRE_ID in range(6):
        print(f"------------------------INCRE_ID: {INCRE_ID}-----------------------------------")
        data_dir = f'../data/data/{INCRE_ID}'  # TODO 注意，需要手动更换数据集（特征提取预训练数据集、特征提取）
        feat_dir = f'../data/feat/{INCRE_ID}'  # TODO 注意，需要手动更换数据集（MLP预训练集、MLP增量学习各阶段、测试集）
        AE_dir = '../data/model'
        model_dir = './weights'
        result_dir = './results'
        cuda_device = 1
        print(f'buffer_len:{len(buffer)}')
        print("******************feature*********************")
        AE.get_feat.main(data_dir, AE_dir, feat_dir, 'be', cuda_device)  # get_feature
        AE.get_feat.main(data_dir, AE_dir, feat_dir, 'ma', cuda_device)
        AE.get_feat.main(data_dir, AE_dir, feat_dir, 'test_part', cuda_device)
        AE.get_feat.main(data_dir, AE_dir, feat_dir, 'testSet', cuda_device)
        be = np.load(os.path.join(feat_dir, 'be.npy'))[:, :32]  # TODO 注意，无法区分预训练数据集和增量数据集
        ma = np.load(os.path.join(feat_dir, 'ma.npy'))[:, :32]
        print("benign data :", be.shape, "malicious data :", ma.shape)
        train_data = np.concatenate([be, ma], axis=0)
        train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)
        train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)
        print("******************init model*********************")
        etguard = ETGuard(cuda_device)
        if INCRE_ID == 0:
            print("******************pretrain*********************")
            etguard.pretrain(train_dataset, batch_size, model_dir, buffer, cuda_device)
            print("******************predict*********************")
            etguard = torch.load(os.path.join(model_dir, f'{NAME}_0.pkl'))
            etguard.predict_part(feat_dir, result_dir, cuda_device)
            etguard.predict_set(feat_dir, result_dir, cuda_device)
        else:
            print("******************continual learn*********************")
            etguard = torch.load(os.path.join(model_dir, f'{NAME}_{INCRE_ID - 1}.pkl'))
            etguard.continual_learn(train_dataset, batch_size, model_dir, buffer, cuda_device)
            print("******************predict*********************")
            etguard = torch.load(os.path.join(model_dir, f'{NAME}_{INCRE_ID}.pkl'))
            etguard.predict_part(feat_dir, result_dir, cuda_device)
            etguard.predict_set(feat_dir, result_dir, cuda_device)

    # INCRE_ID = 0
    # data_dir = f'../data/data/{INCRE_ID}'  # TODO 注意，需要手动更换数据集（特征提取预训练数据集、特征提取）
    # feat_dir = f'../data/feat/{INCRE_ID}'  # TODO 注意，需要手动更换数据集（MLP预训练集、MLP增量学习各阶段、测试集）
    # AE_dir = '../data/model'
    # model_dir = './weights'
    # result_dir = './results'
    # buffer_size = 200
    # buffer = Buffer(buffer_size)  # 初始化buffer
    # cuda_device = 0
    # print("******************feature*********************")
    # AE.get_feat.main(data_dir, AE_dir, feat_dir, 'be', cuda_device)  # get_feature
    # AE.get_feat.main(data_dir, AE_dir, feat_dir, 'ma', cuda_device)
    # AE.get_feat.main(data_dir, AE_dir, feat_dir, 'test_part', cuda_device)
    # AE.get_feat.main(data_dir, AE_dir, feat_dir, 'testSet', cuda_device)
    # be = np.load(os.path.join(feat_dir, 'be.npy'))[:, :32]  # TODO 注意，无法区分预训练数据集和增量数据集
    # ma = np.load(os.path.join(feat_dir, 'ma.npy'))[:, :32]
    # print("benign data :", be.shape, "malicious data :", ma.shape)
    # train_data = np.concatenate([be, ma], axis=0)
    # train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)
    # train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)
    # print("******************init model*********************")
    # etguard = etguard()
    # if INCRE_ID == 0:
    #     print("******************pretrain*********************")
    #     etguard.pretrain(train_dataset, batch_size, model_dir, buffer, cuda_device)
    #     print("******************predict*********************")
    #     etguard = torch.load(os.path.join(model_dir, f'{NAME}_0.pkl'))
    #     etguard.predict_part(feat_dir, result_dir, cuda_device)
    #     etguard.predict_set(feat_dir, result_dir, cuda_device)
    # else:
    #     print("******************continual learn*********************")
    #     etguard = torch.load(os.path.join(model_dir, f'{NAME}_{INCRE_ID-1}.pkl'))
    #     etguard.continual_learn(train_dataset, batch_size, model_dir, buffer, cuda_device)
    #     print("******************predict*********************")
    #     etguard = torch.load(os.path.join(model_dir, f'{NAME}_{INCRE_ID}.pkl'))
    #     etguard.predict_part(feat_dir, result_dir, cuda_device)
    #     etguard.predict_set(feat_dir, result_dir, cuda_device)

