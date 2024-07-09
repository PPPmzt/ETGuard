from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os

batch_size = 256
Max_epochs = 1000


def main(data_dir, model_dir, device):
    # get raw time-series data of training traffic data
    train_data_be = np.load(os.path.join(data_dir, 'be.npy'))
    train_data_ma = np.load(os.path.join(data_dir, 'ma.npy'))
    train_data = np.concatenate([train_data_be[:, :50], train_data_ma[:, :50]], axis=0)  # 为什么是前50
    np.random.shuffle(train_data)

    #train_data = np.load(os.path.join(data_dir, 'all.npy'))[:, :50]
    print(train_data_be.shape, train_data_ma.shape)

    total_size, input_size = train_data.shape
    print(total_size)
    device_id = int(device)
    print(device_id)
    torch.cuda.set_device(device_id)
    max_epochs = 100
    #max_epochs = Max_epochs * 200 // total_size
    dagmm = LSTM_AE_GMM(
        input_size=input_size,  # FIXME 输入特征的维度需要增加
        max_len=2000,
        emb_dim=32,
        hidden_size=8,
        dropout=0.2,
        est_hidden_size=64,
        est_output_size=8,
        device=device_id,
    ).cuda()

    dagmm.train_mode()
    optimizer = torch.optim.Adam(dagmm.parameters(), lr=1e-2)
    for epoch in range(max_epochs):
        for batch in range(total_size // batch_size + 1):
            if batch * batch_size >= total_size:
                break
            optimizer.zero_grad()
            input = train_data[batch_size * batch: batch_size * (batch + 1)]
            loss = dagmm.loss(torch.Tensor(input).long().cuda())
            print(f"epoch: {epoch}, batch: {batch},loss:{loss}")
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss)
        if (epoch + 1) % 5 == 0:  # modified
            dagmm.to_cpu()
            dagmm = dagmm.cpu()
            torch.save(dagmm, os.path.join(model_dir, f'gru_ae_{epoch+1}.pkl'))
            dagmm.to_cuda(device_id)
            dagmm = dagmm.cuda()
