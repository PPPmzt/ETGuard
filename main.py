import os
import sys

sys.path.append('../RAPIER-master')
# import classifier
from Classifier import train, predict
import AE
from buffer import Buffer


def main(data_dir, model_dir, feat_dir, result_dir, buffer_size, cuda):
    # AE.train.main(data_dir, model_dir, cuda)  # AE pre-train
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)  # get_feature
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'testSet', cuda)

    buffer = Buffer(buffer_size)  # 初始化buffer
    INCRE = False
    train.main(feat_dir, model_dir, INCRE, buffer, cuda)
    # INCRE = True
    # train.main(feat_dir, model_dir, INCRE, buffer, cuda)

    predict.main(feat_dir, model_dir, result_dir, cuda)


if __name__ == '__main__':
    data_dir = 'data/data/3'  # TODO 注意，需要手动更换数据集（特征提取预训练数据集、特征提取）
    feat_dir = 'data/feat/3'  # TODO 注意，需要手动更换数据集（MLP预训练集、MLP增量学习各阶段、测试集）
    model_dir = 'data/model'
    result_dir = 'data/result'
    buffer_size = 200
    cuda = 0

    main(data_dir, model_dir, feat_dir, result_dir, buffer_size, cuda)
