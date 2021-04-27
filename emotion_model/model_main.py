# model_main.py
import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from utils import load_data_labled, load_data_unlabled, labels_norm, tokenization
from data import ReviewDataset
from preprocess import Preprocess
from model import LSTM_Net
from train import training

label_map = {'anger': 0, 'fear': 1, 'sadness': 2, 'love': 3, 'joy': 4, 'surprise': 5}
path_prefix = './'
# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练和验证数据
train_x_path = os.path.join(path_prefix, 'data/train.txt')
valid_x_path = os.path.join(path_prefix, 'data/val.txt')
testing_data_path = os.path.join(path_prefix, 'data/test_data.txt')

# w2v模型地址
w2v_path = os.path.join(path_prefix, 'model/w2v_all.model')  # 處理 word to vec emotion_model 的路徑

# 模型存放位置
model_dir = os.path.join(path_prefix, 'model/') # emotion_model directory for checkpoint emotion_model

# set some important parameters
# -sentence length :: auto handling the sentence length with padding and masking.
# note:better to set a suitable for our different input sentence.
# -fix_embedding :: allow the LSTM change the word embedding or not
# -batch_size :: training batch size may influence the result of training
# -epoch :: how many times we train our emotion_model with whole data
# -lr :: learning rate default 0.001 is ok.

sen_len = 60
fix_embedding = True
batch_size = 32
epoch = 100
lr = 0.001

if __name__ == '__main__':
    # 把训练raw数据放进来
    print("loading data ...")
    train_x, train_labels = load_data_labled(train_x_path)
    valid_x, valid_labels =load_data_labled(valid_x_path)
    #testing_data = load_data_unlabled(testing_data_path)

    # 将sentence处理成tokens; 支持不去stop words
    tokens_train = tokenization(train_x + valid_x, del_stop=False)
    #tokens_test = tokenization(testing_data, del_stop=False)

    # 处理labels
    train_labels = labels_norm(train_labels, label_map)
    valid_labels = labels_norm(valid_labels, label_map)

    # 基于已有w2v模型对tokens做id转换, 以便在后面的embedding层中, 能拿到vectors
    preprocess = Preprocess(tokens_train, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    labels = preprocess.labels_to_tensor(train_labels+valid_labels)

    # 生成一个model对象,用来跑模型
    # 2021/3/18 新增attention机制
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
    model = model.to(device)

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_train, X_val, y_train, y_val = model_selection.train_test_split(train_x, labels, test_size=0.2, random_state=233)

    # generate Dataset instance
    train_dataset = ReviewDataset(X=X_train, y=y_train)
    val_dataset = ReviewDataset(X=X_val, y=y_val)

    # batch of tensor
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 0)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 0)

    # training
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)