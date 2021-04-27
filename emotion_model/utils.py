import torch
import os
import pandas as pn
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_data_labled(path=''):
    text, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            text.append(line.strip('\n').split(';')[0])
            labels.append(line.strip('\n').split(';')[1])
    return text, labels

def load_data_unlabled(path=''):
    text = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            text.append(line.strip('\n'))
    return text

def labels_norm(labels, label_map, norm=True):
    if norm:
        # change the labels to numbers
        labels = [label_map[l] for l in labels]
    else:
        # change numbers to labels
        map_label = [i for i in label_map.keys()]
        labels = [map_label[l] for l in labels]
    return labels

def tokenization(data, del_stop=True):
    data = [sent.lower() for sent in data]
    data = [word_tokenize(sent) for sent in data]

    clean_data = []

    word_Lemmatized = WordNetLemmatizer()


    if del_stop:
        stop_words = stopwords.words('english')
        # stop_words.append()
        for idx, sent in enumerate(data):
            new_sent = [word_Lemmatized.lemmatize(w) for w in sent if w not in stop_words]
            clean_data.append(new_sent)
    else:
        for idx, sent in enumerate(data):
            new_sent = [word_Lemmatized.lemmatize(w) for w in sent]
            clean_data.append(new_sent)
    return clean_data


def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    # torch.max()[1]
    # [1] :: return the highest value index
    n2labels = torch.max(outputs, dim=1)[1]
    correct = torch.sum(torch.eq(n2labels, labels)).item()
    return correct