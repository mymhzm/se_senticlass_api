import os
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
from utils import *
import pandas as pd
from preprocess import Preprocess
from data import ReviewDataset

w2v_path = os.path.join(dir_path, 'model/w2v_all.model')
sen_len = 60
label_map = {'anger': 0, 'fear': 1, 'sadness': 2, 'love': 3, 'joy': 4, 'surprise': 5}

# 最简单的接口版本
# 暂不支持多条查询（看看是否需要新增多条支持, 已经支持？
# 暂只能每次预测前必须进行一次初始化（0427已优化
class lstm_classifier():
    def __init__(self, model_trained="ckpt.model", batch_size = 256, device = "cuda:0"):
        self.model = torch.load(os.path.join(dir_path, "model/" + model_trained))
        self.batch_size = batch_size
        self.device = device
        self.test_loader = None

    def data_init(self):
        # 做embedding
        try:
            # 获得原word2vec的word-idx映射关系(这个信息和已训练好的embedding layer里的idx一致)
            self.preprocess = Preprocess(sen_len, w2v_path=w2v_path)
            # 做输入端embedding层
            self.preprocess.make_embedding(load=True)
            print("classifier initialized !")
        except BaseException as e:
            print(e)

    def data_handle(self, input_sentences):
        text = []
        for row in input_sentences.split("\n"):
            text.append(row.strip('\n'))
        tokens_test = tokenization(text)
        tokens_test = self.preprocess.sentence_word2idx(tokens_test)
        test_dataset = ReviewDataset(X=tokens_test, y=None)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size, # 后续可支持批量预测
                                                  shuffle=False
                                                  )

    def predict(self, input_sentences):
        # 处理输入数据
        self.data_handle(input_sentences)
        self.model.eval()
        res_output = []
        with torch.no_grad():
            for inputs in self.test_loader:
                inputs = inputs.to(self.device, dtype=torch.long)
                outputs = self.model(inputs)
                #outputs = outputs.squeeze()
                # 返回softmax后最大概率索引, 对应的类别
                outputs = torch.max(outputs, dim=1)[1]
                res_output += outputs.int().tolist()
        outputs = labels_norm(outputs, label_map, norm=False)
        return outputs

if __name__ == "__main__":
    print("api demo")
    lstm_action = lstm_classifier()
    lstm_action.data_init()
    while True:
        input_sentences = input("input your words:\n")
        #input_sentences = '''you are so funny, i like you \n you are so ugly, holly shit'''
        #print(input_sentences)

        outputs = lstm_action.predict(input_sentences)

        print(input_sentences)
        print("Emotion:", outputs, "\n")

        # for idx, pack in enumerate(zip(input_sentences, outputs)):
        #     print("No.", idx, "  ", pack[0])
        #     print("Emotion:", pack[1], "\n")