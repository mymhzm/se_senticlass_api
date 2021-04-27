# 特征工程
import torch
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sen_len, w2v_path='./model/w2v_all.model'):
        self.w2v_path = w2v_path

        # 规范句子长度
        self.sen_len = sen_len

        # 一定要理解下面三个东东的含义
        self.word2idx = {}  # 序号-对应word
        self.idx2word = [] # 对应word-序号
        self.embedding_matrix = [] # 对应vector, 按行排列形成matrix
        # 为什么要index ： 如果都用word内存消耗大,如果不用word不好进行匹配

    def get_w2v_model(self):
        # 把已经训练好的w2v模型读进来
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size #统一使用词向量维数

    def add_embedding(self, word):
        ## 用于加入某个词的vector, 随机赋值就可以了
        # 符号"<PAD>" 用于补全短句子, "<UNK>"用于标记未见过的词
        # 把word加进embedding, 并赋予他一个随机生成的 representation vector
        # 注意word 只会是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim) # 随机给一个embedding
        torch.nn.init.uniform_(vector) # 初始化nn网络
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        ## 获取一个完整的word2vec embedding
        ## 以便后面model里面使用
        #print("Get embedding ...")
        # 已经训练好的 Word2vec word embedding （请记住这种描述，embedding其实就是有了wrod2vec的关系表）
        # 可以用vector来代表任意一个word
        if load:
            #print("loading word to vec emotion_model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作一个 word2idx 的 dictionary 便于由word读idx
        # 制作一个 idx2word 的 list 便于由idx读word
        # 制作一个 word2vector 的 list 只能由idx读vectors
        for i, word in enumerate(self.embedding.wv.vocab): # .wv.vocab 读取字典id-word
            #print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['he'] = 1
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx) #获取当前词的序号,技巧:直接用len
            self.idx2word.append(word) #获取当前词
            self.embedding_matrix.append(self.embedding[word]) #获取w2v model的vectors
        #print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 把 "<PAD>" 跟 "<UNK>" 加进 embedding 里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        #print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.sen_len:
            # 超出规定长度就截掉
            sentence = sentence[:self.sen_len]
        else:
            # 不够规定长度就补充, "<PAD>"代表补充的单词
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        #assert len(sentence) == self.sen_len #做一个验证而已，确保生效
        return sentence

    def sentence_word2idx(self, sentences):
        # 把句子里面的字转成相对应的index
        sentence_list = []
        for i, sen in enumerate(sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()): #处理unk
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        # 由于都是index组成的sentences，如[[3,1,87], [42,20], ...]
        # 转成long类型（int64）的tensor
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把 labels 转成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)