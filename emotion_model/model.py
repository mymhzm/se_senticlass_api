# emotion_model.py
## 定义模型
# 這個 block 是要拿來訓練的模型
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# emotion_model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 制作 embedding layer, 重点：仅在这里放入了embedding的词向量,意味着训练时只需要input word idx即可
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1)) # 声明了embedding的数量和维数
        self.embedding.weight = torch.nn.Parameter(embedding)

        # 是否练 embedding layer，
        self.embedding.weight.requires_grad = False if fix_embedding else True

        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # attention输出的权重和偏移量
        self.w_omega = nn.Parameter((torch.zeros(self.hidden_dim * self.num_layers).cuda()), requires_grad=True)
        self.b_omega = nn.Parameter((torch.zeros(self.hidden_dim * self.num_layers).cuda()), requires_grad=True) ## nn.Parameter加入训练

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True) # batch_first=True要设置，不然默认是[seq_len, batch_size, embedding]

        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim,hidden_dim // 2),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(hidden_dim // 2, 6),
                                         )

    def attention_net(self, lstm_output):
        # hidden : [batch_size, hidden_dim * num_directions(=2), 1(=n_layer)]

        # 以lstm最后一层作为query，对原sentence做attention，计算阿尔法分布，提取句子重要信息
        # [batch, 1, hidden_dim]
        final_state = lstm_output[:, -1, :]

        # [batch, seq_len, hidden_dim]  seq_len个state的输出, 维度hidden_dim
        k = lstm_output # 每个batch的每个Outputs作为key值, （同时也当成后面的values

        # [batch, hidden_dim, 1]
        q = final_state.view(-1, self.hidden_dim , 1) # 用最后一个时间点作为query

        # q和k叉乘，并保留原batch，得到每一个句子对应的预阿尔法值（seq_len解读）
        # attn_weights : [batch, seq_len, 1] ->压缩 [batch_size, seq_len]
        attn_weights = torch.bmm(k, q).squeeze(2)

        # 对每个batch求出他的α分布
        # 求出来的分布值就是如下表示, 值的个数就是seq_len, 对应的就是每一个words的关注度!
        # [batch, seq_len]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # [batch, seq_len, hidden_dim] and [batch, seq_len]
        # [batch, hidden_dim, seq_len] and [batch, seq_len, 1]
        # transpose交换两个维度
        # qkv = [batch, hidden_dim,  1] -> [batch, hidden_dim]
        qkv = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        # 给attention层加上weights和bias，一起参与训练，加强表达
        # [batch, hidden_dim] 那么dot product需要拼成一样就可以计算了
        fit_w = self.w_omega.expand(qkv.shape[0], self.hidden_dim)
        fit_b = self.b_omega.expand(qkv.shape[0], self.hidden_dim)
        attention = torch.mul(qkv, fit_w) + fit_b
        return attention # attention : [batch_size, hidden_dim]

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        """
        _ 还会输出h_n和c_n
        h_n为每个batch最后一个state的输出但是可以通过x[:, -1, :]来取得
        c_n是每个batch最后一个state的cell的状态（一般用不到嗷）
        output:batch_size * len_sen * hidden_size
        h_n: (1(单向)*self.num_layers(lstm的层数)) * batch_size * hidden_size
        c_n: (1(单向)*self.num_layers) * batch_size * hidden_size
        
        x 的 dimension (batch, seq_len, hidden_dim) 因为设置了batch_first=True
        取最后一个state的LSTM的输出, 来替代整个sentence的意思 :: x = x[:, -1, :] 和 h_n[-1]是一样的, 但是h_n要提前定义
        这里是attention模块，用最后一个state作为query去match每个state的outputs，然后和outputs叉乘得到attention值
        """
        x = self.attention_net(x)
        x = self.classifier(x)
        return x