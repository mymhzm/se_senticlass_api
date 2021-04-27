# 用于训练word to vector的word embedding
from utils import load_data_labled, load_data_unlabled, tokenization
from gensim.models import word2vec

def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=250, window=9, min_count=5, workers=12, iter=10, sg=1)
    return model

if __name__ == "__main__":
    print("loading tarining data ...")
    train_x, _ = load_data_labled('./data/train.txt')
    validata, _ = load_data_labled('./data/val.txt')

    print("loading testing data ...")
    test_x = load_data_unlabled('./data/test_data.txt')

    tokens_all = tokenization(train_x + validata + test_x, del_stop=False)

    model = train_word2vec(tokens_all)

    print("saving emotion_model ...")
    model.save('./emotion_model/w2v_all.emotion_model')