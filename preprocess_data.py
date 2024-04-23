import numpy as np
from glob import glob
from tqdm import tqdm
import gc



def load_data(data_dir:str, emb_dir:str, min_length:int, max_length:int):

    print("Preprocessing data...")

    #IMDBデータセットの読み込み
    train_pos_file = glob(data_dir + '/train/pos/*.txt')
    train_neg_file = glob(data_dir + '/train/neg/*.txt')
    test_pos_file = glob(data_dir + '/test/pos/*.txt')
    test_neg_file = glob(data_dir + '/test/neg/*.txt')

    all_files = train_pos_file + train_neg_file + test_pos_file + test_neg_file

    data = []

    print("Loading data...")
    for file in tqdm(all_files):
        with open(file, "r", encoding="utf-8") as f:
            contenxt = f.read()
            data.append(contenxt.strip().split())

    labels = [1] * len(train_pos_file) + [0] * len(train_neg_file) + [1] * len(test_pos_file) + [0] * len(test_neg_file)

    #Warmupのため、トークン数がmin_length以下のデータを削除
    if min_length:
        data, labels = zip(*[(d, l) for d, l in zip(data, labels) if len(d) >= min_length])
        data = list(data)
        labels = list(labels)

    #パディング処理
    print("Padding data...")
    padded_data = padding_token(data, max_length)

    del data
    gc.collect()

    #データの単語埋め込み化
    print("Embedding data...")
    emb_data = embedding(emb_dir, padded_data)

    print(len(emb_data). len(emb_data[0]), len(emb_data[0][0]))

    preprocessed_data = np.array(emb_data, dtype=np.float32)
    preprocessed_labels = np.array(labels, dtype=np.int32)


    print("Preprocessing data done!")
    return preprocessed_data, preprocessed_labels


def padding_token(data, max_len):
    pad_token = "<PAD>"

    for i, d in tqdm(enumerate(data)):
        if len(d) < max_len:
            data[i] = d + [pad_token] * (max_len - len(d))
        else:
            data[i] = d[:max_len]

    return data

def embedding(emb_dir, data, pad_token="<PAD>", unk_token="<UNK>"):

    emb_file = glob(emb_dir + '/*.txt')

    print(emb_file)

    #単語埋め込みの読み込み
    embeddings_index = {}
    with open("glove.840B.300d/glove.840B.300d.txt", 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            values = line.split()
            print(f"values: {values}")
            break
            word = values[0]
            if len(values[1:]) != 300:
                skip_len = len(values[1:]) - 300
                values = values[skip_len:]
            vector = np.array(values[1:], "float32")
            embeddings_index[word] = vector
        #embeddings_index[pad_token] = [0.0] * len(emb)
        #embeddings_index[unk_token] = [0.1] * len(emb)
    f.close()

    #単語埋め込みの作成
    sentence_list = []

    #データの単語埋め込み化（文章数×最大トークン数×埋め込み次元）
    for sentence in tqdm(data):
        emb_sentence = []
        for word in sentence:
            if word in embeddings_index:
                emb_sentence.append(embeddings_index[word])
            else:
                emb_sentence.append(embeddings_index[unk_token])
        sentence_list.append(emb_sentence)

    return sentence_list

def main():
    data_dir = 'aclImdb'
    emb_dir = 'glove.840B.300d'
    min_length = 800
    max_length = 1000
    preprocessed_data, preprocessed_labels = load_data(data_dir, emb_dir, min_length, max_length)
    print(preprocessed_data.shape)
    print(preprocessed_labels.shape)

if __name__ == '__main__':
    main()