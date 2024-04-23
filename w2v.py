import urllib.request
import zipfile
import re
import MeCab
from gensim.models import word2vec, Word2Vec, KeyedVectors
from gensim.models.poincare import PoincareModel, PoincareRelations
import numpy as np
import collections

# url = 'https://www.aozora.gr.jp/cards/000148/files/794_ruby_4237.zip'  # 三四郎
title = '銀河鉄道の夜'
fname = '43737_ruby_19028.zip'
url = 'https://www.aozora.gr.jp/cards/000081/files/{}'.format(fname)  # 銀河鉄道の夜


def download_novel():
    # zipファイルダウンロード
    urllib.request.urlretrieve(url, fname)

    # ダウンロードしたzipの解凍
    with zipfile.ZipFile(fname, 'r') as myzip:
        myzip.extractall()
        # 解凍後のファイルからデータ読み込み
        for myfile in myzip.infolist():
            # 解凍後ファイル名取得
            filename = myfile.filename
            # ファイルオープン時にencodingを指定してsjisの変換をする
            with open(filename, encoding='sjis') as file:
                text = file.read()

    # ファイル整形
    # ヘッダ部分の除去
    text = re.split('\-{5,}',text)[2]
    # フッタ部分の除去
    text = re.split('底本：',text)[0]
    # | の除去
    text = text.replace('|', '')
    # ルビの削除
    text = re.sub('《.+?》', '', text)
    # 入力注の削除
    text = re.sub('［＃.+?］', '',text)
    # 空行の削除
    text = re.sub('\n\n', '\n', text)
    text = re.sub('\r', '', text)

    # 頭の100文字の表示
    print(text[:100])
    # 見やすくするため、空行
    print("\n\n")
    # 後ろの100文字の表示
    print(text[-100:])

    with open('{}.txt'.format(title), 'w') as f:
        f.write(text)


def tokenizer(text):
    t = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    t.parse("")
    m = t.parseToNode(text)
    tokens = []
    while m:
        token_data = m.feature.split(",")
        token = [m.surface]
        for data in token_data:
            token.append(data)
        tokens.append(token)
        m = m.next
    tokens.pop(0)
    tokens.pop(-1)
    return tokens


def extract(text, mode):
    words = []

    for token in tokenizer(text):
        if mode == 0:
            if token[1]=='名詞' or token[1]=='動詞'or token[1]=='形容詞':
                words.append(token)
        elif mode == 1:
            if token[1]!='記号' and token[0]!='\n' and not ('〝' in token[0]) and not ('(' in token[0]) and not (')' in token[0]) and not ('〟' in token[0])and not ('。' in token[0]):
                words.append(token)
        elif mode == 2:
            if token[1]!='記号' and token[0]!='\n' and not ('〝' in token[0]) and not ('(' in token[0]) and not (')' in token[0]) and not ('〟' in token[0])and not ('。' in token[0]):
                words.append(token[0])
    return words


def text2tokens(filename):
    nov = open('{}.txt'.format(filename), "r")
    lines = nov.readlines()
    word_list = []

    for line in lines:
        word_list.extend(extract(line, 2))

    with open('{}_tokens.txt'.format(filename), 'w') as f:
        text = ','.join(word_list)
        f.write(text)


def load_tokens(filename):
    file = open(filename, "r")
    text = file.read()
    tokens = text.split(',')

    return tokens


def train_model(list, model_name):
    # sg:1=skip_gram その他はCBOW, min_count: 最小頻度, window: 単語間の最大距離
    # model = Word2Vec([list], sg=1, size=100, window=15, min_count=1)
    model = Word2Vec([list], sg=1, size=200, window=10, min_count=20, iter=5)
    model.save("./{}.model".format(model_name))


def retrain(list, model_name):
    model = Word2Vec.load("{}.model".format(model_name))
    model.build_vocab(list, update=True)
    model.train(list, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./{}_retrain.model".format(model_name))


def load_data():
    word_list = load_tokens('{}_tokens.txt'.format(title))
    model = Word2Vec.load("./{}.model".format(title))
    data = []
    print(model.wv.most_similar('カムパネルラ'))
    # print(c)

    for word in word_list:
        data.append(model.wv[word])

    return np.array(data)


def load_data_poincare():
    word_list = load_tokens('{}_tokens.txt'.format(title))
    model = PoincareModel.load("./{}_poincare_5.model".format(title))
    data = []
    # print(c)
    print(model.kv.most_similar('カムパネルラ'))

    for word in word_list:
        data.append(model.kv[word])

    return np.array(data)


def vec2word(vectors):
    word_list = []
    counts = []
    ranks = []
    model = Word2Vec.load("./{}.model".format(title))

    all_word = load_tokens('{}_tokens.txt'.format(title))
    c = collections.Counter(all_word)
    values, cs = zip(*c.most_common())

    for vec in vectors:
        word = model.wv.most_similar([vec], [], 1)[0][0]
        word_list.append(word)
        counts.append(c[word])
        ranks.append(values.index(word)+1)

    return ' '.join(word_list), np.array(counts), ranks


def vec2word_poincare(vectors):
    word_list = []
    counts = []
    ranks = []
    model = PoincareModel.load("./{}_poincare_5.model".format(title))

    all_word = load_tokens('{}_tokens.txt'.format(title))
    # c = collections.Counter(all_word)
    # values, cs = zip(*c.most_common())

    for vec in vectors:
        # word = model.kv.most_similar([vec], [], 1)[0][0]
        word = model.kv.most_similar([vec])[0][0]
        word_list.append(word)
        # counts.append(c[word])
        # ranks.append(values.index(word)+1)

    # return ' '.join(word_list), np.array(counts), ranks
    return ' '.join(word_list)


def preprocessing():
    download_novel()
    text2tokens(title)


if __name__ == '__main__':
    # preprocessing()
    # word_list = load_tokens('{}_tokens.txt'.format(title))
    # retrain(word_list, 'jawiki')
    model_name = 'jawiki'
    model = Word2Vec.load("./{}.model".format(model_name))
    print(model.wv.most_similar('カムパネルラ'))
    # train_model(word_list, title)
