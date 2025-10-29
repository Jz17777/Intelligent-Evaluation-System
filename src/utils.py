import jieba
from tqdm import tqdm

import config

class JiebaTokenizer:
    unk_token = '<UNK>'
    pad_token = '<PAD>'
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)

        self.word2index = {word: index for index,word in enumerate(vocab_list)}
        self.index2word = {index: word for index,word in enumerate(vocab_list)}

        self.unk_index = self.word2index[self.unk_token]
        self.pad_index = self.word2index[self.pad_token]

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """
        创建词表
        :param sentences: 传入的句子[“我喜欢踢足球”,"天天都在吃饭"]
        :param vocab_file: 保存词表路径
        :return:
        """
        vocab = set()
        for sentence in tqdm(sentences,desc= 'Creating Vocab_list'):
            for word in jieba.lcut(sentence):
                if word.strip() != '':
                    vocab.add(word)
        vocab = [cls.pad_token]+[cls.unk_token]+list(vocab)
        print('vocab_size:', len(vocab))

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write(word + '\n')
            print('Vocab_list saved', len(vocab))

    @classmethod
    def from_vocab(cls, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f if line.strip()]

        print("词表加载完成")
        return cls(vocab_list)

    def encode(self, sentence, seq_len):
        words = jieba.lcut(sentence)

        if len(words) > seq_len:
            words = words[:seq_len]
        if len(words) < seq_len:
            words += [self.pad_token] * (seq_len - len(words))

        return [self.word2index.get(word, self.unk_index) for word in words]