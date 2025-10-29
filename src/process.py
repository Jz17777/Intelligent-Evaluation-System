import pandas as pd
import config
import sklearn
from utils import JiebaTokenizer

def process():
    df = pd.read_csv(config.RAW_DATA_DIR/'online_shopping_10_cats.csv', usecols=["review","label"])

    df = df.dropna()

    train_dataset, test_dataset = sklearn.model_selection.train_test_split(df, test_size=0.2, stratify=df["label"])

    JiebaTokenizer.build_vocab(df["review"], config.PROCESSED_DATA_DIR/'vocab.txt')

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR/'vocab.txt')

    #构建数据集
    train_dataset["review"] = train_dataset["review"].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    train_dataset.to_json(config.PROCESSED_DATA_DIR/'train_dataset.json', orient='records',lines=True)

    test_dataset["review"] = test_dataset["review"].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    test_dataset.to_json(config.PROCESSED_DATA_DIR/"test_dataset.json", orient='records',lines=True)


if __name__ == '__main__':
    process()