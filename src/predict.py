import torch

import config
from model import GRU
from utils import JiebaTokenizer

def run_predict():
    #加载资源
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR/'vocab.txt')

    model = GRU(vocab_size=tokenizer.vocab_size,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_LAYERS,
                bidirectional=config.BIDIRECTIONAL,
                pad_token=tokenizer.pad_index).to(device)

    model.load_state_dict(torch.load(config.MODELS_DIR/'best_GRU_model.pt'))

    print("请输入评价，输入q或者quit退出")
    while True:
        user_input = input('>')
        if user_input in ['q', 'quit']:
            print('程序已退出')
            break
        if user_input.strip() == '':
            print("请输入评价")
            continue

        result = predict(model, tokenizer, user_input, device)
        if result > 0.5:
            print("正向评价(置信度：{})".format(result))
        else:
            print("负面评价(置信度：{})".format(1 - result))

def predict_batch(input_tensor, model):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        return torch.sigmoid(output).tolist()



def predict(model, tokenizer, user_input, device):
    #处理输入
    index_list = tokenizer.encode(user_input, config.SEQ_LEN)
    input_tensor = torch.tensor([index_list]).to(device)

    batch_result = predict_batch(input_tensor, model)
    return batch_result[0]

if __name__ == '__main__':
    run_predict()