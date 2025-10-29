import torch

import config
from model import GRU
from dataset import get_dataloader
from utils import JiebaTokenizer
from train import eval_one_eopch

def run_eval():
    #选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    #tokenizer
    tokenizer = JiebaTokenizer.from_vocab(vocab_file=config.PROCESSED_DATA_DIR/'vocab.txt')

    #dataloader
    train_dataloader, eval_dataloader = get_dataloader()

    #model
    model = GRU(vocab_size = tokenizer.vocab_size,
                embedding_dim = config.EMBEDDING_DIM,
                hidden_dim = config.HIDDEN_DIM,
                num_layers = config.NUM_LAYERS,
                bidirectional = config.BIDIRECTIONAL,
                pad_token = tokenizer.pad_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR/'best_GRU_model.pt'))

    loss_fn = torch.nn.BCEWithLogitsLoss()
    avg_loss, accuracy = eval_one_eopch(eval_dataloader, model, loss_fn, device)

    print("avg_loss: ", avg_loss)
    print("accuracy: ", accuracy)

if __name__ == '__main__':
    run_eval()