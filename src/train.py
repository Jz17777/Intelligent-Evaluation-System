import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from utils import JiebaTokenizer
import config
from model import GRU
from dataset import get_dataloader


def train_one_eopch(dataloader, model, loss_function, optimizer, device):
    model.train()
    epoch_total_loss = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()

    return epoch_total_loss / len(dataloader)

def eval_one_eopch(dataloader, model, loss_function, device):
    model.eval()
    epoch_total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.squeeze(-1)
        loss = loss_function(outputs, labels)
        epoch_total_loss += loss.item()

        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds == labels.bool()).sum().item()
        total += labels.numel()

    avg_loss = epoch_total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def train():
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

    #损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()

    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    #tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR/time.strftime("%Y%m%d-%H%M%S"))

    print("开始训练")
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS+1):
        print("===================EPOCH %d=====================" % epoch)

        avg_loss = train_one_eopch(train_dataloader, model, loss_fn, optimizer, device)

        print("loss: ", avg_loss)

        eval_avg_loss, accuracy = eval_one_eopch(eval_dataloader, model, loss_fn, device)

        print("eval_accuracy: ", accuracy)

        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('accuracy', accuracy, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR/'best_GRU_model.pt')
            print("模型保存成功")
        else:
            print("模型无需保存")

    writer.close()

if __name__ == '__main__':
    train()