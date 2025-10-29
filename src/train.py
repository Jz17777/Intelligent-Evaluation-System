import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path
import argparse

from utils import JiebaTokenizer
import config
from model import build_model
from dataset import get_dataloader


def train_one_epoch(dataloader, model, loss_function, optimizer, device):
    model.train()
    epoch_total_loss = 0.0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        if outputs.dim() == 2 and outputs.size(-1) == 1:
            outputs = outputs.squeeze(-1)

        labels = labels.view_as(outputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_total_loss += loss.item()

    return epoch_total_loss / len(dataloader)


@torch.no_grad()
def eval_one_epoch(dataloader, model, loss_function, device):
    model.eval()
    epoch_total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        outputs = model(inputs)
        if outputs.dim() == 2 and outputs.size(-1) == 1:
            outputs = outputs.squeeze(-1)

        labels = labels.view_as(outputs)

        loss = loss_function(outputs, labels)
        epoch_total_loss += loss.item()

        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_loss = epoch_total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="GRU",
        choices=["RNN", "LSTM", "GRU"],
        help="选择循环网络架构 (RNN / LSTM / GRU)"
    )
    return parser.parse_args()


def train():
    args = parse_args()

    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | arch: {args.arch}")

    # 目录确保存在
    Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # tokenizer
    tokenizer = JiebaTokenizer.from_vocab(vocab_file=config.PROCESSED_DATA_DIR / 'vocab.txt')

    # dataloader
    train_dataloader, eval_dataloader = get_dataloader()

    # 构建模型（自动调用 model.py 中的模块）
    model = build_model(
        arch=args.arch,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        bidirectional=config.BIDIRECTIONAL,
        pad_token=tokenizer.pad_index,
        dropout=getattr(config, "DROPOUT", 0.1),
    ).to(device)

    # 损失函数 & 优化器
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # TensorBoard
    log_dir = Path(config.LOGS_DIR) / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir))

    print("开始训练")
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, config.EPOCHS + 1):
        print(f"=================== EPOCH {epoch} =====================")

        train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_acc = eval_one_epoch(eval_dataloader, model, loss_fn, device)

        print(f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | val_acc: {val_acc:.4f}")

        # TensorBoard 分组记录
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('metrics/val_accuracy', val_acc, epoch)

        # 按验证集 loss 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_path = Path(config.MODELS_DIR) / f"best_{args.arch.upper()}_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"模型保存成功: {save_path}")
        else:
            print("模型无需保存")

    print(f"训练结束。best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f}")
    writer.close()


if __name__ == '__main__':
    train()
