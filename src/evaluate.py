import argparse
from pathlib import Path
import torch

import config
from model import build_model
from dataset import get_dataloader
from utils import JiebaTokenizer
from train import eval_one_epoch

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model on the validation set.")
    p.add_argument("--arch", type=str, default="GRU", choices=["RNN", "LSTM", "GRU"],
                   help="选择循环网络架构 (RNN / LSTM / GRU)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="权重文件路径；若不提供，则默认使用 models/best_{ARCH}_model.pt")
    return p.parse_args()


def run_eval():
    args = parse_args()

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | arch: {args.arch}")

    # tokenizer
    tokenizer = JiebaTokenizer.from_vocab(vocab_file=config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 只需要验证集
    _, eval_dataloader = get_dataloader()

    # 构建模型
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

    # 权重路径
    ckpt_path = Path(args.ckpt) if args.ckpt else (Path(config.MODELS_DIR) / f"best_{args.arch.upper()}_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到权重文件：{ckpt_path}")

    # 加载权重（兼容不同 PyTorch 版本）
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)  # 新版推荐
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)  # 旧版回退

    model.load_state_dict(state)
    model.eval()

    # 评估
    loss_fn = torch.nn.BCEWithLogitsLoss()
    avg_loss, accuracy = eval_one_epoch(eval_dataloader, model, loss_fn, device)

    print(f"avg_loss: {avg_loss:.6f}")
    print(f"accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    run_eval()