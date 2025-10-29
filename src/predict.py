import argparse
from pathlib import Path
import torch

import config
from model import build_model          # ✅ 使用工厂函数创建 RNN/LSTM/GRU
from utils import JiebaTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Interactive sentiment prediction.")
    p.add_argument("--arch", type=str, default="GRU", choices=["RNN", "LSTM", "GRU"],
                   help="选择循环网络架构 (RNN / LSTM / GRU)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="权重文件路径；不提供则默认使用 models/best_{ARCH}_model.pt")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="正向评价判定阈值（默认 0.5）")
    p.add_argument("--max_len", type=int, default=None,
                   help="可选：覆盖 config.SEQ_LEN 的最大长度")
    return p.parse_args()


@torch.no_grad()
def predict_batch(input_tensor: torch.Tensor, model: torch.nn.Module):
    model.eval()
    logits = model(input_tensor)
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits)      # [B]
    return probs.tolist()


def predict_one(model, tokenizer, text: str, device, max_len: int = None) -> float:
    # 编码并对齐长度
    seq_len = max_len if max_len is not None else getattr(config, "SEQ_LEN", None)
    if seq_len is None:
        raise ValueError("未找到序列长度，请在 --max_len 指定或在 config.SEQ_LEN 中定义。")

    index_list = tokenizer.encode(text, seq_len)
    input_tensor = torch.tensor([index_list], dtype=torch.long, device=device)
    prob = predict_batch(input_tensor, model)[0]
    return float(prob)


def run_predict():
    args = parse_args()

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | arch: {args.arch}")

    # 资源
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 构建模型（与训练一致）
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
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 交互式预测
    print("请输入评价内容，输入 q / quit / exit 退出")
    while True:
        try:
            user_input = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print("\n程序已退出")
            break

        if user_input.lower() in {'q', 'quit', 'exit'}:
            print("程序已退出")
            break
        if not user_input:
            print("请输入评价内容")
            continue

        prob_pos = predict_one(model, tokenizer, user_input, device, max_len=args.max_len)
        label = "正向评价" if prob_pos >= args.threshold else "负面评价"
        conf = prob_pos if prob_pos >= args.threshold else (1.0 - prob_pos)
        print(f"{label}（置信度：{conf:.3f}，正向概率：{prob_pos:.3f}）")


if __name__ == '__main__':
    run_predict()