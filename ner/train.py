import argparse
import os
from typing import *

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from model import NLLModel
from utils import set_seed, collate_fn
from prepro import read_conll, LABEL_TO_ID
from torch.cuda.amp import autocast, GradScaler
import seqeval.metrics
# import wandb

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def train(args: Any, model: nn.Module, train_features: List[Dict[str, Any]],
          benchmarks: List[Tuple[str, List[Dict[str, Any]]]]):
    """
    训练模型。

    Args:
        args (Any): 提供模型配置信息的对象，该对象包含了模型需要的各种配置。
        model (nn.Module): 要训练的模型。
        train_features (List[Dict[str, Any]]): 用于训练的特征列表。
        benchmarks (List[Tuple[str, List[Dict[str, Any]]]]): 用于评估的数据集列表。

    """

    # 创建DataLoader，用于批量处理训练数据
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=True)

    # 计算总训练步数和warmup步数
    total_steps = int(
        len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    # 初始化优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 初始化梯度缩放器，用于动态缩放梯度以防止在反向传播过程中出现太大或太小的梯度
    scaler = GradScaler()

    num_steps = 0  # 初始化步数计数器
    # 进行多个epoch的训练
    for epoch in range(int(args.num_train_epochs)):
        # 遍历每一个batch
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()  # 将模型设置为训练模式
            # 根据步数更新alpha_t的值
            if num_steps < args.alpha_warmup_ratio * total_steps:
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha
            # 将每个batch的数据移动到设备上
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with autocast():  # 使用自动混合精度进行训练，可以节省显存并加速训练
                outputs = model(**batch)  # 前向传播得到模型输出
            loss = outputs[0] / args.gradient_accumulation_steps  # 计算损失
            scaler.scale(loss).backward()  # 反向传播计算梯度
            # 每隔gradient_accumulation_steps步进行一次参数更新
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1  # 更新步数计数器
                scaler.unscale_(optimizer)  # 取消梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)  # 对梯度进行裁剪，防止梯度爆炸
                scaler.step(optimizer)  # 更新模型参数
                scaler.update()  # 更新梯度缩放器的状态
                scheduler.step()  # 更新学习率
                model.zero_grad()  # 清空模型的梯度
                # 记录损失（如果你使用wandb）
                # wandb.log({'loss': loss.item()}, step=num_steps)
            # 如果是最后一步，对所有的评估数据集进行评估
            if step == len(train_dataloader) - 1:
                for tag, features in benchmarks:
                    results = evaluate(args, model, features, tag=tag)
                    # 记录评估结果（如果你使用wandb）
                    # wandb.log(results, step=num_steps)


def evaluate(args: Any, model: nn.Module, features: List[Dict[str, Any]], tag: str = "dev") -> Dict[
    str, float]:
    """
    评估模型的性能。

    Args:
        args (Any): 提供模型配置信息的对象，该对象包含了模型需要的各种配置。
        model (nn.Module): 要评估的模型。
        features (List[Dict[str, Any]]): 用于评估的特征列表。
        tag (str, optional): 用于标记评估的数据集。默认为"dev"。

    Returns:
        Dict[str, float]: 评估结果，包含了F1分数。
    """

    # 创建DataLoader，用于批量处理特征数据
    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, drop_last=False)
    preds, keys = [], []  # 初始化预测列表和标签列表
    # 遍历每一个batch
    for batch in dataloader:
        model.eval()  # 将模型设置为评估模式
        # 将每个batch的数据移动到设备上
        batch = {key: value.to(args.device) for key, value in batch.items()}
        keys += batch['labels'].cpu().numpy().flatten().tolist()  # 将标签添加到keys列表，注意这里把所有句子的标注都合并在了仪器（拉平）
        batch['labels'] = None  # 清空batch中的标签
        with torch.no_grad():  # 关闭梯度计算
            logits = model(**batch)[0]  # 前向传播得到logits
            # 计算每个logit的argmax得到预测结果，并添加到preds列表
            preds += np.argmax(logits.cpu().numpy(), axis=-1).tolist()

    # 去掉keys和preds中标签为-1的项
    preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != -1]))
    # 将预测结果和标签从ID转为实际的标签
    preds = [ID_TO_LABEL[pred] for pred in preds]
    keys = [ID_TO_LABEL[key] for key in keys]
    model.zero_grad()  # 清空模型的梯度
    f1 = seqeval.metrics.f1_score([keys], [preds])  # 计算F1分数
    output = {
        tag + "_f1": f1,  # 将F1分数添加到输出结果中
    }
    print(output)  # 打印输出结果
    return output  # 返回输出结果


def main():
    """
    主程序入口，用于解析命令行参数，读取数据，初始化模型和tokenizer，然后开始训练模型。
    """

    # 初始化一个命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加各种命令行参数
    parser.add_argument("--data_dir", default="./data", type=str)  # 数据目录
    parser.add_argument("--model_name_or_path", default="../cache/bert-base-cased", type=str)  # 预训练模型名称或路径
    parser.add_argument("--max_seq_length", default=512, type=int)  # 输入序列的最大长度

    # 训练配置参数
    parser.add_argument("--batch_size", default=64, type=int)  # 批处理大小
    parser.add_argument("--learning_rate", default=1e-5, type=float)  # 学习率
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)  # 梯度积累步数
    parser.add_argument("--eps", default=1e-6, type=float)  # 优化器的epsilon参数
    parser.add_argument("--max_grad_norm", default=1.0, type=float)  # 梯度裁剪的阈值
    parser.add_argument("--warmup_ratio", default=0.1, type=float)  # 学习率预热比例
    parser.add_argument("--dropout_prob", default=0.1, type=float)  # Dropout层的丢弃比例
    parser.add_argument("--num_train_epochs", default=50.0, type=float)  # 训练的轮数
    parser.add_argument("--seed", type=int, default=42)  # 随机种子，保证实验可复现
    parser.add_argument("--num_class", type=int, default=9)  # 分类的类别数

    # Weights & Biases项目名称
    parser.add_argument("--project_name", type=str, default="NLL-IE-NER")
    parser.add_argument("--n_model", type=int, default=2)  # 模型数
    parser.add_argument("--alpha", type=float, default=50.0)  # NLL损失的权重
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)  # NLL损失权重的预热比例

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化Weights & Biases，如果你不使用W&B，你可以注释掉这行代码
    # wandb.init(project=args.project_name)

    # 确定使用的设备是CPU还是GPU，并获取GPU数量
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # 设置随机种子，以确保结果的可复现性
    set_seed(args)

    # 加载预训练模型的tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = NLLModel(args)

    # 读取训练集、验证集和测试集
    train_file = os.path.join(args.data_dir, "train.txt")
    dev_file = os.path.join(args.data_dir, "dev.txt")
    test_file = os.path.join(args.data_dir, "test.txt")
    testre_file = os.path.join(args.data_dir, "conllpp_test.txt")

    # 读取并处理CoNLL格式的数据
    train_features = read_conll(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read_conll(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read_conll(test_file, tokenizer, max_seq_length=args.max_seq_length)
    testre_features = read_conll(testre_file, tokenizer, max_seq_length=args.max_seq_length)

    # 定义评估的数据集
    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
        ("test_rev", testre_features)
    )

    # 开始训练模型
    train(args, model, train_features, benchmarks)



if __name__ == "__main__":
    main()
