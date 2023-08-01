from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


def kl_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    计算KL散度，用于衡量两个概率分布之间的差异。

    Args:
        p (torch.Tensor): 第一个概率分布。
        q (torch.Tensor): 第二个概率分布。

    Returns:
        torch.Tensor: 计算出的KL散度。
    """
    # 对每个元素加上一个非常小的值防止log(0)，然后计算KL散度
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)


class NERModel(nn.Module):
    """
    NERModel类，继承自PyTorch的nn.Module类，用于命名实体识别任务。

    Attributes:
        model (nn.Module): 使用 AutoModel 自动加载的预训练模型。
        dropout (nn.Module): 用于dropout操作。
        classifier (nn.Module): 用于分类的全连接层。
        loss_fnt (nn.Module): 用于计算损失的函数。
    """

    def __init__(self, args: Any):
        """
        初始化NERModel类。

        Args:
            args (Any): 提供模型配置信息的对象，该对象包含了模型需要的各种配置。
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.args = args  # 保存配置信息
        # 从预训练模型库中加载配置，并设置标签类别数量
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class)
        # 从预训练模型库中加载模型
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(args.dropout_prob)  # 创建Dropout层
        self.classifier = nn.Linear(config.hidden_size, args.num_class)  # 创建分类层
        # 创建交叉熵损失函数，忽略标签为-1的输入
        self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        """
        定义模型的前向传播。

        Args:
            input_ids (torch.Tensor): 输入的token ids。
            attention_mask (torch.Tensor): 用于屏蔽padding的mask。
            labels (Optional[torch.Tensor], optional): 真实的序列标注。默认为None。

        Returns:
            tuple: 包含损失和logits的元组，若labels不空，则返回(loss, logits)，否则返回logits
        """
        h, *_ = self.model(input_ids, attention_mask, return_dict=False)  # 获取模型的输出
        h = self.dropout(h)  # 进行dropout操作
        c = self.args.num_class  # 获取类别数量
        logits = self.classifier(h)  # 通过分类器得到logits
        logits = logits.view(-1, c)  # 调整logits的形状
        outputs = (logits,)  # 将logits放入输出元组
        if labels is not None:  # 如果有标签，那么计算损失
            labels = labels.view(-1)  # 调整标签的形状
            loss = self.loss_fnt(logits, labels)  # 计算损失
            outputs = (loss,) + outputs  # 将损失添加到输出元组的开始
        return outputs  # 返回输出元组


class NLLModel(nn.Module):
    """
    NLLModel类，继承自PyTorch的nn.Module类，使用多个NERModel并计算其平均输出。

    Attributes:
        models (nn.ModuleList): NERModel的列表。
        device (list): 设备列表，用于存放模型。
        loss_fnt (nn.Module): 用于计算损失的函数。
    """

    def __init__(self, args: Any):
        """
        初始化NLLModel类。

        Args:
            args (Any): 提供模型配置信息的对象，该对象包含了模型需要的各种配置。
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.args = args  # 保存配置信息
        self.models = nn.ModuleList()  # 创建ModuleList用于存放多个NERModel
        # 根据gpu数量分配模型，如果没有gpu则为空列表
        self.device = [i % args.n_gpu for i in range(args.n_model)] if args.n_gpu != 0 else []
        self.loss_fnt = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
        for i in range(args.n_model):  # 根据配置中的模型数量，创建多个NERModel
            model = NERModel(args)  # 创建NERModel
            model.to(self.device[i])  # 将模型移动到指定设备
            self.models.append(model)  # 将模型添加到models列表

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        """
        定义模型的前向传播。

        Args:
            input_ids (torch.Tensor): 输入的token ids。
            attention_mask (torch.Tensor): 用于屏蔽padding的mask。
            labels (Optional[torch.Tensor], optional): 真实标签。默认为None。

        Returns:
            tuple: (多个模型的汇总loss, logits)，logits.shape=[batch_size*sequence_length, num_labels]
        """
        # 如果没有标签，则只返回第一个模型的输出
        if labels is None:
            return self.models[0](input_ids=input_ids, attention_mask=attention_mask)
        else:
            num_models = len(self.models)  # 获取模型数量
            outputs = []
            # 对每个模型进行前向传播
            for i in range(num_models):
                output = self.models[i](
                    input_ids=input_ids.to(self.device[i]),
                    attention_mask=attention_mask.to(self.device[i]),
                    labels=labels.to(self.device[i]) if labels is not None else None,
                )
                # 将输出移到主设备
                output = tuple([o.to(0) for o in output])
                outputs.append(output)  # 将输出添加到outputs列表
            model_output = outputs[0]  # 获取第一个模型的输出
            # 计算所有模型输出的平均损失
            loss = sum([output[0] for output in outputs]) / num_models
            logits = [output[1] for output in outputs]  # 获取所有模型的logits
            probs = [F.softmax(logit, dim=-1) for logit in logits]  # 对logits进行softmax得到概率
            avg_prob = torch.stack(probs, dim=0).mean(0)  # 计算所有模型概率的平均值
            # 创建mask用于屏蔽标签为-1的输入
            mask = (labels.view(-1) != -1).to(logits[0])  # todo(这里可能要改，因为我们的特殊字符编码不是-1，而是0)
            # 计算平均概率和每个模型概率的KL散度，然后乘以mask并取平均
            reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs]) / num_models
            # 计算reg_loss的总和，除以mask的数量，如果mask的数量为0，则加上一个很小的值防止除0错误
            reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
            # 将损失和正则化损失相加，正则化损失的权重由配置中的alpha_t决定
            loss = loss + self.args.alpha_t * reg_loss
            model_output = (loss,) + model_output[1:]  # 更新模型输出的损失
        return model_output  # 返回模型输出
