from models.base_model import BaseModel

import torch
from models.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
from torch.optim import Adam
from sample import sample, truncate

import matplotlib.pyplot as plt

import os
import torch
import json

from transformers import PreTrainedTokenizerFast
import tranception
from tranception import config, model_pytorch

# 生成模型配置
# 设置超参数(生成模型)
sequence_batch_size = 16  # 每次采样数目
num_epochs = 2000  # 训练周期数
max_length = 512  # 最大序列长度
top_p = 0.9  # top-p采样的参数0.9
temp = 1.5  # 温度采样参数0.8
context = "1"  # 例如："MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGH"
model_checkpoint_path = './checkpoints/progen2-small'  # 模型checkpoint的路径
tokenizer_file = 'tokenizer.json'  # Tokenizer文件的路径
learning_rate = 0.00001

# 设置设备，例如 'cpu' 或 'cuda:0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化模型和tokenizer(生成模型)
progen = ProGenForCausalLM.from_pretrained(model_checkpoint_path).to(device)
tokenizer = Tokenizer.from_file(tokenizer_file)

# 定义序列生成的参数
# context = "1"  # 例如："MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGH"
# max_length = 256  # 可根据需要调整
# num_return_sequences = 2  # 你想生成的序列数量
# top_p = 0.95  # 累积概率阈值（"nucleus sampling"）
# temp = 0.7  # 采样温度
pad_token_id = tokenizer.encode('<|pad|>').ids[0]  # pad token id

class SequenceModel(BaseModel):
    """
    Model for handling sequence data.
    """
    def train(self, data):
        # Implement training for sequence data
        pass

    def predict(self, data):
        # Implement prediction for sequence data
        pass

    # Implement save and load methods if necessary
