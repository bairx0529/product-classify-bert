import torch
from transformers import AutoTokenizer

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataset, DatasetType
from runner.predict import predict

# 准备资源
# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型
model = ProductClassifier().to(device)
model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

# 数据集
dataset = get_dataset(DatasetType.TRAIN)
class_label = dataset.features['label']


def predict_title(text):
    return predict(text, model, tokenizer, device, class_label)