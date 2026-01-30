import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader, DatasetType
from runner.predict import predict_batch


def evaluate_model(model, dataloader, device):
    all_labels = []
    all_predictions = []

    for batch in tqdm(dataloader, desc='评估'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].tolist()
        predict_result = predict_batch(input_ids, attention_mask, model)

        all_labels.extend(label)
        all_predictions.extend(predict_result)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1}


def run_evaluate():
    # 准备资源
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型
    model = ProductClassifier().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))

    # 数据集
    dataloader = get_dataloader(DatasetType.TEST)

    result = evaluate_model(model, dataloader, device)
    print("========== 评估结果 ==========")
    print(f'accuracy:{result["accuracy"]:.4f}')
    print(f'precision:{result["precision"]:.4f}')
    print(f'recall:{result["recall"]:.4f}')
    print(f'f1:{result["f1"]:.4f}')
    print("=============================")
