import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader, DatasetType


class EarlyStopping:
    def __init__(self, patience=2):
        self.best_loss = float('inf')
        self.counter = 0  # 连续超过best_loss的次数
        self.patience = patience

    def should_stop(self, avg_loss, model, path):
        if avg_loss < self.best_loss:
            self.counter = 0
            self.best_loss = avg_loss
            torch.save(model.state_dict(), path)
            print("保存最优模型")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False


def run_one_epoch(model, dataloader, loss_function, device, scaler=None, optimizer=None, is_train=True):
    total_loss = 0

    if is_train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_train):  # 在训练时计算梯度/验证时不计算梯度
        for batch in tqdm(dataloader, desc=('训练' if is_train else '验证')):
            input_ids = batch['input_ids'].to(device)  # [batch_size, seq_len]
            attention_mask = batch['attention_mask'].to(device)  # [batch_size, seq_len]
            label = batch['label'].to(device)  # [batch_size]

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                # 前向传播
                outputs = model(input_ids, attention_mask)
                # outputs.shape = [batch_size, num_classes]

                # 计算损失
                loss = loss_function(outputs, label)

            # 反向传播
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

            total_loss += loss.item()
    return total_loss / len(dataloader)

def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    train_dataloader = get_dataloader(DatasetType.TEST)  # 仅为测试
    valid_dataloader = get_dataloader(DatasetType.VALID)

    # 模型
    model = ProductClassifier(freeze_bert=True).to(device)

    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 日志
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 早停策略
    early_stopping = EarlyStopping()

    # 梯度缩放器
    scaler = torch.amp.GradScaler(device.type)

    start_epoch = 1

    # 检查是否存在checkpoint文件
    # checkpoint文件路径
    checkpoint_path = config.MODELS_DIR / 'checkpoint.pt'
    if checkpoint_path.exists():
        print("存在checkpoint文件，断点续跑")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        early_stopping.best_loss = checkpoint['best_loss']
        early_stopping.counter = checkpoint['counter']
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("不存在checkpoint文件，从头开始训练")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"========== Epoch {epoch} ==========")
        # 训练一轮
        train_avg_loss = run_one_epoch(model, train_dataloader, loss_function, device, scaler, optimizer, is_train=True)
        # 验证一轮
        valid_avg_loss = run_one_epoch(model, valid_dataloader, loss_function, device, is_train=False)

        print(f"训练集Loss: {train_avg_loss:.4f}")
        print(f"验证集Loss: {valid_avg_loss:.4f}")

        writer.add_scalar('Loss/Train', train_avg_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_avg_loss, epoch)

        if early_stopping.should_stop(valid_avg_loss, model, config.MODELS_DIR / 'best.pt'):
            print("早停策略触发，训练提前结束")
            break

        # 保存训练状态（checkpoint）
        checkpoint = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(),
                      "best_loss": early_stopping.best_loss,
                      "counter": early_stopping.counter,
                      "epoch": epoch}
        torch.save(checkpoint, config.MODELS_DIR / 'checkpoint.pt')

    writer.close()


