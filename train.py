import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import get_loaders
from model import ConvLSTMModel
import os
import torch.optim.lr_scheduler as lr_scheduler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.internal.api_implementation')

# 配置
csv_file = 'your label path'
image_dir = 'your dataset path'
batch_size = 128
num_epochs = 100
save_path = './model_weights.pth'  # 保存模型权重的路径
checkpoint_path = './checkpoint.pth'  # 保存 checkpoint 的路径

import csv

def train_and_validate(start_epoch, num_epochs, train_loader, val_loader, model, criterion, optimizer, scheduler,
                       device):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')  # 初始的最佳验证损失

    # 从中间继续训练时，使用之前保存的最佳验证损失
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        best_val_loss = checkpoint['best_val_loss']

    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            model.train()  # 设置为训练模式
            running_train_loss = 0.0
            total_batches = len(train_loader)

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据加载到 CUDA

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                # 打印每个 batch 的进度
                if (batch_idx + 1) % 10 == 0:  # 每 10 个 batch 打印一次
                    print(f"  Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item():.4f}")

            avg_train_loss = running_train_loss / total_batches
            with open('train_loss.csv',mode='a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow([avg_train_loss])
            train_losses.append(avg_train_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

            # 验证过程
            model.eval()  # 设置为评估模式
            running_val_loss = 0.0
            total_val_batches = len(val_loader)
            with torch.no_grad():
                for val_batch_idx, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / total_val_batches
            val_losses.append(avg_val_loss)
            with open('val_loss.csv',mode='a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow([avg_val_loss])
            print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

            # 保存最优的模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                # 保存模型和优化器的状态
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                print(f"Model weights and checkpoint saved to {save_path}")

            scheduler.step()
            print(f"Learning rate: {scheduler.get_last_lr()}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving loss curves...")

    finally:
        # 绘制损失曲线
        plt.figure()
        plt.plot(range(start_epoch, start_epoch + len(train_losses)), train_losses, label='Train Loss')
        plt.plot(range(start_epoch, start_epoch + len(val_losses)), val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_validation_loss.png')  # 保存图像
        plt.show()


if __name__ == '__main__':
    # 获取数据加载器
    train_loader, val_loader = get_loaders(csv_file, image_dir, batch_size)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数、优化器
    model = ConvLSTMModel().to(device)  # 将模型加载到 CUDA
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # 将初始学习率改为0.005
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 使用学习率调度器

    # 检查是否有已保存的 checkpoint
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从保存的epoch之后继续训练
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}")

    # 执行训练
    train_and_validate(start_epoch, num_epochs, train_loader, val_loader, model, criterion, optimizer, scheduler,
                       device)
