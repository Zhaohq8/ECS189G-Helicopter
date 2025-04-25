'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 150  # 增加训练轮数
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3  # 调整学习率
    # defines momentum hyperparameter
    momentum = 0.9

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # 增加网络宽度和深度
        self.fc_layer_1 = nn.Linear(784, 256)  # 增加第一层神经元数
        self.bn1 = nn.BatchNorm1d(256)         # 批归一化
        self.activation_func_1 = nn.LeakyReLU(0.1)  # 使用LeakyReLU代替ReLU

        self.fc_layer_2 = nn.Linear(256, 128)  # 添加中间层
        self.bn2 = nn.BatchNorm1d(128)         # 批归一化
        self.activation_func_2 = nn.LeakyReLU(0.1)

        self.fc_layer_3 = nn.Linear(128, 10)   # 输出层

        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.3)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # 第一个隐藏层
        h1 = self.fc_layer_1(x)
        if len(h1.shape) > 1 and h1.shape[0] > 1:  # 确保批量大小>1才使用BatchNorm
            h1 = self.bn1(h1)
        h1 = self.activation_func_1(h1)
        h1 = self.dropout(h1)

        # 第二个隐藏层
        h2 = self.fc_layer_2(h1)
        if len(h2.shape) > 1 and h2.shape[0] > 1:
            h2 = self.bn2(h2)
        h2 = self.activation_func_2(h2)
        h2 = self.dropout(h2)

        # 输出层 - 不在这里应用softmax，而是在损失函数中处理
        logits = self.fc_layer_3(h2)
        return logits

    def fit(self, X, y):
        # 创建结果目录
        os.makedirs('../../result/stage_2_result', exist_ok=True)

        # List to store loss values during training
        losses = []
        train_accuracies = []

        # 优化器使用AdamW
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # L2正则化
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # 使用CrossEntropyLoss，内部已包含LogSoftmax和NLLLoss
        loss_function = nn.CrossEntropyLoss()

        # 准确率评估器
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # 将数据转换为张量并缓存
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        # 早停设置
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.max_epoch):
            # 训练模式
            self.train(mode=True)  # 使用mode=True参数来避免递归

            # 前向传播
            logits = self.forward(X_tensor)

            # 计算损失
            train_loss = loss_function(logits, y_tensor)
            losses.append(train_loss.item())

            # 反向传播和优化
            optimizer.zero_grad()
            train_loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()

            # 计算准确率
            with torch.no_grad():
                # 评估模式
                self.eval()
                pred_probs = torch.nn.functional.softmax(logits, dim=1)
                pred_labels = pred_probs.max(1)[1]
                accuracy = (pred_labels == y_tensor).float().mean().item()
                train_accuracies.append(accuracy)

            # 学习率调度
            scheduler.step(train_loss)

            # 检查早停
            if train_loss.item() < best_loss:
                best_loss = train_loss.item()
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = self.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    # 加载最佳模型状态
                    self.load_state_dict(best_model_state)
                    break

            if epoch % 2 == 0:
                print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {train_loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 加载最佳模型
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        # 保存训练损失和准确率图表
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy over Epochs')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('../../result/stage_2_result/training_curves.png')
        plt.close()

    def test(self, X):
        # 评估模式
        self.eval()

        # 不计算梯度
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X))
            logits = self.forward(X_tensor)
            pred_probs = torch.nn.functional.softmax(logits, dim=1)
            pred_labels = pred_probs.max(1)[1]

        return pred_labels

    def run(self):
        print('method running...')

        # 数据归一化
        print('--preprocessing data...')
        self.data['train']['X'] = np.array(self.data['train']['X'], dtype=np.float32) / 255.0
        self.data['test']['X'] = np.array(self.data['test']['X'], dtype=np.float32) / 255.0

        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])  # 使用fit替代train

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y.tolist(), 'true_y': self.data['test']['y']}
