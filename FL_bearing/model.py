"""
轴承故障诊断模型集合
包含多种适用于振动信号的模型:
1. BearingCNN - 基础一维CNN
2. BearingResNet - 残差网络版本
3. BearingLSTM - LSTM时序模型
4. BearingCNN_LSTM - CNN+LSTM混合模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 基础 CNN 模型
# ============================================================================
class BearingCNN(nn.Module):
    """
    用于轴承故障诊断的一维CNN模型
    输入: (batch_size, num_channels, signal_length) - 振动信号
    输出: (batch_size, num_classes) - 故障类别概率
    """
    def __init__(self, num_classes=4, signal_length=1024, num_channels=1):
        super(BearingCNN, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=64, stride=8, padding=28)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第三个卷积块
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算全连接层的输入维度
        with torch.no_grad():
            dummy = torch.randn(1, num_channels, signal_length)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            self.fc_input_dim = x.flatten(1).shape[1]

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 卷积层
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # 展平
        x = x.flatten(1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_features(self, x):
        """获取特征表示（用于可视化）"""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x


# ============================================================================
# ResNet 残差块
# ============================================================================
class ResidualBlock(nn.Module):
    """一维残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class BearingResNet(nn.Module):
    """
    用于轴承故障诊断的一维ResNet模型
    """
    def __init__(self, num_classes=4, signal_length=1024, num_channels=1):
        super(BearingResNet, self).__init__()

        self.in_channels = 32

        # 初始卷积
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=64, stride=8,
                               padding=28, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 残差层
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        # 计算全连接层维度
        with torch.no_grad():
            dummy = torch.randn(1, num_channels, signal_length)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            x = self.avgpool(x)
            self.fc_input_dim = x.flatten(1).shape[1]

        # 全连接层
        self.fc = nn.Linear(self.fc_input_dim, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


# ============================================================================
# LSTM 模型
# ============================================================================
class BearingLSTM(nn.Module):
    """
    用于轴承故障诊断的LSTM模型
    """
    def __init__(self, num_classes=4, signal_length=1024, num_channels=1,
                 hidden_size=128, num_layers=2):
        super(BearingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_channels = num_channels

        # LSTM层
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, dropout=0.3)

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # 输入 x shape: (batch_size, num_channels, signal_length)
        # 需要转换为: (batch_size, signal_length, num_channels)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]

        # 全连接层
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def get_features(self, x):
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = F.relu(self.fc1(out))
        return out


# ============================================================================
# CNN-LSTM 混合模型
# ============================================================================
class BearingCNN_LSTM(nn.Module):
    """
    CNN-LSTM混合模型: CNN提取特征，LSTM建模时序
    """
    def __init__(self, num_classes=4, signal_length=1024, num_channels=1,
                 hidden_size=64, num_layers=1):
        super(BearingCNN_LSTM, self).__init__()

        # CNN特征提取部分
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=64, stride=8, padding=28)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算CNN输出的时间步数
        with torch.no_grad():
            dummy = torch.randn(1, num_channels, signal_length)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            self.cnn_output_channels = x.shape[1]
            self.cnn_output_length = x.shape[2]

        # LSTM部分
        self.lstm = nn.LSTM(input_size=self.cnn_output_channels,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # CNN特征提取
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # 转换为LSTM输入格式: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步
        out = lstm_out[:, -1, :]

        # 全连接
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def get_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = F.relu(self.fc1(out))
        return out


# ============================================================================
# 模型工厂函数
# ============================================================================
MODEL_REGISTRY = {
    'cnn': BearingCNN,
    'resnet': BearingResNet,
    'lstm': BearingLSTM,
    'cnn_lstm': BearingCNN_LSTM,
}


def get_model(model_name='cnn', num_classes=4, signal_length=1024, num_channels=1, **kwargs):
    """
    获取模型实例的工厂函数

    Args:
        model_name: 模型名称 ('cnn', 'resnet', 'lstm', 'cnn_lstm')
        num_classes: 类别数
        signal_length: 信号长度
        num_channels: 输入通道数
        **kwargs: 其他模型特定参数

    Returns:
        模型实例
    """
    model_class = MODEL_REGISTRY.get(model_name.lower())
    if model_class is None:
        raise ValueError(f"未知的模型类型: {model_name}, 可用的模型: {list(MODEL_REGISTRY.keys())}")

    return model_class(num_classes=num_classes, signal_length=signal_length,
                      num_channels=num_channels, **kwargs)


# ============================================================================
# 联邦学习聚合算法
# ============================================================================
def aggregate_models(model_states, weights=None):
    """
    FedAvg 模型聚合

    Args:
        model_states: 各client的模型参数字典列表
        weights: 各client的权重（如样本数量），None则平均

    Returns:
        聚合后的模型参数字典
    """
    if weights is None:
        weights = [1.0 / len(model_states)] * len(model_states)
    else:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    aggregated_state = {}
    for key in model_states[0].keys():
        aggregated_state[key] = sum(
            weight * state[key]
            for weight, state in zip(weights, model_states)
        )

    return aggregated_state


