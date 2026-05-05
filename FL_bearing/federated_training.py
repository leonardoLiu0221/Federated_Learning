import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
from model import get_model, aggregate_models
from data_loader import create_federated_clients, split_train_test, FaultType, FAULT_NAMES

DEFAULT_CONFIG = {
    'data_root': 'E:\\FL\\Data',
    'dataset_type': 'xjtu',
    'model_name': 'cnn',
    'signal_length': 1024,
    'overlap': 128,
    'num_classes': 3,

    'batch_size': 64,
    'local_epochs': 5,
    'global_rounds': 15,
    'lr': 0.001,
    'weight_decay': 1e-4,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'save_dir': 'results',
}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train_local_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device).float(), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return model.state_dict(), len(train_loader.dataset)

def evaluate_model(model, test_loader, device, num_classes=3):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).long()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = 100. * correct / total
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(num_classes)))

    return accuracy, cm

def run_federated_training(config: Dict, output_prefix: str = ''):
    set_seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    print("[1/4] 加载客户端数据...")
    clients = create_federated_clients(
        dataset_type=config['dataset_type'],
        data_root=config['data_root'],
        signal_length=config['signal_length'],
        overlap=config['overlap'],
        num_classes=config['num_classes']
    )
    split_clients = split_train_test(clients, test_size=0.2, random_state=config['seed'])

    first_client = next(iter(split_clients.values()))
    num_channels = first_client['X_train'].shape[1]
    print(f"输入通道数: {num_channels}")

    client_loaders = {}
    client_test_loaders = {}
    for client_name, data in split_clients.items():
        train_dataset = TensorDataset(
            torch.from_numpy(data['X_train']),
            torch.from_numpy(data['y_train'])
        )
        test_dataset = TensorDataset(
            torch.from_numpy(data['X_test']),
            torch.from_numpy(data['y_test'])
        )
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        client_loaders[client_name] = {'loader': train_loader, 'num_samples': len(train_dataset)}
        client_test_loaders[client_name] = test_loader
        print(f"  {client_name}: {len(data['X_train'])}/{len(data['X_test'])} 样本")

    client_names = list(client_loaders.keys())

    print("\n[2/4] 初始化全局模型...")
    global_model = get_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        signal_length=config['signal_length'],
        num_channels=num_channels
    ).to(config['device'])
    criterion = nn.CrossEntropyLoss()

    print("\n[3/4] 开始联邦学习训练...")
    best_global_acc = 0.0
    best_model_state = None

    for round_idx in range(config['global_rounds']):
        print(f"\n[轮次 {round_idx + 1}/{config['global_rounds']}]")

        local_model_states = []
        client_weights = []

        for client_name in client_names:
            local_model = get_model(
                model_name=config['model_name'],
                num_classes=config['num_classes'],
                signal_length=config['signal_length'],
                num_channels=num_channels
            ).to(config['device'])
            local_model.load_state_dict(global_model.state_dict())

            optimizer = optim.Adam(
                local_model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )

            state_dict, num_samples = train_local_model(
                local_model,
                client_loaders[client_name]['loader'],
                criterion,
                optimizer,
                config['local_epochs'],
                config['device']
            )
            local_model_states.append(state_dict)
            client_weights.append(num_samples)

        aggregated_state = aggregate_models(local_model_states)
        global_model.load_state_dict(aggregated_state)

        print("  全局模型在各客户端测试集上的准确率:")
        all_acc = []
        for client_name in client_names:
            acc, _ = evaluate_model(
                global_model,
                client_test_loaders[client_name],
                config['device'],
                num_classes=config['num_classes']
            )
            all_acc.append(acc)
            print(f"    {client_name}: {acc:.2f}%")

    print("\n[4/4] 训练完成!")
    print("\n最终全局模型在各客户端测试集上的表现:")
    for client_name in client_names:
        acc, cm = evaluate_model(
            global_model,
            client_test_loaders[client_name],
            config['device'],
            num_classes=config['num_classes']
        )
        print(f"\n  {client_name}:")
        print(f"    准确率: {acc:.2f}%")
        print("    混淆矩阵:")
        for i, row in enumerate(cm):
            fault_name = FAULT_NAMES.get(i, f"Class {i}")
            print(f"      {fault_name:<8}: {row}")
    best_model_file = os.path.join(config['save_dir'], f'{output_prefix}best_model.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'accuracy': best_global_acc,
        'num_channels': num_channels
    }, best_model_file)
    print(f"  最佳模型已保存: {best_model_file}")
    return global_model


def main():
    parser = argparse.ArgumentParser(description='轴承故障诊断联邦学习训练')
    parser.add_argument('--config', type=str, default=None, help='配置JSON文件路径')
    parser.add_argument('--dataset', type=str, default='xjtu',
                       choices=['xjtu'], help='数据集类型')
    parser.add_argument('--model', type=str, default='cnn', help='模型类型')
    parser.add_argument('--rounds', type=int, default=None, help='全局轮数')
    parser.add_argument('--local-epochs', type=int, default=None, help='本地训练轮数')
    parser.add_argument('--output-prefix', type=str, default='', help='输出文件前缀')

    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))

    config['dataset_type'] = args.dataset
    config['model_name'] = args.model
    if args.rounds is not None:
        config['global_rounds'] = args.rounds
    if args.local_epochs is not None:
        config['local_epochs'] = args.local_epochs

    if not args.output_prefix:
        args.output_prefix = f"{config['dataset_type']}_{config['model_name']}_"

    try:
        run_federated_training(config, args.output_prefix)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
