import os
import sys
import time
import json
import argparse
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import aggregate_models, get_model


FAULT_NAMES = {0: '内圈', 1: '外圈', 2: '保持架'}

DEFAULT_CONFIG = {
    'train_data_root': "E:\\FL\\Data\\Clients_train_2",
    'test_data_root': "E:\\FL\\Data\\Clients_test",
    'client_dirs': None,
    'dataset_type': 'xjtu',
    'model_name': 'cnn',
    'signal_length': 1024,
    'num_classes': 3,

    'batch_size': 64,
    'local_epochs': 3,
    'global_rounds': 6,
    'clients_per_round': None,
    'lr': 0.001,
    'weight_decay': 1e-4,

    'enable_detection': True,
    'detection_start_round': 1,
    'dq_batches': None,
    'min_benign_clients': 1,
    'dq_center_gap_threshold': 0.01,
    'save_dq_plots': True,
    'dq_plot_dir': 'dq_plots',

    'generator_hidden_dim': 128,
    'generator_lr': 0.001,
    'generator_weight_decay': 1e-4,
    'generator_steps': 20,
    'generator_batch_size': 32,
    'lambda_gen': 0.5,
    'feature_norm_weight': 1e-4,

    'test_size': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'save_dir': 'results_case1',
}


class LabelToFeatureGenerator(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, one_hot_labels):
        return self.net(one_hot_labels)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def parse_nullable_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if str(value).lower() in {'none', 'all'}:
        return None
    return int(value)


def parse_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if not value.strip():
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def discover_client_dirs(data_root: str, client_dirs: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    if client_dirs:
        discovered = []
        for client_dir in client_dirs:
            full_path = client_dir if os.path.isabs(client_dir) else os.path.join(data_root, client_dir)
            if not os.path.isdir(full_path):
                raise ValueError(f'客户端目录不存在: {full_path}')
            if not os.path.exists(os.path.join(full_path, 'data.npy')) or not os.path.exists(os.path.join(full_path, 'labels.npy')):
                raise ValueError(f'客户端目录缺少 data.npy 或 labels.npy: {full_path}')
            discovered.append((os.path.basename(os.path.normpath(full_path)), full_path))
        return discovered

    if not os.path.isdir(data_root):
        raise ValueError(f'数据根目录不存在: {data_root}')

    discovered = []
    for name in sorted(os.listdir(data_root)):
        full_path = os.path.join(data_root, name)
        if not os.path.isdir(full_path):
            continue
        if os.path.exists(os.path.join(full_path, 'data.npy')) and os.path.exists(os.path.join(full_path, 'labels.npy')):
            discovered.append((name, full_path))

    if not discovered:
        raise ValueError(f'未在 {data_root} 下发现包含 data.npy 和 labels.npy 的客户端目录')
    return discovered


def load_client_split(data_root: str, client_dirs: Optional[List[str]], num_classes: int, split_name: str):
    clients = {}
    for client_name, client_dir in discover_client_dirs(data_root, client_dirs):
        X = np.load(os.path.join(client_dir, 'data.npy'))
        y = np.load(os.path.join(client_dir, 'labels.npy')).astype(np.int64)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        clients[client_name] = {
            'data': X,
            'labels': y,
            'description': f'{split_name} - {client_name}',
        }
        print(f"  {client_name}: {len(X)} 样本, 标签分布: {np.bincount(y, minlength=num_classes)}")
    return clients


def load_global_test_data(test_data_root: str, num_classes: int):
    data_path = os.path.join(test_data_root, 'data.npy')
    labels_path = os.path.join(test_data_root, 'labels.npy')
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise ValueError(f'测试集目录缺少 data.npy 或 labels.npy: {test_data_root}')

    X = np.load(data_path)
    y = np.load(labels_path).astype(np.int64)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    print(f"  公共测试集: {len(X)} 样本, 标签分布: {np.bincount(y, minlength=num_classes)}")
    return X, y


def prepare_client_data(train_clients: Dict):
    split_clients = {}
    for client_name in sorted(train_clients.keys()):
        train_data = train_clients[client_name]
        split_clients[client_name] = {
            'X_train': train_data['data'],
            'y_train': train_data['labels'],
            'description': train_data['description'],
        }
    return split_clients


def build_loaders(split_clients: Dict, test_data: Tuple[np.ndarray, np.ndarray], config: Dict):
    client_loaders = {}
    for client_name, data in split_clients.items():
        train_dataset = TensorDataset(
            torch.from_numpy(data['X_train']),
            torch.from_numpy(data['y_train']),
        )
        client_loaders[client_name] = {
            'loader': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True),
            'num_samples': len(train_dataset),
        }

    X_test, y_test = test_data
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    return client_loaders, test_loader


def get_predictor_head(model):
    if hasattr(model, 'fc2'):
        return model.fc2
    if hasattr(model, 'fc'):
        return model.fc
    raise ValueError('当前模型未找到可用预测头，需包含 fc2 或 fc')


def infer_feature_dim(model, sample_shape: Tuple[int, int], device: str) -> int:
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, *sample_shape, device=device)
        features = model.get_features(dummy)
    return features.shape[1]


def one_hot(labels, num_classes: int, device: str):
    return F.one_hot(labels.long(), num_classes=num_classes).float().to(device)


def compute_client_dq(global_model, generator, client_loader, config: Dict, device: str) -> float:
    global_model.eval()
    generator.eval()
    predictor = get_predictor_head(global_model)
    matched = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(client_loader):
            if config['dq_batches'] is not None and batch_idx >= config['dq_batches']:
                break

            data = data.to(device).float()
            target = target.to(device).long()

            real_logits = global_model(data)
            real_pred = real_logits.argmax(dim=1)

            generated_features = generator(one_hot(target, config['num_classes'], device))
            generated_logits = predictor(generated_features)
            generated_pred = generated_logits.argmax(dim=1)

            matched += (real_pred == generated_pred).sum().item()
            total += target.size(0)

    return float(matched / total) if total > 0 else 0.0


def detect_benign_clients(dq_scores: Dict[str, float], config: Dict):
    client_names = list(dq_scores.keys())
    if len(client_names) < 2:
        return client_names, []

    values = np.array([[dq_scores[name]] for name in client_names], dtype=np.float64)
    if np.allclose(values, values[0]):
        return client_names, []

    kmeans = KMeans(n_clusters=2, random_state=config['seed'], n_init=10)
    labels = kmeans.fit_predict(values)
    centers = kmeans.cluster_centers_.reshape(-1)
    center_gap = float(np.max(centers) - np.min(centers))
    if center_gap < float(config['dq_center_gap_threshold']):
        return client_names, []

    malicious_cluster = int(np.argmin(centers))

    benign = [name for name, label in zip(client_names, labels) if label != malicious_cluster]
    suspicious = [name for name, label in zip(client_names, labels) if label == malicious_cluster]

    min_benign = min(int(config['min_benign_clients']), len(client_names))
    if len(benign) < min_benign:
        ranked = sorted(client_names, key=lambda name: dq_scores[name], reverse=True)
        benign = ranked[:min_benign]
        suspicious = [name for name in client_names if name not in benign]

    return benign, suspicious


def sample_clients_for_round(benign_clients: List[str], clients_per_round: Optional[int], rng: np.random.Generator):
    if not benign_clients:
        raise ValueError('没有可用于本轮训练的良性客户端')
    if clients_per_round is None or clients_per_round >= len(benign_clients):
        return list(benign_clients)
    return rng.choice(benign_clients, size=clients_per_round, replace=False).tolist()


def train_local_mcdfl(local_model, generator, train_loader, criterion, optimizer, config: Dict, device: str):
    local_model.train()
    generator.eval()
    predictor = get_predictor_head(local_model)
    total_samples = 0
    total_loss = 0.0
    correct = 0

    for _ in range(config['local_epochs']):
        for data, target in train_loader:
            data = data.to(device).float()
            target = target.to(device).long()

            optimizer.zero_grad()

            real_logits = local_model(data)
            loss_real = criterion(real_logits, target)

            with torch.no_grad():
                generated_features = generator(one_hot(target, config['num_classes'], device)).detach()
            generated_logits = predictor(generated_features)
            loss_generated = criterion(generated_logits, target)

            loss = loss_real + float(config['lambda_gen']) * loss_generated
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * target.size(0)
            correct += (real_logits.argmax(dim=1) == target).sum().item()
            total_samples += target.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0
    return local_model.state_dict(), len(train_loader.dataset), avg_loss, accuracy


def train_generator_on_server(global_model, generator, optimizer_g, criterion, config: Dict, device: str):
    global_model.eval()
    generator.train()
    predictor = get_predictor_head(global_model)

    for param in global_model.parameters():
        param.requires_grad_(False)

    total_loss = 0.0
    for _ in range(config['generator_steps']):
        labels = torch.randint(
            low=0,
            high=config['num_classes'],
            size=(config['generator_batch_size'],),
            device=device,
        )
        labels_one_hot = one_hot(labels, config['num_classes'], device)

        optimizer_g.zero_grad()
        generated_features = generator(labels_one_hot)
        logits = predictor(generated_features)
        ce_loss = criterion(logits, labels)
        norm_loss = generated_features.pow(2).mean()
        loss = ce_loss + float(config['feature_norm_weight']) * norm_loss
        loss.backward()
        optimizer_g.step()
        total_loss += loss.item()

    for param in global_model.parameters():
        param.requires_grad_(True)

    return total_loss / max(1, config['generator_steps'])


def evaluate_model(model, test_loader, device: str, num_classes: int = 3):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            probs = F.softmax(output, dim=1)
            predicted = output.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    sensitivity_values = []
    specificity_values = []
    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fn = cm[class_idx, :].sum() - tp
        fp = cm[:, class_idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        if tp + fn > 0:
            sensitivity_values.append(tp / (tp + fn))
        if tn + fp > 0:
            specificity_values.append(tn / (tn + fp))

    sensitivity = 100.0 * float(np.mean(sensitivity_values)) if sensitivity_values else 0.0
    specificity = 100.0 * float(np.mean(specificity_values)) if specificity_values else 0.0

    if total > 0:
        y_true_one_hot = np.eye(num_classes)[labels]
        mse = float(mean_squared_error(y_true_one_hot, probs))
    else:
        mse = 0.0

    try:
        auc = float(roc_auc_score(labels, probs, labels=list(range(num_classes)), multi_class='ovr', average='macro'))
    except ValueError:
        auc = None

    metrics = {
        'accuracy': accuracy,
        'mse': mse,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
    }
    return metrics, cm


def save_dq_cluster_plot(
    dq_scores: Dict[str, float],
    benign_clients: List[str],
    suspicious_clients: List[str],
    save_dir: str,
    output_prefix: str,
    round_num: int,
    plot_dir_name: str = 'dq_plots',
):
    if not dq_scores:
        return None

    plot_dir = os.path.join(save_dir, plot_dir_name)
    os.makedirs(plot_dir, exist_ok=True)

    client_names = list(dq_scores.keys())
    x = np.arange(1, len(client_names) + 1)
    y = np.array([dq_scores[name] for name in client_names], dtype=np.float64)
    colors = ['tab:red' if name in suspicious_clients else 'tab:blue' for name in client_names]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, c=colors, s=80)
    plt.plot(x, y, color='gray', linestyle='--', alpha=0.5)
    for xi, yi in zip(x, y):
        plt.annotate(
            f'{yi:.4f}',
            xy=(xi, yi),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
            fontsize=8,
        )
    plt.xticks(x, client_names, rotation=30)
    plt.xlabel('Client Index')
    plt.ylabel('Data Quality')
    plt.title(f'DQ Clustering - Round {round_num}')
    plt.ylim(max(0.0, float(np.min(y)) - 0.05), min(1.05, float(np.max(y)) + 0.05))
    plt.grid(True, linestyle='--', alpha=0.3)

    if benign_clients:
        plt.scatter([], [], c='tab:blue', s=80, label='Benign')
    if suspicious_clients:
        plt.scatter([], [], c='tab:red', s=80, label='Suspicious')
    plt.legend()
    plt.tight_layout()

    plot_file = os.path.join(plot_dir, f'{output_prefix}round_{round_num:03d}_dq_cluster.png')
    plt.savefig(plot_file, dpi=200)
    plt.close()
    return plot_file


def save_metrics(metrics: List[Dict], save_dir: str, output_prefix: str):
    metrics_file = os.path.join(save_dir, f'{output_prefix}metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics_file


def run_mcdfl_training(config: Dict, output_prefix: str = 'mcdfl_'):
    set_seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)
    rng = np.random.default_rng(config['seed'])

    print('加载训练集客户端数据')
    train_clients = load_client_split(config['train_data_root'], config.get('client_dirs'), config['num_classes'], '训练集')
    print('加载公共测试集数据')
    test_data = load_global_test_data(config['test_data_root'], config['num_classes'])
    client_names = list(train_clients.keys())

    split_clients = prepare_client_data(train_clients)
    first_client = next(iter(split_clients.values()))
    num_channels = first_client['X_train'].shape[1]
    sample_shape = (num_channels, first_client['X_train'].shape[2])
    print(f'  输入通道数: {num_channels}')

    client_loaders, test_loader = build_loaders(split_clients, test_data, config)
    for client_name, data in split_clients.items():
        train_dist = np.bincount(data['y_train'], minlength=config['num_classes'])
        print(f"  {client_name}: 训练 {len(data['X_train'])}, 训练标签分布={train_dist}")

    print('初始化全局模型和条件生成器...')
    global_model = get_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        signal_length=config['signal_length'],
        num_channels=num_channels,
    ).to(config['device'])
    feature_dim = infer_feature_dim(global_model, sample_shape, config['device'])
    predictor_name = 'fc2' if hasattr(global_model, 'fc2') else 'fc'
    print(f'  潜在特征维度: {feature_dim}, 预测头: {predictor_name}')

    generator = LabelToFeatureGenerator(
        num_classes=config['num_classes'],
        feature_dim=feature_dim,
        hidden_dim=config['generator_hidden_dim'],
    ).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    generator_optimizer = optim.Adam(
        generator.parameters(),
        lr=config['generator_lr'],
        weight_decay=config['generator_weight_decay'],
    )

    best_global_acc = 0.0
    best_model_state = deepcopy(global_model.state_dict())
    best_generator_state = deepcopy(generator.state_dict())
    metrics = []

    print('开始 MCDFL 训练...')
    for round_idx in range(config['global_rounds']):
        round_num = round_idx + 1
        print(f'\n[轮次 {round_num}/{config["global_rounds"]}]')

        dq_scores = {}
        dq_plot_file = None
        if config['enable_detection'] and round_num >= int(config['detection_start_round']):
            for client_name in client_names:
                dq_scores[client_name] = compute_client_dq(
                    global_model,
                    generator,
                    client_loaders[client_name]['loader'],
                    config,
                    config['device'],
                )
            benign_clients, suspicious_clients = detect_benign_clients(dq_scores, config)
            dq_text = ', '.join(f'{name}: {score:.4f}' for name, score in dq_scores.items())
            print(f'  DQ: {dq_text}')
            print(f'  判定良性客户端: {benign_clients}')
            print(f'  判定可疑客户端: {suspicious_clients}')
            if config.get('save_dq_plots', True):
                dq_plot_file = save_dq_cluster_plot(
                    dq_scores,
                    benign_clients,
                    suspicious_clients,
                    config['save_dir'],
                    output_prefix,
                    round_num,
                    config.get('dq_plot_dir', 'dq_plots'),
                )
                print(f'  DQ 聚类图已保存: {dq_plot_file}')
        else:
            benign_clients = client_names
            suspicious_clients = []
            print('  本轮未启用检测，所有客户端视为良性')

        selected_clients = sample_clients_for_round(
            benign_clients,
            parse_nullable_int(config.get('clients_per_round')),
            rng,
        )
        print(f'  本轮训练客户端: {selected_clients}')

        local_model_states = []
        client_weights = []
        local_stats = {}

        for client_name in selected_clients:
            local_model = get_model(
                model_name=config['model_name'],
                num_classes=config['num_classes'],
                signal_length=config['signal_length'],
                num_channels=num_channels,
            ).to(config['device'])
            local_model.load_state_dict(global_model.state_dict())

            optimizer = optim.Adam(
                local_model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay'],
            )
            state_dict, num_samples, avg_loss, train_acc = train_local_mcdfl(
                local_model,
                generator,
                client_loaders[client_name]['loader'],
                criterion,
                optimizer,
                config,
                config['device'],
            )
            local_model_states.append(state_dict)
            client_weights.append(num_samples)
            local_stats[client_name] = {'loss': avg_loss, 'train_acc': train_acc}
            print(f'    {client_name}: loss={avg_loss:.4f}, train_acc={train_acc:.2f}%')

        aggregated_state = aggregate_models(local_model_states, weights=client_weights)
        global_model.load_state_dict(aggregated_state)

        generator_loss = train_generator_on_server(
            global_model,
            generator,
            generator_optimizer,
            criterion,
            config,
            config['device'],
        )
        print(f'  生成器服务器端 loss: {generator_loss:.4f}')

        print('  全局模型在公共测试集上的评估指标:')
        test_metrics, _ = evaluate_model(
            global_model,
            test_loader,
            config['device'],
            num_classes=config['num_classes'],
        )
        auc_text = f"{test_metrics['auc']:.4f}" if test_metrics['auc'] is not None else 'N/A'
        print(
            f"  Accuracy={test_metrics['accuracy']:.2f}%, "
            f"MSE={test_metrics['mse']:.6f}, "
            f"Sensitivity={test_metrics['sensitivity']:.2f}%, "
            f"Specificity={test_metrics['specificity']:.2f}%, "
            f"AUC={auc_text}"
        )
        mean_metrics = test_metrics
        mean_acc = mean_metrics['accuracy']
        if mean_acc > best_global_acc:
            best_global_acc = mean_acc
            best_model_state = deepcopy(global_model.state_dict())
            best_generator_state = deepcopy(generator.state_dict())
            print(f'  新最佳模型: 平均准确率 {mean_acc:.2f}%')

        metrics.append({
            'round': round_num,
            'dq_scores': dq_scores,
            'benign_clients': benign_clients,
            'suspicious_clients': suspicious_clients,
            'dq_plot_file': dq_plot_file,
            'selected_clients': selected_clients,
            'local_stats': local_stats,
            'generator_loss': generator_loss,
            'test_metrics': test_metrics,
            'mean_metrics': mean_metrics,
            'mean_accuracy': mean_acc,
        })

    print('最终评估...')
    global_model.load_state_dict(best_model_state)
    final_test_metrics, cm = evaluate_model(
        global_model,
        test_loader,
        config['device'],
        num_classes=config['num_classes'],
    )
    auc_text = f"{final_test_metrics['auc']:.4f}" if final_test_metrics['auc'] is not None else 'N/A'
    print(
        f"  Accuracy={final_test_metrics['accuracy']:.2f}%, "
        f"MSE={final_test_metrics['mse']:.6f}, "
        f"Sensitivity={final_test_metrics['sensitivity']:.2f}%, "
        f"Specificity={final_test_metrics['specificity']:.2f}%, "
        f"AUC={auc_text}"
    )
    print('  混淆矩阵:')
    for i, row in enumerate(cm):
        fault_name = FAULT_NAMES.get(i, f'Class {i}')
        print(f'    {fault_name:<8}: {row}')

    print('保存结果...')
    best_model_file = os.path.join(config['save_dir'], f'{output_prefix}best_model.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'generator_state_dict': best_generator_state,
        'config': config,
        'accuracy': best_global_acc,
        'final_test_metrics': final_test_metrics,
        'num_channels': num_channels,
        'feature_dim': feature_dim,
        'client_names': client_names,
    }, best_model_file)
    metrics_file = save_metrics(metrics, config['save_dir'], output_prefix)
    print(f'  最佳模型已保存: {best_model_file}')
    print(f'  训练记录已保存: {metrics_file}')
    return global_model, generator, metrics


def load_config(args):
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config.update(json.load(f))

    if args.train_data_root is not None:
        config['train_data_root'] = args.train_data_root
    if args.test_data_root is not None:
        config['test_data_root'] = args.test_data_root
    if args.client_dirs is not None:
        config['client_dirs'] = parse_list(args.client_dirs)
    if args.clients_per_round is not None:
        config['clients_per_round'] = parse_nullable_int(args.clients_per_round)
    if args.detection_start_round is not None:
        config['detection_start_round'] = args.detection_start_round
    if args.lambda_gen is not None:
        config['lambda_gen'] = args.lambda_gen
    if args.generator_steps is not None:
        config['generator_steps'] = args.generator_steps
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir

    config['clients_per_round'] = parse_nullable_int(config.get('clients_per_round'))
    config['client_dirs'] = parse_list(config.get('client_dirs'))
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='配置 JSON 文件路径')
    parser.add_argument('--train-data-root', type=str, default=None, help='训练集客户端数据根目录')
    parser.add_argument('--test-data-root', type=str, default=None, help='测试集客户端数据根目录')
    parser.add_argument('--client-dirs', type=str, default=None, help='逗号分隔的客户端目录名')
    parser.add_argument('--clients-per-round', type=str, default=None, help='每轮采样客户端数，all/none 表示全部')
    parser.add_argument('--detection-start-round', type=int, default=None, help='从第几轮开始检测')
    parser.add_argument('--lambda-gen', type=float, default=None, help='本地生成特征损失权重')
    parser.add_argument('--generator-steps', type=int, default=None, help='每轮服务器端生成器更新步数')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')
    parser.add_argument('--save-dir', type=str, default=None, help='结果保存目录')
    parser.add_argument('--output-prefix', type=str, default='mcdfl_', help='输出文件前缀')

    args = parser.parse_args()
    config = load_config(args)

    try:
        start_time = time.time()
        run_mcdfl_training(config, args.output_prefix)
        print(f'\n总耗时: {time.time() - start_time:.2f} 秒')
    except KeyboardInterrupt:
        print('\n\n训练被用户中断')
        sys.exit(1)
    except Exception as e:
        print(f'\n\n错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
