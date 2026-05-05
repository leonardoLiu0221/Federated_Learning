import os
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import pandas as pd


class FaultType:
    INNER_RACE = 0
    OUTER_RACE = 1
    CAGE = 2


FAULT_NAMES = {
    FaultType.INNER_RACE: "内圈故障",
    FaultType.OUTER_RACE: "外圈故障",
    FaultType.CAGE: "保持架故障",
}


class XJTUSYLoader:
    CONDITIONS = {
        '37.5Hz11kN': {'rotating_speed': 37.5, 'radial_force': 11},
    }

    BEARING_FAULT_MAP = {
        ('37.5Hz11kN', 'Bearing2_1'): FaultType.INNER_RACE,
        ('37.5Hz11kN', 'Bearing2_2'): FaultType.OUTER_RACE,
        ('37.5Hz11kN', 'Bearing2_3'): FaultType.CAGE,
        ('37.5Hz11kN', 'Bearing2_4'): FaultType.OUTER_RACE,
        ('37.5Hz11kN', 'Bearing2_5'): FaultType.OUTER_RACE,
    }

    def __init__(self, data_dir: str, signal_length: int = 1024, overlap: int = 0,
                 use_horizontal: bool = True, use_vertical: bool = False):

        self.data_dir = data_dir
        self.signal_length = signal_length
        self.overlap = overlap
        self.use_horizontal = use_horizontal
        self.use_vertical = use_vertical

    def _segment_signal(self, signal: np.ndarray) -> np.ndarray:
        step = self.signal_length - self.overlap
        segments = []
        for start_idx in range(0, len(signal) - self.signal_length + 1, step):
            segment = signal[start_idx:start_idx + self.signal_length]
            segments.append(segment)
        return np.array(segments)

    def _get_fault_type(self, condition: str, bearing_name: str) -> Optional[int]:
        key = (condition, bearing_name)
        return self.BEARING_FAULT_MAP.get(key)

    def load_data(self, conditions: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        all_segments = []
        all_labels = []

        if conditions is None:
            conditions = list(self.CONDITIONS.keys())

        for condition in conditions:
            cond_dir = os.path.join(self.data_dir, condition)
            if not os.path.exists(cond_dir):
                continue

            for bearing_name in os.listdir(cond_dir):
                if not bearing_name.startswith('Bearing'):
                    continue

                bearing_dir = os.path.join(cond_dir, bearing_name)
                if not os.path.isdir(bearing_dir):
                    continue

                fault_type = self._get_fault_type(condition, bearing_name)

                if fault_type is None:
                    print(f"  跳过复合故障: {condition}/{bearing_name}")
                    continue

                csv_files = sorted([f for f in os.listdir(bearing_dir) if f.endswith('.csv')],
                                  key=lambda x: int(os.path.splitext(x)[0]))

                if not csv_files:
                    continue

                for csv_file in csv_files:
                    csv_path = os.path.join(bearing_dir, csv_file)
                    try:
                        df = pd.read_csv(csv_path)

                        signals = []
                        if self.use_horizontal and 'Horizontal_vibration_signals' in df.columns:
                            signals.append(df['Horizontal_vibration_signals'].values)
                        if self.use_vertical and 'Vertical_vibration_signals' in df.columns:
                            signals.append(df['Vertical_vibration_signals'].values)

                        if not signals:
                            continue
                        min_len = min(len(s) for s in signals)
                        for i in range(len(signals)):
                            signals[i] = signals[i][:min_len]

                        segments_list = [self._segment_signal(s) for s in signals]

                        min_samples = min(len(segs) for segs in segments_list)
                        segments_list = [segs[:min_samples] for segs in segments_list]
                        combined = np.stack(segments_list, axis=1)
                        all_segments.extend(combined)

                        num_samples = len(segments_list[0]) if len(segments_list) > 0 else 0
                        all_labels.extend([fault_type] * num_samples)

                    except Exception as e:
                        print(f"  警告: 无法读取 {csv_path}: {e}")
                        continue

        X = np.array(all_segments)
        y = np.array(all_labels)

        return X, y


def create_federated_clients(dataset_type: str = 'xjtu',
                              data_root: str = 'E:\\FL\\Data\\Data',
                              signal_length: int = 1024,
                              overlap: int = 0,
                              num_classes: int = 3,
                              num_clients: int = 3,
                              random_state: int = 42) -> Dict:
    """
    客户端划分策略:
    将 XJTU-SY 37.5Hz11kN 工况数据（包含全部3个故障类别）随机划分为 num_clients 个客户端
    """
    clients = {}
    xjtu_dir = os.path.join(data_root, 'XJTU-SY_Bearing_Datasets')
    # 只加载 37.5Hz11kN 工况数据（包含全部3个类别）
    xjtu_loader = XJTUSYLoader(xjtu_dir, signal_length=signal_length, overlap=overlap,
                               use_horizontal=True, use_vertical=False)
    X, y = xjtu_loader.load_data(conditions=['37.5Hz11kN'])

    print(f"  总数据量: {len(X)} 样本")
    print(f"  标签分布: {np.bincount(y)}")

    # 按类别分层划分，确保每个客户端都有完整的三个类别
    np.random.seed(random_state)

    # 对每个类别分别进行索引收集和划分
    indices_by_class = {}
    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        indices_by_class[cls] = cls_indices

    # 将每个类别的索引分成 num_clients 份
    client_indices = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_indices = indices_by_class[cls]
        split_size = len(cls_indices) // num_clients
        for i in range(num_clients):
            start = i * split_size
            end = (i + 1) * split_size if i < num_clients - 1 else len(cls_indices)
            client_indices[i].extend(cls_indices[start:end])

    # 创建客户端
    for i in range(num_clients):
        indices = np.array(client_indices[i])
        np.random.shuffle(indices)  # 打乱每个客户端内部的样本顺序

        client_name = f'Client{i+1}_37.5Hz11kN'
        clients[client_name] = {
            'data': X[indices],
            'labels': y[indices],
            'description': f'XJTU-SY - 37.5Hz11kN - Part {i+1}'
        }
        print(f"  {client_name}: {len(indices)} 样本, 标签分布: {np.bincount(y[indices])}")

    return clients


def split_train_test(client_data: Dict, test_size: float = 0.2,
                    random_state: int = 42) -> Dict:
    result = {}
    for client_name, data in client_data.items():
        X = data['data']
        y = data['labels']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        result[client_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'description': data['description']
        }
    return result


if __name__ == '__main__':
    try:
        clients = create_federated_clients(
            dataset_type='both',
            signal_length=1024,
            overlap=128,
            num_classes=3
        )

        split_clients = split_train_test(clients)

        for name, data in split_clients.items():
            print(f"{name}:")
            print(f"  描述: {data['description']}")
            print(f"  训练集: {len(data['X_train'])} 样本")
            print(f"  测试集: {len(data['X_test'])} 样本")
            print(f"  标签分布: {np.bincount(data['y_train'])}")
            print(f"  数据形状: {data['X_train'].shape}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
