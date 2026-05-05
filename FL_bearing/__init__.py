from .data_loader import (
    XJTUSYLoader,
    FaultType,
    FAULT_NAMES,
    create_federated_clients,
    split_train_test
)
from .model import (
    BearingCNN,
    BearingResNet,
    BearingLSTM,
    BearingCNN_LSTM,
    get_model,
    aggregate_models
)

__version__ = '1.0.0'
