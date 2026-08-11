from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ModelConfig:

    img_size: int = 128
    in_channels: int = 3
    out_channels: int = 6
    kernel_size: tuple = (3, 3)
    dropout_rate: float = 5e-2
    start_log_std: float = -0.5

@dataclass(frozen=True, slots=True)
class TrainingConfig:

    lr: float = 1e-3
    batch_size: int = 128
    
