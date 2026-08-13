import flax.struct as struct
from typing import Tuple

@struct.dataclass
class ActorConfig:

    img_size: int = 128
    output_features: int = 6
    features: Tuple[int, ...] = (32, 64, 32, 16, 4)
    dense_features: Tuple[int, ...] = (1024, 512, 256, 64)
    kernel_size: tuple = (3, 3)
    dropout_rate: float = 5e-2
    start_log_std: float = -0.5


@struct.dataclass
class CriticConfig:

    img_size: int = 128,
    features: Tuple[int, ...] = (32, 64, 32, 16, 4)
    dense_features: Tuple[int, ...] = (1024, 512, 256, 64)
    kernel_size: tuple = (3, 3)
    dropout_rate: float = 5e-2

    
