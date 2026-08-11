from flax import nnx
import jax.numpy as jnp
from ppo_cfg import ModelConfig

class ActorNetwork(nnx.Module):

    def __init__(self, model_cfg: ModelConfig, rngs: nnx.Rngs):

        super().__init__()

        self.conv1 = nnx.Conv(model_cfg.in_channels, 32, model_cfg.kernel_size, rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.dropout1 = nnx.Dropout(model_cfg.dropout_rate, rngs=rngs)

        self.conv2 = nnx.Conv(32, 16, model_cfg.kernel_size, rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(num_features=16, rngs=rngs)
        self.dropout1 = nnx.Dropout(model_cfg.dropout_rate, rngs=rngs)

        self.conv3 = nnx.Conv(16, 8, model_cfg.kernel_size, rngs=rngs)
        self.batch_norm3 = nnx.BatchNorm(num_features=8, rngs=rngs)
        self.dropout3 = nnx.Dropout(model_cfg.dropout_rate, rngs=rngs)

        self.conv4 = nnx.Conv(8, 4, model_cfg.kernel_size, rngs=rngs)
        self.batch_norm4 = nnx.BatchNorm(num_features=4, rngs=rngs)
        self.dropout4 = nnx.Dropout(model_cfg.dropout_rate, rngs=rngs)

        self.linear1 = nnx.Linear(4*(model_cfg.img_size**2), 2048)
        self.linear2 = nnx.Linear(4*(model_cfg.img_size**2), 512)
        self.linear3 = nnx.Linear(4*(model_cfg.img_size**2), 128)
        self.linear3 = nnx.Linear(4*(model_cfg.img_size**2), 128)


        self._weight_init()

    def _weight_init(self):
        pass

    def __call__(self, x: jnp.Array) -> jnp.Array:
        pass

        

