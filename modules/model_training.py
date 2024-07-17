import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from nextgenjax.model import NextGenModel
from nextgenjax.layers import DenseLayer, ConvolutionalLayer
from nextgenjax.custom_layers import CustomLayer
from nextgenjax.optimizers import sgd, adam
from nextgenjax.activations import relu, tanh
import haiku as hk

class ModelTrainer(NextGenModel):
    def __init__(self, num_layers, hidden_size, num_heads, dropout_rate, use_relative_attention=False, use_gradient_checkpointing=False, use_mixed_precision=False):
        super().__init__(num_layers, hidden_size, num_heads, dropout_rate, use_relative_attention, use_gradient_checkpointing, use_mixed_precision)
        self.action_size = None

    def set_action_size(self, action_size):
        self.action_size = action_size

    @nn.compact
    def __call__(self, x):
        x = super().__call__(x)
        if self.action_size is not None:
            x = DenseLayer(features=self.action_size)(x)
        return x

def create_train_state(rng, input_shape, action_size, learning_rate=0.0005):
    def forward_fn(x):
        model = ModelTrainer(num_layers=3, hidden_size=64, num_heads=4, dropout_rate=0.1)
        model.set_action_size(action_size)
        return model(x)

    transformed_forward = hk.transform(forward_fn)
    params = transformed_forward.init(rng, jnp.ones(input_shape))
    tx = adam(learning_rate)
    return train_state.TrainState.create(apply_fn=transformed_forward.apply, params=params, tx=tx)

class ModelTrainerWrapper:
    def __init__(self, input_shape, n_actions):
        rng = jax.random.PRNGKey(0)
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.state = create_train_state(rng, (1,) + input_shape, n_actions)
        self.target_params = self.state.params
        self.forward = hk.transform(lambda x: ModelTrainer(num_layers=3, hidden_size=64, num_heads=4, dropout_rate=0.1)(x))

    @jax.jit
    def _forward(self, params, x):
        return self.forward.apply(params, None, x)

    def predict(self, state):
        if not isinstance(state, jnp.ndarray):
            state = jnp.array(state)
        state = state.astype(jnp.float32)
        if state.ndim == 3:
            state = jnp.expand_dims(state, axis=0)
        elif state.ndim != 4:
            raise ValueError(f"Expected state with 3 or 4 dimensions, got {state.ndim}")
        return jnp.argmax(self._forward(self.state.params, state), axis=1)

    @jax.jit
    def _update_step(self, state, batch):
        def loss_fn(params):
            states, actions, rewards, next_states, dones = batch
            q_values = self._forward(params, states)
            next_q_values = self._forward(self.target_params, next_states)
            next_q_values = jnp.max(next_q_values, axis=1)
            targets = rewards + (1 - dones) * 0.99 * next_q_values
            predictions = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze()
            return jnp.mean((targets - predictions) ** 2)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = jnp.array(states).astype(jnp.float32)
        next_states = jnp.array(next_states).astype(jnp.float32)
        actions = jnp.array(actions).astype(jnp.int32)
        rewards = jnp.array(rewards).astype(jnp.float32)
        dones = jnp.array(dones).astype(jnp.float32)

        batch = (states, actions, rewards, next_states, dones)
        self.state, loss = self._update_step(self.state, batch)
        return loss.item()

    def update_target(self):
        self.target_params = self.state.params