import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

class ModelTrainer(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2), padding=((2, 2), (2, 2)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_size)(x)
        return x

def create_train_state(rng, action_size, learning_rate=0.0005):
    model = ModelTrainer(action_size=action_size)
    params = model.init(rng, jnp.ones([1, 84, 84, 4]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

class ModelTrainerWrapper:
    def __init__(self, input_shape, n_actions):
        rng = jax.random.PRNGKey(0)
        self.state = create_train_state(rng, n_actions)
        self.target_params = self.state.params

    @jax.jit
    def _forward(self, params, x):
        return self.state.apply_fn({'params': params}, x)

    def predict(self, state):
        state = jnp.array(state).astype(jnp.float32)
        state = jnp.expand_dims(state, axis=0)
        return jnp.argmax(self._forward(self.state.params, state), axis=1)

    @jax.jit
    def _update_step(self, state, batch):
        def loss_fn(params):
            states, actions, rewards, next_states, dones = batch
            q_values = state.apply_fn({'params': params}, states)
            next_q_values = state.apply_fn({'params': self.target_params}, next_states)
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