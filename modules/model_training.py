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
import random

# GA configs
pop_size = 50
mutation_rate_min = 0.01
mutation_rate_max = 0.5
crossover_rate_min = 0.5
crossover_rate_max = 1
min_elite_size = 2
max_elite_size = 5
tournament_size_min = 2
tournament_size_max = 10

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

def create_train_state(rng, input_shape, action_size, learning_rate=0.001):
    def forward_fn(x):
        model = ModelTrainer(num_layers=3, hidden_size=64, num_heads=4, dropout_rate=0.1)
        model.set_action_size(action_size)
        return model(x)

    transformed_forward = hk.transform(forward_fn)
    params = transformed_forward.init(rng, jnp.ones(input_shape))

    # Use Adam optimizer with learning rate schedule
    schedule_fn = optax.exponential_decay(init_value=learning_rate, transition_steps=1000, decay_rate=0.9)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule_fn)
    )

    return train_state.TrainState.create(apply_fn=transformed_forward.apply, params=params, tx=tx)

class ModelTrainerWrapper:
    def __init__(self, input_shape, n_actions):
        rng = jax.random.PRNGKey(0)
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.state = create_train_state(rng, (1,) + input_shape, n_actions)
        self.target_params = self.state.params
        self.forward = hk.transform(lambda x: ModelTrainer(num_layers=3, hidden_size=64, num_heads=4, dropout_rate=0.1)(x))
        self.population = self._initialize_population()
        self.best_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0

    def _initialize_population(self):
        return [self._mutate_params(self.state.params) for _ in range(pop_size)]

    def _mutate_params(self, params):
        mutation_rate = random.uniform(mutation_rate_min, mutation_rate_max)
        return jax.tree_map(lambda x: x + jax.random.normal(jax.random.PRNGKey(0), x.shape) * mutation_rate, params)

    def _crossover(self, parent1, parent2):
        crossover_rate = random.uniform(crossover_rate_min, crossover_rate_max)
        return jax.tree_map(lambda x, y: jnp.where(jax.random.uniform(jax.random.PRNGKey(0), x.shape) < crossover_rate, x, y), parent1, parent2)

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

        # Evaluate population
        losses = [self._update_step(train_state.TrainState.create(apply_fn=self.state.apply_fn, params=params, tx=self.state.tx), batch)[1] for params in self.population]

        # Selection
        elite_size = random.randint(min_elite_size, max_elite_size)
        elites = sorted(zip(self.population, losses), key=lambda x: x[1])[:elite_size]

        # Crossover and Mutation
        new_population = [elite[0] for elite in elites]
        while len(new_population) < pop_size:
            parent1 = self._tournament_selection(self.population, losses)
            parent2 = self._tournament_selection(self.population, losses)
            child = self._crossover(parent1, parent2)
            child = self._mutate_params(child)
            new_population.append(child)

        self.population = new_population
        self.state = train_state.TrainState.create(apply_fn=self.state.apply_fn, params=self.population[0], tx=self.state.tx)

        min_loss = min(losses)

        # Early stopping
        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print("Early stopping triggered")
            return None

        return min_loss

    def _tournament_selection(self, population, fitnesses):
        tournament_size = random.randint(tournament_size_min, tournament_size_max)
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        return min(tournament, key=lambda x: x[1])[0]

    def update_target(self):
        self.target_params = self.state.params

    def compress_model(self):
        # Implement model compression techniques here
        # For example, pruning, quantization, or knowledge distillation
        pass
