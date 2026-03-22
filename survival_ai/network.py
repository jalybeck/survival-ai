"""Pure-Python feedforward neural network for later action selection and learning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass


@dataclass(slots=True)
class LayerParameters:
    """Stores one dense layer's trainable parameters and gradients."""

    weights: list[list[float]]
    biases: list[float]
    weight_gradients: list[list[float]]
    bias_gradients: list[float]


@dataclass(slots=True)
class ForwardTrace:
    """Stores intermediate values from the latest forward pass."""

    inputs: list[float]
    pre_activations: list[list[float]]
    activations: list[list[float]]
    outputs: list[float]


class SimpleMLP:
    """Small dense neural network implemented with plain Python lists."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        seed: int | None = None,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive.")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self._rng = random.Random(seed)
        self.layers = self._initialize_layers()
        self.last_trace: ForwardTrace | None = None

    def forward(self, inputs: list[float]) -> list[float]:
        """Run a forward pass and return raw action scores from the output layer."""

        if len(inputs) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} inputs but received {len(inputs)}."
            )

        activations = [list(inputs)]
        pre_activations: list[list[float]] = []
        current_values = list(inputs)

        for layer_index, layer in enumerate(self.layers):
            z_values = self._dense_forward(current_values, layer.weights, layer.biases)
            pre_activations.append(z_values)

            # Hidden layers use ReLU so the network can learn non-linear policies.
            # The output layer stays linear because later phases may apply different
            # training targets or action-selection strategies on top of raw scores.
            if layer_index < len(self.layers) - 1:
                current_values = [self._relu(value) for value in z_values]
            else:
                current_values = z_values
            activations.append(current_values)

        self.last_trace = ForwardTrace(
            inputs=list(inputs),
            pre_activations=pre_activations,
            activations=activations,
            outputs=list(current_values),
        )
        return list(current_values)

    def backward(self, output_gradients: list[float]) -> list[float]:
        """Backpropagate output gradients and accumulate parameter gradients."""

        if self.last_trace is None:
            raise RuntimeError("forward() must be called before backward().")
        if len(output_gradients) != self.output_size:
            raise ValueError(
                f"Expected {self.output_size} output gradients but received {len(output_gradients)}."
            )

        current_gradients = list(output_gradients)

        for layer_index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_index]
            layer_inputs = self.last_trace.activations[layer_index]
            layer_pre_activations = self.last_trace.pre_activations[layer_index]

            if layer_index < len(self.layers) - 1:
                current_gradients = [
                    gradient * self._relu_derivative(z_value)
                    for gradient, z_value in zip(
                        current_gradients,
                        layer_pre_activations,
                        strict=False,
                    )
                ]

            for output_index, neuron_gradient in enumerate(current_gradients):
                layer.bias_gradients[output_index] += neuron_gradient
                for input_index, input_value in enumerate(layer_inputs):
                    layer.weight_gradients[output_index][input_index] += neuron_gradient * input_value

            propagated_gradients = [0.0 for _ in range(len(layer_inputs))]
            for input_index in range(len(layer_inputs)):
                for output_index, neuron_gradient in enumerate(current_gradients):
                    propagated_gradients[input_index] += (
                        layer.weights[output_index][input_index] * neuron_gradient
                    )
            current_gradients = propagated_gradients

        return current_gradients

    def update(self, learning_rate: float) -> None:
        """Apply accumulated gradients and then reset them to zero."""

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        for layer in self.layers:
            for output_index in range(len(layer.weights)):
                layer.biases[output_index] -= learning_rate * layer.bias_gradients[output_index]
                layer.bias_gradients[output_index] = 0.0
                for input_index in range(len(layer.weights[output_index])):
                    layer.weights[output_index][input_index] -= (
                        learning_rate * layer.weight_gradients[output_index][input_index]
                    )
                    layer.weight_gradients[output_index][input_index] = 0.0

    def save(self, path: str) -> None:
        """Save model architecture and parameters to a JSON file."""

        payload = {
            "input_size": self.input_size,
            "hidden_sizes": list(self.hidden_sizes),
            "output_size": self.output_size,
            "layers": [
                {
                    "weights": layer.weights,
                    "biases": layer.biases,
                }
                for layer in self.layers
            ],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load(self, path: str) -> None:
        """Load model architecture and parameters from a JSON file."""

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload["input_size"] != self.input_size:
            raise ValueError("Saved input_size does not match the current network.")
        if tuple(payload["hidden_sizes"]) != self.hidden_sizes:
            raise ValueError("Saved hidden_sizes do not match the current network.")
        if payload["output_size"] != self.output_size:
            raise ValueError("Saved output_size does not match the current network.")

        loaded_layers: list[LayerParameters] = []
        for layer_payload in payload["layers"]:
            weights = [[float(value) for value in row] for row in layer_payload["weights"]]
            biases = [float(value) for value in layer_payload["biases"]]
            loaded_layers.append(
                LayerParameters(
                    weights=weights,
                    biases=biases,
                    weight_gradients=[[0.0 for _ in row] for row in weights],
                    bias_gradients=[0.0 for _ in biases],
                )
            )

        self.layers = loaded_layers
        self.last_trace = None

    def inspect_forward(self, inputs: list[float]) -> ForwardTrace:
        """Run a forward pass and return cached layer-by-layer values for inspection."""

        self.forward(inputs)
        if self.last_trace is None:
            raise RuntimeError("Forward trace was not captured.")
        return self.last_trace

    def _initialize_layers(self) -> list[LayerParameters]:
        """Create all dense layers with small random initial weights."""

        sizes = (self.input_size, *self.hidden_sizes, self.output_size)
        layers: list[LayerParameters] = []

        for input_count, output_count in zip(sizes[:-1], sizes[1:], strict=False):
            weight_limit = (6.0 / max(1, input_count + output_count)) ** 0.5
            weights = [
                [
                    self._rng.uniform(-weight_limit, weight_limit)
                    for _ in range(input_count)
                ]
                for _ in range(output_count)
            ]
            biases = [0.0 for _ in range(output_count)]
            layers.append(
                LayerParameters(
                    weights=weights,
                    biases=biases,
                    weight_gradients=[[0.0 for _ in range(input_count)] for _ in range(output_count)],
                    bias_gradients=[0.0 for _ in range(output_count)],
                )
            )
        return layers

    @staticmethod
    def _dense_forward(
        inputs: list[float],
        weights: list[list[float]],
        biases: list[float],
    ) -> list[float]:
        """Compute one dense layer output."""

        outputs: list[float] = []
        for neuron_weights, bias in zip(weights, biases, strict=False):
            value = bias
            for weight, input_value in zip(neuron_weights, inputs, strict=False):
                value += weight * input_value
            outputs.append(value)
        return outputs

    @staticmethod
    def _relu(value: float) -> float:
        """Apply the ReLU activation function."""

        return value if value > 0.0 else 0.0

    @staticmethod
    def _relu_derivative(value: float) -> float:
        """Return the derivative of ReLU with respect to its pre-activation."""

        return 1.0 if value > 0.0 else 0.0


def softmax(logits: list[float]) -> list[float]:
    """Convert raw scores into a probability distribution."""

    if not logits:
        return []
    max_logit = max(logits)
    exps = [pow(2.718281828459045, logit - max_logit) for logit in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exps]


def masked_softmax(logits: list[float], allowed_indices: list[int]) -> list[float]:
    """Apply softmax only over allowed action indices and zero out the rest."""

    probabilities = [0.0 for _ in logits]
    if not allowed_indices:
        return probabilities

    masked_logits = [logits[index] for index in allowed_indices]
    masked_probabilities = softmax(masked_logits)
    for index, probability in zip(allowed_indices, masked_probabilities, strict=False):
        probabilities[index] = probability
    return probabilities


def select_index_from_probabilities(probabilities: list[float], rng: random.Random) -> int:
    """Sample one index from a categorical distribution."""

    roll = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if roll <= cumulative:
            return index
    return max(range(len(probabilities)), key=lambda idx: probabilities[idx])
