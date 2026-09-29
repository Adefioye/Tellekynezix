"""A minimal differentiable PennyLane/PyTorch hybrid classifier."""

from collections.abc import Sequence

import pennylane as qml
import torch
from torch import nn

from .circuits import BACKEND_NAME, N_QUBITS

DEFAULT_SEED = 7
DEFAULT_QUANTUM_LAYERS = 2
N_QUANTUM_FEATURES = 2**N_QUBITS


def _make_quantum_layer(n_layers: int) -> qml.qnn.TorchLayer:
    """Create a trainable two-qubit layer backed by a local simulator."""

    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")

    device = qml.device(BACKEND_NAME, wires=N_QUBITS)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        # A single probability measurement gives TorchLayer a stable
        # (batch_size, 2**N_QUBITS) result across PennyLane versions.
        return qml.probs(wires=range(N_QUBITS))

    weight_shapes = {"weights": (n_layers, N_QUBITS)}
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class HybridQuantumClassifier(nn.Module):
    """Classical preprocessing and output layers around a quantum layer."""

    def __init__(
        self,
        input_features: int = 2,
        output_classes: int = 2,
        quantum_layers: int = DEFAULT_QUANTUM_LAYERS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be at least 1")
        if output_classes < 2:
            raise ValueError("output_classes must be at least 2")

        # Keep construction deterministic without altering the caller's global
        # random-number-generator state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.classical_input = nn.Linear(input_features, N_QUBITS)
            self.quantum = _make_quantum_layer(quantum_layers)
            self.classical_output = nn.Linear(N_QUANTUM_FEATURES, output_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores for a batch of input features."""

        if features.ndim != 2:
            raise ValueError("features must have shape (batch_size, input_features)")

        angles = torch.tanh(self.classical_input(features)) * torch.pi
        quantum_features = self.quantum(angles)
        return self.classical_output(quantum_features)


def make_toy_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic XOR-style dataset for the training example."""

    features = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    return features, targets


def train_classifier(
    model: HybridQuantumClassifier,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 20,
    learning_rate: float = 0.1,
) -> Sequence[float]:
    """Train the example model and return the initial and per-epoch losses."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    with torch.no_grad():
        initial_loss = loss_function(model(features), targets).item()
    loss_history = [initial_loss]

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_history.append(loss_function(model(features), targets).item())

    return loss_history
