"""Tests for the differentiable PennyLane/PyTorch integration."""

import torch
from torch import nn

from tellekynezix_qml import HybridQuantumClassifier
from tellekynezix_qml.hybrid_model import make_toy_dataset, train_classifier


def test_hybrid_model_produces_finite_batch_output() -> None:
    model = HybridQuantumClassifier(seed=7)
    features, _ = make_toy_dataset()

    output = model(features)

    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()


def test_loss_backpropagates_through_quantum_parameters() -> None:
    model = HybridQuantumClassifier(seed=7)
    features, targets = make_toy_dataset()

    loss = nn.CrossEntropyLoss()(model(features), targets)
    loss.backward()

    quantum_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("quantum.")
    ]
    assert quantum_gradients
    assert all(gradient is not None for gradient in quantum_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in quantum_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in quantum_gradients)


def test_optimizer_updates_trainable_parameters() -> None:
    model = HybridQuantumClassifier(seed=7)
    features, targets = make_toy_dataset()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(features), targets)
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(initial_parameters[name], parameter.detach())
        for name, parameter in model.named_parameters()
        if name.startswith("quantum.")
    )


def test_training_is_reproducible_and_reduces_loss() -> None:
    features, targets = make_toy_dataset()
    first_model = HybridQuantumClassifier(seed=7)
    second_model = HybridQuantumClassifier(seed=7)

    first_losses = train_classifier(first_model, features, targets, epochs=10)
    second_losses = train_classifier(second_model, features, targets, epochs=10)

    assert first_losses == second_losses
    assert first_losses[-1] < first_losses[0]
