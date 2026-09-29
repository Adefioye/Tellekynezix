"""Train the minimal hybrid quantum/classical classifier."""

import torch

from tellekynezix_qml import BACKEND_NAME, HybridQuantumClassifier
from tellekynezix_qml.hybrid_model import make_toy_dataset, train_classifier


def main() -> None:
    model = HybridQuantumClassifier(seed=7)
    features, targets = make_toy_dataset()
    losses = train_classifier(model, features, targets)

    model.eval()
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()

    if losses[-1] >= losses[0]:
        raise RuntimeError("Hybrid model training did not reduce the loss")

    print(f"Quantum backend: {BACKEND_NAME}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Training accuracy: {accuracy:.0%}")
    print("Hybrid PennyLane/PyTorch training completed successfully.")


if __name__ == "__main__":
    main()
