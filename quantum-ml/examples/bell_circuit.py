"""Run the local two-qubit Bell-state circuit."""

import math

from quantum_ml import BACKEND_NAME, bell_state_probabilities
from tqdm import tqdm


def main() -> None:
    with tqdm(total=1, desc="Running Bell circuit", unit="circuit") as progress:
        probabilities = tuple(float(value) for value in bell_state_probabilities())
        progress.update()
    expected_probabilities = (0.5, 0.0, 0.0, 0.5)
    labels = ("|00>", "|01>", "|10>", "|11>")

    print(f"Quantum backend: {BACKEND_NAME}")
    for label, probability in zip(labels, probabilities, strict=True):
        print(f"P({label}) = {probability:.6f}")

    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
        raise RuntimeError("Bell-state probabilities do not sum to one")
    if not all(
        math.isclose(actual, expected, abs_tol=1e-8)
        for actual, expected in zip(
            probabilities, expected_probabilities, strict=True
        )
    ):
        raise RuntimeError("Bell circuit did not produce the expected state")

    print("Bell circuit executed successfully.")


if __name__ == "__main__":
    main()
