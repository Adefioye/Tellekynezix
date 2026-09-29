"""Small quantum circuits used to verify the local PennyLane environment."""

import pennylane as qml

BACKEND_NAME = "default.qubit"
N_QUBITS = 2

_bell_device = qml.device(BACKEND_NAME, wires=N_QUBITS)


@qml.qnode(_bell_device)
def bell_state_probabilities():
    """Prepare a Bell state and return exact computational-basis probabilities.

    The state is (|00> + |11>) / sqrt(2), so the ideal probability vector is
    [0.5, 0.0, 0.0, 0.5]. The simulator uses analytic execution because no
    finite shot count is configured.
    """

    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=range(N_QUBITS))

