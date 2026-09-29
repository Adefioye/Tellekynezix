# Tellekynezix Quantum Machine Learning Environment

This component provides a reproducible local environment for introductory
quantum machine learning experiments. It uses PennyLane for quantum circuits,
PyTorch for classical machine learning, and PennyLane's built-in
`default.qubit` simulator. No quantum-hardware account, API key, or cloud
service is required.

## Requirements

- Python 3.11, 3.12, or 3.13
- `pip`

The project pins PennyLane, PyTorch, tqdm, and pytest to tested versions in
`pyproject.toml`. A virtual environment keeps these dependencies separate from
the rest of Tellekynezix.

## Installation

Run these commands from the repository root.

### macOS and Linux

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./quantum-ml[dev]"
python -m pip check
```

Use `python3.12` or `python3.13` in the first command if that is the supported
version installed on your machine.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".\quantum-ml[dev]"
python -m pip check
```

## Run the Bell circuit

```sh
python quantum-ml/examples/bell_circuit.py
```

The example creates the Bell state
`(|00> + |11>) / sqrt(2)` and should report probabilities close to:

```text
Quantum backend: default.qubit
P(|00>) = 0.500000
P(|01>) = 0.000000
P(|10>) = 0.000000
P(|11>) = 0.500000
Bell circuit executed successfully.
```

The simulator runs analytically, so these probabilities do not contain
finite-shot sampling noise. A progress bar reports when circuit execution is
complete.

## Run the hybrid model

```sh
python quantum-ml/examples/train_hybrid_model.py
```

The example trains a small classifier on an XOR-style dataset. A classical
PyTorch layer converts each input into two rotation angles, a trainable
two-qubit circuit generates computational-basis probabilities, and another
PyTorch layer produces class scores. PyTorch backpropagation updates both the
classical and quantum parameters.

The training command displays epoch progress and the latest loss value.

Exact loss values can vary if the example seed or dependency versions change,
but the output should end with:

```text
Hybrid PennyLane/PyTorch training completed successfully.
```

This example validates gradient flow and environment integration; it is not a
benchmark or a production classifier.

## Run the tests

```sh
python -m pytest quantum-ml/tests -v
```

The tests verify that:

- The Bell-state probabilities are correct and normalized.
- The hybrid model produces finite outputs with the expected shape.
- Loss gradients reach the trainable quantum parameters.
- An optimizer step changes model parameters.
- Seeded training is reproducible and reduces the toy loss.

The same checks run in GitHub Actions with Python 3.11, 3.12, and 3.13.

## Project layout

```text
quantum-ml/
├── examples/                  # Executable Bell and hybrid demonstrations
├── src/quantum_ml/            # Reusable circuit and model code
├── tests/                     # Fast local-simulator tests
├── pyproject.toml             # Python and dependency definition
└── README.md                  # This guide
```

## Troubleshooting

If imports fail, confirm that the virtual environment is active and reinstall
the component:

```sh
python -m pip install -e "./quantum-ml[dev]"
```

If the environment was created with an unsupported Python version, remove the
`.venv` directory, recreate it with Python 3.11 through 3.13, and reinstall.
The `.venv` directory is ignored by Git and must not be committed.

Deactivate the environment when finished:

```sh
deactivate
```
