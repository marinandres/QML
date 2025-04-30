# Quantum Machine Learning Comparison

This project aims to compare the performance of classical and quantum machine learning models using the Airline Passenger dataset. It implements a Classical Recurrent Neural Network (RNN) using PyTorch and a Quantum Recurrent Neural Network (QRNN) using Qiskit. The goal is to analyze and visualize the differences in model architecture, performance, and the underlying quantum systems.

## Project Structure

```
quantum-ml-comparison
├── src
│   ├── data
│   │   └── airline_passengers.py       # Functions for loading and preprocessing the dataset
│   ├── models
│   │   ├── classical_rnn.py             # Classical RNN implementation
│   │   └── quantum_rnn.py                # Quantum RNN implementation
│   ├── utils
│   │   ├── data_preprocessing.py         # Utility functions for data preprocessing
│   │   └── visualization.py               # Functions for visualizing model results
│   ├── train.py                          # Script for training the models
│   └── evaluate.py                       # Script for evaluating model performance
├── notebooks
│   ├── model_comparison.ipynb            # Jupyter notebook for model performance comparison
│   └── quantum_circuit_visualization.ipynb # Jupyter notebook for visualizing quantum circuits
├── config
│   └── ibm_config.json                   # Configuration for IBM Quantum Platform
├── requirements.txt                      # Python package dependencies
├── .gitignore                            # Files and directories to ignore by Git
└── README.md                             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd quantum-ml-comparison
   ```

2. **Install the required packages:**
   Create a virtual environment and install the dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd quantum-ml-comparison

# On macOS/Linux
python -m venv env
source env/bin/activate

# On Windows
python -m venv env
.\env\Scripts\activate

{
    "api_key": "YOUR_IBM_QUANTUM_API_KEY",
    "url": "https://quantum-api.ibm.com/api",
    "backend": "ibmq_qasm_simulator",
    "project_id": "YOUR_PROJECT_ID",
    "max_execution_time": 300,
    "shots": 1024
}

python -m src.train --model classical --hidden_size 32 --epochs 50
python -m src.train --model quantum --hidden_size 32 --n_qubits 5 --n_qlayers 1 --epochs 50

Available parameters:

--model: Model type (classical or quantum)
--input_size: Input feature size (default: 1)
--hidden_size: Hidden state size (default: 32)
--n_qubits: Number of qubits for quantum model (default: 5)
--n_qlayers: Number of quantum layers (default: 1)
--batch_size: Batch size for training (default: 32)
--seq_length: Sequence length for time series (default: 12)
--lr: Learning rate (default: 0.01)
--epochs: Number of training epochs (default: 100)
--save_path: Path to save model and results (default: 'models')

## Evaluating Models
Evaluate the classical RNN model:

python -m src.evaluate --model classical --model_path models/classical_rnn_h32.pth --hidden_size 32
python -m src.evaluate --model quantum --model_path models/quantum_rnn_h32_q5_l1.pth --hidden_size 32 --n_qubits 5 --n_qlayers 1

## Visualizing Results
Use Jupyter notebooks for interactive visualizations:

jupyter notebook notebooks/model_comparison.ipynb
jupyter notebook notebooks/quantum_circuit_visualization.ipynb

## Usage

- To train the models, run:
  ```
  python src/train.py
  ```

- To evaluate the models, run:
  ```
  python src/evaluate.py
  ```

- Use the Jupyter notebooks in the `notebooks` directory for model comparison and quantum circuit visualization.

## Conclusion

This project provides a framework for exploring the capabilities of quantum machine learning in comparison to classical approaches. By utilizing the Airline Passenger dataset, it aims to highlight the strengths and weaknesses of both methodologies in a clear and visual manner.


## Implementation Details
Classical RNN
The classical RNN is a simple recurrent neural network implemented in PyTorch. It processes time series data and predicts future values.

Quantum RNN
The quantum RNN extends the classical RNN by integrating a quantum circuit layer:

Classical Pre-processing: Input data is processed via a classical linear layer
Quantum Processing: Data passes through a parameterized quantum circuit using Qiskit
Measurement: Quantum circuit outputs are measured as expectation values of the Z-operator
Classical Post-processing: Quantum outputs are processed via another classical linear layer
The quantum circuit consists of:

Data encoding using rotation gates
Parameterized rotation gates (RX, RZ, RX) for each qubit
Entanglement via CNOT gates in a cyclic pattern
Additional parameterized rotation gates
Measurements in the Z-basis
## Dataset
The project uses the Airline Passenger dataset, which contains monthly totals of international airline passengers from 1949 to 1960. The data is univariate time series, making it suitable for testing sequential models.

## Results
After training both models, the results are saved in the models directory:

Model weights (.pth files)
Training history (.pkl files)
Loss curves (.png files)
The evaluation script generates:

Performance metrics (MSE, RMSE, MAE)
Inference time statistics
Prediction vs. actual value plots

## Quantum Circuit Explanation

Explanation of the Quantum Circuit
The quantum circuit shown in the image is a variational quantum circuit designed for a Quantum Recurrent Neural Network (QRNN). Here's a detailed breakdown of its components:

1. Input Encoding
Each qubit (q0 to q4) is initialized in the ground state |0⟩.
The input values [0.5, 0.6, 0.7, 0.8, 0.9] are encoded into the circuit using RX gates. These gates rotate the qubit state around the X-axis of the Bloch sphere by an angle proportional to the input value. Mathematically: [ RX(\theta) = e^{-i \frac{\theta}{2} X} ] where ( X ) is the Pauli-X matrix, and ( \theta ) is the rotation angle (e.g., 0.5 for q0).
2. Variational Layers
The circuit has 1 layer of trainable parameters (weights), represented by additional RX and RZ gates. These gates apply rotations around the X-axis and Z-axis, respectively: [ RZ(\phi) = e^{-i \frac{\phi}{2} Z} ] where ( Z ) is the Pauli-Z matrix, and ( \phi ) is the rotation angle.
Each qubit has its own set of trainable parameters for RX and RZ gates. For example, q0 has RX(0.1), RZ(0.2), and RX(0.3).
3. Entanglement
CNOT gates (blue "+" symbols) are used to entangle the qubits. Entanglement creates quantum correlations between qubits, allowing the circuit to capture complex relationships in the data.
The entanglement pattern in the circuit is a linear chain, where each qubit is entangled with its neighbor.
4. Measurement
After the variational layer and entanglement, the qubits are measured in the computational basis (|0⟩ and |1⟩). The measurement results are classical bitstrings (e.g., 10010).
Explanation of the Measurement Results
The measurement results represent the frequencies of different bitstrings observed after simulating the circuit. Here's how to interpret them:

1. Quantum State Evolution
The quantum state of the system evolves as the gates are applied. For example:
After the RX gates, the state of each qubit is a superposition of |0⟩ and |1⟩.
The RZ gates and entanglement operations further modify the amplitudes and phases of the quantum state.
The final state of the system is a superposition of all possible bitstrings (e.g., |00000⟩, |00001⟩, ..., |11111⟩), with complex amplitudes ( \alpha_i ) for each bitstring ( |i⟩ ): [ |\psi⟩ = \sum_{i=0}^{2^n - 1} \alpha_i |i⟩ ] where ( |\alpha_i|^2 ) is the probability of measuring bitstring ( |i⟩ ).
2. Measurement Probabilities
When the circuit is measured, the quantum state collapses to one of the bitstrings with a probability proportional to ( |\alpha_i|^2 ).
For example, the bitstring 00000 was observed 220 times, which suggests it has the highest probability.
3. Results Breakdown
The measurement results are:

Each key represents a bitstring (e.g., 10010), and the value represents the number of times it was observed in the simulation.
The most frequent bitstrings (e.g., 00000, 10001, 11110) correspond to the quantum states with the highest probabilities.
4. Mathematical Representation
The measurement results can be summarized as a probability distribution: [ P(|i⟩) = \frac{\text{count}(|i⟩)}{\text{total counts}} ] For example:

Total counts = ( 1024 ) (sum of all values).
Probability of 00000: [ P(|00000⟩) = \frac{220}{1024} \approx 0.215 ]
Quantum Terms and Concepts
Superposition: Each qubit is in a superposition of |0⟩ and |1⟩ after the RX gates.
Entanglement: The CNOT gates create entanglement, which is essential for capturing correlations in the data.
Measurement: The quantum state collapses to a classical bitstring upon measurement, with probabilities determined by the amplitudes of the quantum state.
Key Insights for Presentation
The circuit encodes input data into quantum states, applies trainable transformations, and measures the results.
The measurement results reflect the probabilistic nature of quantum mechanics.
The most frequent bitstrings (e.g., 00000) correspond to the quantum states with the highest probabilities, which are influenced by the input data and trainable parameters.