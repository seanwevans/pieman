# Pieman
<img width="256" alt="A pieman" src="https://github.com/user-attachments/assets/69ddd1fd-1329-4820-96f8-677d478a07c5" />


Pieman is a simple neural network with a configurable number of hidden layers,
optimized using AVX vectorization for performance.

## Features

- **Multi-layer Neural Network**: Configurable number of hidden layers.
- **AVX Vectorization**: Uses AVX instructions to accelerate forward and backward propagation.
- **Aligned Memory Allocation**: Ensures optimal data alignment for SIMD operations.
- **Model Persistence**: Supports saving and loading model weights and biases.
- **Training Monitoring**: Reports loss at set intervals.

## Requirements

- **Compiler**: GCC with AVX support (`-mavx -mavx2`)
- **Libraries**: Standard C libraries (`stdio.h`, `stdlib.h`, `math.h`, `immintrin.h`)
- **OS**: Linux/macOS/Windows (with proper AVX support)

For the optional `beta.py` demo, install the Python dependencies:

```sh
pip install -r requirements.txt
```

## Compilation

Use the provided `Makefile` to build the project:

```sh
make
```

## Usage

Run the executable:

```sh
./nnet
```

The training will start and display epoch-wise loss values. The trained model is saved as `model.bin`.

## Command-line Options

You can override default parameters at runtime:

| Option | Description | Default |
| ------ | ----------- | ------- |
| `-l`   | Number of hidden layers | `28` |
| `-s`   | Neurons per hidden layer | `28` |
| `-r`   | Learning rate | `1e-2` |
| `-e`   | Maximum epochs | `1e9` |

## Model Architecture

- **Input Layer**: 784 neurons (default)
- **Hidden Layers**: 28 layers, each with 28 neurons
- **Output Layer**: 10 neurons
- **Activation Function**: Sigmoid
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Gradient Descent

All layers allocate memory using 32‑byte alignment so that 256‑bit AVX
instructions can operate on four `double` values at a time. The code pads each
layer's size to a multiple of four and performs matrix multiplications using
`_mm256_load_pd` and `_mm256_fmadd_pd` for efficient fused multiply‑add
operations. A helper `horizontal_sum` function reduces the AVX registers to a
scalar result, enabling the forward and backward passes to remain vectorized.

## Example Output

```
👨‍🎓 784 params
┏━━━━━━━━━━━━━━━━━━━━━━┓
┃   Epoch  Loss        ┃
┣━━━━━━━━━━━━━━━━━━━━━━┫
┃       1  0.306073244 ┃
┃   10000  0.000067958 ┃
┃   20000  0.000032532 ┃
┃   30000  0.000021231 ┃
┃   40000  0.000015706 ┃
┃   50000  0.000012440 ┃
┃   60000  0.000010286 ┃
┃   61650  0.000010000 ┃
┗━━━━━━━━━━━━━━━━━━━━━━┛
🏋️ Training Time: 1.16 seconds
💾 Model saved: model.bin
```

The snippet above was captured after building the project with `make` and
running `./nnet`. Training stops once the loss drops below the threshold defined
by `MAX_ACCEPTABLE_LOSS`, which with the default settings happens at roughly
61k epochs.

## Saving and Loading the Model

### Save Model

The model is saved automatically after training. Check the return value to
ensure the operation succeeded:

```c
int rc = save_model("model.bin");
if (rc < 0) {
    fprintf(stderr, "Failed to save model (error %d)\n", rc);
}
// On success you'll see:
// 💾 model.bin
// On failure an error message is printed and rc will be negative.
```

### Load Model

To load a saved model and verify it loads correctly:

```c
rc = load_model("model.bin");
if (rc < 0) {
    fprintf(stderr, "Failed to load model (error %d)\n", rc);
}
// If loading succeeds no message is printed by the function.
```

Call this after `initialize_network()` to restore the weights and biases before
training or inference.

## Customization

Modify these macros in `main.c` to adjust the default model:

```c
#define INPUT_SIZE_ORIGINAL 784
#define DEFAULT_HIDDEN_LAYERS 28
#define DEFAULT_HIDDEN_SIZE_ORIGINAL 28
#define OUTPUT_SIZE_ORIGINAL 10
#define DEFAULT_LEARNING_RATE 1e-2
#define DEFAULT_MAX_EPOCHS 1e9
#define MAX_ACCEPTABLE_LOSS 1e-5
#define REPORT_FREQUENCY 10000
```

## Optional: BetaNet Example

The repository also includes `beta.py`, a small PyTorch script that
demonstrates a modern framework for neural networks. To try it out, install the
Python dependencies and run the script:

```sh
pip install numpy torch matplotlib scipy
python3 beta.py
```

This step is completely optional and may require a machine with sufficient
memory and AVX2 support.

## Cleanup

Remove compiled binaries:

```sh
make clean
```

## Testing

Run the minimal test suite with:

```sh
make test
```

## License

This project is licensed under the MIT license.

## `beta.py`

`beta.py` is a small PyTorch example that learns the parameters of a Beta
distribution from randomly generated inputs. It demonstrates how to build a
simple neural network using `numpy`, `torch`, `scipy`, and `matplotlib`.
