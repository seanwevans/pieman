# Pieman

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

## Model Architecture

- **Input Layer**: 784 neurons (default)
- **Hidden Layers**: 28 layers, each with 28 neurons
- **Output Layer**: 10 neurons
- **Activation Function**: Sigmoid
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Gradient Descent

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

## Saving and Loading the Model

### Save Model

The model is saved automatically after training:

```c
save_model("model.bin");
```

### Load Model

To load a saved model:

```c
load_model("model.bin");
```

## Customization

Modify these macros in `main.c` to adjust the model:

```c
#define INPUT_SIZE_ORIGINAL 784
#define HIDDEN_LAYERS 28
#define HIDDEN_SIZE_ORIGINAL 28
#define OUTPUT_SIZE_ORIGINAL 10
#define LEARNING_RATE 1e-2
#define MAX_EPOCHS 1e9
#define MAX_ACCEPTABLE_LOSS 1e-5
#define REPORT_FREQUENCY 10000
```

## Cleanup

Remove compiled binaries:

```sh
make clean
```

## License

This project is licensed under the MIT license.
