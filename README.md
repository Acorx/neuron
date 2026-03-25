# neuron 🧠

> CPU-Native AI Training & Inference Library

Train and run neural networks on CPU. No GPU required.

## Key Innovations

### 1. Quantization-Aware Training (QAT)
Train directly in 4-bit precision. Weights never leave the quantized domain.
**8x less memory, 8x less compute.**

### 2. State-Free Optimizer (SignSGD)
No momentum, no variance. Just the sign of the gradient.
**0 extra memory** (vs Adam's 2x overhead).

### 3. Cache-Tiled Matrix Multiplication
Operations optimized for CPU cache hierarchy (L1/L2/L3).
**1+ GFLOPS on modern CPUs.**

### 4. Quantized Inference
Dequantize on-the-fly during matrix multiply.
**No separate inference engine needed.**

## Usage

```go
package main

import (
    "github.com/Acorx/neuron/nn"
    "github.com/Acorx/neuron/optim"
    "github.com/Acorx/neuron/train"
    "github.com/Acorx/neuron/tensor"
)

func main() {
    // Create model
    model := nn.NewSimpleModel(2, 64, 10)

    // Train with QAT
    trainer := train.Trainer{
        Model:     model,
        Optim:     optim.NewSignSGD(0.01),
        Epochs:    50,
        QATStart:  30,  // Quantize at epoch 30
        GroupSize: 32,
    }
    trainer.Train(inputs, targets, 2, 10)

    // Inference
    x := tensor.New([]float32{1.0, 2.0}, 1, 2)
    output := model.Forward(x)
}
```

## Benchmark

```
MatMul 128x128:  4ms  (1.00 GFLOPS)
MatMul 256x256: 33ms  (1.02 GFLOPS)
MatMul 512x512: 261ms (1.03 GFLOPS)
```

## Architecture

```
neuron/
├── tensor/     Quantized tensors, cache-tiled ops
├── nn/         Neural network layers (Linear, Embedding, LayerNorm)
├── optim/      Memory-efficient optimizers (SignSGD, AdamW, SGD)
├── train/      Training loop with QAT support
└── examples/   Demo: 3-class spiral classifier
```

## Philosophy

> "Change the paradigm. Train on CPU. Make AI accessible."

- Every developer should be able to train models
- No GPU? No problem.
- 4-bit precision is enough for most tasks
- Cache matters more than cores

## License

MIT
