# neuron 🧠

> **Neural networks from mathematical formulas. No GPU needed.**

Instead of storing billions of weights, we compute them from a tiny formula.
64 parameters → 99 weights generated on-the-fly.

## What This Is

A research prototype exploring: **can we generate neural network weights from compact mathematical representations instead of storing them?**

### The Formula

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

- **αₖ** (64 numbers): learnable coefficients ← the only thing we store
- **ω, φ** (fixed): random Fourier basis functions
- **Result**: generates any weight matrix from 64 numbers

### Results

```
Stored:    64 params (Fourier coefficients)
Virtual:   99 params (computed weights)
Training:  Finite-difference gradients (no backprop!)
Loss:      1.001 → 0.660 (300 epochs, CPU only)
Accuracy:  33% → 51.1% (spiral classification)
Time:      5 minutes on CPU
```

## Usage

```go
import "github.com/Acorx/neuron/fourier"

// Create network: 2 inputs → 16 hidden → 3 outputs
// Generated from 64 Fourier coefficients
net := fourier.New(2, 16, 3, 1, 32)

// Train with finite-difference gradients
net.Train(inputs, targets, 300, 0.03)

// Inference
output := net.Forward(input)
```

## Project Structure

```
neuron/
├── fourier/
│   └── fourier.go    ← The core: Fourier weight generation + FD training
├── examples/
│   └── demo.go       ← Spiral classification demo
├── README.md
├── go.mod
└── LICENSE
```

## How It Works

1. **Weight Generation**: For each layer, compute all weights using the Fourier formula
   - Position (row, col, layer) → Fourier features → dot product with α → weight
   - Vectorized: all weights for a layer in one pass

2. **Training**: Finite-difference gradient estimation
   - For each parameter αₖ: compute loss(αₖ+ε) and loss(αₖ-ε)
   - Gradient ≈ (loss⁺ - loss⁻) / 2ε
   - Update: αₖ -= lr × gradient
   - No backpropagation needed!

3. **Inference**: Generate weights on-the-fly, multiply with input

## Why This Matters

| | Traditional | neuron |
|---|---|---|
| Weight storage | GB of memory | Bytes |
| Training | GPU + backprop | CPU + finite diff |
| Inference | Memory-bound | Compute-bound |
| Portability | Needs GPU | Any CPU |

## Limitations (Honest)

- Training is slow (finite diff needs 2×N forward passes per step)
- Accuracy not competitive with traditional NNs yet
- Works best for small models (1K-100K virtual params)
- Research prototype, not production-ready

## Next Steps

- [ ] Adaptive learning rate scheduling
- [ ] More Fourier components for expressiveness
- [ ] Test on MNIST / harder tasks
- [ ] Vectorize weight generation with SIMD
- [ ] Hybrid: fractal prior + residual corrections

## License

MIT
