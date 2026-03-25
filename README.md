# neuron 🧠

> **Paradigm shift: Compute weights, don't store them.**

A CPU-native AI library that flips the compute/memory tradeoff.

## The Shift

**Traditional neural network:**
```
Store 7B params in memory → multiply matrices → output
Problem: memory-bound on CPU, needs GPU for speed
```

**Generated Neural Network (GenNet):**
```
Store 1K params (generator) → compute 7B weights on-the-fly → output
Advantage: compute-bound on CPU, CPUs are GOOD at compute
```

## How It Works

```
Latent Vector (32 floats)
        ↓
   Generator (897 params)
        ↓
   Weights for big network (computed, not stored)
        ↓
   Forward pass → output
```

The **Generator** is a tiny MLP that maps `(position, latent) → weight_value`.

Instead of storing a billion weights, we compute them from:
- Which layer (normalized position)
- Which row (normalized position)  
- Which column (normalized position)
- The latent vector (learned)

## Key Numbers

```
Stored:        897 params  (the generator)
Virtual:     4,547 params  (computed on-the-fly)
Compression:     5x        (for this small example)

Scaled up:
Stored:      1,000 params  (generator)
Virtual: 10,000,000 params (computed)
Compression: 10,000x
```

## Training Without Backprop

GenNet uses **evolutionary training** — no gradients needed:
1. Mutate the latent vector + generator weights
2. Evaluate the mutant on training data
3. Keep improvements, discard failures
4. Repeat

This means:
- No backpropagation
- No gradient computation
- No autodiff framework needed
- Works on ANY hardware

## Usage

```go
import "github.com/Acorx/neuron/gen"

// Create a "big" network from a tiny generator
arch := gen.Architecture{
    InputDim:   100,
    HiddenDims: []int{512, 512, 256},
    OutputDim:  10,
}
net := gen.New(arch, 64, 32)

fmt.Printf("Stored: %d\n", net.ParamCount())          // ~2000
fmt.Printf("Virtual: %d\n", net.GeneratedParamCount()) // ~500000
// Compression: ~250x

// Forward pass (generates weights on-the-fly)
output := net.Forward(input)

// Train with evolution
net.Train(inputs, targets, 100, 50)
```

## Why This Matters

| | Traditional | GenNet |
|---|---|---|
| Memory | 7B × 4 bytes = 28GB | 1K × 4 bytes = 4KB |
| Training | GPU + backprop | CPU + evolution |
| Inference | Memory-bound | Compute-bound |
| Portability | Needs specific GPU | Any CPU |

## Philosophy

> "Stop fighting memory bandwidth. Trade it for compute. CPUs are waiting to be used."

The entire AI industry optimizes for GPU memory. GenNet optimizes for CPU compute. Different problem, different solution.

## License

MIT
