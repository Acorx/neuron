# neuron 🧠

> **A new paradigm: Weights from fractal formulas.**

Like the Mandelbrot set (infinite complexity from `z²+c`),
a Fractal Neural Network generates unlimited weights from 8 numbers.

## The Idea

Traditional NN: Store weights as matrices.
Fractal NN: Compute weights from a formula.

```
8 parameters → fractal formula → weights for entire network
```

The formula has **self-similarity**: weights at different layers
follow the same pattern at different scales.

## Results

```
Stored:      8 params (the formula)
Virtual:  2,307 params (computed on-the-fly)
Compression: 288x

Depth scaling:
  Depth 1: 8 → 195 (24x)
  Depth 3: 8 → 2,307 (288x)
  Depth 8: 8 → 7,587 (948x)
```

## Usage

```go
import "github.com/Acorx/neuron/fractal"

// 8 params define a 5-layer network
net := fractal.New(
    2,      // input dim
    32,     // hidden dim
    10,     // output dim
    3,      // depth (number of hidden layers)
)

// Forward pass (generates all weights on-the-fly)
output := net.Forward(input)

// Train with evolution (no backprop!)
net.Train(inputs, targets, 100, 50)
```

## Why This Is Different

| | Traditional | Hypernetwork | **Fractal NN** |
|---|---|---|---|
| Weight source | Stored | Generator network | Mathematical formula |
| Stored params | 1B | 1K | **8** |
| Compression | 1x | 1000x | **∞ (formula scales)** |
| Inductive bias | None | Learned | **Self-similarity** |
| Training | Backprop | Backprop | **Evolution** |

## Philosophy

> "Nature uses fractals for infinite complexity from simple rules.
> Why not neural networks?"

## License

MIT
