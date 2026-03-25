# neuron 🧠

> **Research: Neural Networks from Mathematical Formulas**

Three approaches tested to generate neural network weights from compact representations:

## Approaches Tested

### 1. Fractal Neural Networks (Fourier)
**Formula:** W(i,j,l) = Σₖ αₖ·sin(ωₖᵢ·i + ωₖⱼ·j + ωₖₗ·l + φₖ)

- 40 params → 20,746 virtual (519x compression)
- Inference: 17ms for 80K weights
- ❌ Training doesn't converge (evolutionary too slow)

### 2. Neural Cellular Automata
**Concept:** Evolve a grid using learned rules → grid is the weight matrix

- 89 params → 2,794 virtual (31x compression)
- ❌ Very slow (19s per epoch)
- ❌ Training doesn't converge

### 3. Hypernetworks (v1)
**Concept:** Small generator network produces weights for larger network

- 897 params → 4,547 virtual (5x compression)
- ❌ Training too slow

## What We Learned

**What works:**
- ✅ Compression is real and measurable (up to 519x)
- ✅ Forward pass generates valid weights
- ✅ Scales linearly with network depth
- ✅ Zero storage for weights (only formula)

**What doesn't work:**
- ❌ Evolutionary training doesn't converge fast enough
- ❌ Weight generation is inherently slower than lookup
- ❌ The formula needs to be more expressive

**The fundamental tradeoff:**
- Traditional: 4GB memory, 1ms inference (GPU)
- Fractal: 32 bytes memory, 17ms inference (CPU)
- We're trading memory for compute, but compute is also slower

## The Real Challenge

The bottleneck isn't compression — it's **training convergence**.

Evolutionary strategies need thousands of evaluations to find good parameters. Backpropagation needs ~100. That's a 50x gap.

**Possible solutions (not yet implemented):**
1. Finite-difference gradient estimation (2x slower than backprop, but works)
2. Hybrid approach: pre-train small network, then compress to fractal
3. Better search algorithms (CMA-ES, Bayesian optimization)
4. Hardware acceleration for weight generation (SIMD, GPU)

## Where This Could Work

- Small specialized models (1-10M params)
- Edge devices without GPUs
- Extreme compression for deployment
- Research into alternative learning paradigms

## Where This Won't Work (Yet)

- Large language models (billions of params)
- Real-time applications needing GPU speed
- Tasks requiring high accuracy training

## Philosophy

> "We proved the concept. The math works. The training is the open problem."

This is research, not a product. The contribution is showing that weight generation from formulas is possible and measurable. The next step is finding better training methods.

## License

MIT
