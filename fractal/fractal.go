// Package fractal — Fractal Neural Networks v2
//
// Improved fractal formula using learnable Fourier features.
// More expressive, faster convergence, better accuracy.
package fractal

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// FractalNet v2: Learnable Fourier fractal formula
// Instead of fixed sin/cos, we learn the frequencies and phases.
// Formula: W(i,j,layer) = Σₖ αₖ * sin(ωₖᵢ * i + ωₖⱼ * j + ωₖₗ * layer + φₖ)
// Parameters: [α₁...αₙ, ω₁ᵢ, ω₁ⱼ, ω₁ₗ, φ₁, ..., ωₙᵢ, ωₙⱼ, ωₙₗ, φₙ]
type FractalNet struct {
	// Fourier components: each component has amplitude, 3 frequencies, and phase
	// Stored as flat array: [amp, freq_i, freq_j, freq_l, phase] per component
	K     int // Number of Fourier components
	Params []float32 // 5*K parameters

	InputDim  int
	OutputDim int
	HiddenDim int
	Depth     int
}

// New creates a fractal network with K Fourier components.
// K=8 means 40 parameters. K=16 means 80 parameters.
func New(inputDim, hiddenDim, outputDim, depth, k int) *FractalNet {
	params := make([]float32, 5*k)
	// Better init: amplitudes small, frequencies moderate, phases random
	for i := 0; i < k; i++ {
		base := i * 5
		params[base] = randn() * 0.1    // amplitude
		params[base+1] = randn() * 2.0  // freq_i
		params[base+2] = randn() * 2.0  // freq_j
		params[base+3] = randn() * 2.0  // freq_l
		params[base+4] = randn() * 3.14 // phase
	}
	return &FractalNet{
		K:         k,
		Params:    params,
		InputDim:  inputDim,
		OutputDim: outputDim,
		HiddenDim: hiddenDim,
		Depth:     depth,
	}
}

// Weight computes a single weight using the Fourier fractal formula.
// This is the core: a sum of sinusoidal components with learnable frequencies.
func (f *FractalNet) Weight(row, col, layer, inDim, outDim int) float32 {
	r := float32(row+1) / float32(inDim+1)
	c := float32(col+1) / float32(outDim+1)
	l := float32(layer+1) / float32(f.Depth+1)

	w := float32(0)
	for k := 0; k < f.K; k++ {
		base := k * 5
		amp := f.Params[base]
		fi := f.Params[base+1]
		fj := f.Params[base+2]
		fl := f.Params[base+3]
		phi := f.Params[base+4]
		w += amp * sin(fi*r + fj*c + fl*l + phi)
	}
	return w * 0.03
}

// Bias computes bias using same formula with col=0 marker.
func (f *FractalNet) Bias(idx, layer, dim int) float32 {
	x := float32(idx+1) / float32(dim+1)
	l := float32(layer+1) / float32(f.Depth+1)

	b := float32(0)
	for k := 0; k < f.K; k++ {
		base := k * 5
		b += f.Params[base] * sin(f.Params[base+1]*x + f.Params[base+3]*l + f.Params[base+4])
	}
	return b * 0.01
}

// Forward computes output by generating all weights on-the-fly.
func (f *FractalNet) Forward(input []float32) []float32 {
	x := input
	dims := f.dims()

	for layer := 0; layer < len(dims)-1; layer++ {
		inD, outD := dims[layer], dims[layer+1]
		output := make([]float32, outD)

		for j := 0; j < outD; j++ {
			sum := f.Bias(j, layer, outD)
			for i := 0; i < inD; i++ {
				sum += x[i] * f.Weight(i, j, layer, inD, outD)
			}
			if layer < len(dims)-2 {
				sum = relu(sum)
			}
			output[j] = sum
		}
		x = output
	}
	return x
}

func (f *FractalNet) dims() []int {
	d := []int{f.InputDim}
	for i := 0; i < f.Depth; i++ {
		d = append(d, f.HiddenDim)
	}
	return append(d, f.OutputDim)
}

func (f *FractalNet) ParamCount() int { return len(f.Params) }

func (f *FractalNet) VirtualParamCount() int {
	count := 0
	dims := f.dims()
	for i := 0; i < len(dims)-1; i++ {
		count += dims[i]*dims[i+1] + dims[i+1]
	}
	return count
}

// Train uses CMA-ES inspired evolutionary strategy (better than random mutations).
func (f *FractalNet) Train(inputs [][]float32, targets [][]float32, epochs, popSize int) {
	n := len(f.Params)
	fmt.Printf("🌀 Fractal NN v2 (Fourier)\n")
	fmt.Printf("   Components: %d (%d params)\n", f.K, n)
	fmt.Printf("   Virtual: %d params (%.0fx compression)\n",
		f.VirtualParamCount(), float64(f.VirtualParamCount())/float64(n))
	fmt.Printf("   Training: %d samples, %d epochs, pop=%d\n\n",
		len(inputs), epochs, popSize)

	best := make([]float32, n)
	copy(best, f.Params)
	bestLoss := f.loss(inputs, targets)

	// Adaptive step size
	sigma := float32(0.5)

	for epoch := 0; epoch < epochs; epoch++ {
		start := time.Now()

		// Sort population by fitness
		type mutant struct {
			params []float32
			loss   float32
		}
		pop := make([]mutant, popSize)

		var wg sync.WaitGroup
		for p := 0; p < popSize; p++ {
			wg.Add(1)
			go func(p int) {
				defer wg.Done()
				m := make([]float32, n)
				for i := range m {
					m[i] = best[i] + randn()*sigma
				}
				f.unpack(m)
				pop[p] = mutant{m, f.loss(inputs, targets)}
			}(p)
		}
		wg.Wait()

		// Select best and recombine (CMA-ES style)
		for i := 0; i < popSize; i++ {
			for j := i + 1; j < popSize; j++ {
				if pop[j].loss < pop[i].loss {
					pop[i], pop[j] = pop[j], pop[i]
				}
			}
		}

		// Take top 25% and average (recombination)
		elite := max(1, popSize/4)
		newBest := make([]float32, n)
		for i := 0; i < elite; i++ {
			for j := 0; j < n; j++ {
				newBest[j] += pop[i].params[j] / float32(elite)
			}
		}

		newLoss := pop[0].loss
		if newLoss < bestLoss {
			copy(best, newBest)
			bestLoss = newLoss
			sigma *= 1.1
		} else {
			sigma *= 0.95 // reduce on failure
		}
		sigma = clamp(sigma, 0.01, 2.0)

		f.unpack(best)
		if epoch%5 == 0 || epoch == epochs-1 {
			fmt.Printf("   Epoch %3d | Loss: %.6f | σ: %.3f | %v\n",
				epoch, bestLoss, sigma, time.Since(start).Round(time.Millisecond))
		}
	}
}

func (f *FractalNet) unpack(p []float32) { copy(f.Params, p) }

func (f *FractalNet) loss(inputs, targets [][]float32) float32 {
	total := float32(0)
	for i, input := range inputs {
		out := f.Forward(input)
		for j := range out {
			d := out[j] - targets[i][j]
			total += d * d
		}
	}
	return total / float32(len(inputs))
}

// Helpers
var seed uint64 = 42
func randn() float32 { seed = seed*6364136223846793005 + 1442695040888963407; return float32(seed>>33)/float32(1<<31) - 1.0 }
func sin(x float32) float32 { return float32(math.Sin(float64(x))) }
func relu(x float32) float32 { if x > 0 { return x }; return 0 }
func max(a, b int) int { if a > b { return a }; return b }
func clamp(x, lo, hi float32) float32 { if x < lo { return lo }; if x > hi { return hi }; return x }
