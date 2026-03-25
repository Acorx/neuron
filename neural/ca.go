// Package neural implements Neural Cellular Automata for weight generation.
//
// BREAKTHROUGH IDEA: Use cellular automata to generate neural network weights.
//
// Conway's Game of Life produces infinite complexity from 2 simple rules.
// We LEARN the rules to produce the exact weight patterns we need.
//
// The grid evolves: seed → rule¹ → rule² → ... → ruleᴺ = weights
// Only the seed and the rule are stored. The weights are computed.
//
// A cellular automaton can be Turing-complete.
// A learned CA can theoretically generate ANY weight pattern.
package neural

import (
	"fmt"
	"math"
	"time"
)

// CA is a Neural Cellular Automata weight generator.
type CA struct {
	// The learnable rule: maps (cell, neighbors) → new cell value
	// Rule is parameterized by a small set of coefficients
	// Rule(cell, N, NE, E, SE, S, SW, W, NW) = Σₖ wₖ * activation(Σⱼ aₖⱼ * inputⱼ + bₖ)
	// This is a tiny 1-hidden-layer MLP applied to each cell
	RuleWeights1 []float32 // (9 inputs) × (hidden) — input to hidden
	RuleBias1    []float32 // hidden
	RuleWeights2 []float32 // (hidden) × 1 — hidden to output
	RuleBias2    float32   // output bias

	HiddenSize int // Hidden layer size in the rule

	// Network architecture
	Depth     int // How many CA steps to evolve
	InputDim  int
	OutputDim int
	HiddenDim int
}

// NewCA creates a new Neural Cellular Automata weight generator.
func NewCA(inputDim, hiddenDim, outputDim, depth, ruleHidden int) *CA {
	// Rule: 9 inputs (cell + 8 neighbors) → hidden → 1 output
	rw1 := make([]float32, 9*ruleHidden)
	rb1 := make([]float32, ruleHidden)
	rw2 := make([]float32, ruleHidden)

	for i := range rw1 {
		rw1[i] = randn() * 0.3
	}
	for i := range rb1 {
		rb1[i] = randn() * 0.1
	}
	for i := range rw2 {
		rw2[i] = randn() * 0.3
	}

	return &CA{
		RuleWeights1: rw1,
		RuleBias1:    rb1,
		RuleWeights2: rw2,
		RuleBias2:    randn() * 0.1,
		HiddenSize:   ruleHidden,
		Depth:        depth,
		InputDim:     inputDim,
		OutputDim:    outputDim,
		HiddenDim:    hiddenDim,
	}
}

// Weight computes a single weight by evolving the CA.
// The position (row, col) seeds the CA, then we evolve it N steps.
// The final cell value is the weight.
func (ca *CA) Weight(row, col, layer, inDim, outDim int) float32 {
	// Seed: normalized position + layer
	r := float32(row+1) / float32(inDim+1)
	c := float32(col+1) / float32(outDim+1)
	l := float32(layer+1) / float32(ca.Depth+1)

	// Create a small grid (3x3 = 9 cells)
	grid := [9]float32{r, c, l, r * c, r * l, c * l, r * r, c * c, l * l}

	// Evolve the grid for N steps
	for step := 0; step < ca.Depth; step++ {
		newGrid := [9]float32{}
		for i := 0; i < 9; i++ {
			// Get neighbors (wrapping)
			neighbors := [9]float32{
				grid[i],                                          // self
				grid[(i+1)%9], grid[(i+2)%9], grid[(i+3)%9],     // right neighbors
				grid[(i+4)%9], grid[(i+5)%9], grid[(i+6)%9],     // bottom neighbors
				grid[(i+7)%9], grid[(i+8)%9],                     // left neighbors
			}

			// Apply the learned rule (tiny MLP)
			// Hidden layer
			hidden := make([]float32, ca.HiddenSize)
			for h := 0; h < ca.HiddenSize; h++ {
				sum := ca.RuleBias1[h]
				for j := 0; j < 9; j++ {
					sum += neighbors[j] * ca.RuleWeights1[j*ca.HiddenSize+h]
				}
				hidden[h] = tanh(sum)
			}

			// Output layer
			out := ca.RuleBias2
			for h := 0; h < ca.HiddenSize; h++ {
				out += hidden[h] * ca.RuleWeights2[h]
			}
			newGrid[i] = out
		}
		grid = newGrid
	}

	// The evolved grid encodes the weight
	return grid[0] * 0.05
}

// Bias computes a bias value.
func (ca *CA) Bias(idx, layer, dim int) float32 {
	x := float32(idx+1) / float32(dim+1)
	l := float32(layer+1) / float32(ca.Depth+1)

	grid := [9]float32{x, l, x * l, x * x, l * l, 0, 0, 0, 0}
	for step := 0; step < ca.Depth; step++ {
		newGrid := [9]float32{}
		for i := 0; i < 9; i++ {
			neighbors := [9]float32{
				grid[i], grid[(i+1)%9], grid[(i+2)%9],
				grid[(i+3)%9], grid[(i+4)%9], grid[(i+5)%9],
				grid[(i+6)%9], grid[(i+7)%9], grid[(i+8)%9],
			}
			hidden := make([]float32, ca.HiddenSize)
			for h := 0; h < ca.HiddenSize; h++ {
				sum := ca.RuleBias1[h]
				for j := 0; j < 9; j++ {
					sum += neighbors[j] * ca.RuleWeights1[j*ca.HiddenSize+h]
				}
				hidden[h] = tanh(sum)
			}
			out := ca.RuleBias2
			for h := 0; h < ca.HiddenSize; h++ {
				out += hidden[h] * ca.RuleWeights2[h]
			}
			newGrid[i] = out
		}
		grid = newGrid
	}
	return grid[0] * 0.02
}

// Forward computes output by generating all weights via CA evolution.
func (ca *CA) Forward(input []float32) []float32 {
	x := input
	dims := ca.dims()

	for layer := 0; layer < len(dims)-1; layer++ {
		inD, outD := dims[layer], dims[layer+1]
		output := make([]float32, outD)
		for j := 0; j < outD; j++ {
			sum := ca.Bias(j, layer, outD)
			for i := 0; i < inD; i++ {
				sum += x[i] * ca.Weight(i, j, layer, inD, outD)
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

func (ca *CA) dims() []int {
	d := []int{ca.InputDim}
	for i := 0; i < ca.Depth; i++ {
		d = append(d, ca.HiddenDim)
	}
	return append(d, ca.OutputDim)
}

func (ca *CA) ParamCount() int {
	return len(ca.RuleWeights1) + len(ca.RuleBias1) + len(ca.RuleWeights2) + 1
}

func (ca *CA) VirtualParamCount() int {
	count := 0
	dims := ca.dims()
	for i := 0; i < len(dims)-1; i++ {
		count += dims[i]*dims[i+1] + dims[i+1]
	}
	return count
}

// Train optimizes the CA rule using evolution with crossover.
func (ca *CA) Train(inputs [][]float32, targets [][]float32, epochs, popSize int) {
	n := ca.ParamCount()
	fmt.Printf("🧬 Neural Cellular Automata\n")
	fmt.Printf("   Rule params: %d\n", n)
	fmt.Printf("   Virtual params: %d (%.0fx)\n",
		ca.VirtualParamCount(), float64(ca.VirtualParamCount())/float64(n))
	fmt.Printf("   Evolution steps: %d\n\n", ca.Depth)

	best := ca.pack()
	bestLoss := ca.loss(inputs, targets)
	sigma := float32(0.3)

	for epoch := 0; epoch < epochs; epoch++ {
		start := time.Now()

		// Generate and evaluate population
		type result struct{ params []float32; loss float32 }
		results := make([]result, popSize)

		for p := 0; p < popSize; p++ {
			mutant := make([]float32, n)
			for i := range mutant {
				mutant[i] = best[i] + randn()*sigma
			}
			ca.unpack(mutant)
			results[p] = result{mutant, ca.loss(inputs, targets)}
		}

		// Sort by fitness
		for i := 0; i < popSize; i++ {
			for j := i + 1; j < popSize; j++ {
				if results[j].loss < results[i].loss {
					results[i], results[j] = results[j], results[i]
				}
			}
		}

		// Recombine top 25%
		elite := max(1, popSize/4)
		newBest := make([]float32, n)
		for i := 0; i < elite; i++ {
			for j := 0; j < n; j++ {
				newBest[j] += results[i].params[j] / float32(elite)
			}
		}

		if results[0].loss < bestLoss {
			copy(best, newBest)
			bestLoss = results[0].loss
			sigma *= 1.05
		} else {
			sigma *= 0.95
		}
		sigma = clamp(sigma, 0.01, 1.0)

		ca.unpack(best)
		if epoch%5 == 0 || epoch == epochs-1 {
			fmt.Printf("   Epoch %3d | Loss: %.6f | Best: %.6f | σ: %.3f | %v\n",
				epoch, results[0].loss, bestLoss, sigma, time.Since(start).Round(time.Millisecond))
		}
	}
}

func (ca *CA) pack() []float32 {
	p := make([]float32, 0, ca.ParamCount())
	p = append(p, ca.RuleWeights1...)
	p = append(p, ca.RuleBias1...)
	p = append(p, ca.RuleWeights2...)
	p = append(p, ca.RuleBias2)
	return p
}

func (ca *CA) unpack(p []float32) {
	o := 0
	copy(ca.RuleWeights1, p[o:o+len(ca.RuleWeights1)])
	o += len(ca.RuleWeights1)
	copy(ca.RuleBias1, p[o:o+len(ca.RuleBias1)])
	o += len(ca.RuleBias1)
	copy(ca.RuleWeights2, p[o:o+len(ca.RuleWeights2)])
	o += len(ca.RuleWeights2)
	ca.RuleBias2 = p[o]
}

func (ca *CA) loss(inputs, targets [][]float32) float32 {
	total := float32(0)
	for i, input := range inputs {
		out := ca.Forward(input)
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
func tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }
func relu(x float32) float32 { if x > 0 { return x }; return 0 }
func max(a, b int) int { if a > b { return a }; return b }
func clamp(x, lo, hi float32) float32 { if x < lo { return lo }; if x > hi { return hi }; return x }
