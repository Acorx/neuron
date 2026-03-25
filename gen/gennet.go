// Package gen implements Generated Neural Networks (GenNet).
//
// PARADIGM SHIFT: Instead of storing weights, we COMPUTE them.
//
// Traditional NN:  Store 1B parameters → multiply → output
// GenNet:          Store 1K parameters → generate 1B weights on-the-fly → output
//
// A tiny "generator" network produces the weights of a much larger
// "generated" network. Only the generator is trained and stored.
//
// This flips the compute/memory tradeoff:
// - CPUs are good at compute, bad at memory bandwidth
// - GenNet trades memory for compute → better fit for CPU
// - A "1 billion parameter" model fits in kilobytes
package gen

import (
	"fmt"
	"math"
	"time"
)

// GenNet is a Generated Neural Network.
// The weights of the actual network are computed on-the-fly
// by the Generator, never stored.
type GenNet struct {
	Latent    []float32  // Latent vector (what we train)
	Generator *Generator // Tiny network that produces weights
	Arch      Architecture
}

// Architecture defines the structure of the generated network.
type Architecture struct {
	InputDim  int
	HiddenDims []int
	OutputDim int
}

// Generator is a tiny MLP that maps (position, latent) → weight value.
// This is the ONLY thing we store and train.
type Generator struct {
	Layers []GenLayer
}

// GenLayer is a layer in the generator.
type GenLayer struct {
	Weights []float32
	Bias    []float32
	InDim   int
	OutDim  int
}

// New creates a GenNet.
// latentDim: size of the latent vector (typically 64-256)
// genHidden: hidden size of the generator (typically 32-128)
func New(arch Architecture, latentDim, genHidden int) *GenNet {
	// Create generator: maps (layer_idx + row_idx + col_idx + latent) → weight
	// Input to generator: [layer_onehot..., row_norm, col_norm, latent...]
	// For simplicity, we'll encode position as normalized floats
	genInputDim := 3 + latentDim // layer_frac, row_frac, col_frac + latent

	gen := &Generator{
		Layers: []GenLayer{
			newGenLayer(genInputDim, genHidden),
			newGenLayer(genHidden, genHidden),
			newGenLayer(genHidden, 1), // output: single weight value
		},
	}

	// Initialize latent
	latent := make([]float32, latentDim)
	for i := range latent {
		latent[i] = randn() * 0.1
	}

	return &GenNet{
		Latent:    latent,
		Generator: gen,
		Arch:      arch,
	}
}

// Forward computes the output by generating all weights on-the-fly.
func (g *GenNet) Forward(input []float32) []float32 {
	x := input
	layerIdx := 0

	// Process through each layer of the generated network
	allDims := append([]int{g.Arch.InputDim}, g.Arch.HiddenDims...)
	allDims = append(allDims, g.Arch.OutputDim)

	for l := 0; l < len(allDims)-1; l++ {
		inDim := allDims[l]
		outDim := allDims[l+1]

		// Generate weights for this layer on-the-fly
		weights := g.generateWeights(l, len(allDims)-2, inDim, outDim)
		bias := g.generateBias(l, len(allDims)-2, outDim)

		// Matrix multiply: output = x * W + b
		output := make([]float32, outDim)
		for j := 0; j < outDim; j++ {
			sum := bias[j]
			for i := 0; i < inDim; i++ {
				sum += x[i]*weights[i*outDim+j]
			}
			// ReLU for hidden layers
			if l < len(allDims)-2 {
				if sum < 0 {
					sum = 0
				}
			}
			output[j] = sum
		}
		x = output
		layerIdx++
	}

	return x
}

// generateWeights computes all weights for a layer using the generator.
// Parallelized across rows for speed.
func (g *GenNet) generateWeights(layerIdx, totalLayers, inDim, outDim int) []float32 {
	weights := make([]float32, inDim*outDim)
	layerFrac := float32(layerIdx) / float32(max(totalLayers, 1))

	// Parallelize over rows
	type result struct {
		row int
		vals []float32
	}
	ch := make(chan result, inDim)

	for i := 0; i < inDim; i++ {
		go func(i int) {
			rowVals := make([]float32, outDim)
			rowFrac := float32(i) / float32(max(inDim, 1))
			for j := 0; j < outDim; j++ {
				colFrac := float32(j) / float32(max(outDim, 1))
				genInput := make([]float32, 0, 3+len(g.Latent))
				genInput = append(genInput, layerFrac, rowFrac, colFrac)
				genInput = append(genInput, g.Latent...)
				w := g.Generator.Forward(genInput)
				rowVals[j] = w[0] * 0.1
			}
			ch <- result{i, rowVals}
		}(i)
	}

	for i := 0; i < inDim; i++ {
		r := <-ch
		copy(weights[r.row*outDim:], r.vals)
	}

	return weights
}

// generateBias computes bias for a layer.
func (g *GenNet) generateBias(layerIdx, totalLayers, outDim int) []float32 {
	bias := make([]float32, outDim)
	layerFrac := float32(layerIdx) / float32(max(totalLayers, 1))

	for j := 0; j < outDim; j++ {
		colFrac := float32(j) / float32(max(outDim, 1))

		genInput := make([]float32, 0, 3+len(g.Latent))
		genInput = append(genInput, layerFrac, 1.0, colFrac) // row_frac=1 for bias
		genInput = append(genInput, g.Latent...)

		w := g.Generator.Forward(genInput)
		bias[j] = w[0] * 0.1
	}
	return bias
}

// Forward runs the generator network.
func (gen *Generator) Forward(input []float32) []float32 {
	x := input
	for _, layer := range gen.Layers {
		output := make([]float32, layer.OutDim)
		for j := 0; j < layer.OutDim; j++ {
			sum := layer.Bias[j]
			for i := 0; i < layer.InDim; i++ {
				sum += x[i]*layer.Weights[i*layer.OutDim+j]
			}
			// Tanh for intermediate, linear for output
			if layer.OutDim > 1 {
				sum = tanh(sum)
			}
			output[j] = sum
		}
		x = output
	}
	return x
}

// ParamCount returns total stored parameters.
func (g *GenNet) ParamCount() int {
	count := len(g.Latent)
	for _, layer := range g.Generator.Layers {
		count += len(layer.Weights) + len(layer.Bias)
	}
	return count
}

// GeneratedParamCount returns the number of parameters in the generated network.
func (g *GenNet) GeneratedParamCount() int {
	count := 0
	allDims := append([]int{g.Arch.InputDim}, g.Arch.HiddenDims...)
	allDims = append(allDims, g.Arch.OutputDim)
	for i := 0; i < len(allDims)-1; i++ {
		count += allDims[i]*allDims[i+1] + allDims[i+1] // weights + bias
	}
	return count
}

// Train trains the GenNet using evolutionary strategy (no backprop needed!).
// This is the SECOND paradigm shift: train without gradients.
func (g *GenNet) Train(inputs [][]float32, targets [][]float32, epochs int, population int) {
	fmt.Printf("🧬 GenNet Training (Evolutionary)\n")
	fmt.Printf("   Stored params: %d\n", g.ParamCount())
	fmt.Printf("   Generated params: %d\n", g.GeneratedParamCount())
	fmt.Printf("   Compression: %.0fx\n", float64(g.GeneratedParamCount())/float64(g.ParamCount()))
	fmt.Printf("   Population: %d, Epochs: %d\n\n", population, epochs)

	// Pack all trainable params into a single vector
	allParams := g.packParams()
	bestParams := make([]float32, len(allParams))
	copy(bestParams, allParams)
	bestLoss := g.computeLoss(inputs, targets)

	sigma := float32(0.1) // mutation strength
	learningRate := float32(0.01)

	for epoch := 0; epoch < epochs; epoch++ {
		start := time.Now()
		improvements := 0

		for p := 0; p < population; p++ {
			// Create mutant
			mutant := make([]float32, len(bestParams))
			noise := make([]float32, len(bestParams))
			for i := range mutant {
				noise[i] = randn() * sigma
				mutant[i] = bestParams[i] + noise[i]
			}

			// Evaluate mutant
			g.unpackParams(mutant)
			mutantLoss := g.computeLoss(inputs, targets)

			if mutantLoss < bestLoss {
				// Accept improvement
				for i := range bestParams {
					bestParams[i] += learningRate * noise[i]
				}
				bestLoss = mutantLoss
				improvements++
			}
		}

		g.unpackParams(bestParams)
		elapsed := time.Since(start)

		if epoch%10 == 0 || epoch == epochs-1 {
			fmt.Printf("   Epoch %3d | Loss: %.6f | Improvements: %d/%d | %v\n",
				epoch, bestLoss, improvements, population, elapsed.Round(time.Millisecond))
		}

		// Decay mutation strength
		sigma *= 0.999
	}
}

func (g *GenNet) packParams() []float32 {
	var params []float32
	params = append(params, g.Latent...)
	for _, layer := range g.Generator.Layers {
		params = append(params, layer.Weights...)
		params = append(params, layer.Bias...)
	}
	return params
}

func (g *GenNet) unpackParams(params []float32) {
	latentDim := len(g.Latent)
	copy(g.Latent, params[:latentDim])
	offset := latentDim

	for i := range g.Generator.Layers {
		layer := &g.Generator.Layers[i]
		wLen := len(layer.Weights)
		copy(layer.Weights, params[offset:offset+wLen])
		offset += wLen
		bLen := len(layer.Bias)
		copy(layer.Bias, params[offset:offset+bLen])
		offset += bLen
	}
}

func (g *GenNet) computeLoss(inputs [][]float32, targets [][]float32) float32 {
	totalLoss := float32(0)
	for i, input := range inputs {
		output := g.Forward(input)
		for j := range output {
			diff := output[j] - targets[i][j]
			totalLoss += diff * diff
		}
	}
	return totalLoss / float32(len(inputs))
}

// --- Helpers ---

func newGenLayer(in, out int) GenLayer {
	w := make([]float32, in*out)
	scale := float32(math.Sqrt(2.0 / float64(in)))
	for i := range w {
		w[i] = randn() * scale
	}
	return GenLayer{
		Weights: w,
		Bias:    make([]float32, out),
		InDim:   in,
		OutDim:  out,
	}
}

var seed uint64 = 42

func randn() float32 {
	seed = seed*6364136223846793005 + 1442695040888963407
	return float32(seed>>33) / float32(1<<31) - 1.0
}

func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
