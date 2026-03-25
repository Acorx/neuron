// Fractal NN v2 — Pushing the limits
package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/fractal"
)

func main() {
	fmt.Println("🌀 Fractal Neural Networks v2 — Pushing Limits")
	fmt.Println("================================================")
	fmt.Println()
	fmt.Println("Formula: W(i,j,l) = Σₖ αₖ·sin(ωₖᵢ·i + ωₖⱼ·j + ωₖₗ·l + φₖ)")
	fmt.Println("Learnable Fourier components = learnable frequencies")
	fmt.Println()

	// === 1. Compression at different scales ===
	fmt.Println("📊 Compression Scaling (K=8, 40 params):")
	fmt.Println()
	for _, d := range []int{1, 2, 3, 5, 8, 10, 15, 20} {
		n := fractal.New(10, 32, 10, d, 8)
		fmt.Printf("   Depth %2d: %d stored → %d virtual (%.0fx)\n",
			d, n.ParamCount(), n.VirtualParamCount(),
			float64(n.VirtualParamCount())/float64(n.ParamCount()))
	}

	// === 2. More Fourier components = more expressive ===
	fmt.Println()
	fmt.Println("🔬 Expressiveness (depth 3, hidden 32):")
	fmt.Println()
	for _, k := range []int{4, 8, 16, 32, 64} {
		n := fractal.New(10, 32, 10, 3, k)
		fmt.Printf("   K=%2d: %d params → %d virtual (%.0fx)\n",
			k, n.ParamCount(), n.VirtualParamCount(),
			float64(n.VirtualParamCount())/float64(n.ParamCount()))
	}

	// === 3. Classification test ===
	fmt.Println()
	fmt.Println("🎯 Classification (K=16, 80 params, depth 2, hidden 32):")
	fmt.Println()
	net := fractal.New(2, 32, 3, 2, 16)
	inputs, targets := makeSpirals(90, 3)

	start := time.Now()
	net.Train(inputs, targets, 30, 60)
	fmt.Printf("\n⏱️  %v\n", time.Since(start).Round(time.Millisecond))

	correct := 0
	for i, input := range inputs {
		if argmax(net.Forward(input)) == argmax(targets[i]) {
			correct++
		}
	}
	fmt.Printf("📊 Accuracy: %.1f%% (%d/%d)\n\n", float64(correct)/float64(len(inputs))*100, correct, len(inputs))

	// === 4. Inference speed at different scales ===
	fmt.Println("⚡ Inference Speed:")
	fmt.Println()
	for _, d := range []int{2, 5, 10, 20} {
		n := fractal.New(10, 64, 10, d, 8)
		in := make([]float32, 10)
		start := time.Now()
		iters := 500
		for i := 0; i < iters; i++ {
			n.Forward(in)
		}
		perIter := time.Since(start) / time.Duration(iters)
		fmt.Printf("   Depth %2d (%6d virtual): %v/iter\n", d, n.VirtualParamCount(), perIter)
	}

	// === 5. Extrapolation: what would 1B params look like ===
	fmt.Println()
	fmt.Println("🔮 Extrapolation to 1B params:")
	fmt.Println()
	fmt.Println("   A transformer with 1B params:")
	fmt.Println("   - Traditional: 4GB RAM, needs A100 GPU")
	fmt.Println("   - Fractal (K=64, 320 params): 1.2KB + compute")
	fmt.Println("   - Compression: ~3,000,000x")
	fmt.Println()
	fmt.Println("   The tradeoff:")
	fmt.Println("   - 1B weight computations per forward pass")
	fmt.Println("   - On CPU: ~1-10 seconds per token")
	fmt.Println("   - On GPU (traditional): ~1ms per token")
	fmt.Println()
	fmt.Println("   Use case: not replacing GPT-4 for chat.")
	fmt.Println("   Use case: running specialized models on phones.")
	fmt.Println("   Use case: edge AI where GPUs don't exist.")
}

func makeSpirals(n, c int) ([][]float32, [][]float32) {
	var in, out [][]float32
	p := n / c
	for ci := 0; ci < c; ci++ {
		for i := 0; i < p; i++ {
			r := float32(i) / float32(p) * 5
			t := float32(ci)*2*float32(math.Pi)/float32(c) + float32(i)*0.1
			in = append(in, []float32{
				r*cos(t) + noise(), r*sin(t) + noise(),
			})
			target := make([]float32, c)
			target[ci] = 1
			out = append(out, target)
		}
	}
	return in, out
}
func cos(x float32) float32 { return float32(math.Cos(float64(x))) }
func sin(x float32) float32 { return float32(math.Sin(float64(x))) }
func noise() float32        { return (float32(time.Now().UnixNano()%1000)/1000.0 - 0.5) * 0.2 }
func argmax(d []float32) int { m := 0; for i, v := range d { if v > d[m] { m = i } }; return m }
