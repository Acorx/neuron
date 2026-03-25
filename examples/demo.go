package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/fourier"
)

func main() {
	fmt.Println("🔬 Fourier Neural Networks — Vectorized")
	fmt.Println("========================================")
	fmt.Println()
	fmt.Println("Key innovation: generate ALL weights via matrix ops,")
	fmt.Println("not one-by-one. Finite-difference gradients.")
	fmt.Println()

	// Compression
	fmt.Println("📊 Compression:")
	fmt.Println()
	for _, k := range []int{8, 16, 32, 64} {
		f := fourier.New(10, 32, 10, 3, k)
		fmt.Printf("   K=%2d: %d alpha → %d virtual (%.0fx)\n",
			k, f.ParamCount(), f.VirtualParamCount(),
			float64(f.VirtualParamCount())/float64(f.ParamCount()))
	}

	// Classification
	fmt.Println()
	fmt.Println("🎯 Classification (K=32, depth 2, hidden 16):")
	fmt.Println()
	net := fourier.New(2, 16, 3, 2, 32)
	inputs, targets := makeSpirals(60, 3)

	start := time.Now()
	net.Train(inputs, targets, 20, 0.01)
	fmt.Printf("\n⏱️  %v\n", time.Since(start).Round(time.Millisecond))

	correct := 0
	for i, input := range inputs {
		if argmax(net.Forward(input)) == argmax(targets[i]) {
			correct++
		}
	}
	fmt.Printf("📊 Accuracy: %.1f%% (%d/%d)\n", float64(correct)/float64(len(inputs))*100, correct, len(inputs))
}

func makeSpirals(n, c int) ([][]float32, [][]float32) {
	var in, out [][]float32
	p := n / c
	for ci := 0; ci < c; ci++ {
		for i := 0; i < p; i++ {
			r := float32(i) / float32(p) * 5
			t := float32(ci)*2*float32(math.Pi)/float32(c) + float32(i)*0.1
			in = append(in, []float32{r*cos(t) + noise(), r*sin(t) + noise()})
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
