package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/neural"
)

func main() {
	fmt.Println("🧬 Neural Cellular Automata for AI")
	fmt.Println("====================================")
	fmt.Println()
	fmt.Println("Conway's Game of Life: 2 rules → infinite complexity")
	fmt.Println("Neural CA: learned rule → weight patterns")
	fmt.Println("Only the RULE is stored. Weights are evolved.")
	fmt.Println()

	// === Compression ===
	fmt.Println("📊 Rule Size vs Generated Weights:")
	fmt.Println()
	for _, rh := range []int{2, 4, 8, 16} {
		ca := neural.NewCA(10, 32, 10, 3, rh)
		fmt.Printf("   Rule hidden=%2d: %d rule params → %d virtual (%.0fx)\n",
			rh, ca.ParamCount(), ca.VirtualParamCount(),
			float64(ca.VirtualParamCount())/float64(ca.ParamCount()))
	}

	// === Classification ===
	fmt.Println()
	fmt.Println("🎯 Classification (rule hidden=8, depth 3):")
	fmt.Println()
	ca := neural.NewCA(2, 16, 3, 3, 8)
	inputs, targets := makeSpirals(60, 3)

	start := time.Now()
	ca.Train(inputs, targets, 30, 40)
	fmt.Printf("\n⏱️  %v\n", time.Since(start).Round(time.Millisecond))

	correct := 0
	for i, input := range inputs {
		if argmax(ca.Forward(input)) == argmax(targets[i]) {
			correct++
		}
	}
	fmt.Printf("📊 Accuracy: %.1f%% (%d/%d)\n", float64(correct)/float64(len(inputs))*100, correct, len(inputs))

	// === Inference speed ===
	fmt.Println()
	fmt.Println("⚡ Inference:")
	fmt.Println()
	ca2 := neural.NewCA(10, 32, 10, 3, 4)
	in := make([]float32, 10)
	start = time.Now()
	for i := 0; i < 500; i++ {
		ca2.Forward(in)
	}
	fmt.Printf("   %v per pass (%d virtual params)\n", time.Since(start)/500, ca2.VirtualParamCount())
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
