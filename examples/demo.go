// Fractal Neural Network Demo
package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/fractal"
)

func main() {
	fmt.Println("🌀 Fractal Neural Network")
	fmt.Println("=========================")
	fmt.Println()
	fmt.Println("8 parameters → fractal formula → unlimited weights")
	fmt.Println()

	// === Compression scaling ===
	fmt.Println("📊 Compression Scaling:")
	fmt.Println()
	for _, d := range []int{1, 2, 3, 4, 5, 6, 8, 10, 15, 20} {
		n := fractal.New(10, 32, 10, d)
		fmt.Printf("   Depth %2d: %d stored → %d virtual (%.0fx)\n",
			d, n.ParamCount(), n.VirtualParamCount(),
			float64(n.VirtualParamCount())/float64(n.ParamCount()))
	}

	// === Quick classification ===
	fmt.Println()
	fmt.Println("🎯 Quick Classification (depth 2, hidden 16):")
	fmt.Println()

	net := fractal.New(2, 16, 3, 2)
	inputs, targets := makeSpirals(60, 3)

	start := time.Now()
	net.Train(inputs, targets, 20, 40)
	fmt.Printf("\n⏱️  %v\n", time.Since(start).Round(time.Millisecond))

	correct := 0
	for i, input := range inputs {
		if argmax(net.Forward(input)) == argmax(targets[i]) {
			correct++
		}
	}
	fmt.Printf("📊 Accuracy: %.1f%%\n", float64(correct)/float64(len(inputs))*100)

	// === Inference speed ===
	fmt.Println()
	fmt.Println("⚡ Inference Speed:")
	fmt.Println()
	n := fractal.New(10, 32, 10, 3)
	in := make([]float32, 10)
	start = time.Now()
	for i := 0; i < 1000; i++ {
		n.Forward(in)
	}
	fmt.Printf("   %v per forward pass\n", time.Since(start)/1000)

	// === The vision ===
	fmt.Println()
	fmt.Println("💡 Vision:")
	fmt.Println()
	fmt.Println("   Traditional 1B-param model: 4GB RAM (needs GPU)")
	fmt.Println("   Fractal 1B-param model:     32 bytes + compute")
	fmt.Println()
	fmt.Println("   Trade memory for compute.")
	fmt.Println("   CPUs love compute. Hate memory bandwidth.")
	fmt.Println("   Fractal NNs are built for CPUs.")
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
func cos(x float32) float32  { return float32(math.Cos(float64(x))) }
func sin(x float32) float32  { return float32(math.Sin(float64(x))) }
func noise() float32         { return (float32(time.Now().UnixNano()%1000)/1000.0 - 0.5) * 0.2 }
func argmax(d []float32) int { m := 0; for i, v := range d { if v > d[m] { m = i } }; return m }
