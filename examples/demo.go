package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/fourier"
)

func main() {
	fmt.Println("🔬 Fourier NN — Convergence Test")
	fmt.Println("==================================")
	fmt.Println()

	// Small network, many epochs, higher LR
	net := fourier.New(2, 16, 3, 1, 32)
	inputs, targets := makeSpirals(90, 3)

	fmt.Printf("Params: %d alpha → %d virtual (%.0fx)\n\n",
		net.ParamCount(), net.VirtualParamCount(),
		float64(net.VirtualParamCount())/float64(net.ParamCount()))

	start := time.Now()
	net.Train(inputs, targets, 300, 0.03)
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
