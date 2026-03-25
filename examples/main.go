// neuron demo — Generated Neural Networks (fixed)
package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/gen"
)

func main() {
	fmt.Println("🧠 neuron — Generated Neural Networks")
	fmt.Println("======================================")
	fmt.Println()
	fmt.Println("PARADIGM: Compute weights, don't store them")
	fmt.Println()

	// Big generated network, tiny generator
	arch := gen.Architecture{
		InputDim:   2,
		HiddenDims: []int{64, 64},
		OutputDim:  3,
	}
	net := gen.New(arch, 32, 16) // latent=32, gen_hidden=16

	fmt.Printf("Stored params (generator):  %d\n", net.ParamCount())
	fmt.Printf("Generated params (virtual): %d\n", net.GeneratedParamCount())
	fmt.Printf("Compression: %.0fx\n\n", float64(net.GeneratedParamCount())/float64(net.ParamCount()))

	// Generate data
	inputs, targets := generateSpirals(60, 3)

	// Train
	start := time.Now()
	net.Train(inputs, targets, 20, 20)
	fmt.Printf("\n⏱️  %v\n", time.Since(start).Round(time.Millisecond))

	// Test
	correct := 0
	for i, input := range inputs {
		output := net.Forward(input)
		pred := argmax(output)
		if pred == argmax(targets[i]) {
			correct++
		}
	}
	fmt.Printf("📊 Accuracy: %.1f%% (%d/%d)\n", float64(correct)/float64(len(inputs))*100, correct, len(inputs))
}

func generateSpirals(n, classes int) ([][]float32, [][]float32) {
	var inputs [][]float32
	var targets [][]float32
	perClass := n / classes
	for c := 0; c < classes; c++ {
		for i := 0; i < perClass; i++ {
			r := float32(i) / float32(perClass) * 5
			t := float32(c)*2*float32(math.Pi)/float32(classes) + float32(i)*0.1
			x := r*float32(math.Cos(float64(t))) + noise()
			y := r*float32(math.Sin(float64(t))) + noise()
			inputs = append(inputs, []float32{x, y})
			target := make([]float32, classes)
			target[c] = 1.0
			targets = append(targets, target)
		}
	}
	return inputs, targets
}

func noise() float32 {
	return (float32(time.Now().UnixNano()%1000)/1000.0 - 0.5) * 0.2
}

func argmax(data []float32) int {
	maxIdx := 0
	for i, v := range data {
		if v > data[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
