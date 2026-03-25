package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/fractal"
)

func main() {
	fmt.Println("🌀 Fractal Neural Networks")
	fmt.Println("==========================")
	fmt.Println()
	fmt.Println("REVOLUTION: Weights from a fractal formula.")
	fmt.Println("Like Mandelbrot: z²+c = infinite complexity.")
	fmt.Println("8 params → formula → unlimited weights.")
	fmt.Println()

	// Create fractal network
	// 8 parameters define a network with 5 layers
	net := fractal.New(2, 32, 3, 3)

	fmt.Printf("Stored:     %d params (the formula)\n", net.ParamCount())
	fmt.Printf("Virtual:    %d params (computed on-the-fly)\n", net.VirtualParamCount())
	fmt.Printf("Compression: %.0fx\n\n", float64(net.VirtualParamCount())/float64(net.ParamCount()))

	// Generate data
	inputs, targets := generateSpirals(60, 3)

	// Train
	start := time.Now()
	net.Train(inputs, targets, 30, 50)
	fmt.Printf("\n⏱️  %v\n", time.Since(start).Round(time.Millisecond))

	// Test
	correct := 0
	for i, input := range inputs {
		out := net.Forward(input)
		if argmax(out) == argmax(targets[i]) {
			correct++
		}
	}
	fmt.Printf("📊 Accuracy: %.1f%% (%d/%d)\n",
		float64(correct)/float64(len(inputs))*100, correct, len(inputs))

	// Show what happens with different depths
	fmt.Println("\n🔬 Depth Scaling:")
	for d := 1; d <= 8; d++ {
		n := fractal.New(2, 32, 3, d)
		fmt.Printf("   Depth %d: %d stored → %d virtual (%.0fx)\n",
			d, n.ParamCount(), n.VirtualParamCount(),
			float64(n.VirtualParamCount())/float64(n.ParamCount()))
	}
}

func generateSpirals(n, classes int) ([][]float32, [][]float32) {
	var in [][]float32
	var out [][]float32
	perClass := n / classes
	for c := 0; c < classes; c++ {
		for i := 0; i < perClass; i++ {
			r := float32(i) / float32(perClass) * 5
			t := float32(c)*2*float32(math.Pi)/float32(classes) + float32(i)*0.1
			in = append(in, []float32{
				r*float32(math.Cos(float64(t))) + noise(),
				r*float32(math.Sin(float64(t))) + noise(),
			})
			target := make([]float32, classes)
			target[c] = 1
			out = append(out, target)
		}
	}
	return in, out
}

func noise() float32 {
	return (float32(time.Now().UnixNano()%1000)/1000.0 - 0.5) * 0.2
}

func argmax(data []float32) int {
	m := 0
	for i, v := range data {
		if v > data[m] {
			m = i
		}
	}
	return m
}
