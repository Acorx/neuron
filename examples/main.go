// neuron demo — Train a classifier on CPU
package main

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/nn"
	"github.com/Acorx/neuron/optim"
	"github.com/Acorx/neuron/tensor"
	"github.com/Acorx/neuron/train"
)

func main() {
	fmt.Println("🧠 neuron — CPU-Native AI Training Library")
	fmt.Println("==========================================")
	fmt.Println()

	inputDim := 2
	hiddenDim := 64
	outputDim := 3
	numSamples := 300

	inputs, targets := generateSpirals(numSamples, outputDim)

	model := nn.NewSimpleModel(inputDim, hiddenDim, outputDim)
	fmt.Printf("Model: %d → %d → %d (%d parameters)\n\n",
		inputDim, hiddenDim, outputDim, model.ParamCount())

	trainer := train.Trainer{
		Model:     model,
		Optim:     optim.NewSignSGD(0.01),
		Epochs:    50,
		Verbose:   false,
		QATStart:  30,
		GroupSize: 32,
	}

	start := time.Now()
	trainer.Train(inputs, targets, inputDim, outputDim)
	fmt.Printf("\n⏱️  Total: %v\n", time.Since(start).Round(time.Millisecond))

	correct := 0
	for i, input := range inputs {
		x := tensor.New(input, 1, inputDim)
		logits := model.Forward(x)
		pred := argmax(logits.Data)
		if pred == targets[i] {
			correct++
		}
	}
	fmt.Printf("📊 Accuracy: %.1f%% (%d/%d)\n", float64(correct)/float64(numSamples)*100, correct, numSamples)

	fmt.Println()
	train.Benchmark()
}

func generateSpirals(n, numClasses int) ([][]float32, []int) {
	var inputs [][]float32
	var targets []int
	perClass := n / numClasses

	for c := 0; c < numClasses; c++ {
		for i := 0; i < perClass; i++ {
			r := float32(i) / float32(perClass) * 5
			t := float32(c)*2*float32(math.Pi)/float32(numClasses) + float32(i)*0.1
			x := r*cos(t) + noise()
			y := r*sin(t) + noise()
			inputs = append(inputs, []float32{x, y})
			targets = append(targets, c)
		}
	}
	return inputs, targets
}

func cos(x float32) float32 { return float32(math.Cos(float64(x))) }
func sin(x float32) float32 { return float32(math.Sin(float64(x))) }

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
