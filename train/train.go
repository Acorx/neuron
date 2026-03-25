// Package train provides the training loop with quantization-aware training.
package train

import (
	"fmt"
	"math"
	"time"

	"github.com/Acorx/neuron/nn"
	"github.com/Acorx/neuron/optim"
	"github.com/Acorx/neuron/tensor"
)

// Loss functions

// CrossEntropy computes cross-entropy loss for classification.
func CrossEntropy(logits *tensor.Tensor, targets []int) float32 {
	batch, classes := logits.Shape[0], logits.Shape[1]
	probs := tensor.Softmax(logits)
	totalLoss := float32(0)

	for i := 0; i < batch; i++ {
		p := probs.Data[i*classes+targets[i]]
		if p < 1e-10 {
			p = 1e-10
		}
		totalLoss -= float32(math.Log(float64(p)))
	}
	return totalLoss / float32(batch)
}

// MSELoss computes mean squared error loss.
func MSELoss(pred, target *tensor.Tensor) float32 {
	size := pred.Size()
	sum := float32(0)
	for i := 0; i < size; i++ {
		diff := pred.Data[i] - target.Data[i]
		sum += diff * diff
	}
	return sum / float32(size)
}

// CrossEntropyGrad computes gradient of cross-entropy loss.
func CrossEntropyGrad(logits *tensor.Tensor, targets []int) *tensor.Tensor {
	batch, classes := logits.Shape[0], logits.Shape[1]
	probs := tensor.Softmax(logits)
	grad := make([]float32, len(probs.Data))

	for i := 0; i < batch; i++ {
		for j := 0; j < classes; j++ {
			g := probs.Data[i*classes+j]
			if j == targets[i] {
				g -= 1
			}
			grad[i*classes+j] = g / float32(batch)
		}
	}
	return tensor.New(grad, logits.Shape...)
}

// MSELossGrad computes gradient of MSE loss.
func MSELossGrad(pred, target *tensor.Tensor) *tensor.Tensor {
	size := pred.Size()
	grad := make([]float32, size)
	for i := 0; i < size; i++ {
		grad[i] = 2 * (pred.Data[i] - target.Data[i]) / float32(size)
	}
	return tensor.New(grad, pred.Shape...)
}

// Trainer manages the training loop.
type Trainer struct {
	Model    *nn.SimpleModel
	Optim    optim.Optimizer
	Epochs   int
	Verbose  bool
	QATStart int // Epoch to start quantization-aware training
	GroupSize int
}

// Train trains the model on given data.
func (t *Trainer) Train(inputs [][]float32, targets []int, inputDim, outputDim int) {
	batchSize := len(inputs)

	fmt.Printf("🧠 Training on CPU\n")
	fmt.Printf("   Model: %d parameters\n", t.Model.ParamCount())
	fmt.Printf("   Data: %d samples\n", batchSize)
	fmt.Printf("   Epochs: %d\n", t.Epochs)
	if t.QATStart > 0 {
		fmt.Printf("   QAT starts at epoch %d (group size %d)\n", t.QATStart, t.GroupSize)
	}
	fmt.Println()

	for epoch := 0; epoch < t.Epochs; epoch++ {
		start := time.Now()

		// Start QAT
		if epoch == t.QATStart && t.QATStart > 0 {
			fmt.Printf("   ⚡ Enabling Quantization-Aware Training (4-bit)\n")
			t.Model.QuantizeAll(t.GroupSize)
		}

		totalLoss := float32(0)
		correct := 0

		// Simple batch training (no mini-batching for simplicity)
		for i := 0; i < batchSize; i++ {
			// Forward
			x := tensor.New(inputs[i], 1, inputDim)
			logits := t.Model.Forward(x)
			loss := CrossEntropy(logits, []int{targets[i]})
			totalLoss += loss

			// Check accuracy
			pred := argmax(logits.Data)
			if pred == targets[i] {
				correct++
			}

			// Backward (simplified - manual gradient computation)
			grad := CrossEntropyGrad(logits, []int{targets[i]})

			// Manual backprop through layers
			t.backprop(x, grad)
		}

		avgLoss := totalLoss / float32(batchSize)
		accuracy := float32(correct) / float32(batchSize) * 100
		elapsed := time.Since(start)

		if t.Verbose || epoch%10 == 0 || epoch == t.Epochs-1 {
			fmt.Printf("   Epoch %3d | Loss: %.4f | Acc: %.1f%% | %v\n",
				epoch, avgLoss, accuracy, elapsed.Round(time.Millisecond))
		}
	}
}

// Simple manual backprop for the 2-layer model.
func (t *Trainer) backprop(x *tensor.Tensor, grad *tensor.Tensor) {
	// This is a simplified backprop for demonstration.
	// In a real library, you'd use automatic differentiation.

	// For the last layer: dW = x^T * grad
	// For the first layer: chain rule through ReLU

	layers := t.Model.Layers
	lin1 := layers[0].(*nn.Linear)
	lin2 := layers[1].(*nn.Linear)

	// Gradient for layer 2 weights
	// x2 is the output of layer 1 + ReLU
	x1 := lin1.Forward(x)
	x1Act := tensor.ReLU(x1)

	// dW2 = x1Act^T * grad
	gradW2 := tensor.New(nil, lin2.OutFeatures, lin2.InFeatures)
	for i := 0; i < lin2.OutFeatures; i++ {
		for j := 0; j < lin2.InFeatures; j++ {
			gradW2.Data[i*lin2.InFeatures+j] = grad.Data[i] * x1Act.Data[j]
		}
	}

	// Gradient through ReLU
	gradX1 := tensor.New(nil, 1, lin2.InFeatures)
	for j := 0; j < lin2.InFeatures; j++ {
		g := float32(0)
		for i := 0; i < lin2.OutFeatures; i++ {
			g += grad.Data[i] * lin2.Weight.Data[i*lin2.InFeatures+j]
		}
		if x1.Data[j] > 0 {
			gradX1.Data[j] = g
		}
	}

	// Gradient for layer 1 weights
	gradW1 := tensor.New(nil, lin1.OutFeatures, lin1.InFeatures)
	for i := 0; i < lin1.OutFeatures; i++ {
		for j := 0; j < lin1.InFeatures; j++ {
			gradW1.Data[i*lin1.InFeatures+j] = gradX1.Data[i] * x.Data[j]
		}
	}

	// Update weights
	t.Optim.Step(
		[]*tensor.Tensor{lin1.Weight, lin1.Bias, lin2.Weight, lin2.Bias},
		[]*tensor.Tensor{gradW1, tensor.Zeros(lin1.OutFeatures), gradW2, grad},
	)
}

func argmax(data []float32) int {
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// Benchmark benchmarks the library on CPU.
func Benchmark() {
	fmt.Println("🔬 neuron Benchmark")
	fmt.Println("===================")

	// Matrix multiplication benchmark
	sizes := []int{128, 256, 512}
	for _, size := range sizes {
		a := tensor.Randn(size, size)
		b := tensor.Randn(size, size)

		start := time.Now()
		_ = tensor.MatMul(a, b)
		elapsed := time.Since(start)

		flops := 2 * size * size * size
		gflops := float64(flops) / elapsed.Seconds() / 1e9

		fmt.Printf("   MatMul %dx%d: %v (%.2f GFLOPS)\n", size, size, elapsed.Round(time.Millisecond), gflops)
	}

	fmt.Println()

	// Quantized vs full-precision
	fmt.Println("Quantized MatMul (8-bit):")
	for _, size := range []int{256, 512} {
		a := tensor.Randn(size, size)
		b := tensor.Randn(size, size)
		b.Quantize(32)

		start := time.Now()
		_ = tensor.MatMulQuant(a, b)
		elapsed := time.Since(start)

		fmt.Printf("   MatMulQuant %dx%d: %v\n", size, size, elapsed.Round(time.Millisecond))
	}
}
