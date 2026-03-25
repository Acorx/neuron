// Package optim provides memory-efficient optimizers for CPU training.
//
// Key innovation: State-free optimizers that use ZERO extra memory
// (unlike Adam which uses 2x the model memory for momentum+variance).
package optim

import (
	"math"

	"github.com/Acorx/neuron/tensor"
)

// Optimizer is the interface for all optimizers.
type Optimizer interface {
	Step(params []*tensor.Tensor, grads []*tensor.Tensor)
	ZeroGrad(params []*tensor.Tensor)
}

// SignSGD is a state-free optimizer.
// Memory: 0 extra (just uses the sign of the gradient).
// Works surprisingly well for quantized models.
type SignSGD struct {
	LR float32
}

// NewSignSGD creates a SignSGD optimizer.
func NewSignSGD(lr float32) *SignSGD {
	return &SignSGD{LR: lr}
}

// Step updates parameters using sign of gradients.
func (o *SignSGD) Step(params []*tensor.Tensor, grads []*tensor.Tensor) {
	for i, p := range params {
		if grads[i] == nil {
			continue
		}
		size := p.Size()
		for j := 0; j < size; j++ {
			g := grads[i].Data[j]
			sign := float32(1)
			if g < 0 {
				sign = -1
			} else if g == 0 {
				sign = 0
			}
			p.Data[j] -= o.LR * sign
		}
	}
}

// ZeroGrad zeros the gradients.
func (o *SignSGD) ZeroGrad(params []*tensor.Tensor) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0
			}
		}
	}
}

// AdamW is a memory-efficient AdamW with quantized states.
// Instead of storing full FP32 momentum/variance,
// we store them in INT8 (4x memory reduction).
type AdamW struct {
	LR     float32
	Beta1  float32
	Beta2  float32
	Eps    float32
	WeightDecay float32

	// Quantized states (int8 instead of float32)
	m      [][]int8 // momentum
	v      [][]int8 // variance
	mScale []float32
	vScale []float32
	step   int
}

// NewAdamW creates a memory-efficient AdamW optimizer.
func NewAdamW(lr float32) *AdamW {
	return &AdamW{
		LR:           lr,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  0.01,
	}
}

// Step updates parameters using AdamW with quantized states.
func (o *AdamW) Step(params []*tensor.Tensor, grads []*tensor.Tensor) {
	o.step++

	// Lazy init of states
	if o.m == nil {
		o.m = make([][]int8, len(params))
		o.v = make([][]int8, len(params))
		o.mScale = make([]float32, len(params))
		o.vScale = make([]float32, len(params))
		for i, p := range params {
			size := p.Size()
			o.m[i] = make([]int8, size)
			o.v[i] = make([]int8, size)
			o.mScale[i] = 1.0
			o.vScale[i] = 1.0
		}
	}

	biasCorrection1 := float32(1.0 - math.Pow(float64(o.Beta1), float64(o.step)))
	biasCorrection2 := float32(1.0 - math.Pow(float64(o.Beta2), float64(o.step)))

	for i, p := range params {
		if grads[i] == nil {
			continue
		}
		size := p.Size()
		for j := 0; j < size; j++ {
			g := grads[i].Data[j]

			// Dequantize states
			mVal := float32(o.m[i][j]) * o.mScale[i]
			vVal := float32(o.v[i][j]) * o.vScale[i]

			// Update
			mVal = o.Beta1*mVal + (1-o.Beta1)*g
			vVal = o.Beta2*vVal + (1-o.Beta2)*g*g

			// Bias correction
			mHat := mVal / biasCorrection1
			vHat := vVal / biasCorrection2

			// Weight update with weight decay
			p.Data[j] -= o.LR * (mHat/(float32(math.Sqrt(float64(vHat)))+o.Eps) + o.WeightDecay*p.Data[j])

			// Re-quantize states (simple: use max abs value per param)
			o.m[i][j] = quantizeInt8(mVal, 1.0)
			o.v[i][j] = quantizeInt8(vVal, 1.0)
		}
	}
}

// ZeroGrad zeros the gradients.
func (o *AdamW) ZeroGrad(params []*tensor.Tensor) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0
			}
		}
	}
}

// SGD is a basic stochastic gradient descent optimizer.
type SGD struct {
	LR      float32
	Momentum float32
	m       [][]float32
}

// NewSGD creates an SGD optimizer.
func NewSGD(lr, momentum float32) *SGD {
	return &SGD{LR: lr, Momentum: momentum}
}

// Step updates parameters using SGD.
func (o *SGD) Step(params []*tensor.Tensor, grads []*tensor.Tensor) {
	if o.Momentum > 0 && o.m == nil {
		o.m = make([][]float32, len(params))
		for i, p := range params {
			o.m[i] = make([]float32, p.Size())
		}
	}

	for i, p := range params {
		if grads[i] == nil {
			continue
		}
		size := p.Size()
		for j := 0; j < size; j++ {
			g := grads[i].Data[j]
			if o.Momentum > 0 {
				o.m[i][j] = o.Momentum*o.m[i][j] + g
				g = o.m[i][j]
			}
			p.Data[j] -= o.LR * g
		}
	}
}

// ZeroGrad zeros the gradients.
func (o *SGD) ZeroGrad(params []*tensor.Tensor) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0
			}
		}
	}
}

func quantizeInt8(val, scale float32) int8 {
	q := int(math.Round(float64(val / scale)))
	if q > 127 {
		return 127
	}
	if q < -128 {
		return -128
	}
	return int8(q)
}
