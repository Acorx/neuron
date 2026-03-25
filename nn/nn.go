// Package nn provides neural network layers with quantization-aware training.
package nn

import (
	"math"

	"github.com/Acorx/neuron/tensor"
)

// Layer is the interface for all neural network layers.
type Layer interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
	QuantizeAll(groupSize int)
}

// Linear is a fully connected layer with optional quantization.
type Linear struct {
	Weight *tensor.Tensor // (outFeatures, inFeatures)
	Bias   *tensor.Tensor // (outFeatures)
	InFeatures  int
	OutFeatures int
	Quantized   bool
}

// NewLinear creates a new linear layer.
func NewLinear(inFeatures, outFeatures int) *Linear {
	w := tensor.Randn(outFeatures, inFeatures)
	b := tensor.Zeros(outFeatures)
	return &Linear{
		Weight:      w,
		Bias:        b,
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
	}
}

// Forward computes y = xW^T + b
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	// x is (batch, inFeatures)
	// Weight is (outFeatures, inFeatures)
	// We need x * W^T = (batch, outFeatures)

	// Transpose weight
	wT := tensor.New(nil, l.InFeatures, l.OutFeatures)
	for i := 0; i < l.OutFeatures; i++ {
		for j := 0; j < l.InFeatures; j++ {
			wT.Data[j*l.OutFeatures+i] = l.Weight.Data[i*l.InFeatures+j]
		}
	}

	var out *tensor.Tensor
	if l.Quantized {
		out = tensor.MatMulQuant(x, wT)
	} else {
		out = tensor.MatMul(x, wT)
	}

	// Add bias
	batch := x.Shape[0]
	for i := 0; i < batch; i++ {
		for j := 0; j < l.OutFeatures; j++ {
			out.Data[i*l.OutFeatures+j] += l.Bias.Data[j]
		}
	}

	return out
}

// Parameters returns all trainable parameters.
func (l *Linear) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{l.Weight, l.Bias}
}

// QuantizeAll quantizes all parameters.
func (l *Linear) QuantizeAll(groupSize int) {
	l.Weight.Quantize(groupSize)
	l.Quantized = true
}

// Embedding is a lookup table for discrete tokens.
type Embedding struct {
	Weight *tensor.Tensor // (vocabSize, embedDim)
	VocabSize int
	EmbedDim  int
}

// NewEmbedding creates a new embedding layer.
func NewEmbedding(vocabSize, embedDim int) *Embedding {
	w := tensor.Randn(vocabSize, embedDim)
	return &Embedding{
		Weight:    w,
		VocabSize: vocabSize,
		EmbedDim:  embedDim,
	}
}

// Forward looks up embeddings for given token IDs.
func (e *Embedding) Forward(ids []int) *tensor.Tensor {
	seqLen := len(ids)
	result := make([]float32, seqLen*e.EmbedDim)
	for i, id := range ids {
		copy(result[i*e.EmbedDim:(i+1)*e.EmbedDim],
			e.Weight.Data[id*e.EmbedDim:(id+1)*e.EmbedDim])
	}
	return tensor.New(result, seqLen, e.EmbedDim)
}

// Parameters returns all trainable parameters.
func (e *Embedding) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{e.Weight}
}

// QuantizeAll quantizes all parameters.
func (e *Embedding) QuantizeAll(groupSize int) {
	e.Weight.Quantize(groupSize)
}

// LayerNorm is a layer normalization.
type LayerNorm struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
	Dim    int
	Eps    float32
}

// NewLayerNorm creates a new layer normalization.
func NewLayerNorm(dim int) *LayerNorm {
	return &LayerNorm{
		Weight: tensor.Ones(dim),
		Bias:   tensor.Zeros(dim),
		Dim:    dim,
		Eps:    1e-5,
	}
}

// Forward applies layer normalization.
func (ln *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 2 {
		panic("layernorm requires 2D tensor")
	}
	rows, cols := x.Shape[0], x.Shape[1]
	result := make([]float32, len(x.Data))

	for i := 0; i < rows; i++ {
		// Compute mean
		mean := float32(0)
		for j := 0; j < cols; j++ {
			mean += x.Data[i*cols+j]
		}
		mean /= float32(cols)

		// Compute variance
		variance := float32(0)
		for j := 0; j < cols; j++ {
			diff := x.Data[i*cols+j] - mean
			variance += diff * diff
		}
		variance /= float32(cols)

		// Normalize
		std := float32(math.Sqrt(float64(variance + ln.Eps)))
		for j := 0; j < cols; j++ {
			result[i*cols+j] = ((x.Data[i*cols+j] - mean) / std) * ln.Weight.Data[j] + ln.Bias.Data[j]
		}
	}

	return tensor.New(result, x.Shape...)
}

// Parameters returns all trainable parameters.
func (ln *LayerNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{ln.Weight, ln.Bias}
}

// QuantizeAll quantizes all parameters.
func (ln *LayerNorm) QuantizeAll(groupSize int) {
	// LayerNorm weights are usually kept in full precision
}

// SimpleModel is a basic feedforward model for demonstration.
type SimpleModel struct {
	Layers []Layer
}

// NewSimpleModel creates a simple 2-layer MLP.
func NewSimpleModel(inputDim, hiddenDim, outputDim int) *SimpleModel {
	return &SimpleModel{
		Layers: []Layer{
			NewLinear(inputDim, hiddenDim),
			NewLinear(hiddenDim, outputDim),
		},
	}
}

// Forward runs the full forward pass.
func (m *SimpleModel) Forward(x *tensor.Tensor) *tensor.Tensor {
	for i, layer := range m.Layers {
		x = layer.Forward(x)
		// Apply ReLU after each layer except the last
		if i < len(m.Layers)-1 {
			x = tensor.ReLU(x)
		}
	}
	return x
}

// Parameters returns all trainable parameters.
func (m *SimpleModel) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

// QuantizeAll quantizes all layers.
func (m *SimpleModel) QuantizeAll(groupSize int) {
	for _, layer := range m.Layers {
		layer.QuantizeAll(groupSize)
	}
}

// ParamCount returns total number of parameters.
func (m *SimpleModel) ParamCount() int {
	count := 0
	for _, p := range m.Parameters() {
		count += p.Size()
	}
	return count
}
