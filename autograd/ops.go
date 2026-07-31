package autograd

import (
	"math"

	tensor "github.com/redxdager/go-tensor/tensors"
)

func Add(a, b *tensor.Tensor) *tensor.Tensor {
	out := tensor.Add(a, b)
	requiresGrad := a.RequiresGrad || b.RequiresGrad
	out.RequiresGrad = requiresGrad

	if requiresGrad {
		out.Grad = tensor.Zeros(out.Shape...)
		out.Parents = []*tensor.Tensor{a, b}

		out.Backward = func() {
			// Gradient for A
			if a.RequiresGrad {
				a.Grad = tensor.Add(a.Grad, out.Grad)
			}
			// Gradient for B (Handles BroadCast Summation if B is [1, N])
			if b.RequiresGrad {
				if len(b.Shape) == 2 && b.Shape[0] == 1 && out.Grad.Shape[0] > 1 {
					// Sum gradients along the batch dimension (axis 0)
					batchSize := out.Grad.Shape[0]
					dim := out.Grad.Shape[1]
					for j := 0; j < dim; j++ {
						sum := 0.0
						for i := 0; i < batchSize; i++ {
							sum += out.Grad.Data[i*dim+j]
						}
						b.Grad.Data[j] += sum
					}
				} else {
					b.Grad = tensor.Add(b.Grad, out.Grad)
				}
			}
		}
	}

	return out
}

// MatMul performs matrix multiplication: Out = A @ B
func MatMul(a, b *tensor.Tensor) *tensor.Tensor {
	outVal := tensor.MatMul(a, b)
	requiresGrad := a.RequiresGrad || b.RequiresGrad

	out := &tensor.Tensor{
		Data:         outVal.Data,
		Shape:        outVal.Shape,
		Strides:      outVal.Strides,
		Parents:      []*tensor.Tensor{a, b},
		RequiresGrad: requiresGrad,
	}

	if requiresGrad {
		out.Grad = tensor.Zeros(out.Shape...)
		out.Backward = func() {
			// dL/dA = dL/dOut @ B^T
			if a.RequiresGrad {
				bT := tensor.Transpose(b)
				da := tensor.MatMul(out.Grad, bT)
				a.Grad = tensor.Add(a.Grad, da)
			}
			// dL/dB = A^T @ dL/dOut
			if b.RequiresGrad {
				aT := tensor.Transpose(a)
				db := tensor.MatMul(aT, out.Grad)
				b.Grad = tensor.Add(b.Grad, db)
			}
		}
	}

	return out
}

// Mean computes the scalar mean across all elements in the tensor.
func Mean(t *tensor.Tensor) *tensor.Tensor {
	size := float64(t.Size())
	sum := 0.0
	for _, val := range t.Data {
		sum += val
	}

	out := &tensor.Tensor{
		Data:         []float64{sum / size},
		Shape:        []int{1},
		Strides:      []int{1},
		Parents:      []*tensor.Tensor{t},
		RequiresGrad: t.RequiresGrad,
	}

	if t.RequiresGrad {
		out.Grad = tensor.Zeros(1)
		out.Backward = func() {
			gradScalar := out.Grad.Data[0] / size
			for i := range t.Grad.Data {
				t.Grad.Data[i] += gradScalar
			}
		}
	}

	return out
}

// ReLU applies element-wise Rectified Linear Unit activation: Out = max(0, X)
func ReLU(t *tensor.Tensor) *tensor.Tensor {
	outData := make([]float64, len(t.Data))
	for i, val := range t.Data {
		if val > 0 {
			outData[i] = val
		} else {
			outData[i] = 0
		}
	}

	out := &tensor.Tensor{
		Data:         outData,
		Shape:        t.Shape,
		Strides:      t.Strides,
		Parents:      []*tensor.Tensor{t},
		RequiresGrad: t.RequiresGrad,
	}

	if t.RequiresGrad {
		out.Grad = tensor.Zeros(out.Shape...)
		out.Backward = func() {
			for i, val := range t.Data {
				if val > 0 {
					t.Grad.Data[i] += out.Grad.Data[i]
				}
			}
		}
	}

	return out
}

// Sigmoid applies element-wise Sigmoid activation: Out = 1 / (1 + e^-X)
func Sigmoid(t *tensor.Tensor) *tensor.Tensor {
	outData := make([]float64, len(t.Data))
	for i, val := range t.Data {
		outData[i] = 1.0 / (1.0 + math.Exp(-val))
	}

	out := &tensor.Tensor{
		Data:         outData,
		Shape:        t.Shape,
		Strides:      t.Strides,
		Parents:      []*tensor.Tensor{t},
		RequiresGrad: t.RequiresGrad,
	}

	if t.RequiresGrad {
		out.Grad = tensor.Zeros(out.Shape...)
		out.Backward = func() {
			// dL/dX = dL/dOut * Out * (1 - Out)
			for i := range t.Data {
				sig := out.Data[i]
				t.Grad.Data[i] += out.Grad.Data[i] * sig * (1.0 - sig)
			}
		}
	}

	return out
}
