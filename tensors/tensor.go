package tensor

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/redxdager/go-tensor/activations"
)

type Tensor struct {
	Data    []float64
	Shape   []int
	Strides []int
}

type Stringer interface {
	String() string
}

var rng = rand.New(rand.NewSource((time.Now().UnixNano())))

func ManualSeed(seed int64) {
	rng = rand.New(rand.NewSource(seed))
}

func calcStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func totalSize(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

func New(shape ...int) *Tensor {
	size := totalSize(shape)
	return &Tensor{
		Data:    make([]float64, size),
		Shape:   shape,
		Strides: calcStrides(shape),
	}
}

func Zeros(shape ...int) *Tensor {
	return New(shape...)
}

func Ones(shape ...int) *Tensor {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = 1.0
	}
	return t
}

func Rand(shape ...int) *Tensor {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = rng.Float64()
	}
	return t
}
func Randn(shape ...int) *Tensor {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = rng.NormFloat64()
	}
	return t
}
func FromSlice(data []float64, shape ...int) *Tensor {
	if totalSize(shape) != len(data) {
		panic(fmt.Sprintf("shape %v does not match data length %d", shape, len(data)))
	}
	t := New(shape...)
	copy(t.Data, data)
	return t
}

func (t *Tensor) FlatIndex(indices ...int) int {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("expected %d indices, got %d", len(t.Shape), len(indices)))
	}
	index := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			panic(fmt.Sprintf("index %d out of bounds for axis %d with size %d", idx, i, t.Shape[i]))
		}
		index += idx * t.Strides[i]
	}
	return index
}
func (t *Tensor) Get(indices ...int) float64 {
	return t.Data[t.FlatIndex(indices...)]
}

func (t *Tensor) Set(val float64, indices ...int) {
	t.Data[t.FlatIndex(indices...)] = val
}

func (t *Tensor) Apply(fn func(float64) float64) *Tensor {
	out := New(t.Shape...)
	for i, val := range t.Data {
		out.Data[i] = fn(val)
	}
	return out
}

func (t *Tensor) Reshape(newShape ...int) *Tensor {
	if totalSize(newShape) != len(t.Data) {
		panic(fmt.Sprintf("cannot reshape tensor of size %d into shape %v", len(t.Data), newShape))
	}
	return &Tensor{
		Data:    t.Data,
		Shape:   newShape,
		Strides: calcStrides(newShape),
	}
}

func Add(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor shape mismatched for addition need the same shape nxn !")
	}
	out := New(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

func MatMul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic(fmt.Sprintf("MatMul expects 2D tensors, got %dD and %dD", len(a.Shape), len(b.Shape)))
	}

	m, n1 := a.Shape[0], a.Shape[1]
	n2, p := b.Shape[0], b.Shape[1]

	if n1 != n2 {
		panic(fmt.Sprintf("cannot multiply shapes (%d, %d) and (%d, %d)", m, n1, n2, p))
	}

	out := New(m, p)

	// Direct flat-slice indexing for speed and safety
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n1; k++ {
				aIdx := i*a.Strides[0] + k*a.Strides[1]
				bIdx := k*b.Strides[0] + j*b.Strides[1]
				sum += a.Data[aIdx] * b.Data[bIdx]
			}
			outIdx := i*out.Strides[0] + j*out.Strides[1]
			out.Data[outIdx] = sum
		}
	}

	return out
}

func (t *Tensor) Relu() *Tensor {
	out := New(t.Shape...)
	for i, val := range t.Data {
		out.Data[i] = activations.ReLu(val)
	}
	return out
}

func (t *Tensor) Sigmoid() *Tensor {
	out := New(t.Shape...)
	for i, val := range t.Data {
		out.Data[i] = activations.Sigmoid(val)
	}
	return out
}

func (t *Tensor) LeakyReLU(lr float64) *Tensor {
	out := New(t.Shape...)
	for i, val := range t.Data {
		out.Data[i] = activations.LeakyReLU(val, lr)
	}
	return out
}

func (t *Tensor) Tanh() *Tensor {
	out := New(t.Shape...)
	for i, val := range t.Data {
		out.Data[i] = activations.Tanh(val)
	}
	return out
}

// This function is to beautify showing the tensors in the output hehe!! :)
func (t *Tensor) String() string {
	if len(t.Shape) == 0 {
		return "Tensor([])"
	}

	if len(t.Shape) == 1 {
		strVals := make([]string, len(t.Data))
		for i, v := range t.Data {
			strVals[i] = fmt.Sprintf("%.4f", v)
		}
		return fmt.Sprintf("Tensor([%s])", strings.Join(strVals, ", "))
	}

	if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]
		var sb strings.Builder
		sb.WriteString("Tensor([\n")

		for i := 0; i < rows; i++ {
			sb.WriteString("  [")
			rowVals := make([]string, cols)
			for j := 0; j < cols; j++ {
				idx := i*t.Strides[0] + j*t.Strides[1]
				rowVals[j] = fmt.Sprintf("%.4f", t.Data[idx])
			}
			sb.WriteString(strings.Join(rowVals, ", "))
			sb.WriteString("]")
			if i < rows-1 {
				sb.WriteString(",\n")
			}
		}
		sb.WriteString(fmt.Sprintf("\n], shape=%v)", t.Shape))
		return sb.String()
	}

	strVals := make([]string, len(t.Data))
	for i, v := range t.Data {
		strVals[i] = fmt.Sprintf("%.4f", v)
	}
	return fmt.Sprintf("Tensor([%s], shape=%v)", strings.Join(strVals, ", "), t.Shape)
}
