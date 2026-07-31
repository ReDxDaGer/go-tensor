package tensor

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/redxdager/go-tensor/activations"
)

type Tensor struct {
	Data         []float64
	Shape        []int
	Strides      []int
	RequiresGrad bool
	Grad         *Tensor
	Parents      []*Tensor
	Backward     func()
}

type TensorOption func(*Tensor)

type Stringer interface {
	String() string
}

var rng = rand.New(rand.NewSource((time.Now().UnixNano())))

func ManualSeed(seed int64) {
	rng = rand.New(rand.NewSource(seed))
}

func WithGrad() TensorOption {
	return func(t *Tensor) {
		t.RequiresGrad = true
		t.Grad = Zeros(t.Shape...)
	}
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

func NewWithOpts(shape []int, opts ...TensorOption) *Tensor {
	t := New(shape...)
	for _, opt := range opts {
		opt(t)
	}
	return t
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

func (t *Tensor) Size() int {
	return totalSize(t.Shape)
}

func FromSlice(data []float64, shape ...int) *Tensor {
	if len(data) != totalSize(shape) {
		panic("tensors: data length does not match tensor shape dimensions")
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
	// 1. Exact Shape Match
	if sameShape(a.Shape, b.Shape) {
		outData := make([]float64, len(a.Data))
		for i := range a.Data {
			outData[i] = a.Data[i] + b.Data[i]
		}
		return FromSlice(outData, a.Shape...)
	}

	// 2. Broadcast Bias: A is [Batch, Dim], B is [1, Dim]
	if len(a.Shape) == 2 && len(b.Shape) == 2 && b.Shape[0] == 1 && a.Shape[1] == b.Shape[1] {
		batchSize := a.Shape[0]
		dim := a.Shape[1]
		outData := make([]float64, len(a.Data))

		for i := 0; i < batchSize; i++ {
			for j := 0; j < dim; j++ {
				idx := i*dim + j
				outData[idx] = a.Data[idx] + b.Data[j]
			}
		}
		return FromSlice(outData, a.Shape...)
	}

	panic(fmt.Sprintf("tensors: shape mismatch for addition (%v vs %v)", a.Shape, b.Shape))
}

func sameShape(s1, s2 []int) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
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

func Transpose(t *Tensor) *Tensor {
	if len(t.Shape) != 2 {
		panic("tensors: Transpose currently supports 2D matrices only")
	}

	rows, cols := t.Shape[0], t.Shape[1]
	out := New(cols, rows)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out.Data[j*rows+i] = t.Data[i*cols+j]
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

// String provides a PyTorch-like string representation of the Tensor.
func (t *Tensor) String() string {
	if t == nil || len(t.Data) == 0 {
		return "Tensor([])"
	}

	// 1. Format Extra Metadata (RequiresGrad & Grad.Data)
	var metaParts []string
	if t.RequiresGrad {
		metaParts = append(metaParts, "requires_grad=true")
		if t.Grad != nil && len(t.Grad.Data) > 0 {
			gradStr := formatSlice(t.Grad.Data, 6)
			metaParts = append(metaParts, fmt.Sprintf("grad=[%s]", gradStr))
		}
	}

	metaSuffix := ""
	if len(metaParts) > 0 {
		metaSuffix = ", " + strings.Join(metaParts, ", ")
	}

	// 2. 0D / Scalar or Empty Shape
	if len(t.Shape) == 0 {
		return fmt.Sprintf("Tensor(%.4f%s)", t.Data[0], metaSuffix)
	}

	// 3. 1D Vector
	if len(t.Shape) == 1 {
		return fmt.Sprintf("Tensor([%s]%s)", formatSlice(t.Data, 6), metaSuffix)
	}

	// 4. 2D Matrix
	if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]

		// Ensure strides exist to avoid index panics
		stride0, stride1 := cols, 1
		if len(t.Strides) == 2 {
			stride0, stride1 = t.Strides[0], t.Strides[1]
		}

		var sb strings.Builder
		sb.WriteString("Tensor([\n")

		maxRowsToShow := 8
		truncateRows := rows > maxRowsToShow

		for i := 0; i < rows; i++ {
			if truncateRows && i >= 4 && i < rows-4 {
				if i == 4 {
					sb.WriteString("  ...\n")
				}
				continue
			}

			sb.WriteString("  [")
			rowVals := make([]float64, cols)
			for j := 0; j < cols; j++ {
				idx := i*stride0 + j*stride1
				if idx < len(t.Data) {
					rowVals[j] = t.Data[idx]
				}
			}
			sb.WriteString(formatSlice(rowVals, 6))
			sb.WriteString("]")
			if i < rows-1 {
				sb.WriteString(",\n")
			}
		}
		sb.WriteString(fmt.Sprintf("\n], shape=%v%s)", t.Shape, metaSuffix))
		return sb.String()
	}

	// 5. Fallback for N-dimensional Tensors
	return fmt.Sprintf("Tensor([%s], shape=%v%s)", formatSlice(t.Data, 6), t.Shape, metaSuffix)
}

// Helper func to help display the format
func formatSlice(data []float64, maxItems int) string {
	if len(data) == 0 {
		return ""
	}

	if len(data) <= maxItems {
		strVals := make([]string, len(data))
		for i, v := range data {
			strVals[i] = fmt.Sprintf("%.4f", v)
		}
		return strings.Join(strVals, ", ")
	}

	// Truncate long vectors (e.g. [1.0000, 2.0000, 3.0000, ..., 8.0000, 9.0000, 10.0000])
	half := maxItems / 2
	head := make([]string, half)
	tail := make([]string, half)

	for i := 0; i < half; i++ {
		head[i] = fmt.Sprintf("%.4f", data[i])
		tail[i] = fmt.Sprintf("%.4f", data[len(data)-half+i])
	}

	return fmt.Sprintf("%s, ..., %s", strings.Join(head, ", "), strings.Join(tail, ", "))
}
