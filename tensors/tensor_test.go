package tensor

import (
	"math"
	"strings"
	"testing"
)

// Helper function to check float equality within a tolerance
func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestTensorCreation(t *testing.T) {
	// 1. Test FromSlice and Strides calculation
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	tensor := FromSlice(data, 2, 3)

	if len(tensor.Shape) != 2 || tensor.Shape[0] != 2 || tensor.Shape[1] != 3 {
		t.Fatalf("Expected shape [2, 3], got %v", tensor.Shape)
	}

	if len(tensor.Strides) != 2 || tensor.Strides[0] != 3 || tensor.Strides[1] != 1 {
		t.Fatalf("Expected strides [3, 1], got %v", tensor.Strides)
	}

	// 2. Test Zeros and Ones
	z := Zeros(3, 3)
	for _, val := range z.Data {
		if val != 0.0 {
			t.Errorf("Expected zeros, got %f", val)
		}
	}

	o := Ones(2, 4)
	for _, val := range o.Data {
		if val != 1.0 {
			t.Errorf("Expected ones, got %f", val)
		}
	}
}

func TestMatMulShapeAndValues(t *testing.T) {
	// Matrix A (2x3) @ Matrix B (3x2) -> Result (2x2)
	// A = [[1, 2, 3],
	//      [4, 5, 6]]
	// B = [[7, 8],
	//      [9, 1],
	//      [2, 3]]
	aData := []float64{1, 2, 3, 4, 5, 6}
	bData := []float64{7, 8, 9, 1, 2, 3}

	A := FromSlice(aData, 2, 3)
	B := FromSlice(bData, 3, 2)

	// Result should be:
	// [[1*7 + 2*9 + 3*2,  1*8 + 2*1 + 3*3],
	//  [4*7 + 5*9 + 6*2,  4*8 + 5*1 + 6*3]]
	// = [[31, 19],
	//    [85, 55]]
	expected := []float64{31, 19, 85, 55}

	res := MatMul(A, B)

	if res.Shape[0] != 2 || res.Shape[1] != 2 {
		t.Fatalf("Expected output shape [2, 2], got %v", res.Shape)
	}

	for i, val := range res.Data {
		if !almostEqual(val, expected[i], 1e-5) {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}

func TestBroadcastingAddition(t *testing.T) {
	// A (2x3) + B (1x3) -> Bias broadcasting across batches
	aData := []float64{
		1, 2, 3,
		4, 5, 6,
	}
	bData := []float64{10, 20, 30}

	A := FromSlice(aData, 2, 3)
	B := FromSlice(bData, 1, 3)

	res := Add(A, B)

	expected := []float64{
		11, 22, 33,
		14, 25, 36,
	}

	for i, val := range res.Data {
		if !almostEqual(val, expected[i], 1e-5) {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}

func TestStringFormatting(t *testing.T) {
	// Create a tensor requiring gradients and verify string output format
	tensor := FromSlice([]float64{1.23456, 2.34567, 3.45678, 4.56789}, 2, 2)
	tensor.RequiresGrad = true
	tensor.Grad = Zeros(2, 2)
	tensor.Grad.Data[0] = 0.5 // Simulate gradient tracking

	str := tensor.String()

	t.Logf("Formatted Output:\n%s", str)

	// Checks
	if !strings.Contains(str, "requires_grad=true") {
		t.Errorf("Expected 'requires_grad=true' in string output")
	}
	if !strings.Contains(str, "grad=[0.5000") {
		t.Errorf("Expected gradient data formatted in string output")
	}
	if !strings.Contains(str, "shape=[2 2]") {
		t.Errorf("Expected shape metadata in string output")
	}
}

func TestLargeTensorTruncationInString(t *testing.T) {
	// Create a large 1D tensor (50 elements) and verify ellipsis truncation
	data := make([]float64, 50)
	for i := range data {
		data[i] = float64(i)
	}

	largeTensor := FromSlice(data, 50)
	str := largeTensor.String()

	t.Logf("Truncated Output:\n%s", str)

	if !strings.Contains(str, "...") {
		t.Errorf("Expected truncation '...' for large tensor printing")
	}
}
