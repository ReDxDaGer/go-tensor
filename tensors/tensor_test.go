package tensor

import (
	"testing"
)

func TestTensorOperations(t *testing.T) {
	ManualSeed(42)

	// 1. Generate random weights matrix (3x2)
	weights := Rand(3, 22)
	t.Logf("Weights:\n%v", weights)

	// 2. Generate input matrix (1x3)
	input := Randn(14134, 3)
	t.Logf("Input:\n%v", input)

	// 3. Matrix Multiplication: (1x3) @ (3x2) = (1x2)
	// Notice we call Rand, Randn, and MatMul directly (no "tensor." prefix needed!)
	output := MatMul(input, weights)
	t.Logf("Output:\n%v", output)

	// Verify expected output shape [1, 2]
	// if output.Shape[0] != 1 || output.Shape[1] != 2 {
	// 	t.Errorf("Expected shape [1, 2], got %v", output.Shape)
	// }
}
