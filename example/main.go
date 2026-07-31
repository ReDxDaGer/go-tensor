package main

import (
	"fmt"
	"log"

	"github.com/redxdager/go-tensor/autograd"
	"github.com/redxdager/go-tensor/dataset"
	tensor "github.com/redxdager/go-tensor/tensors"
)

func main() {

	fmt.Println("\n[1/4] Loading CSV Dataset & Initializing DataLoader...")

	// Load dataset: target is column index 2
	ds, err := dataset.LoadCSVDataset("data.csv", 2, true)
	if err != nil {
		log.Fatalf("Failed to load CSV dataset: %v", err)
	}

	fmt.Printf("✓ Dataset loaded successfully! Total Samples: %d\n", ds.Len())

	// Initialize DataLoader
	loader := dataset.NewDataLoader(ds, 2, true)

	fmt.Println("\n[2/4] Initializing Model Weights & Biases (RequiresGrad = true)...")

	// Input Dim = 2, Hidden Dim = 3, Output Dim = 1
	W1 := tensor.Randn(2, 3)
	W1.RequiresGrad = true
	W1.Grad = tensor.Zeros(W1.Shape...)

	B1 := tensor.Zeros(1, 3)
	B1.RequiresGrad = true
	B1.Grad = tensor.Zeros(B1.Shape...)

	W2 := tensor.Randn(3, 1)
	W2.RequiresGrad = true
	W2.Grad = tensor.Zeros(W2.Shape...)

	B2 := tensor.Zeros(1, 1)
	B2.RequiresGrad = true
	B2.Grad = tensor.Zeros(B2.Shape...)

	fmt.Printf("  - W1 Shape: %v | B1 Shape: %v\n", W1.Shape, B1.Shape)
	fmt.Printf("  - W2 Shape: %v | B2 Shape: %v\n", W2.Shape, B2.Shape)

	fmt.Println("\n[3/4] Running Forward Pass across DataLoader Batches...")

	batchCount := 0
	for loader.HasNext() {
		xBatch, yBatch, ok := loader.NextBatch()
		if !ok {
			break
		}
		batchCount++

		fmt.Printf("\n--- Processing Batch #%d (Batch Size: %d) ---\n", batchCount, xBatch.Shape[0])

		// Reset accumulated gradients before forward pass
		autograd.ZeroGrad(W1, B1, W2, B2)

		// Layer 1: H1 = ReLU( (X @ W1) + B1 )
		xw1 := autograd.MatMul(xBatch, W1)
		z1 := autograd.Add(xw1, B1)
		h1 := autograd.ReLU(z1)

		// Layer 2 (Output): Pred = Sigmoid( (H1 @ W2) + B2 )
		h1w2 := autograd.MatMul(h1, W2)
		z2 := autograd.Add(h1w2, B2)
		pred := autograd.Sigmoid(z2)

		// Compute Loss: MSE = Mean( (Pred - Target)^2 )
		diff := autograd.Add(pred, neg(yBatch)) // Pred - Y
		sqDiff := autograd.MatMul(tensor.Transpose(diff), diff)
		loss := autograd.Mean(sqDiff)

		fmt.Printf("  Batch %d Loss: %.6f\n", batchCount, loss.Data[0])

		fmt.Println("  Executing Autograd Reverse Topological Backward Pass...")
		autograd.Backward(loss)

		fmt.Println("  ✓ Gradients Computed:")
		fmt.Printf("    • dL/dW1 (Shape %v):\n%v\n", W1, W1)
		fmt.Printf("    • dL/dB1 (Shape %v):\n%v\n", B1.Grad.Shape, B1.Grad.Data)
		fmt.Printf("    • dL/dW2 (Shape %v):\n%v\n", W2.Grad.Shape, W2.Grad.Data)
		fmt.Printf("    • dL/dB2 (Shape %v):\n%v\n", B2.Grad.Shape, B2.Grad.Data)
	}

}

// Negation helper function: -X
func neg(t *tensor.Tensor) *tensor.Tensor {
	outData := make([]float64, len(t.Data))
	for i, val := range t.Data {
		outData[i] = -val
	}
	return tensor.FromSlice(outData, t.Shape...)
}
