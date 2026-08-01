# go-tensor 🚀

A high-performance, lightweight N-dimensional tensor and deep learning framework written in pure Go. Inspired by PyTorch.

<!--[![Go Reference](https://pkg.go.dev/badge/github.com/redxdager/go-tensor.svg)](https://pkg.go.dev/github.com/redxdager/go-tensor)
[![Go Report Card](https://goreportcard.com/badge/github.com/redxdager/go-tensor)](https://goreportcard.com/report/github.com/redxdager/go-tensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)-->

## Features

- **Contiguous Memory Tensors** — Strided N-dimensional array operations backed by an efficient flat slice memory layout.
- **PyTorch-Style Formatting** — Human-readable multi-dimensional printing via `fmt.Stringer`, so tensors print cleanly with `fmt.Println` out of the box.
- **Modular Activations** — Integrated package for `ReLU`, `LeakyReLU`, `Sigmoid`, and `Tanh`.
- **Autograd (Reverse-Mode Automatic Differentiation)** — Build a dynamic computation graph with `RequiresGrad` tensors and call `autograd.Backward` to compute gradients via reverse topological traversal — no manual backward passes required.
- **Datasets & DataLoader** — Load tabular data straight from CSV files and iterate over it in shuffled, batched form, PyTorch `DataLoader`-style.
- **Pure Go** — Zero heavy C dependencies, fast compilation, and ready for Go backends.

## Installation

```bash
go get github.com/redxdager/go-tensor
```

## Quick Start

```go
package main

import (
	"fmt"

	"github.com/redxdager/go-tensor/tensors"
)

func main() {
	// Initialize random weights (3x2) and input (1x3)
	weights := tensors.Rand(3, 2)
	input := tensors.Randn(1, 3)

	// Matrix multiplication
	output := tensors.MatMul(input, weights)

	// Apply LeakyReLU
	activated := output.LeakyReLU(0.01)

	fmt.Println("Resulting Tensor:")
	fmt.Println(activated)
}
```

## Loading Data from CSV

`go-tensor` can load datasets directly from CSV files and feed them through a shuffled, batched `DataLoader` — no manual parsing required.

```go
package main

import (
	"fmt"
	"log"

	"github.com/redxdager/go-tensor/dataset"
)

func main() {
	// LoadCSVDataset(path, targetColumnIndex, hasHeader)
	ds, err := dataset.LoadCSVDataset("data.csv", 3, true)
	if err != nil {
		log.Fatalf("error loading CSV dataset: %v", err)
	}
	fmt.Printf("Loaded %d samples\n", ds.Len())

	// Wrap it in a DataLoader for shuffled, batched iteration
	loader := dataset.NewDataLoader(ds, 4, true)

	for epoch := 1; epoch <= 2; epoch++ {
		loader.Reset() // reshuffles indices and resets the cursor
		for loader.HasNext() {
			xBatch, yBatch, ok := loader.NextBatch()
			if !ok {
				break
			}
			fmt.Println(xBatch) // inputs
			fmt.Println(yBatch) // targets
		}
	}
}
```

### How it works

| Component | Description |
|---|---|
| `dataset.LoadCSVDataset(path, targetCol, hasHeader)` | Parses a CSV file into a `Dataset`, splitting each row into input features and a target column. |
| `dataset.Dataset` | Interface any data source implements — just `Len() int` and `Get(idx int) Sample`. |
| `dataset.Sample` | A single `(Input, Target)` pair, mirroring PyTorch's `__getitem__` convention. |
| `dataset.NewDataLoader(ds, batchSize, shuffle)` | Wraps a `Dataset` for shuffled, batched iteration. |
| `loader.Reset()` | Resets the cursor to the start of an epoch and reshuffles indices if `shuffle` is enabled. |
| `loader.HasNext()` / `loader.NextBatch()` | Standard iterator pattern for pulling batches as `*tensor.Tensor` pairs. |

Because `dataset.Dataset` is just an interface, you can implement your own backing source (in-memory slices, JSON, a database cursor, etc.) and it will work with `DataLoader` automatically — CSV is just the built-in convenience loader.

## Autograd & Training a Small Network

`go-tensor` now ships an `autograd` package implementing reverse-mode automatic differentiation. Mark tensors with `RequiresGrad = true`, build a forward pass using `autograd.*` ops (which record a computation graph as they run), then call `autograd.Backward` once on your loss to populate `.Grad` on every parameter — no hand-written derivatives needed.

```go
package main

import (
	"fmt"
	"log"

	"github.com/redxdager/go-tensor/autograd"
	"github.com/redxdager/go-tensor/dataset"
	tensor "github.com/redxdager/go-tensor/tensors"
)

func main() {
	// 1. Load data
	ds, err := dataset.LoadCSVDataset("data.csv", 2, true)
	if err != nil {
		log.Fatalf("failed to load CSV dataset: %v", err)
	}
	loader := dataset.NewDataLoader(ds, 2, true)

	// 2. Initialize parameters that require gradients
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

	for loader.HasNext() {
		xBatch, yBatch, ok := loader.NextBatch()
		if !ok {
			break
		}

		// Always clear accumulated gradients before a new forward pass
		autograd.ZeroGrad(W1, B1, W2, B2)

		// Forward pass: Hidden = ReLU(X @ W1 + B1), Pred = Sigmoid(Hidden @ W2 + B2)
		h1 := autograd.ReLU(autograd.Add(autograd.MatMul(xBatch, W1), B1))
		pred := autograd.Sigmoid(autograd.Add(autograd.MatMul(h1, W2), B2))

		// Loss (MSE-style)
		diff := autograd.Add(pred, neg(yBatch))
		loss := autograd.Mean(autograd.MatMul(tensor.Transpose(diff), diff))
		fmt.Printf("loss: %.6f\n", loss.Data[0])

		// Backward pass: walks the graph in reverse topological order
		autograd.Backward(loss)

		// Gradients are now populated on every RequiresGrad tensor
		fmt.Println("dL/dW1:", W1.Grad.Data)
		fmt.Println("dL/dB1:", B1.Grad.Data)
		fmt.Println("dL/dW2:", W2.Grad.Data)
		fmt.Println("dL/dB2:", B2.Grad.Data)
	}
}

func neg(t *tensor.Tensor) *tensor.Tensor {
	out := make([]float64, len(t.Data))
	for i, v := range t.Data {
		out[i] = -v
	}
	return tensor.FromSlice(out, t.Shape...)
}
```

### Autograd API

| Component | Description |
|---|---|
| `Tensor.RequiresGrad` | Marks a tensor as a leaf parameter that should accumulate gradients. |
| `Tensor.Grad` | Holds the accumulated gradient tensor; must be initialized (e.g. `tensor.Zeros(shape...)`) before the first backward pass. |
| `autograd.MatMul`, `autograd.Add`, `autograd.ReLU`, `autograd.Sigmoid`, `autograd.Mean` | Differentiable ops — each records itself on a dynamic computation graph as it executes, alongside the raw tensor math. |
| `autograd.ZeroGrad(params...)` | Clears accumulated gradients on the given tensors; call this at the start of every batch/step so gradients don't accumulate across iterations. |
| `autograd.Backward(loss)` | Traverses the computation graph in reverse topological order from `loss`, populating `.Grad` on every upstream `RequiresGrad` tensor via the chain rule. |

This is enough to hand-roll a training loop (forward → loss → `Backward` → your own SGD update) today. A built-in `optimizers` package (SGD, Adam) is next on the roadmap so the parameter-update step doesn't have to be written by hand.

## Project Layout

```
go-tensor/
├── go.mod
├── README.md
├── LICENSE
├── activations/
│   └── activation_functions.go
├── autograd/
│   └── autograd.go
├── dataset/
│   ├── dataset.go
│   ├── dataloader.go
│   └── csv_loader.go
├── tensors/
│   ├── tensor.go
│   └── tensor_test.go
└── example/
    └── main.go
```

## Running Tests

```bash
go test ./... -v
```

## Documentation

Full API documentation is available on [pkg.go.dev](https://pkg.go.dev/github.com/redxdager/go-tensor).

## Roadmap

- [x] Core N-dimensional tensor engine
- [x] Activation functions
- [x] CSV dataset loading + DataLoader
- [x] Autograd / reverse-mode backpropagation
- [ ] Optimizers (SGD, Adam)
- [ ] Layer abstractions (Linear, Sequential)

## Contributing

Issues and pull requests are welcome. Please make sure `go test ./... -v` passes and run `gofmt` before submitting.

## License

Released under the [MIT License](LICENSE).
