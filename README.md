# go-tensor 🚀

A high-performance, lightweight N-dimensional tensor and deep learning framework written in pure Go. Inspired by PyTorch.

<!--[![Go Reference](https://pkg.go.dev/badge/github.com/redxdager/go-tensor.svg)](https://pkg.go.dev/github.com/redxdager/go-tensor)
[![Go Report Card](https://goreportcard.com/badge/github.com/redxdager/go-tensor)](https://goreportcard.com/report/github.com/redxdager/go-tensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)-->

## Features

- **Contiguous Memory Tensors** — Strided N-dimensional array operations backed by an efficient flat slice memory layout.
- **PyTorch-Style Formatting** — Human-readable multi-dimensional printing via `fmt.Stringer`, so tensors print cleanly with `fmt.Println` out of the box.
- **Modular Activations** — Integrated package for `ReLU`, `LeakyReLU`, `Sigmoid`, and `Tanh`.
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

`go-tensor` can now load datasets directly from CSV files and feed them through a shuffled, batched \`DataLoader\` — no manual parsing required.

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
| `dataset.LoadCSVDataset(path, targetCol, hasHeader)` | Parses a CSV file into a \`Dataset\`, splitting each row into input features and a target column. |
| `dataset.Dataset` | Interface any data source implements — just \`Len() int\` and \`Get(idx int) Sample\`. |
| `dataset.Sample` | A single \`(Input, Target)\` pair, mirroring PyTorch's \`__getitem__\` convention. |
| `dataset.NewDataLoader(ds, batchSize, shuffle)` | Wraps a \`Dataset\` for shuffled, batched iteration. |
| `loader.Reset()` | Resets the cursor to the start of an epoch and reshuffles indices if \`shuffle\` is enabled. |
| `loader.HasNext()` / \`loader.NextBatch()\` | Standard iterator pattern for pulling batches as \`*tensor.Tensor\` pairs. |

Because \`dataset.Dataset\` is just an interface, you can implement your own backing source (in-memory slices, JSON, a database cursor, etc.) and it will work with \`DataLoader\` automatically — CSV is just the built-in convenience loader.

## Project Layout

```
go-tensor/
├── go.mod
├── README.md
├── LICENSE
├── activations/
│   └── activation_functions.go
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
- [ ] Autograd / backpropagation
- [ ] Optimizers (SGD, Adam)
- [ ] Layer abstractions (Linear, Sequential)

## Contributing

Issues and pull requests are welcome. Please make sure \`go test ./... -v\` passes and run \`gofmt\` before submitting.

## License

Released under the [MIT License](LICENSE).
