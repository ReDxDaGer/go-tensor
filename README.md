# go-tensor 🚀

A high-performance, lightweight N-dimensional tensor and deep learning framework written in pure Go. Inspired by PyTorch.

<!--[![Go Reference](https://pkg.go.dev/badge/github.com/redxdager/go-tensor.svg)](https://pkg.go.dev/github.com/redxdager/go-tensor)
[![Go Report Card](https://goreportcard.com/badge/github.com/redxdager/go-tensor)](https://goreportcard.com/report/github.com/redxdager/go-tensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)-->

## Features

- **Contiguous Memory Tensors** — Strided N-dimensional array operations backed by an efficient flat slice memory layout.
- **PyTorch-Style Formatting** — Human-readable multi-dimensional printing.
- **Modular Activations** — Integrated package for `ReLU`, `LeakyReLU`, `Sigmoid`, and `Tanh`.
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

## Project Layout

```
go-tensor/
├── go.mod
├── README.md
├── LICENSE
├── activations/
│   └── activation_functions.go
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

## Contributing

Issues and pull requests are welcome. Please make sure `go test ./... -v` passes and run `gofmt` before submitting.

## License

Released under the [MIT License](LICENSE).
