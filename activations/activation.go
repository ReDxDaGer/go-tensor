package activations

import (
	"math"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x))
}

func ReLu(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}

func Tanh(x float64) float64 {
	numerator := math.Exp(x) - math.Exp(-x)
	denominator := math.Exp(x) + math.Exp(-x)

	return numerator / denominator
}

func ReLU_derivative(x float64) float64 {
	if x > 0 {
		return 1.0
	} else {
		return 0.0
	}
}

func LeakyReLU(x float64, alpha float64) float64 {
	if x > 0 {
		return x
	} else {
		return alpha * x
	}
}

func LeakyReluDerivative(x float64, alpha float64) float64 {
	if x > 0 {
		return 1.0
	} else {
		return alpha
	}
}

func ApplySlice(input []float64, fn func(float64) float64) []float64 {
	output := make([]float64, len(input))
	for i, val := range input {
		output[i] = fn(val)
	}
	return output
}
