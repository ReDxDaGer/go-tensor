package dataset

type Sample struct {
	Input  []float64
	Target []float64
}

type Dataset interface {
	Len() int
	Get(index int) Sample
}

type TensorDataset struct {
	Input  [][]float64
	Target [][]float64
}

func NewTensorDataset(input [][]float64, target [][]float64) *TensorDataset {
	if len(input) != len(target) {
		panic("Inputs and targets must have the same length")
	}
	return &TensorDataset{Input: input, Target: target}
}

func (d *TensorDataset) Len() int {
	return len(d.Input)
}
func (d *TensorDataset) Get(index int) Sample {
	return Sample{
		Input:  d.Input[index],
		Target: d.Target[index],
	}
}
