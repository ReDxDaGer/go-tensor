package dataset

import (
	"math/rand"

	tensor "github.com/redxdager/go-tensor/tensors"
)

type DataLoader struct {
	dataset   Dataset
	batchSize int
	Shuffle   bool
	indices   []int
	cursor    int
}

func NewDataLoader(ds Dataset, batchSize int, shuffle bool) *DataLoader {
	indices := make([]int, ds.Len())
	for i := range indices {
		indices[i] = i
	}

	loader := &DataLoader{
		dataset:   ds,
		batchSize: batchSize,
		Shuffle:   shuffle,
		indices:   indices,
		cursor:    0,
	}

	loader.Reset()
	return loader
}

func (dl *DataLoader) Reset() {
	dl.cursor = 0
	if dl.Shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}
}

func (dl *DataLoader) HasNext() bool {
	return dl.cursor < dl.dataset.Len()
}

func (dl *DataLoader) NextBatch() (*tensor.Tensor, *tensor.Tensor, bool) {
	if !dl.HasNext() {
		return nil, nil, false
	}

	end := dl.cursor + dl.batchSize
	if end > dl.dataset.Len() {
		end = dl.dataset.Len()
	}

	batchIndices := dl.indices[dl.cursor:end]
	currentBatchSize := len(batchIndices)
	dl.cursor = end

	firstSample := dl.dataset.Get(batchIndices[0])
	inputDim := len(firstSample.Input)
	targetDim := len(firstSample.Target)

	xData := make([]float64, 0, currentBatchSize*inputDim)
	yData := make([]float64, 0, currentBatchSize*targetDim)

	for _, idx := range batchIndices {
		s := dl.dataset.Get(idx)
		xData = append(xData, s.Input...)
		yData = append(yData, s.Target...)
	}

	xTensor := tensor.FromSlice(xData, currentBatchSize, inputDim)
	yTensor := tensor.FromSlice(yData, currentBatchSize, targetDim)

	return xTensor, yTensor, true
}
