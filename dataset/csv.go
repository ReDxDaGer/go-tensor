package dataset

import (
	"encoding/csv"
	"os"
	"strconv"
)

// LoadCSVDataset reads a CSV file and separates feature columns from label columns
func LoadCSVDataset(filePath string, targetColIndex int, hasHeader bool) (*TensorDataset, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	startRow := 0
	if hasHeader {
		startRow = 1
	}

	var inputs [][]float64
	var targets [][]float64

	for i := startRow; i < len(records); i++ {
		row := records[i]
		var inputRow []float64
		var targetRow []float64

		for colIdx, valStr := range row {
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil {
				continue // skip invalid parse
			}

			if colIdx == targetColIndex {
				targetRow = append(targetRow, val)
			} else {
				inputRow = append(inputRow, val)
			}
		}

		inputs = append(inputs, inputRow)
		targets = append(targets, targetRow)
	}

	return NewTensorDataset(inputs, targets), nil
}
