package main

import (
	"fmt"
	"log"
	"os"

	"github.com/redxdager/go-tensor/dataset"
)

// createDummyCSV writes a small sample dataset to disk if it doesn't
// already exist, so the example runs out of the box with no setup.
func createDummyCSV(filename string) error {
	if _, err := os.Stat(filename); err == nil {
		// File already exists — nothing to do.
		return nil
	}

	content := `feature1,feature2,feature3,label
1.0,2.0,3.0,0.0
4.0,5.0,6.0,1.0
7.0,8.0,9.0,0.0
10.0,11.0,12.0,1.0
13.0,14.0,15.0,0.0
16.0,17.0,18.0,1.0
19.0,20.0,21.0,0.0
22.0,23.0,24.0,1.0
25.0,26.0,27.0,0.0
28.0,29.0,30.0,1.0`

	return os.WriteFile(filename, []byte(content), 0644)
}

func main() {
	const csvPath = "data.csv"

	// 1. Ensure sample data exists.
	if err := createDummyCSV(csvPath); err != nil {
		log.Fatalf("failed to create dummy CSV: %v", err)
	}

	// 2. Load the dataset from CSV.
	// Signature: LoadCSVDataset(path string, targetCol int, hasHeader bool)
	fmt.Println("=== 1. Loading CSV Dataset ===")
	ds, err := dataset.LoadCSVDataset(csvPath, 3, true)
	if err != nil {
		log.Fatalf("error loading CSV dataset: %v", err)
	}
	fmt.Printf("Loaded dataset with %d total samples.\n\n", ds.Len())

	// 3. Wrap it in a DataLoader for shuffled, batched iteration.
	const batchSize = 4
	const shuffle = true
	loader := dataset.NewDataLoader(ds, batchSize, shuffle)

	fmt.Println("=== 2. Iterating Batches Across 2 Epochs ===")
	for epoch := 1; epoch <= 2; epoch++ {
		fmt.Printf("\n--- Epoch %d ---\n", epoch)
		loader.Reset() // Resets the cursor and reshuffles indices.

		batchNum := 1
		for loader.HasNext() {
			xBatch, yBatch, ok := loader.NextBatch()
			if !ok {
				break
			}

			fmt.Printf("[Batch %d]\n", batchNum)
			fmt.Printf("  Inputs:\n%v\n", xBatch)
			fmt.Printf("  Targets:\n%v\n", yBatch)
			batchNum++
		}
	}

	fmt.Println("\n✅ CSV loading and DataLoader verification complete!")
}
