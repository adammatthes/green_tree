package tensor

import (
	"testing"
)

func TestConfusionMatrixCreation(t *testing.T) {
	t1, _ := InitTensor64(2, 2)
	t2, _ := InitTensor64(2, 2)

	t1.Data = []float64{1.0, 1.0, 1.0, 1.0}

	matrix, err := GenerateConfusionMatrix(t1, t2)
	if err != nil {
		t.Errorf("Failed to generate confusion matrix: %v\n", err)
	}

	if matrix.FalseNegatives != uint64(4) {
		t.Errorf("Unexpected values in confusion matrix: %v", matrix)
	}
}
