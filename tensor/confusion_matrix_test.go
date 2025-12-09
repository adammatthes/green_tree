package tensor

import (
	"testing"
	"math"
)

func TestFalseNegative(t *testing.T) {
	t1, _ := InitTensor64(2, 2)
	t2, _ := InitTensor64(2, 2)

	t1.Data = []float64{1.0, 1.0, 1.0, 1.0}

	matrix, err := GenerateConfusionMatrix(t1, t2)
	if err != nil {
		t.Errorf("Failed to generate confusion matrix: %v\n", err)
	}

	if matrix.FalseNegatives != uint64(4) {
		t.Errorf("Unexpected values for false negative: %v\n", matrix.FalseNegatives)
	}
}

func TestAllCorrect(t *testing.T) {
	t1, _ := InitTensor64(2, 2)
	t2, _ := InitTensor64(2, 2)
	t1.Data = []float64{1.0, 1.0, 1.0, 1.0}
	t2.Data = []float64{1.0, 1.0, 1.0, 1.0}

	matrix, err := GenerateConfusionMatrix(t1, t2)
	if err != nil {
		t.Errorf("Failed to generate confusion matrix: %v\n", err)
	}

	if matrix.TruePositives != uint64(4) {
		t.Errorf("Unexpected value for true positive: %v\n", matrix.TruePositives)
	}
}

func TestTrueNegative(t *testing.T) {
	t1, _ := InitTensor64(2, 2)
	t2, _ := InitTensor64(2, 2)

	matrix, err := GenerateConfusionMatrix(t1, t2)
	if err != nil {
		t.Errorf("Failed to generate confusion matrix: %v\n", err)
	}

	if matrix.TrueNegatives != uint64(4) {
		t.Errorf("Unexpected value for True Negative: %v\n", matrix.TrueNegatives)
	}
}

func TestAllWrong(t *testing.T) {
	t1, _ := InitTensor64(2, 2)
	t2, _ := InitTensor64(2, 2)

	t2.Data = []float64{1.0, 1.0, 1.0, 1.0}

	matrix, err := GenerateConfusionMatrix(t1, t2)
	if err != nil {
		t.Errorf("Failed to generate confusion matrix: %v\n", err)
	}

	if matrix.FalsePositives != uint64(4) {
		t.Errorf("Unexpected value for False positive: %v\n", matrix.FalsePositives)
	}
}

func TestMixedScenario(t *testing.T) {
	t1, _ := InitTensor64(10)
	t2, _ := InitTensor64(10)

	t1.Data = []float64{1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0}
	t2.Data = []float64{1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}

	matrix, err := GenerateConfusionMatrix(t1, t2)
	if err != nil {
		t.Errorf("Failed to generate confusion matrix during mixed test: %v\n", err)
	}

	if matrix.TruePositives != uint64(4) {
		t.Errorf("Unexpected number of true positives: %v\n", matrix.TruePositives)
	}

	if matrix.FalsePositives != uint64(2) {
		t.Errorf("Unexpected number of false positives: %v\n", matrix.FalsePositives)
	}

	if matrix.TrueNegatives != uint64(3) {
		t.Errorf("Unexpected number of true negatives: %v\n", matrix.TrueNegatives)
	}

	if matrix.FalseNegatives != uint64(1) {
		t.Errorf("Unexpected number of false negatives: %v\n", matrix.FalseNegatives)
	}

	tol := 1e-5

	if math.Abs(matrix.Precision() - 0.666666) > tol {
		t.Errorf("Unexpected precision score: %v\n", matrix.Precision())
	}

	if math.Abs(matrix.Recall() - 0.8) > tol {
		t.Errorf("Unexpected recall score: %v\n", matrix.Recall())
	}

	if math.Abs(matrix.F1Score() - 0.727272) > tol {
		t.Errorf("Unexpected f1 score: %v\n", matrix.F1Score())
	}
}
