package tensor

import (
	"testing"
	"math"
)

func TestFitStatistics(t *testing.T) {
	tol := 1e-5

	input, _ := InitTensor64(4, 2)
	if input.Shape[0] != 4 || input.Shape[1] != 2 || input.Strides[0] != 2 || input.Strides[1] != 1 {
		t.Errorf("Unexpected Shape or Strides: %v %v", input.Shape, input.Strides)
	}

	input.Data = []float64{10.0, 0.0, 20.0, 10.0, 30.0, 20.0, 40.0, 30.0}

	expectedMu := []float64{25.0, 15.0}
	expectedSigma := []float64{11.1803398875, 11.1803398875}

	scaler := &StandardScaler[float64, uint64]{}

	err := scaler.FitStatistics(input)
	if err != nil {
		t.Errorf("FitStatistics failed: %v", err)
	}

	if len(scaler.Mu) != 2 {
		t.Errorf("Unexpected length of Mu: %v", scaler.Mu)
	}

	for i := range scaler.Mu {
		if math.Abs(scaler.Mu[i] - expectedMu[i]) > tol {
			t.Errorf("Unexpected Mean Values: %v", scaler.Mu)
		}
	}

	if len(scaler.Sigma) != 2 {
		t.Errorf("Unexpected length of Sigma: %v", scaler.Sigma)
	}

	for i := range scaler.Sigma {
		if math.Abs(scaler.Sigma[i] - expectedSigma[i]) > tol {
			t.Errorf("Unexpected standard deviation values: %v", scaler.Sigma)
		}
	}
}

func TestScalerTransform(t *testing.T) {
	input, _ := InitTensor64(4, 2)
	input.Data = []float64{10.0, 0.0, 20.0, 10.0, 30.0, 20.0, 40.0, 30.0}

	scaler := &StandardScaler[float64, uint64]{}
	err := scaler.FitStatistics(input)
	if err != nil {
		t.Errorf("FitStatistics failed: %v\n", err)
	}

	transformed, err := scaler.Transform(input)
	if err != nil {
		t.Errorf("Transform using scaler failed: %v\n", err)
	}

	expected := []float64{-1.3416, -1.3416, -0.4472, -0.4472, 0.4472, 0.4472, 1.3416, 1.3416}

	tol := 1e-4

	for n := 0; n < len(expected); n++ {
		if math.Abs(transformed.Data[n] - expected[n]) > tol {
			t.Errorf("Unexpected transform values: %v", transformed.Data)
		}
	}
}
