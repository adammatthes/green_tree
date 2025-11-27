package tensor

import (
	"testing"
	"math/rand"
)

func targetOutput(x1, x2 float64) float64 {
	return (10 + (5 * x1) - (2 * x2)) + (rand.Float64() - 1.0)
}

func TestInitLinearRegressionModel(t *testing.T) {
	lrm, err := InitLinearRegressionModel(uint(10), 0.05, uint(100))
	if err != nil {
		t.Errorf("Failed to init Linear Regression Model")
	}

	if lrm.LearningRate != 0.05 || lrm.MaxIterations != 100 {
		t.Errorf("Linear Regrssion Model Values not initialized correctly")
	}

	if lrm.Weights.Shape[0] != uint(10) || lrm.Weights.Shape[1] != uint(1) {
		t.Errorf("Shape of Linear Regression weights incorrect")
	}

	for n := 0; n < len(lrm.Weights.Data); n++ {
		val := lrm.Weights.Data[n]
		if val > 1.0 || val < -1.0 {
			t.Errorf("Unexpected weight value: %v is out of range.", val)
			break
		}
	}
}

func TestFitLinearRegressionModel(t *testing.T) {
	features, err := InitTensor[float64, uint]([]uint{100, 3})
	if err != nil {
		t.Errorf("Failed to create features tensor during Fit test: %v\n", err)
	}

	for n := uint(0); n < features.Shape[0]; n++ {
		features.Set([]uint{n, 0}, 1.0)
		features.Set([]uint{n, 1}, rand.Float64() * 10.0 - 9.0)
		features.Set([]uint{n, 2}, rand.Float64() * 10.0 - 9.0)
	}

	targets, err := InitTensor[float64, uint]([]uint{100, 1})
	if err != nil {
		t.Errorf("Failed to create targets during Fit test: %v\n", err)
	}

	for n := uint(0); n < features.Shape[0]; n++ {
		x1, _ := features.Get([]uint{n, 1})
		x2, _ := features.Get([]uint{n, 2})
		targets.Data[n] = targetOutput(x1, x2)
	}

	lrm, err := InitLinearRegressionModel(uint(3), 0.01, uint(5000))
	if err != nil {
		t.Errorf("Failed to init Linear Regression Model during Fit test: %v\n", err)
	}

	err = lrm.Fit(features, targets)
	if err != nil {
		t.Errorf("Error during Fit method: %v\n", err)
	}

	if lrm.Weights.Data[0] < 9.9 || lrm.Weights.Data[0] > 10.1 || lrm.Weights.Data[1] < 4.9 || lrm.Weights.Data[1] > 5.1 || lrm.Weights.Data[2] < -2.1 || lrm.Weights.Data[2] > -1.9 {
		t.Errorf("Values don't match expected model")
	}
}
