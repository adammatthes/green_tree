package tensor

import (
	"testing"
	"math/rand"
	"math"
)

func targetOutput(x1, x2 float64) float64 {
	return (10 + (5 * x1) - (2 * x2)) + (rand.Float64() - 1.0)
}

func TestInitLinearRegressionModel(t *testing.T) {
	lrm, err := InitLinearRegressionModel(uint(10), 0.05, 0.9, uint(100))
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
	numSamples := uint(1000)
	learningRate := 0.0000001
	momentum := 0.9
	maxIterations := uint(20000)

	expectedWeights := []float64{10.0, 5.0, -2.0}

	xBase, _ := InitRandomTensor([]uint{numSamples, 2}, 10.0)

	y, err := InitTargetTensor[float64, uint](xBase, expectedWeights)
	if err != nil {
		t.Errorf("Failed to init target tensor in Fit test")
	}

	xAug, err := xBase.AugmentBias()
	if err != nil {
		t.Errorf("AugmentBias failed in Fit test: %v\n", err)
	}

	if xAug.Data[0] != 1.0 {
		t.Errorf("Unexpected Value from AugmentBias: %v\n", xAug.Data[0])
	}

	numFeatures := xAug.Shape[1]
	lrm, err := InitLinearRegressionModel[float64, uint](numFeatures, learningRate, momentum, maxIterations)
	if err != nil {
		t.Errorf("Regression model init failed in Fit test: %v\n", err)
	}

	err = lrm.Fit(xAug, y)
	if err != nil {
		t.Errorf("Fit failed: %v\n", err)
	}

	epsilon := 0.05

	learnedWeights := lrm.Weights.Data

	if len(learnedWeights) != len(expectedWeights) {
		t.Errorf("Mismatch of expected weights: got %v, want %v\n", len(learnedWeights), len(expectedWeights))
	}

	for n, expected := range expectedWeights {
		actual := learnedWeights[n]

		diff := math.Abs(actual - expected)

		if diff > epsilon {
			t.Errorf("Weight failed convergence. Expected %v, got %v\n", expected, actual)
		}
	}

	if !lrm.Weights.Valid() {
		t.Errorf("Model contains invalid weights, i.e., NaN, Inf: %v\n", lrm.Weights)
	}
}
