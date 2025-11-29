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
	lrm, err := InitLinearRegressionModel(uint(10), 0.05, 0.9, 5.0, uint(100))
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
	learningRate := 0.00001
	momentum := 0.9
	threshold := 5.0
	maxIterations := uint(50000)

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
	lrm, err := InitLinearRegressionModel[float64, uint](
		numFeatures, learningRate, momentum, threshold, maxIterations)
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

func TestPredict(t *testing.T) {
	numSamples := uint(1000)
	learningRate := 0.0001
	momentum := 0.95
	threshold := 5.0
	maxIterations := uint(100000)

	expectedWeights := []float64{10.0, 5.0, -2.0}

	xBase, _ := InitRandomTensor[float64, uint]([]uint{numSamples, 2}, 10.0)
	yTargets, _ := InitTargetTensor[float64, uint](xBase, expectedWeights)

	xAug, _ := xBase.AugmentBias()

	lrm, _ := InitLinearRegressionModel[float64, uint](
		xAug.Shape[1], learningRate, momentum, threshold, maxIterations)

	err := lrm.Fit(xAug, yTargets)
	if err != nil {
		t.Errorf("Failed to fit model during Predict test: %v", err)
	}

	predictions, err := lrm.Predict(xBase)
	if err != nil {
		t.Errorf("Predict method failed: %v", err)
	}

	if predictions.Shape[0] != numSamples || predictions.Shape[1] != 1 {
		t.Errorf("Unexpected Predict output shape. Got %v, expected [%d, 1]", predictions.Shape, numSamples)
	}

	rmse, err := RootMeanSquareError(predictions, yTargets)
	if err != nil {
		t.Errorf("RMSE error: %v", err)
	}

	noiseFloor := 0.6

	if float64(rmse) > noiseFloor {
		t.Errorf("RMSE higher than expected. Got %.4f, expected below %.2f", rmse, noiseFloor)
	}

	epsilon := 1.1

	numFailed := 0.0
	diffSum := 0.0

	for n := uint(0); n < numSamples; n++ {
		actual := yTargets.Data[n]
		predicted := predictions.Data[n]

		diff := math.Abs(float64(actual - predicted))

		if diff > epsilon {
			//t.Errorf("Prediction failed at sample %v. Predicted %v, Actual %v", n, predicted, actual)
			numFailed += 1
			diffSum += diff
		}
	}

	if numFailed > 0 {
		t.Errorf("Average diff of expected versus actual: %v", diffSum / numFailed)
	}
}

func TestRMSE(t *testing.T) {
	predictions := &Tensor[float64, uint]{
		Data: []float64{10.0, 20.0, 30.0},
		Shape: []uint{3, 1},
		Strides: []uint{1, 1},
	}

	targets := &Tensor[float64, uint]{
		Data: []float64{11.0, 18.0, 27.0},
		Shape: []uint{3, 1},
		Strides: []uint{1, 1},
	}

	expectedRMSE := 2.1602
	epsilon :=1e-4

	actualRMSE, err := RootMeanSquareError(predictions, targets)
	if err != nil {
		t.Errorf("Failed to calculate RMSE: %v", err)
	}

	diff := math.Abs(actualRMSE - expectedRMSE)

	if diff > epsilon {
		t.Errorf("Diff of expected versus actual RMSE to large: %v > %v", diff, epsilon)
	}
}
