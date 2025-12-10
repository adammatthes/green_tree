package tensor

import (
	"testing"
	"math"
)

func TestInitLogisticRegression(t *testing.T) {
	lrm, err := InitLogisticRegression[float64, uint](uint(5), 0.0, 0.01, 10000)
	if err != nil {
		t.Errorf("Failed to create Logistic Regression Model")
	}

	if len(lrm.Weights.Data) != 5 {
		t.Errorf("Unexpected size of weights in logistic regression model. Expected 5, got %v\n", len(lrm.Weights.Data))
	}

	if lrm.Bias != 0.0 {
		t.Errorf("Unexpected Bias Value: %v\n", lrm.Bias)
	}

	if lrm.LearningRate != 0.01 {
		t.Errorf("Unexpected Learning Rate Value: %v\n", lrm.LearningRate)
	}

	if lrm.NumIterations != 10000 {
		t.Errorf("Unexpected Number of Iterations value: %v\n", lrm.NumIterations)
	}
}

func TestFitLogisticRegression(t *testing.T) {
	tol := 1e-2

	featuresData := []float64{1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0}
	features, _ := InitTensor64(4, 2)
	features.Data = featuresData

	targetData := []float64{1.0, 0.0, 0.0, 0.0}
	targets, _ := InitTensor64(4, 1)
	targets.Data = targetData

	numFeatures := uint64(2)
	bias := 0.0
	learningRate := 0.01
	numIterations := uint64(10000)

	model, err := InitLogisticRegression[float64, uint64](numFeatures, bias, learningRate, numIterations)

	if err != nil {
		t.Errorf("Failed to init logistic regression before Fit: %v\n", err)
	}

	initialPredict, _ := model.Predict(features)
	initialCost, _ := CalculateCost(initialPredict, targets)

	err = model.Fit(features, targets)
	if err != nil {
		t.Errorf("Error during logisitc Fit: %v\n", err)
	}

	for n := 1; n < len(model.CostHistory); n++ {
		curr := model.CostHistory[n]
		if curr > model.CostHistory[n - 1] {
			t.Errorf("Cost not declining at index %v", n)
		}
	}

	finalPredict, _ := model.Predict(features)
	finalCost, _ := CalculateCost(finalPredict, targets)

	if finalCost >= initialCost || finalCost > 0.1 {
		t.Errorf("Model failed to converge. Initial cost: %v, Final cost: %v\n", initialCost, finalCost)
	}

	predictedLabels, err := Classify(finalPredict, 0.5)
	if err != nil {
		t.Errorf("Classify failed: %v\n", err)
	}

	correctPredictions := 0

	for n := 0; n < len(predictedLabels.Data); n++ {
		if predictedLabels.Data[n] == targets.Data[n] {
			correctPredictions++
		}
	}

	accuracy := float64(correctPredictions) / float64(len(targets.Data))

	if math.Abs(accuracy - 1.0) > tol {
		t.Errorf("Model does not have high accuracy. Got %.2f, want 1.0", accuracy)
	}

}

func TestPredictLogisticRegression(t *testing.T) {
	tol := 1e-6

	input, _ := InitTensor64(2, 2)
	input.Data = []float64{1.0, 1.0, -1.0, -1.0}

	weights, _ := InitTensor64(2, 1)
	weights.Data = []float64{1.0, 1.0}

	bias := -1.0

	model, err := InitLogisticRegression[float64, uint64](uint64(2), bias, 0.01, uint64(100000))
	if err != nil {
		t.Errorf("Failed to init model during Predict Logistic Regression: %v\n", err)
	}

	model.Weights = weights

	expected := []float64{0.731058578, 0.047425873}

	predicted, err := model.Predict(input)
	if err != nil {
		t.Errorf("Predict method failed: %v\n", err)
	}

	if len(predicted.Data) != len(expected) {
		t.Errorf("predicted does not match length of expected: got %v expected 2", len(predicted.Data))
	}

	for n := 0; n < len(expected); n++ {
		if math.Abs(predicted.Data[n] - expected[n]) > tol {
			t.Errorf("Unexpected values in prediction: got %v expected %v", predicted.Data, expected)
		}
	}

}
