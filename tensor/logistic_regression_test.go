package tensor

import (
	"testing"
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

	featuresData := []float64{1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0}
	features, _ := InitTensor64(4, 2)
	features.Data = featuresData

	targetData := []float64{1.0, 0.0, 0.0, 0.0}
	targets, _ := InitTensor64(4, 1)
	targets.Data = targetData

	numFeatures := uint64(2)
	bias := 0.0
	learningRate := 0.1
	numIterations := uint64(1000)

	model, err := InitLogisticRegression[float64, uint64](numFeatures, bias, learningRate, numIterations)

	if err != nil {
		t.Errorf("Failed to init logistic regression before Fit: %v\n", err)
	}

	err = model.Fit(features, targets)
	if err != nil {
		t.Errorf("Error during logisitc Fit: %v\n", err)
	}

	if model.Weights.Data[0] < 0.1 || model.Weights.Data[1] < 0.1 {
		t.Errorf("Unexpected values for weights in model: %v\n", model.Weights.Data)
	}

	if model.Bias > -0.1 {
		t.Errorf("Bias did not converge to a negative value: %v", model.Bias)
	}

}
