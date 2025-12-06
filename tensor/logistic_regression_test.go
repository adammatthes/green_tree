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
