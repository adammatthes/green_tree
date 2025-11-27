package tensor

import (
	"testing"
)

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
}
