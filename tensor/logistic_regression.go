package tensor

import (
	"fmt"
)

type LogisticRegressionModel[T Numeric, S Index] struct {
	Weights *Tensor[T, S]
	Bias T
	LearningRate T
	NumIterations S
}

func InitLogisticRegression[T Numeric, S Index](numFeatures S,
						bias T,
						learningRate T,
						numIterations S) (*LogisticRegressionModel[T, S], error) {
	
	maxVal := float64(0.01)
	weights, err := InitRandomTensor[T, S]([]S{numFeatures, S(1)}, T(maxVal))
	if err != nil {
		return &LogisticRegressionModel[T, S]{}, fmt.Errorf("Failed to create weights tensor during InitLogisticRegression: %v", err)
	}

	model := &LogisticRegressionModel[T, S] {
		Weights:	weights,
		Bias:		bias,
		LearningRate:	learningRate,
		NumIterations:	numIterations,
	}

	return model, nil
	




}
