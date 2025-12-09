package tensor

import (
	"fmt"
)

type ConfusionMatrix[S Index] struct {
	TruePositives	S
	FalsePositives	S
	TrueNegatives	S
	FalseNegatives	S
}

func GenerateConfusionMatrix[T Numeric, S Index](
	actual *Tensor[T, S],
	predicted *Tensor[T, S]) (*ConfusionMatrix[S], error) {
	
	if len(actual.Data) != len(predicted.Data) {
		return &ConfusionMatrix[S]{}, fmt.Errorf("ConfusionMatrix error: Tensors must have equal length to compare")
	}

	numElements := len(actual.Data)
	matrix := &ConfusionMatrix[S]{}

	for n := 0; n < numElements; n++ {
		actualOutcome := actual.Data[n]
		predictedOutcome := predicted.Data[n]


		if actualOutcome == T(1.0) && predictedOutcome == T(1.0) {
			matrix.TruePositives++
		} else if actualOutcome == T(0.0) && predictedOutcome == T(1.0) {
			matrix.FalsePositives++
		} else if actualOutcome == T(0.0) && predictedOutcome == T(0.0) {
			matrix.TrueNegatives++
		} else if actualOutcome == T(1.0) && predictedOutcome == T(0.0) {
			matrix.FalseNegatives++
		} else {
			return &ConfusionMatrix[S]{}, fmt.Errorf("Generation of Confusion matrix expects label values to be 1 or 0")
		}
	}

	return matrix, nil
}

func (cm *ConfusionMatrix[S]) Precision() float64 {
	tp := float64(cm.TruePositives)
	fp := float64(cm.FalsePositives)

	if tp + fp == 0 {
		return 0.0
	}

	return tp / (tp + fp)
}

func (cm *ConfusionMatrix[S]) Recall() float64 {
	tp := float64(cm.TruePositives)
	fn := float64(cm.FalseNegatives)

	if tp + fn == 0 {
		return 0.0
	}

	return tp / (tp + fn)
}

func (cm *ConfusionMatrix[S]) F1Score() float64 {
	precision := cm.Precision()
	recall := cm.Recall()

	if precision + recall == 0 {
		return 0.0
	}

	return 2.0 * (precision * recall) / (precision + recall)
}
