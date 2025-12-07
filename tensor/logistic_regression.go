package tensor

import (
	"fmt"
	"errors"
)

type LogisticRegressionModel[T Numeric, S Index] struct {
	Weights *Tensor[T, S]
	Bias T
	LearningRate T
	NumIterations S
	CostHistory []T
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

	history := make([]T, 0, numIterations)

	model := &LogisticRegressionModel[T, S] {
		Weights:	weights,
		Bias:		bias,
		LearningRate:	learningRate,
		NumIterations:	numIterations,
		CostHistory:	history,
	}

	return model, nil
	




}

func (lrm *LogisticRegressionModel[T, S]) Fit(features *Tensor[T, S], targets *Tensor[T, S]) error {
	inputFirstShape := T(features.Shape[0])
	if inputFirstShape == 0 {
		return errors.New(fmt.Sprintf("Shape value of 0 would cause Zero-division error"))
	}

	for n := S(0); n < lrm.NumIterations; n++ {
		if !lrm.Weights.Valid() {
			fmt.Errorf("NaN or Inf found in Logistic Regression at iteration %v\n", n)
		}


		linearOutput, err := features.Dot(lrm.Weights)
		if err != nil {
			return err
		}

		linearOutput, err = linearOutput.AddScalar(lrm.Bias)
		if err != nil {
			return err
		}

		predictedProbabilities, err := Sigmoid(linearOutput)
		if err != nil {
			return err
		}

		errorTerm, err := predictedProbabilities.Subtract(targets)
		if err != nil {
			return err
		}

		transposedFeatures, err := features.Transpose()
		if err != nil {
			return err
		}

		preGradientWeights, err := transposedFeatures.Dot(errorTerm)
		if err != nil {
			return err
		}

		gradientWeights, err := preGradientWeights.MulScalar(T(1.0) / inputFirstShape)
		if err != nil {
			return err
		}

		errSum, err := errorTerm.Sum()
		if err != nil {
			return err
		}
		gradientBias := errSum / inputFirstShape

		scaledGradient, err := gradientWeights.MulScalar(lrm.LearningRate)

		lrm.Weights, err = lrm.Weights.Subtract(scaledGradient)
		if err != nil {
			return err
		}
		lrm.Bias = lrm.Bias - lrm.LearningRate * gradientBias

		cost, err := CalculateCost(predictedProbabilities, targets)
		if err != nil {
			return err
		}

		lrm.CostHistory = append(lrm.CostHistory, cost)
	}

	return nil
}

func (lrm *LogisticRegressionModel[T, S]) Predict(input *Tensor[T, S]) (*Tensor[T, S], error) {
	linearOutput, err := input.Dot(lrm.Weights)
	if err != nil {
		return &Tensor[T, S]{}, err
	}

	linearOutput, err = linearOutput.AddScalar(lrm.Bias)
	if err != nil {
		return &Tensor[T, S]{}, err
	}

	predicted, err := Sigmoid(linearOutput)
	if err != nil {
		return &Tensor[T, S]{}, err
	}

	return predicted, err
}
