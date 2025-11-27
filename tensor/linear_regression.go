package tensor

import (
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type LinearRegressionModel[T Numeric, S Index] struct {
	Weights		*Tensor[T, S]
	LearningRate 	T
	MaxIterations	S
}

func InitLinearRegressionModel[T Numeric, S Index](
	numFeatures S,
	learningRate T,
	maxIterations S) (*LinearRegressionModel[T, S], error) {

	weightShape := []S{numFeatures, 1}

	weights, err := InitTensor[T, S](weightShape)
	if err != nil {
		return &LinearRegressionModel[T, S]{}, err
	}

	for n := range weights.Data {
		randomVal := rand.Float64() * 2.0 - 1.0
		weights.Data[n] = T(randomVal * 0.01)
	}

	model := LinearRegressionModel[T, S] {
		Weights:	weights,
		LearningRate:	learningRate,
		MaxIterations:	maxIterations,
	}

	return &model, nil
}

func (lrm *LinearRegressionModel[T, S]) Fit(X *Tensor[T, S], Y *Tensor[T, S]) error {
	X_T, err := X.Transpose()
	if err != nil {
		return err
	}

	for n := S(0); n < lrm.MaxIterations; n++ {
		predictions, err := X.Dot(lrm.Weights)
		if err != nil {
			return err
		}

		errorVector, err := predictions.Subtract(Y)
		if err != nil {
			return err
		}

		gradient, err := X_T.Dot(errorVector)
		if err != nil {
			return err
		}

		adjustment, err := gradient.MulScalar(lrm.LearningRate)
		if err != nil {
			return err
		}

		newWeights, err := lrm.Weights.Subtract(adjustment)
		if err != nil {
			return err
		}

		lrm.Weights = newWeights
	}

	return nil
}
