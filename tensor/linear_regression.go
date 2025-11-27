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
