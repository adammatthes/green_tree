package tensor

import (
	"math/rand"
	"time"
	"errors"
	"fmt"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type LinearRegressionModel[T Numeric, S Index] struct {
	Weights		*Tensor[T, S]
	LearningRate 	T
	MaxIterations	S
	MomentumRate	T
	Velocity	*Tensor[T, S]
}

func InitLinearRegressionModel[T Numeric, S Index](
	numFeatures S,
	learningRate T,
	momentumRate T,
	maxIterations S) (*LinearRegressionModel[T, S], error) {

	weightShape := []S{numFeatures, 1}

	weights, err := InitTensor[T, S](weightShape)
	if err != nil {
		return &LinearRegressionModel[T, S]{}, err
	}

	for n := range weights.Data {
		randomVal := rand.Float64() * 1.25 - 1.0
		weights.Data[n] = T(randomVal * 0.000001)
	}

	velocity, err := InitTensor[T, S](weightShape)
	if err != nil {
		return &LinearRegressionModel[T, S]{}, err
	}

	model := LinearRegressionModel[T, S] {
		Weights:	weights,
		LearningRate:	learningRate,
		MaxIterations:	maxIterations,
		MomentumRate: 	momentumRate,
		Velocity:	velocity,
	}

	return &model, nil
}

func (lrm *LinearRegressionModel[T, S]) Fit(X *Tensor[T, S], Y *Tensor[T, S]) error {
	X_T, err := X.Transpose()
	if err != nil {
		return err
	}

	for n := S(0); n < lrm.MaxIterations; n++ {
		if !lrm.Weights.Valid() {
			return errors.New(fmt.Sprintf("NaN or infinity introduced after %v iterations", n))
		}


		predictions, err := X.Dot(lrm.Weights)
		if err != nil {
			return err
		}

		/*if !predictions.Valid() {
			return errors.New(fmt.Sprintf("NaN introduced to predictions after %v iterations", n))
		}*/

		errorVector, err := predictions.Subtract(Y)
		if err != nil {
			return err
		}

		/*if !errorVector.Valid() {
			return errors.New(fmt.Sprintf("NaN introduced to errorVector after %v iterations", n))
		}*/

		gradient, err := X_T.Dot(errorVector)
		if err != nil {
			return err
		}

		/*if !gradient.Valid() {
			return errors.New(fmt.Sprintf("NaN introduced to gradient after %v iterations", n))
		}*/

		velocityOldScaled, err := lrm.Velocity.MulScalar(lrm.MomentumRate)
		if err != nil {
			return err
		}

		gradientScaled, err := gradient.MulScalar(lrm.LearningRate)
		if err != nil {
			return err
		}

		/*if !gradientScaled.Valid() {
			return errors.New(fmt.Sprintf("NaN introduced to gradientScaled after %v iterations", n))
		}*/

		velocityNew, err := velocityOldScaled.Add(gradientScaled)
		if err != nil {
			return err
		}

		lrm.Velocity = velocityNew

		newWeights, err := lrm.Weights.Subtract(velocityNew)
		if err != nil {
			return err
		}

		lrm.Weights = newWeights
	}

	return nil
}
