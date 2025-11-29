package tensor

import (
	"math/rand"
	"math"
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
	ClipThreshold	T
	Velocity	*Tensor[T, S]
}

func InitLinearRegressionModel[T Numeric, S Index](
	numFeatures S,
	learningRate T,
	momentumRate T,
	clipThreshold T,
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
		ClipThreshold: 	clipThreshold,
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
		threshold := float64(lrm.ClipThreshold)
		if threshold > 0 {
			gradientNorm, err := gradient.Norm()
			if err != nil {
				return err
			}

			if math.Abs(float64(gradientNorm)) > threshold {
				scalingFactor := T(threshold / float64(gradientNorm))

				gradient, err = gradient.MulScalar(scalingFactor)
				if err != nil {
					return err
				}
			}
		}

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

func (lrm *LinearRegressionModel[T, S]) Predict(xNew *Tensor[T, S]) (*Tensor[T, S], error) {
	if len(xNew.Shape) != 2 {
		return &Tensor[T, S]{}, errors.New("Predict requires a 2D feature tensor")
	}

	xAug, err := xNew.AugmentBias()
	if err != nil {
		return &Tensor[T, S]{}, errors.New(fmt.Sprintf("AugmentBias failed during Predict: %v", err))
	}

	numFeatures := xAug.Shape[1]
	numWeights := lrm.Weights.Shape[0]

	if numFeatures != numWeights {
		return &Tensor[T, S]{}, errors.New(fmt.Sprintf("Dimensions do not match: %v != %v", numFeatures, numWeights))
	}

	predictions, err := xAug.Dot(lrm.Weights)
	if err != nil {
		return &Tensor[T, S]{}, errors.New(fmt.Sprintf("Predict failed during dot product: %v", err))
	}

	return predictions, nil
}
