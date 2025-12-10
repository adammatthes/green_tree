package tensor

import (
	"fmt"
	"math"
)


type StandardScaler[T Numeric, S Index] struct {
	Mu	[]T
	Sigma	[]T
}

func (scalar *StandardScaler[T, S]) FitStatistics(trainingFeatures *Tensor[T, S]) error {
	if len(trainingFeatures.Shape) != 2 {
		return fmt.Errorf("StandardScalar requires a 2D tensor")
	}

	numSamples := trainingFeatures.Shape[0]
	numFeatures := trainingFeatures.Shape[1]

	scalar.Mu = make([]T, numFeatures)
	scalar.Sigma = make([]T, numFeatures)

	const featureAxis = 1

	for n := S(0); n < numFeatures; n++ {
		columnData, err := trainingFeatures.GetSlice(featureAxis, n)
		if err != nil {
			return fmt.Errorf("Error extracting feature %v: %v", n, err)
		}

		sumOfValues, err := columnData.Sum()
		if err != nil {
			return err
		}
		meanValue := sumOfValues / T(numSamples)
		scalar.Mu[n] = meanValue

		varianceSum := T(0.0)

		stride := columnData.Strides[0]
		dataSlice := columnData.Data

		for n := S(0); n < numSamples; n++ {
			indexInSlice := n * stride
			unscaled := dataSlice[indexInSlice]
			diff := unscaled - meanValue
			varianceSum += diff * diff
		}

		variance := varianceSum / T(numSamples)
		stdDevValue := T(math.Sqrt(float64(variance)))

		if float64(stdDevValue) < 1e-9 {
			stdDevValue = T(1.0)
		}

		scalar.Sigma[n] = stdDevValue
	}
	return nil
}

func (scaler *StandardScaler[T, S]) Transform(input *Tensor[T, S]) (*Tensor[T, S], error) {
	if len(scaler.Mu) == 0 || len(scaler.Sigma) == 0 {
		return &Tensor[T, S]{}, fmt.Errorf("StandardScalar must be fitted (call FitStatistics)")
	}

	if len(input.Shape) != 2 {
		return &Tensor[T, S]{}, fmt.Errorf("Transform requires a 2D tensor")
	}

	numSamples := input.Shape[0]
	numFeatures := input.Shape[1]

	if S(len(scaler.Mu)) != numFeatures {
		return &Tensor[T, S]{}, fmt.Errorf("Expecting %v features, got %v", len(scaler.Mu), numFeatures)
	}

	transformedData, err := InitTensor[T, S](input.Shape)
	if err != nil {
		return &Tensor[T, S]{}, err
	}

	const featureAxis = 1

	for featureIndex := S(0); featureIndex < numFeatures; featureIndex++ {
		featureMean := scaler.Mu[featureIndex]
		featureStdDev := scaler.Sigma[featureIndex]

		columnView, err := input.GetSlice(featureAxis, featureIndex)
		if err != nil {
			return &Tensor[T, S]{}, err
		}

		stride := columnView.Strides[0]
		dataSlice := columnView.Data

		for sampleIndex := S(0); sampleIndex < numSamples; sampleIndex++ {
			indexInSlice := sampleIndex * stride
			unscaled := dataSlice[indexInSlice]

			standardized := (unscaled - featureMean) / featureStdDev

			if math.IsNaN(float64(standardized)) || math.IsInf(float64(standardized), -1) {
				return &Tensor[T, S]{}, fmt.Errorf("NaN/Inf calculated during scaling.")
			}

			newIndex := sampleIndex * numFeatures + featureIndex
			transformedData.Data[newIndex] = standardized
		}
	}

	return transformedData, nil
}
