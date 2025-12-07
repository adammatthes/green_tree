package tensor

import (
	"fmt"
	"math"
)


type StandardScalar[T Numeric, S Index] struct {
	Mu	[]T
	Sigma	[]T
}

func (scalar *StandardScalar[T, S]) FitStatistics(trainingFeatures *Tensor[T, S]) error {
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
