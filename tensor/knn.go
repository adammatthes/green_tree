package tensor

import (
	"sort"
	"errors"
)

type KNN[T Numeric, S Index] struct {
	K			S
	TrainingFeatures	*Tensor[T, S]
	TrainingLabels		[]string
}

type Neighbor[T Numeric] struct {
	Distance	T
	Label		string
}

func FindKNearestLabels[T Numeric, S Index](knn *KNN[T, S], queryPoint *Tensor[T, S]) ([]Neighbor[T], error) {
	
	distances, err := EuclideanDistances(queryPoint, knn.TrainingFeatures)
	if err != nil {
		return nil, err
	}

	numSamples := S(len(distances.Data))
	if numSamples != S(len(knn.TrainingLabels)) {
		return nil, errors.New("distance tensor != label array length")
	}

	neighbors := make([]Neighbor[T], numSamples)
	for n := S(0); n < numSamples; n++ {
		neighbors[n] = Neighbor[T] {
			Distance: 	distances.Data[n],
			Label:		knn.TrainingLabels[n],
		}
	}

	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})

	kNearest := knn.K
	if kNearest > numSamples {
		kNearest = numSamples
	}

	return neighbors[:kNearest], nil
}
