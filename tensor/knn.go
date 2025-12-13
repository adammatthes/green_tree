package tensor

import (
	"sort"
	"errors"
	"fmt"
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

func MajorityVote[T Numeric](neighbors []Neighbor[T]) (string, error) {
	if len(neighbors) == 0 {
		return "", errors.New("Neighbors list is empty")
	}

	counts := make(map[string]int)

	minDistanceByLabel := make(map[string]T)

	for _, n := range neighbors {
		counts[n.Label]++

		if counts[n.Label] == 1 {
			minDistanceByLabel[n.Label] = n.Distance
		}
	}

	predictedLabel := ""
	maxCount := -1
	minDistanceForTieBreak := *new(T)

	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			predictedLabel = label
			minDistanceForTieBreak = minDistanceByLabel[label]
		}

		if count == maxCount {
			if minDistanceByLabel[label] < minDistanceForTieBreak {
				predictedLabel = label
				minDistanceForTieBreak = minDistanceByLabel[label]
			}
		}
	}
	return predictedLabel, nil
}

func (model *KNN[T, S]) Predict(queryPoint *Tensor[T, S]) (string, error) {
	neighbors, err := FindKNearestLabels(model, queryPoint)
	if err != nil {
		return "", fmt.Errorf("FindKNearestNeighbors failed in Predict: %v", err)
	}

	prediction, err := MajorityVote(neighbors)
	if err != nil {
		return "", fmt.Errorf("Majority Vote failed in Predict: %v", err)
	}

	return prediction, nil
}
