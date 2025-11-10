package zip

import (
	"errors"
)

func Zip[T comparable](iterables ...[]T) ([][]T, error) {
	var err error 
	groupSize := 0

	lowestCommonLength := len(iterables[0])

	for _, iter := range iterables {
		currLen := len(iter)
		if currLen < lowestCommonLength {
			lowestCommonLength = currLen
		}
		groupSize += 1
	}

	

	for _, iter := range iterables {
		if len(iter) != lowestCommonLength {
			err = errors.New("Unequal length of iterables; final result is truncated")
			break
		}
	}

	result := make([][]T, lowestCommonLength)
	for i, _ := range result {
		result[i] = make([]T, groupSize)
	}

		
	for i := 0; i < lowestCommonLength; i++ {
		for j := 0; j < groupSize; j++ {
			result[i][j] = iterables[j][i]
		}
	}

	return result, err
}
