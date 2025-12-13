package tensor

import (
	"math"
	"testing"
)

func TestFindKNearestLabels(t *testing.T) {
	t1, _ := InitTensor64(4, 2)
	t1.Data = []float64{5.1, 0.2, 4.9, 0.2, 7.0, 2.5, 6.4, 1.9}

	query, _ := InitTensor64(1, 2)
	query.Data = []float64{6.0, 1.0}

	labels := []string{"A", "B", "B", "A"}

	k := uint64(3)

	expectedNeighbors := []Neighbor[float64] {
		{Distance: 0.984885780179610, Label: "A"},
		{Distance: 1.204159457879203, Label: "A"},
		{Distance: 1.360147050873544, Label: "B"},
	}

	knn := &KNN[float64, uint64]{
		K:			k,
		TrainingFeatures: 	t1,
		TrainingLabels: 	labels,
	}

	actualNeighbors, err := FindKNearestLabels(knn, query)
	if err != nil {
		t.Errorf("FindKNearestNeighbors failed: %v", err)
	}

	if len(actualNeighbors) != len(expectedNeighbors) {
		t.Errorf("Unexpected length from FindKNearestNeighbors result: %v", len(actualNeighbors))
	}

	tol := 1e-5
	for n := range actualNeighbors {
		actual := actualNeighbors[n]
		expected := expectedNeighbors[n]

		if actual.Label != expected.Label {
			t.Errorf("Unexpected label: got %v expected %v", actual.Label, expected.Label)
		}

		diff := math.Abs(actual.Distance - expected.Distance)
		if diff > tol {
			t.Errorf("Unexpected values in KNN result: got %v, expected %v", actualNeighbors, expectedNeighbors)
		}
	}
}

func TestSimpleMajority(t *testing.T) {
	neighbors := []Neighbor[float64] {
		{Distance: 0.5, Label: "Dog"},
		{Distance: 0.6, Label: "Cat"},
		{Distance: 0.7, Label: "Dog"},
		{Distance: 0.8, Label: "Cat"},
		{Distance: 0.9, Label: "Dog"},
	}

	expected := "Dog"

	result, err := MajorityVote(neighbors)
	if err != nil {
		t.Errorf("MajorityVote failed: %v", err)
	}

	if result != expected {
		t.Errorf("Unexpected value from MajorityVote. Got %v, expected %v", result, expected)
	}
}

func TestTieBreaker(t *testing.T) {
	neighbors := []Neighbor[float64] {
		{Distance: 0.1, Label: "Apple"},
		{Distance: 0.2, Label: "Banana"},
		{Distance: 0.3, Label: "Apple"},
		{Distance: 0.4, Label: "Banana"},
	}

	expected := "Apple"

	result, err := MajorityVote(neighbors)
	if err != nil {
		t.Errorf("MajorityVote failed in Tie Breaker: %v", err)
	}

	if result != expected {
		t.Errorf("Unexpected result for Tie-Breaker. Got %v, expected %v", result, expected)
	}
}
