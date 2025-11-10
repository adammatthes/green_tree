package zip

import (
	"testing"
)

func TestZip(t *testing.T) {
	a := []int{1, 2, 3, 4, 5}
	b := []int{6, 7, 8, 9, 0}

	c, err := Zip(a, b)
	if err != nil {
		t.Errorf("Error returning Zip: %v", err)
	}

	if len(c) != 5 {
		t.Errorf("Unexpected number of zipped elements: %v", len(c))
	}

	if len(c[0]) != 2 {
		t.Errorf("Unexpected size of one element: %v %v", len(c[0]), c)
	}

	expectedFirstElement := []int{1, 6}
	if c[0][0] != expectedFirstElement[0] || c[0][1] != expectedFirstElement[1] {
		t.Errorf("First element not expected values: %v", c[0])
	}

	expectedLastElement := []int{5, 0}
	if c[4][0] != expectedLastElement[0] || c[4][1] != expectedLastElement[1] {
		t.Errorf("Last element not expected values: %v", c[4])
	}
}
