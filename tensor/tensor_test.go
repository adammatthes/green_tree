package tensor

import (
	"testing"
)

func testInitTensor(t *testing.T) {
	testTensor, err := InitTensor[int, uint]([]uint{2, 2})
	if err != nil {
		t.Errorf("Could not create new tensor: %v\n", err)
	}

	if testTensor.Shape[0] != 2 {
		t.Errorf("Shape not initialized correctly\n")
	}
}
