package tensor

import (
	"testing"
)

func TestInitTensor(t *testing.T) {
	shape := make([]uint, 2)
	shape[0] = 2
	shape[1] = 2
	testTensor, err := InitTensor[int, uint](shape)
	if err != nil {
		t.Errorf("Could not create new tensor: %v\n", err)
	}

	if testTensor.Shape[0] != 2 {
		t.Errorf("Shape not initialized correctly\n")
	}
}

func TestLinearIndex(t *testing.T) {

	testTensor, err := InitTensor[int, uint]([]uint{5, 5})

	offset, err := testTensor.LinearIndex([]uint{0, 0})
	if err != nil {
		t.Errorf("Error getting offset: %v\n", err)
	}

	if offset != 0 {
		t.Errorf("Offset miscalculated: %v, expected 0\n", offset)
	}

	offset, err = testTensor.LinearIndex([]uint{1, 1})
	if err != nil {
		t.Errorf("Error getting offset: %v\n", err)
	}

	if offset != 6 {
		t.Errorf("Offset miscalculated: %v, expected 6\n", offset)
	}

	offset, err = testTensor.LinearIndex([]uint{4, 4})
	if err != nil {
		t.Errorf("Error getting offset: %v\n", err)
	}

	if offset != 24 {
		t.Errorf("Offset miscalculated: %v, expected 24\n", offset)
	}

	offset, err = testTensor.LinearIndex([]uint{1, 1, 1})
	if err == nil {
		t.Errorf("Mismatched coordinate dimension not caught\n")
	}

	offset, err = testTensor.LinearIndex([]uint{3, 10})
	if err == nil {
		t.Errorf("Out of bounds coordinate value not caught\n")
	}
}

func TestGetMethod(t *testing.T) {
	tt, err := InitTensor[int, uint]([]uint{5, 5})
	if err != nil {
		t.Errorf("Init Tensor failed: %v\n", err)
	}

	tt.Data[4] = 42 // artificially setting for test

	val, err := tt.Get([]uint{0, 4})
	if err != nil {
		t.Errorf("Get method failed: %v\n", err)
	}

	if val != 42 {
		t.Errorf("Did not Get expected value: %v != %v", val, 42)
	}

	val, err = tt.Get([]uint{10, 10})
	if err == nil {
		t.Errorf("Error not relayed from Linear Index in Get\n")
	}

	val, err = tt.Get([]uint{1, 1, 1})
	if err == nil {
		t.Errorf("Error not relayed from Linear Index in Get\n")
	}
}

func TestSetMethod(t *testing.T) {
	tt, err := InitTensor[int, uint]([]uint{5, 5})
	if err != nil {
		t.Errorf("Init Tensor failed: %v\n", err)
	}

	coord := []uint{0, 1}

	err = tt.Set(coord, 42)
	if err != nil {
		t.Errorf("Set failed: %v\n", err)
	}

	val, err := tt.Get(coord)
	if err != nil {
		t.Errorf("Error during Get after Set: %v\n", err)
	}

	if val != 42 {
		t.Errorf("Value was not Set correctly. Expected 42, got %v\n", val)
	}

	err = tt.Set([]uint{1, 1, 1}, 5)
	if err == nil {
		t.Errorf("Error from Linear Index not relayed from Set\n")
	}

	err = tt.Set([]uint{10, 10}, 5)
	if err == nil {
		t.Errorf("Error from Linear Index not relayed from Set\n")
	}
}

func TestTranspose(t *testing.T) {
	tt, err := InitTensor[int, uint]([]uint{2, 5})
	if err != nil {
		t.Errorf("Error InitTensor: %v\n", err)
	}

	if tt.Shape[0] != 2 || tt.Shape[1] != 5 {
		t.Errorf("Shape values not initialized correctly\n")
	}

	tt2, err := tt.Transpose()
	if err != nil {
		t.Errorf("Error during default transposition: %v\n", err)
	}

	if tt2.Shape[0] != 5 || tt2.Shape[1] != 2 {
		t.Errorf("Shape not transposed correctly on default transpose\n")
	}

	tt3, err := InitTensor[int, uint]([]uint{1, 2, 3, 4})
	if err != nil {
		t.Errorf("Error init with 4 dimensions: %v\n", err)
	}

	tt4, err := tt3.Transpose(1, 3, 2, 0)
	if err != nil {
		t.Errorf("Error transposing with custom axes: %v\n", err)
	}

	if tt4.Shape[0] != 2 || tt4.Shape[1] != 4 || tt4.Shape[2] != 3 || tt4.Shape[3] != 1 {
		t.Errorf("Custom transpose incorrect. Got %v expected {2, 4, 3, 1}", tt4.Shape)
	}



}
