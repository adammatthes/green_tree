package tensor

import (
	"testing"
	"fmt"
	"math"
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

func TestDot(t *testing.T) {
	tt1, _ := InitTensor[int, uint]([]uint{2, 2})
	tt1.Set([]uint{0, 0}, 1)
	tt1.Set([]uint{0, 1}, 2)
	tt1.Set([]uint{1, 0}, 3)
	tt1.Set([]uint{1, 1}, 4)

	tt2, _ := InitTensor[int, uint]([]uint{2, 2})
	tt2.Set([]uint{0, 0}, 5)
	tt2.Set([]uint{0, 1}, 6)
	tt2.Set([]uint{1, 0}, 7)
	tt2.Set([]uint{1, 1}, 8)

	result, err := tt1.Dot(tt2)
	if err != nil {
		t.Errorf("Error during 2x2 dot product")
	}

	val, _ := result.Get([]uint{0, 0})
	if val != 19 {
		t.Errorf("Unexpected value for Dot. Got %v, expected 19\n %v\n", val, result.Data)
	}

	val, _ = result.Get([]uint{0, 1})
	if val != 22 {
		t.Errorf("Unexpected value for Dot. Got %v, expected 22\n %v\n", val, result.Data)
	}

	val, _ = result.Get([]uint{1, 0})
	if val != 43 {
		t.Errorf("Unexpected value for Dot. Got %v, expected 43\n %v\n", val, result.Data)
	}

	val, _ = result.Get([]uint{1, 1})
	if val != 50 {
		t.Errorf("Unexpected value for Dot. Got %v, expected 50\n %v\n", val, result.Data)
	}
}

func TestDotAsymmetrical(t *testing.T) {
	t1, _ := InitTensor[int, uint]([]uint{2, 3})
	t1.Set([]uint{0, 0}, 1)
	t1.Set([]uint{0, 1}, 2)
	t1.Set([]uint{0, 2}, 3)
	t1.Set([]uint{1, 0}, 4)
	t1.Set([]uint{1, 1}, 5)
	t1.Set([]uint{1, 2}, 6)
	
	t2, _ := InitTensor[int, uint]([]uint{3, 2})
	t2.Set([]uint{0, 0}, 7)
	t2.Set([]uint{0, 1}, 8)
	t2.Set([]uint{1, 0}, 9)
	t2.Set([]uint{1, 1}, 10)
	t2.Set([]uint{2, 0}, 11)
	t2.Set([]uint{2, 1}, 12)

	result, err := t1.Dot(t2)
	if err != nil {
		t.Errorf("Error performing asymmetrical dot")
	}

	if result.Shape[0] != 2 || result.Shape[1] != 2 {
		t.Errorf("Assymetrical dot product did not result in expected shape")
	}

	val, _ := result.Get([]uint{0, 0})
	if val != 58 {
		t.Errorf("Unexpected value for asymmetrical dot. Got %v expected 58\n %v\n", val, result.Data)
	}

	val, _ = result.Get([]uint{0, 1})
	if val != 64 {
		t.Errorf("Unexpected value for asymmetrical dot. Got %v expected 64\n %v\n", val, result.Data)
	}

	val, _ = result.Get([]uint{1, 0})
	if val != 139 {
		t.Errorf("Unexpected value for asymmetrical dot. Got %v expected 139\n %v\n", val, result.Data)
	}

	val, _ = result.Get([]uint{1, 1})
	if val != 154 {
		t.Errorf("Unexpected value for asymmetrical dot. Got %v expected 154\n %v\n", val, result.Data)
	}
}

func TestBDBatch(t *testing.T) {
	t1, _ := InitTensor[int, uint]([]uint{2, 3, 2})
	for n := 0; n < 6; n++ {
		t1.Data[n] = 1
	}

	for n := 6; n < 12; n++ {
		t1.Data[n] = 3
	}

	fmt.Printf("%v\n", t1.Data)

	t2, _ := InitTensor[int, uint]([]uint{2, 2, 4})
	for n := 0; n < 8; n++ {
		t2.Data[n] = 2
	}
	for n := 8; n < 16; n++ {
		t2.Data[n] = 4
	}

	fmt.Printf("%v\n", t2.Data)

	result, err := t1.Dot(t2)
	if err != nil {
		t.Errorf("Error on BD Batch Dot: %v\n", err)
	}

	for n := uint(0); n < 12; n++ {
		if result.Data[n] != 4 {
			t.Errorf("Unexpected result in BD Batch dot: %v\n", result.Data)
			break
		}
	}

	for n := uint(12); n < 24; n++ {
		if result.Data[n] != 24 {
			t.Errorf("Unexpected result in BD Batch dot: %v\n", result.Data)
			break
		}
	}
}

func TestSubtract(t *testing.T) {
	t1, err := InitTensor[int, uint]([]uint{3, 3, 3})
	if err != nil {
		t.Errorf("InitTensor Failed in Subtract Test")
	}

	for n := 0; n < len(t1.Data); n++ {
		t1.Data[n] = 3
	}

	t2, _ := InitTensor[int, uint]([]uint{3, 3, 3})

	for n := 0; n < len(t2.Data); n++ {
		t2.Data[n] = 2
	}

	result, err := t1.Subtract(t2)
	if err != nil {
		t.Errorf("Tensor subtraction failed")
	}

	for n := 0; n < len(result.Data); n++ {
		if result.Data[n] != 1 {
			t.Errorf("Unexpected Value in Subtraction result: %v != %v", result.Data[n], 1)
		}
	}
}

func TestValidTensor(t *testing.T) {
	tt, err := InitTensor[float64, uint]([]uint{2, 2})
	if err != nil {
		t.Errorf("Failed to init Tensor during Valid test")
	}

	for n := 0; n < len(tt.Data); n++ {
		tt.Data[n] = math.MaxFloat64
	}

	if !tt.Valid() {
		t.Errorf("False negative in validating Tensor values")
	}

	for n := 0; n < len(tt.Data); n++ {
		tt.Data[n] *= 2
	}

	if tt.Valid() {
		t.Errorf("False positive in validating Tensor values")
	}
}

func TestMultiplyScalar(t *testing.T) {
	t1, err := InitTensor[int, uint]([]uint{2, 2})
	if err != nil {
		t.Errorf("Failed init before scalar multiply")
	}

	for n := 0; n < len(t1.Data); n++ {
		t1.Data[n] = 2
	}

	t2, err := t1.MulScalar(2)
	if err != nil {
		t.Errorf("Error occurred during scalar multiply")
	}

	for n := 0; n < len(t2.Data); n++ {
		if t2.Data[n] != 4 {
			t.Errorf("Unexpected value in scalar multiply: %v != %v", t2.Data[n], 4)
		}
	}
}

func TestAddTensor(t *testing.T) {
	t1, err := InitTensor[int, uint]([]uint{2, 2})
	if err != nil {
		t.Errorf("Failed init during Add test")
	}

	for n := 0; n < len(t1.Data); n++ {
		t1.Data[n] = 1
	}

	t2, _ := InitTensor[int, uint]([]uint{2, 2})

	for n := 0; n < len(t2.Data); n++ {
		t2.Data[n] = 2
	}

	result, err := t1.Add(t2)
	if err != nil {
		t.Errorf("Error during addition of Tensors")
	}

	for n := 0; n < len(result.Data); n++ {
		if result.Data[n] != 3 {
			t.Errorf("Unexpected value after Tensor addition: %v != %v", result.Data[n], 3)
		}
	}

}

func TestHadamard(t *testing.T) {
	t1, err := InitTensor[int, uint]([]uint{2, 2})
	if err != nil {
		t.Errorf("Failed init during Hadamard test")
	}

	for n := 0; n < len(t1.Data); n++ {
		t1.Data[n] = 2
	}

	t2, _ := InitTensor[int, uint]([]uint{2, 2})

	for n := 0; n < len(t2.Data); n++ {
		t2.Data[n] = 3
	}

	result, err := t1.Hadamard(t2)
	if err != nil {
		t.Errorf("Hadamard method failed")
	}

	for n := 0; n < len(result.Data); n++ {
		if result.Data[n] != 6 {
			t.Errorf("Unexpected value during Hadamard: %v != %v", result.Data[n], 6)
		}
	}
}

func TestAugmentBias(t *testing.T) {
	t1, err := InitTensor[float64, uint]([]uint{2, 2})
	if err != nil {
		t.Errorf("Failed to init tensor in Augment Bias Test: %v\n", err)
	}

	for n := 0; n < len(t1.Data); n++ {
		t1.Data[n] = 5.0
	}

	result, err := t1.AugmentBias()
	if err != nil {
		t.Errorf("Problem Augmenting Bias: %v\n", err)
	}

	if result.Data[0] > 1.05 {
		t.Errorf("Augmentation unsuccessful. Expected value near 1.0, got %v\n", result.Data[0])
	}
}

func TestRandomTensor(t *testing.T) {
	t1, err := InitRandomTensor[float64, uint]([]uint{100, 100}, 100.0)
	if err != nil {
		t.Errorf("Failed to create a random tensor: %v\n", err)
	}

	for n := 0; n < len(t1.Data); n++ {
		if t1.Data[n] > 100.0 || t1.Data[n] < -100.0 {
			t.Errorf("Value out of range of random tensor: %v", t1.Data[n])
		}
	}
}

func TestTargetTensor(t *testing.T) {
	t1, err := InitTensor[float64, uint]([]uint{3, 2})
	if err != nil {
		t.Errorf("Error creating tensor before target tensor: %v\n", err)
	}

	t1.Data = []float64{10.0, 2.0, 5.0, 4.0, 1.0, 6.0}

	t2, err := InitTargetTensor[float64, uint](t1, []float64{10.0, 2.0, -0.5})
	if err != nil {
		t.Errorf("Error creating target tensor: %v\n", err)
	}

	if t2.Shape[0] != 3 || t2.Shape[1] != 1 {
		t.Errorf("Target tensor does not have expected shape: %v\n", t2.Shape)
	}

	if len(t2.Data) != 3 {
		t.Errorf("Flat array of target tensor did not have expected length")
	}

	if t2.Data[0] < 28.0 || t2.Data[0] > 29.99 || t2.Data[1] < 17.0 || t2.Data[1] > 18.99 || t2.Data[2] < 8.0 || t2.Data[2] > 9.99 {
		t.Errorf("Unexpected values in target tensor: %v\n", t2.Data)
	}
}

func TestNorm(t *testing.T) {
	t1, _ := InitTensor[float64, uint]([]uint{0, 0})
	norm, err := t1.Norm()
	if norm != 0 || err != nil {
		t.Errorf("Unexpected norm value from empty tensor: %v, %v", norm, err)
	}

	t2, _ := InitTensor[float64, uint]([]uint{2, 2})
	t2.Data = []float64{1, 1, 1, 1}

	norm, _ = t2.Norm()
	if norm != 2.0 {
		t.Errorf("Unexpected value from Norm method. Expected %v, got %v", 2.0, norm)
	}
}

func TestR2Score(t *testing.T) {
	targets, _ := InitTensor[float64, uint]([]uint{3, 1})
	targets.Data = []float64{1.0, 2.0, 3.0}

	perfectPred, _ := InitTensor[float64, uint]([]uint{3, 1})
	perfectPred.Data = []float64{1.0, 2.0, 3.0}

	r2Score, err := R2Score(perfectPred, targets)
	if err != nil {
		t.Errorf("Error during R2 Score test: %v", err)
	}

	if r2Score != 1.0 {
		t.Errorf("Unexpected value for perfect match. Got %v, expected 1.0", r2Score)
	}

	worstPred, err := InitTensor[float64, uint]([]uint{3, 1})
	worstPred.Data = []float64{2.0, 2.0, 2.0}

	r2Score, err = R2Score(worstPred, targets)
	if err != nil {
		t.Errorf("Error during worst prediction R2 score: %v", err)
	}

	if r2Score != 0.0 {
		t.Errorf("Unexpected value for worst prediction. Got %v, expected 0.0", r2Score)
	}
}

func TestInit64(t *testing.T) {
	t64, err := InitTensor64(2, 2)
	if err != nil {
		t.Errorf("InitTensor64 failed: %v", err)
	}

	if len(t64.Data) != 4 {
		t.Errorf("Unexpected Length of Tensor64: %v", len(t64.Data))
	}
}

func TestRandom64(t *testing.T) {
	rand64, err := InitRandomTensor64(10.0, 3, 2, 3)
	if err != nil {
		t.Errorf("Failed to init 64-bit random tensor")
	}

	if len(rand64.Shape) != 3 {
		t.Errorf("Unexpected shape of 64-bit, random tensor")
	}
}
