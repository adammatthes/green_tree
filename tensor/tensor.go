package tensor

import (
	"errors"
	"fmt"
	"math/rand"
	"math"
)

type Numeric interface {
	int | int8 | int16 | int32 | int64 | 
	uint | uint8 | uint16 | uint32 | uint64 | 
	float32 | float64
}

type Index interface {
	uint | uint8 | uint16 | uint32 | uint64
}

type Tensor[T Numeric, S Index] struct {
	Shape	[]S
	Strides	[]S
	Data	[]T
}

func InitTensor[T Numeric, S Index](shape []S) (*Tensor[T, S], error) {
	if len(shape) == 0 {
		return &Tensor[T, S]{}, errors.New("Invalid shape for Tensor")
	}

	var totalSize S = S(1);

	for n := 0; n < len(shape); n++ {
		totalSize *= shape[n]
	}

	data := make([]T, totalSize)

	strides := make([]S, len(shape))
	var currentStride S = S(1);

	for n := len(shape) - 1; n >= 0; n-- {
		strides[n] = currentStride
		currentStride *= shape[n]
	}

	result := Tensor[T, S] {
		Shape:		shape,
		Strides:	strides,
		Data:		data}

	return &result, nil
}

func InitRandomTensor[T Numeric, S Index](shape []S, maxVal T) (*Tensor[T, S], error) {
	t, err := InitTensor[T, S](shape)
	if err != nil {
		return &Tensor[T, S]{}, nil
	}

	rangeWidth := 2 * float64(maxVal)
	offset := float64(maxVal)

	for n := range t.Data {
		randomVal := (rand.Float64() * rangeWidth) - offset

		t.Data[n] = T(randomVal)
	}

	return t, nil
}

func InitTargetTensor[T Numeric, S Index](
	xBase *Tensor[T, S],
	weights []T) (*Tensor[T, S], error) {
	if len(xBase.Shape) != 2 {
		return &Tensor[T, S]{}, errors.New("Target Tensor creation requires a 2D tensor")
	}

	numSamples := xBase.Shape[0]

	y, err := InitTensor[T, S]([]S{numSamples, 1})
	if err != nil {
		return &Tensor[T, S]{}, err
	}

	bias := T(weights[0])
	w1 := T(weights[1])
	w2 := T(weights[2])

	rowStride := xBase.Strides[0]

	for n := S(0); n < numSamples; n++ {
		startIdx := n * rowStride

		x1 := xBase.Data[startIdx]
		x2 := xBase.Data[startIdx + 1]

		noise := T(rand.Float64() * 2 - 1.0)

		yN := bias + (w1 * x1) + (w2 * x2) + noise

		y.Data[n] = yN
	}

	return y, nil
}

func (t *Tensor[T, S]) LinearIndex(coord []S) (S, error) {
	if len(coord) != len(t.Shape) {
		return 0, errors.New(fmt.Sprintf("Coordinate dimensions (%v) do not match shape (%v)\n", coord, t.Shape))
	}

	offset := S(0)

	for n := 0; n < len(coord); n++ {
		if coord[n] >= t.Shape[n] {
			return 0, errors.New(fmt.Sprintf("Coordinate dimension (%v) exceeds bounds of shape (%v)\n", coord[n], t.Shape[n]))
		}
		offset += coord[n] * t.Strides[n]
	}

	return offset, nil
}

func (t *Tensor[T, S]) Get(coord []S) (T, error) {
	offset, err := t.LinearIndex(coord)
	if err != nil {
		return *new(T), errors.New(fmt.Sprintf("LinearIndex failed: %v", err))
	}

	return t.Data[offset], nil
}

func (t *Tensor[T, S]) Set(coord []S, val T) error {
	offset, err := t.LinearIndex(coord)
	if err != nil {
		return err
	}

	t.Data[offset] = val

	return nil
}

func (t *Tensor[T, S]) Transpose(axes ...S) (*Tensor[T, S], error) {
	if len(axes) == 0 {
		for n := len(t.Shape) - 1; n >= 0; n-- {
			axes = append(axes, S(n))
		}
	}



	if len(axes) != len(t.Shape) {
		return &Tensor[T, S]{}, errors.New(
			fmt.Sprintf("Number of axes does not match shape of Tensor: %v != %v", len(axes), len(t.Shape)))
	}

	newShape := make([]S, len(t.Shape))
	newStride := make([]S, len(t.Strides))

	for n := 0; n < len(axes); n++ {
		newShape[n] = t.Shape[axes[n]]
		newStride[n] = t.Strides[axes[n]]
	}

	result := Tensor[T, S] {
		Shape:		newShape,
		Strides:	newStride,
		Data:		t.Data,
	}

	return &result, nil


}

func (t *Tensor[T, S]) Dot(other *Tensor[T, S]) (*Tensor[T, S], error) {
	lenA := len(t.Shape)
	lenB := len(other.Shape)

	if lenA < 2 || lenB < 2 {
		return &Tensor[T, S]{}, errors.New("Tensors must at least 2 dimensions for Dot product calculation")
	}

	batchSizeA := S(lenA - 2)
	batchSizeB := S(lenB - 2)

	if batchSizeA != batchSizeB {
		return &Tensor[T, S]{}, errors.New("Batch dimensions do not match")
	}

	for n := S(0); n < batchSizeA; n++ {
		if t.Shape[n] != other.Shape[n] {
			return &Tensor[T, S]{}, errors.New("Shape in batch dimensions do not match") 
		}
	}

	M := t.Shape[batchSizeA]
	K1 := t.Shape[batchSizeA + 1]
	K2 := other.Shape[batchSizeB]
	N := other.Shape[batchSizeB + 1]

	totalMatrixSizeA := M * K1
	totalMatrixSizeB := K2 * N

	if K1 != K2 {
		return &Tensor[T, S]{}, errors.New(fmt.Sprintf("Inner dimensions do not match. %v != %v", K1, K2))
	}

	K := K1

	newShape := make([]S, lenA)
	copy(newShape, t.Shape[:batchSizeA])
	newShape[batchSizeA] = M
	newShape[batchSizeA + 1] = N

	result, err := InitTensor[T, S](newShape)
	if err != nil {
		return &Tensor[T, S]{}, err
	}

	totalMul := S(1)

	for _, dim := range t.Shape[:batchSizeA] {
		totalMul *= dim
	}


	R := M
	C := N
	batchCount := totalMul

	for batchIdx := S(0); batchIdx < batchCount; batchIdx++ {
		batchOffsetA := batchIdx * totalMatrixSizeA
		batchOffsetB := batchIdx * totalMatrixSizeB

		totalMatrixSizeResult := M * N
		batchOffsetResult := batchIdx * totalMatrixSizeResult

		for i := S(0); i < R; i++ {
			for j := S(0); j < C; j++ {
				sum := *new(T)

				for k := S(0); k < K; k++ {
					idxA := batchOffsetA + (i * t.Strides[batchSizeA]) + (k * t.Strides[batchSizeA + 1])
					valA := t.Data[idxA]

					idxB := batchOffsetB + (k * other.Strides[batchSizeB]) + (j * other.Strides[batchSizeB + 1])
					valB := other.Data[idxB]

					sum += valA * valB
				}
				idxResult := batchOffsetResult + (i * result.Strides[batchSizeA]) + (j * result.Strides[batchSizeA + 1])
				result.Data[idxResult] = sum
			}
		}
	}

	return result, nil
}

func (t *Tensor[T, S]) Add(other *Tensor[T, S]) (*Tensor[T, S], error) {
	if len(t.Shape) != len(other.Shape) {
		return &Tensor[T, S]{}, errors.New("Number of dimensions do not match")
	}

	for n := S(0); n < S(len(t.Shape)); n++ {
		if t.Shape[n] != other.Shape[n] {
			return &Tensor[T, S]{}, errors.New("Length of dimension does not match")
		}
	}

	if len(t.Data) != len(other.Data) {
		return &Tensor[T, S]{}, errors.New("Total length of Tensors do not match")
	}

	result, err := InitTensor[T, S](t.Shape)
	if err != nil {
		return &Tensor[T, S]{}, errors.New("Error creating new Tensor before addition")
	}

	for n := S(0); n < S(len(result.Data)); n++ {
		result.Data[n] = t.Data[n] + other.Data[n]
	}

	return result, nil
}

func (t *Tensor[T, S]) Subtract(other *Tensor[T, S]) (*Tensor[T, S], error) {
	if len(t.Shape) != len(other.Shape) {
		return &Tensor[T, S]{}, errors.New("Number of dimensions do not match")
	}

	for n := S(0); n < S(len(t.Shape)); n++ {
		if t.Shape[n] != other.Shape[n] {
			return &Tensor[T, S]{}, errors.New("Shape of dimension does not match")
		}
	}

	if len(t.Data) != len(other.Data) {
		return &Tensor[T, S]{}, errors.New("Length of data in Tensors does not match")
	}

	result, err := InitTensor[T, S](t.Shape)
	if err != nil {
		return &Tensor[T, S]{}, errors.New("Error creating new Tensor before subtraction")
	}

	for n := S(0); n < S(len(t.Data)); n++ {
		result.Data[n] = t.Data[n] - other.Data[n]
	}

	return result, nil
}

func (t *Tensor[T, S]) Valid() bool {
	for n := S(0); n < S(len(t.Data)); n++ {
		val := float64(t.Data[n])

		if math.IsNaN(val) {
			return false
		}

		if math.IsInf(val, 0) {
			return false
		}
	}

	return true
}

func (t *Tensor[T, S]) MulScalar(scalar T) (*Tensor[T, S], error) {
	result, err := InitTensor[T, S](t.Shape)
	if err != nil {
		return &Tensor[T, S]{}, errors.New("Error creating new Tensor before scalar multiply")
	}

	for n := S(0); n < S(len(result.Data)); n++ {
		result.Data[n] = t.Data[n] * scalar
	}

	return result, nil
}

func (t *Tensor[T, S]) Hadamard(other *Tensor[T, S]) (*Tensor[T, S], error) {
	if len(t.Shape) != len(other.Shape) {
		return &Tensor[T, S]{}, errors.New("Tensors must have same number of dimensions")
	}

	for n := S(0); n < S(len(t.Shape)); n++ {
		if t.Shape[n] != other.Shape[n] {
			return &Tensor[T, S]{}, errors.New("Dimensions of Tensors must be same length")
		}
	}

	if len(t.Data) != len(other.Data) {
		return &Tensor[T, S]{}, errors.New("Total length of Tensors does not match")
	}

	result, err := InitTensor[T, S](t.Shape)
	if err != nil {
		return &Tensor[T, S]{}, errors.New("Init of new Tensor failed during Hadamard")
	}

	for n := S(0); n < S(len(result.Data)); n++ {
		result.Data[n] = t.Data[n] * other.Data[n]
	}

	return result, nil
}

func (t *Tensor[T, S]) AugmentBias() (*Tensor[T, S], error) {
	if len(t.Shape) != 2 {
		return &Tensor[T, S]{}, errors.New("Can only augment a two dimensional tensor")
	}

	numSamples := t.Shape[0]
	numFeatures := t.Shape[1]

	newShape := []S{numSamples, numFeatures + 1}

	result, err := InitTensor[T, S](newShape)
	if err != nil {
		return &Tensor[T, S]{}, err
	}

	resultRowStride := result.Strides[0]

	for n := S(0); n < numSamples; n++ {
		biasIndex := n * resultRowStride
		result.Data[biasIndex] = T(1.0)

		resultFeatureStart := biasIndex + 1

		originalFeatureStart := n * t.Strides[0]

		copy(
			result.Data[resultFeatureStart: resultFeatureStart + numFeatures],
			t.Data[originalFeatureStart: originalFeatureStart + numFeatures],
		)
	}

	return result, nil
}

func (t *Tensor[T, S]) Norm() (T, error) {
	if len(t.Data) == 0 {
		return T(0), nil
	}

	var sumOfSquares float64
	for _, val := range t.Data {
		v := float64(val)
		sumOfSquares += v * v
	}

	return T(math.Sqrt(sumOfSquares)), nil
}

func (t *Tensor[T, S]) Mean() (T, error) {
	numElements := len(t.Data)
	if numElements == 0 {
		return T(0), nil
	}

	var sum float64
	for _, val := range t.Data {
		sum += float64(val)
	}

	return T(sum / float64(numElements)), nil
}

func R2Score[T Numeric, S Index](
	predictions *Tensor[T, S],
	targets *Tensor[T, S]) (T, error) {

	yMean, err := targets.Mean()
	if err != nil {
		return T(0), fmt.Errorf("Mean error during R2 function: %v", err)
	}

	residuals, err := targets.Subtract(predictions)
	if err != nil {
		return T(0), fmt.Errorf("Tensor subtraction failed during R2 score: %v", err)
	}

	residualsNorm, _ := residuals.Norm()
	sse := float64(residualsNorm * residualsNorm)

	meanTensor, err := InitTensor[T, S](targets.Shape)
	if err != nil {
		return T(0), err
	}

	for n := range meanTensor.Data {
		meanTensor.Data[n] = yMean
	}

	deviations, err := targets.Subtract(meanTensor)
	if err != nil {
		return T(0), fmt.Errorf("Error calculating deviations for R2 score: %v", err)
	}

	deviationsNorm, _ := deviations.Norm()
	sst := float64(deviationsNorm * deviationsNorm)

	if sst == 0 {
		return T(1.0), nil
	}

	r2 := 1.0 - (sse / sst)

	return T(r2), nil
}

/*

Inverse() Tensor (Matrix Inversion - required for Normal Equation)

*/
