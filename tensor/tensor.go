package tensor

import (
	"errors"
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

	var totalSize S;
	totalSize += 1

	for n := 0; n < len(shape); n++ {
		totalSize *= shape[n]
	}

	data := make([]T, totalSize)

	strides := make([]S, len(shape))
	var currentStride S;
	currentStride += 1;

	for n := len(shape); n >= 0; n-- {
		strides[n] = currentStride
		currentStride *= shape[n]
	}

	result := Tensor[T, S] {
		Shape:		shape,
		Strides:	strides,
		Data:		data}

	return &result, nil
}
