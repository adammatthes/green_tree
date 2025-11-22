package tensor

import (
	"errors"
	"fmt"
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
