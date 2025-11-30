# Tensor

A library for creating and operating on Tensor structures.

Definition:
```go
type Tensor[T Numeric, S Index] struct {
	Shape	[]S  // the length of each dimension
	Strides	[]S  // precalculated values to jump to the next dimension in the flat array
	Data	[]T  // a flat array
}
```

While the data contained in the Tensor can be of pretty much any numeric type, you're likely better off setting to float64, especially if you're applying it to the linear regression model. Shape and Strides slices need to be an unsigned type.

Initialization:

```go
shape := []uint{100, 2}
t, err := InitTensor[float64, uint](shape)
if err != nil {
	// handle error
}

// alternate helper function; sets the types as float64 and uint64
t, err = InitTensor64(shape)
```

While the above example will create a tensor with all values set to zero, you can also make a tensor with random values populated:

```go
shape := []uint{100, 2}
t, err := InitRandomTensor[float64, uint](shape, 10.0) // second argument is max random value
if err != nil {
	// handle error
}

// alternate helper function, similar to InitTensor64
// note the change in argument order to account for variadic uint64
t, err = InitRandomTensor64(10.0, 100, 2)
```

You can also initialize a target tensor that represents your expected values

```go
shape := []uint{100, 2}
base, _ := InitRandomTensor[float64, uint](shape)

weights := []float64{10.0, 5.0, -2.0}
targets, err := InitTargetTensor[float64, uint](base, weights)
if err != nil {
	// handle error
}
```

The tensor also has a variety of utility methods:

```go
val, err := tensor.Get([]uint{0, 4})

err = tensor.Set([]uint{2, 3}, 15.0)

transposed, err := tensor.Transpose(1, 3, 0, 2)  // arguments represent the new order of dimensions; default is to reverse the dimensions

product, err := tensor.Dot(otherTensor)

sum, err := tensor.Add(otherTensor)

diff, err := tensor.Subtract(otherTensor)

if !tensor.Valid() {
	fmt.Errorf("Tensor contains NaN or Inf values")
}

multipliedByScalar, err := tensor.MulScalar(0.2)

hadamard, err := tensor.Hadamard(otherTensor)

augmented, err := tensor.AugmentBias()

norm, err := tensor.Norm()
```

# Linear Regression Model

Definition:

```go
type LinearRegressionModel[T Numeric, S Index] struct {
	Weights		*Tensor[T, S]
	LearningRate	T
	MaxIterations	S
	MomentumRate	T
	ClipThreshold	T
	Velocity	*Tensor[T, S]
}
```

Usage:

```go
numFeatures := 1000
learningRate := 0.0001
momentum := 0.9
clipThreshold := 5.0
maxIterations := 100000

lrm, err := InitLinearRegressionModel[float64, uint](
	numFeatures,
	learningRate,
	momentum,
	clipThreshold,
	maxIterations)

if err != nil {
	// handle error
}

randRange := 10.0
randomTensor, err := InitRandomTensor[float64, uint]([]uint{numFeatures, 2}, randRange)

expectedWeights := []float64{10.0, 5.0, -2.0}
targets, err := InitTargetTensor[float64, uint](randomTensor, expectedWeights)

augmentedFeatures, err := randomTensor.AugmentBias()

err = lrm.Fit(augmentedFeatures, targets)

predictions := lrm.Predict(newFeatures)

rsme, err := rootMeanSquareError(predictions, targets)
```
