# Green Tree

Named after the Green Tree Python, this is a repository dedicated to implementing distinct features of Python in Go

## Set

This package emulates the behaviors of Sets in Python, the major difference being that the generic typing in the implementation requires the user to specify a type upon intialization.

```go
mySet := InitSet[int](nil)
```

You can also provide a slice of values at initialization:

```go
mySet := InitSet[int]([]int{1, 2, 3, 4})
```

You can either add values one at a time:

```go
mySet := InitSet[int](nil)
mySet.Add(1)
mySet.Add(2)
```

Or you can provide a slice to update the Set:

```go
mySet := InitSet[int](nil)
mySet.Update([]int{1, 1, 1, 2, 3, 4})
```

Remove values from a set:
```go
mySet := InitSet[int]([]int{1, 2, 3, 4})
mySet.Remove(1)
mySet.Remove(2)
```

Empty a set:
```go
mySet.Clear()
```

Make a copy of a set:
```go
setCopy := set.Copy()
```

Check if a value is in a Set:
```go
if mySet.Contains(value) {
	// perform logic
}
```

You can perform operations on two sets:
```go
intersect := set1.Intersection(set2) // or set1.And(set2)

difference := set1.Difference(set2) // or set1.Diff(set2)

union := set1.Union(set2) // or set1.Or(set2)

symmetric := set1.SymmetricDifference(set2) // or set1.Xor(set2)
```

You can identify relationships between sets:
```go
if set1.IsDisjoint(set2) {
	// perform logic
} else if set1.IsSubset(set2) {
	// perform logic
} else if set1.IsSuperset(set2) {
	// perform logic
}

if set1.Equals(set2) {
	// perform logic
}
```

## Zip

Emulates the behavior of Python's zip function. Usage:

```go
a := int[]{1, 2, 3, 4}
b := int[]{5, 6, 7, 8}

c, err := Zip(a, b)
```

If any of the provided slices are not of the same length, the resulting slice of slices will have the length of the shortest initial slice, and err will not be nil.

