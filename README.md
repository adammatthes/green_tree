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
