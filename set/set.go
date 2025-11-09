package set

import (
	"fmt"
)

type Set[T comparable] map[T]struct{}

func InitSet[T comparable](iterable []T) Set[T] {
	s := make(Set[T])
	
	if iterable != nil {
		for _, val := range iterable {
			s.Add(val)
		}
	}

	return s
}

func (s Set[T]) Contains(item T) bool {
	_, ok := s[item]
	return ok
}

func (s Set[T]) Add(item T) error {
	s[item] = struct{}{}
	return nil
}

func (s Set[T]) Remove(item T) error {
	delete(s, item)
	return nil
}

func (s Set[T]) Clear() error {
	for k, _ := range s {
		delete(s, k)
	}

	return nil
}

func (s Set[T]) Update(items []T) {
	if items == nil {
		return
	}

	for _, val := range items {
		s.Add(val)
	}
}

func (s Set[T]) Intersection(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	result := InitSet[T](nil)
	for k, _ := range other {
		if s.Contains(k) {
			result.Add(k)
		}
	}

	return result
}

func (s Set[T]) And(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	return s.Intersection(other)
}

func (s Set[T]) Difference(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	result := InitSet[T](nil)
	for k, _ := range s {
		if !other.Contains(k) {
			result.Add(k)
		}
	}

	return result
}

func (s Set[T]) Diff(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	return s.Difference(other)
}

func (s Set[T]) Union(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	result := InitSet[T](nil)
	for k, _ := range s {
		result.Add(k)
	}

	for k, _ := range other {
		result.Add(k)
	}

	return result
}

func (s Set[T]) Or(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	return s.Union(other)
}

func (s Set[T]) SymmetricDifference(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	result := InitSet[T](nil)
	for k, _ := range s {
		if !other.Contains(k) {
			result.Add(k)
		}
	}

	for k, _ := range other {
		if !s.Contains(k) {
			result.Add(k)
		}
	}

	return result
}

func (s Set[T]) Xor(other Set[T]) Set[T] {
	if other == nil {
		return InitSet[T](nil)
	}

	return s.SymmetricDifference(other)
}

func (s Set[T]) Print() {
	fmt.Printf("Set {")
	for k, _ := range s {
		fmt.Printf("%v ", k)
	}
	fmt.Printf("}")
}

func (s Set[T]) IsDisjoint(other Set[T]) bool {
	if other == nil {
		return false
	}

	for k, _ := range s {
		if other.Contains(k) {
			return false
		}
	}

	return true
}

func (s Set[T]) IsSubset(other Set[T]) bool {
	if other == nil {
		return false
	}

	for k, _ := range s {
		if !other.Contains(k) {
			return false
		}
	}

	return true
}

func (s Set[T]) IsSuperset(other Set[T]) bool {
	if other == nil {
		return false
	}

	for k, _ := range other {
		if !s.Contains(k) {
			return false
		}
	}

	return true
}

func (s Set[T]) Copy() Set[T] {
	result := InitSet[T](nil)

	for k, _ := range s {
		result.Add(k)
	}

	return result
}

func (s Set[T]) Equals(other Set[T]) bool {
	return s.IsSubset(other) && s.IsSuperset(other)
}
