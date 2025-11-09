package set

import (
	"fmt"
)

type Set map[any]struct{};

func InitSet(iterable []any) Set {
	s := make(Set)
	
	if iterable != nil {
		for _, val := range iterable {
			s.Add(val)
		}
	}

	return s
}

func (s Set) Contains(item any) bool {
	_, ok := s[item]
	return ok
}

func (s Set) Add(item any) error {
	s[item] = struct{}{}
	return nil
}

func (s Set) Update(items []any) {
	if items == nil {
		return
	}

	for _, val := range items {
		s.Add(val)
	}
}

func (s Set) Intersection(other Set) Set {
	result := InitSet(nil)
	for k, _ := range other {
		if s.Contains(k) {
			result.Add(k)
		}
	}

	return result
}

func (s Set) Difference(other Set) Set {
	result := InitSet(nil)
	for k, _ := range s {
		if !other.Contains(k) {
			result.Add(k)
		}
	}

	return result
}

func (s Set) Union(other Set) Set {
	result := InitSet(nil)
	for k, _ := range s {
		result.Add(k)
	}

	for k, _ := range other {
		result.Add(k)
	}

	return result
}

func (s Set) SymmetricDifference(other Set) Set {
	result := InitSet(nil)
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

func (s Set) Print() {
	for k, _ := range s {
		fmt.Printf("%v ", k)
	}
}
