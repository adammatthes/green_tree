package set

import (
	"testing"
)

func TestSetAdd(t *testing.T) {
	s := InitSet[string](nil)
	s.Add("test")
	if !s.Contains("test") {
		t.Errorf("String not successfully added to Set.")
	}

	if s.Contains("not in Set") {
		t.Errorf("Erroneously found item not in Set.")
	}


	s.Print()
}

func TestSetIntersection(t *testing.T) {
	s1 := InitSet[int](nil)
	s1.Add(1)
	s1.Add(2)

	s2 := InitSet[int](nil)
	s2.Add(2)
	s2.Add(3)

	result := s1.Intersection(s2)

	if !result.Contains(2) {
		t.Errorf("Intersecting number not present")
	}

	if result.Contains(1) || result.Contains(3) {
		t.Errorf("Non-intersecting values present in result")
	}
}

func TestSetDifference(t *testing.T) {
	s1 := InitSet[int](nil)
	s1.Add(1)
	s1.Add(2)
	
	s2 := InitSet[int](nil)
	s2.Add(2)
	s2.Add(3)

	result := s1.Difference(s2)

	if !result.Contains(1) {
		t.Errorf("Expected difference value not present")
	}

	if result.Contains(2) {
		t.Errorf("Value expected to be removed still present")
	}
}

func TestSetUnion(t *testing.T) {
	s1 := InitSet[int](nil)
	s1.Add(1)
	s1.Add(2)

	s2 := InitSet[int](nil)
	s2.Add(3)
	s2.Add(4)

	result := s1.Union(s2)
	if !result.Contains(1) || !result.Contains(2) || !result.Contains(3) || !result.Contains(4) {
		t.Errorf("Missing values in Union of sets")
	}

	if len(result) != 4 {
		t.Errorf("Unexpected length of Union Set")
	}
}

func TestSetSymmetricDifference(t *testing.T) {
	s1 := InitSet[int](nil)
	s1.Add(1)
	s1.Add(2)
	s1.Add(3)

	s2 := InitSet[int](nil)
	s2.Add(3)
	s2.Add(4)
	s2.Add(5)

	result := s1.SymmetricDifference(s2)

	if !result.Contains(1) || !result.Contains(2) || !result.Contains(4) || !result.Contains(5) {
		t.Errorf("Missing values in symmetric difference")
	}

	if result.Contains(3) {
		t.Errorf("Value present in symmetric difference that should not be")
	}

	if len(result) != 4 {
		t.Errorf("Unexpected length of symmetric difference")
	}
}

func TestSetUpdate(t *testing.T) {
	s := InitSet[int](nil)

	s.Update([]int{1, 1, 1, 1, 1})

	if !s.Contains(1) {
		t.Errorf("Value not added during update")
	}

	if len(s) != 1 {
		t.Errorf("Duplicate values added to set")
	}

	if s.Contains(2) || s.Contains(3) || s.Contains(4) {
		t.Errorf("Ghost values before update")
	}

	s.Update([]int{1, 2, 3, 4})

	if !s.Contains(1) || !s.Contains(2) || !s.Contains(3) || !s.Contains(4) {
		t.Errorf("Missing values in update")
	}

	if len(s) != 4 {
		t.Errorf("Unexpected length after update")
	}
}

func TestSetRemove(t *testing.T) {
	s := InitSet[int](nil)
	s.Update([]int{1, 2, 3, 4})
	if !s.Contains(1) {
		t.Errorf("Set not updating")
	}

	s.Remove(1)
	if s.Contains(1) {
		t.Errorf("Removal from Set unsuccessful")
	}

	if !s.Contains(2) || !s.Contains(3) || !s.Contains(4) {
		t.Errorf("Items in Set not intended for removal missing")
	}
}

func TestSetClear(t *testing.T) {
	s := InitSet[int]([]int{1, 2, 3, 4})

	if len(s) != 4 {
		t.Errorf("Set not initializing properly")
	}

	s.Clear()

	if len(s) != 0 {
		t.Errorf("Set not empty after Clear()")
	}
}

func TestSetIsDisjoint(t *testing.T) {
	s1 := InitSet[int]([]int{1, 2, 3, 4})
	s2 := InitSet[int]([]int{5, 6, 7, 8})
	s3 := InitSet[int]([]int{4, 9, 0})

	if !s1.IsDisjoint(s2) {
		t.Errorf("Not recognizing disjoint sets")
	}

	if s1.IsDisjoint(s3) {
		t.Errorf("False positive on disjoint set")
	}
}

func TestSetIsSubset(t *testing.T) {
	s1 := InitSet[int]([]int{1, 2, 3})
	s2 := InitSet[int]([]int{1, 2, 3, 4, 5})
	s3 := InitSet[int]([]int{1, 2, 4, 6})

	if !s1.IsSubset(s2) {
		t.Errorf("Subset not identified")
	}

	if s1.IsSubset(s3) {
		t.Errorf("False positive on subset")
	}
}

func TestSetIsSuperset(t *testing.T) {
	s1 := InitSet[int]([]int{1, 2, 3, 4})
	s2 := InitSet[int]([]int{1, 2})
	s3 := InitSet[int]([]int{4, 5})

	if !s1.IsSuperset(s2) {
		t.Errorf("Superset not identified")
	}

	if s1.IsSuperset(s3) {
		t.Errorf("False positive on superset")
	}
}

func TestSetCopy(t *testing.T) {
	s1 := InitSet[int]([]int{1, 2, 3, 4})
	s2 := s1.Copy()

	if !s1.Equals(s2) {
		t.Errorf("Copy function unsuccessful")
	}
}
