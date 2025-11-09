package set

import (
	"testing"
)

func TestSetAdd(t *testing.T) {
	s := InitSet(nil)
	s.Add("test")
	if !s.Contains("test") {
		t.Errorf("String not successfully added to Set.")
	}

	if s.Contains("not in Set") {
		t.Errorf("Erroneously found item not in Set.")
	}

	s.Add(5)
	if !s.Contains(5) {
		t.Errorf("Number not added")
	}

	s.Print()
}

func TestSetIntersection(t *testing.T) {
	s1 := InitSet(nil)
	s1.Add(1)
	s1.Add(2)

	s2 := InitSet(nil)
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
	s1 := InitSet(nil)
	s1.Add(1)
	s1.Add(2)
	
	s2 := InitSet(nil)
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
	s1 := InitSet(nil)
	s1.Add(1)
	s1.Add(2)

	s2 := InitSet(nil)
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
	s1 := InitSet(nil)
	s1.Add(1)
	s1.Add(2)
	s1.Add(3)

	s2 := InitSet(nil)
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
	s := InitSet(nil)

	if s.Contains(1) || s.Contains(2) || s.Contains(3) || s.Contains(4) {
		t.Errorf("Ghost values before update")
	}

	s.Update([]any{1, 2, 3, 4})

	if !s.Contains(1) || !s.Contains(2) || !s.Contains(3) || !s.Contains(4) {
		t.Errorf("Missing values in update")
	}

	if len(s) != 4 {
		t.Errorf("Unexpected length after update")
	}
}
