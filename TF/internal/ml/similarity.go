package ml

import (
	"math"
)

// Cosine similarity between two sparse vectors (maps). Uses all keys present in either map.
func Cosine(a, b map[int]float64) float64 {
	var dot, suma, sumb float64
	for k, va := range a {
		if vb, ok := b[k]; ok {
			dot += va * vb
		}
		suma += va * va
	}
	// include b-only terms in sumb
	for _, vb := range b {
		sumb += vb * vb
	}
	if suma == 0 || sumb == 0 {
		return 0
	}
	return dot / (math.Sqrt(suma) * math.Sqrt(sumb))
}

// Pearson correlation computed only on common keys (co-rated items).
// If fewer than 2 common keys, returns 0.
// Pearson centrado por usuario (mean-centered) sólo sobre items comunes
func Pearson(a, b map[int]float64) float64 {
	// sacar comunes
	common := 0
	for k := range a {
		if _, ok := b[k]; ok {
			common++
		}
	}
	if common < 2 {
		return 0
	}

	// sacar medias solo de comunes
	var sumA, sumB float64
	for k := range a {
		if _, ok := b[k]; ok {
			sumA += a[k]
			sumB += b[k]
		}
	}
	meanA := sumA / float64(common)
	meanB := sumB / float64(common)

	// Pearson
	var num, denA, denB float64
	for k := range a {
		vb, ok := b[k]
		if !ok {
			continue
		}
		da := a[k] - meanA
		db := vb - meanB
		num += da * db
		denA += da * da
		denB += db * db
	}
	if denA == 0 || denB == 0 {
		return 0
	}
	return num / (math.Sqrt(denA) * math.Sqrt(denB))
}

// Jaccard index on the support (ignora pesos, solo presencia)
func Jaccard(a, b map[int]float64) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 0
	}
	inter := 0
	union := make(map[int]struct{})
	for k := range a {
		union[k] = struct{}{}
		if _, ok := b[k]; ok {
			inter++
		}
	}
	for k := range b {
		union[k] = struct{}{}
	}
	if len(union) == 0 {
		return 0
	}
	return float64(inter) / float64(len(union))
}
