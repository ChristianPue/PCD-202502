package main

// Este archivo solo reexporta funciones para mantener el nodo limpio
// La lógica real está en internal/ml/recommender.go

import "TF/internal/ml"

// Wrapper limpio para calcular score de un solo item (item-based)
func ComputeItemScore(ds *ml.Dataset, userID int, item int, metric ml.SimMetric, neighborK int) float64 {
	itemIndex := ml.BuildItemIndex(ds)
	return ml.RecommendScoreSingle(item, ds.UserRatings[userID], itemIndex, metric, neighborK)
}
