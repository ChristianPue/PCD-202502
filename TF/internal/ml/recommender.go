package ml

import (
	"container/heap"
	"sort"
)

// SimMetric enumerator
type SimMetric int

const (
	CosineSim SimMetric = iota
	PearsonSim
	JaccardSim
)

// ItemScore guarda predicción/score para un item
type ItemScore struct {
	MovieID int
	Score   float64
}

// ----------------- helpers -----------------

// BuildItemIndex: item -> (user->rating)
func BuildItemIndex(ds *Dataset) map[int]map[int]float64 {
	itemIndex := make(map[int]map[int]float64)
	for u, items := range ds.UserRatings {
		for itm, r := range items {
			if _, ok := itemIndex[itm]; !ok {
				itemIndex[itm] = make(map[int]float64)
			}
			itemIndex[itm][u] = r
		}
	}
	return itemIndex
}

// simBetween: dispatch a la función correspondiente
func simBetween(a, b map[int]float64, metric SimMetric) float64 {
	switch metric {
	case CosineSim:
		return Cosine(a, b)
	case PearsonSim:
		return Pearson(a, b)
	case JaccardSim:
		return Jaccard(a, b)
	default:
		return Cosine(a, b)
	}
}

// topKFromMap: ordena y devuelve top K ItemScore
func topKFromMap(scores map[int]float64, k int) []ItemScore {
	top := make([]ItemScore, 0, len(scores))
	for m, sc := range scores {
		top = append(top, ItemScore{MovieID: m, Score: sc})
	}
	sort.Slice(top, func(i, j int) bool { return top[i].Score > top[j].Score })
	if len(top) > k {
		return top[:k]
	}
	return top
}

// ----------------- KNN utilities -----------------

// simple min-heap for top-N neighbors
type neighbor struct {
	id    int
	score float64
}
type minHeap []neighbor

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(neighbor)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func topNneighborsFromScores(scores map[int]float64, n int) []neighbor {
	h := &minHeap{}
	heap.Init(h)
	for id, sc := range scores {
		if h.Len() < n {
			heap.Push(h, neighbor{id: id, score: sc})
		} else if sc > (*h)[0].score {
			heap.Pop(h)
			heap.Push(h, neighbor{id: id, score: sc})
		}
	}
	res := make([]neighbor, h.Len())
	for i := len(res) - 1; i >= 0; i-- {
		res[i] = heap.Pop(h).(neighbor)
	}
	return res
}

// ----------------- Item-based collaborative filtering -----------------

// RecommendItemBased:
// - ds: dataset ya cargado
// - user: userId objetivo
// - topK: cuántas recomendaciones devolver
// - metric: similitud a usar
// - neighborK: cuántos vecinos por candidato considerar (si 0 -> usar todos los items que user calificó)
func RecommendItemBased(ds *Dataset, user int, topK int, metric SimMetric, neighborK int) []ItemScore {
	userRatings, ok := ds.UserRatings[user]
	if !ok {
		return nil
	}
	itemIndex := BuildItemIndex(ds)

	// candidatos = todos los items excepto los ya vistos por user
	candidates := make([]int, 0)
	for it := range itemIndex {
		if _, seen := userRatings[it]; !seen {
			candidates = append(candidates, it)
		}
	}

	scores := make(map[int]float64)

	for _, itemV := range candidates {
		// calcular similitudes entre itemV y los items que user calificó
		simScores := make(map[int]float64, len(userRatings))
		vecB := itemIndex[itemV]
		for itemU := range userRatings {
			vecA := itemIndex[itemU]
			simScores[itemU] = simBetween(vecA, vecB, metric)
		}

		// escoger top neighborK si se solicitó
		var neighbors []neighbor
		if neighborK > 0 {
			neighbors = topNneighborsFromScores(simScores, neighborK)
		} else {
			// convertir todos
			neighbors = make([]neighbor, 0, len(simScores))
			for id, sc := range simScores {
				neighbors = append(neighbors, neighbor{id: id, score: sc})
			}
		}

		num := 0.0
		den := 0.0
		for _, nb := range neighbors {
			r := userRatings[nb.id] // rating del user sobre itemU
			num += nb.score * r
			den += abs(nb.score)
		}
		if den != 0 {
			scores[itemV] = num / den
		} else {
			scores[itemV] = 0
		}
	}

	return topKFromMap(scores, topK)
}

// ----------------- User-based collaborative filtering -----------------

// RecommendUserBased:
// - predice usando los K vecinos usuarios más similares
// - neighborK = cuántos vecinos usuarios considerar
func RecommendUserBased(ds *Dataset, user int, topK int, metric SimMetric, neighborK int) []ItemScore {
	targetRatings, ok := ds.UserRatings[user]
	if !ok {
		return nil
	}
	// construir similitudes entre user y todos los otros users
	userSims := make(map[int]float64)
	for other, ratings := range ds.UserRatings {
		if other == user {
			continue
		}
		userSims[other] = simBetween(targetRatings, ratings, metric)
		if userSims[other] < 0.05 { // este threshold lo vas a tunear luego
			continue
		}
	}

	// seleccionar vecinos top neighborK
	neighbors := topNneighborsFromScores(userSims, neighborK)
	if len(neighbors) == 0 {
		return nil
	}

	// candidatos = items que los vecinos han calificado pero el target no
	candidatesMap := make(map[int]struct{})
	for _, nb := range neighbors {
		other := nb.id
		for it := range ds.UserRatings[other] {
			if _, seen := targetRatings[it]; !seen {
				candidatesMap[it] = struct{}{}
			}
		}
	}
	candidates := make([]int, 0, len(candidatesMap))
	for it := range candidatesMap {
		candidates = append(candidates, it)
	}

	// para cada candidato, agregar weighted avg de vecinos
	scores := make(map[int]float64)
	for _, it := range candidates {
		num := 0.0
		den := 0.0
		for _, nb := range neighbors {
			other := nb.id
			sim := nb.score
			if r, ok := ds.UserRatings[other][it]; ok {
				num += sim * r
				den += abs(sim)
			}
		}
		if den != 0 {
			scores[it] = num / den
		} else {
			scores[it] = 0
		}
	}

	return topKFromMap(scores, topK)
}

// ----------------- util -----------------
func abs(a float64) float64 {
	if a < 0 {
		return -a
	}
	return a
}

func scoreItem(ds *Dataset, userRatings map[int]float64, itemIndex map[int]map[int]float64, itemV int, metric SimMetric, neighborK int) float64 {
	simScores := make(map[int]float64, len(userRatings))
	vecB := itemIndex[itemV]

	for itemU := range userRatings {
		vecA := itemIndex[itemU]
		simScores[itemU] = simBetween(vecA, vecB, metric)
	}

	neighbors := topNneighborsFromScores(simScores, neighborK)

	num := 0.0
	den := 0.0
	for _, nb := range neighbors {
		r := userRatings[nb.id]
		num += nb.score * r
		if nb.score < 0 {
			den -= nb.score
		} else {
			den += nb.score
		}
	}
	if den == 0 {
		return 0
	}
	return num / den
}

func RecommendItemBasedParallel(ds *Dataset, user int, topK int, metric SimMetric, neighborK int, workers int) []ItemScore {
	userRatings, ok := ds.UserRatings[user]
	if !ok {
		return nil
	}

	itemIndex := BuildItemIndex(ds)

	// candidatos
	candidates := make([]int, 0)
	for it := range itemIndex {
		if _, seen := userRatings[it]; !seen {
			candidates = append(candidates, it)
		}
	}

	chunk := len(candidates) / workers
	if chunk == 0 {
		chunk = 1
	}

	out := make(chan map[int]float64, workers)

	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > len(candidates) {
			end = len(candidates)
		}

		go func(slice []int) {
			partial := make(map[int]float64)
			for _, itemV := range slice {
				// *** igual que la lógica normal que ya tienes ***
				partial[itemV] = scoreItem(ds, userRatings, itemIndex, itemV, metric, neighborK)
			}
			out <- partial
		}(candidates[start:end])
	}

	// merge
	scores := make(map[int]float64)
	for i := 0; i < workers; i++ {
		part := <-out
		for k, v := range part {
			scores[k] = v
		}
	}

	return topKFromMap(scores, topK)
}
