package algorithms

import (
	"sync"
)

// JaccardSequential calcula el índice de Jaccard entre todos los usuarios de forma secuencial
func JaccardSequential(users []User) [][]float64 {
	n := len(users)
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		similarity[i][i] = 1.0
		for j := i + 1; j < n; j++ {
			sim := jaccardIndex(users[i], users[j])
			similarity[i][j] = sim
			similarity[j][i] = sim
		}
	}

	return similarity
}

// JaccardConcurrent calcula el índice de Jaccard usando goroutines
func JaccardConcurrent(users []User, numWorkers int) [][]float64 {
	n := len(users)
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
		similarity[i][i] = 1.0
	}

	totalComparisons := (n * (n - 1)) / 2
	bufferSize := min(totalComparisons, numWorkers*100)

	type job struct {
		i, j int
	}
	jobs := make(chan job, bufferSize)

	type result struct {
		i, j int
		sim  float64
	}
	results := make(chan result, bufferSize)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				sim := jaccardIndex(users[j.i], users[j.j])
				results <- result{j.i, j.j, sim}
			}
		}()
	}

	go func() {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				jobs <- job{i, j}
			}
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	var mu sync.Mutex
	for res := range results {
		mu.Lock()
		similarity[res.i][res.j] = res.sim
		similarity[res.j][res.i] = res.sim
		mu.Unlock()
	}

	return similarity
}

// jaccardIndex calcula el índice de Jaccard entre dos usuarios
// Jaccard mide la similitud basándose en juegos en común
func jaccardIndex(u1, u2 User) float64 {
	// Contar intersección (juegos en común)
	intersection := 0
	for appID := range u1.Games {
		if _, exists := u2.Games[appID]; exists {
			intersection++
		}
	}

	// Contar unión (juegos totales únicos entre ambos usuarios)
	union := len(u1.Games) + len(u2.Games) - intersection

	// Evitar división por cero
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// JaccardWeighted calcula un índice de Jaccard ponderado usando rating y playtime
// Esta es una variante que considera no solo presencia/ausencia sino también la intensidad
func JaccardWeighted(u1, u2 User) float64 {
	// Encontrar juegos en común
	var minSum, maxSum float64

	// Crear un set de todos los juegos de ambos usuarios
	allGames := make(map[int]bool)
	for appID := range u1.Games {
		allGames[appID] = true
	}
	for appID := range u2.Games {
		allGames[appID] = true
	}

	// Para cada juego, calcular min y max de las interacciones
	for appID := range allGames {
		var val1, val2 float64

		if game1, exists := u1.Games[appID]; exists {
			val1 = game1.PlaytimeNorm + game1.Rating
		}

		if game2, exists := u2.Games[appID]; exists {
			val2 = game2.PlaytimeNorm + game2.Rating
		}

		if val1 < val2 {
			minSum += val1
			maxSum += val2
		} else {
			minSum += val2
			maxSum += val1
		}
	}

	// Evitar división por cero
	if maxSum == 0 {
		return 0.0
	}

	return minSum / maxSum
}

// JaccardWeightedSequential calcula Jaccard ponderado de forma secuencial
func JaccardWeightedSequential(users []User) [][]float64 {
	n := len(users)
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		similarity[i][i] = 1.0
		for j := i + 1; j < n; j++ {
			sim := JaccardWeighted(users[i], users[j])
			similarity[i][j] = sim
			similarity[j][i] = sim
		}
	}

	return similarity
}

// JaccardWeightedConcurrent calcula Jaccard ponderado usando goroutines
func JaccardWeightedConcurrent(users []User, numWorkers int) [][]float64 {
	n := len(users)
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
		similarity[i][i] = 1.0
	}

	totalComparisons := (n * (n - 1)) / 2
	bufferSize := min(totalComparisons, numWorkers*100)

	type job struct {
		i, j int
	}
	jobs := make(chan job, bufferSize)

	type result struct {
		i, j int
		sim  float64
	}
	results := make(chan result, bufferSize)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				sim := JaccardWeighted(users[j.i], users[j.j])
				results <- result{j.i, j.j, sim}
			}
		}()
	}

	go func() {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				jobs <- job{i, j}
			}
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	var mu sync.Mutex
	for res := range results {
		mu.Lock()
		similarity[res.i][res.j] = res.sim
		similarity[res.j][res.i] = res.sim
		mu.Unlock()
	}

	return similarity
}
