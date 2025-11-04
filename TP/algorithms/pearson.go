package algorithms

import (
	"math"
	"sync"
)

// PearsonSequential calcula la correlación de Pearson entre todos los usuarios de forma secuencial
func PearsonSequential(users []User) [][]float64 {
	n := len(users)
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		similarity[i][i] = 1.0
		for j := i + 1; j < n; j++ {
			sim := pearsonCorrelation(users[i], users[j])
			similarity[i][j] = sim
			similarity[j][i] = sim
		}
	}

	return similarity
}

// PearsonConcurrent calcula la correlación de Pearson usando goroutines
func PearsonConcurrent(users []User, numWorkers int) [][]float64 {
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
				sim := pearsonCorrelation(users[j.i], users[j.j])
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

// pearsonCorrelation calcula la correlación de Pearson entre dos usuarios
func pearsonCorrelation(u1, u2 User) float64 {
	// Encontrar juegos en común
	var commonGames []int
	for appID := range u1.Games {
		if _, exists := u2.Games[appID]; exists {
			commonGames = append(commonGames, appID)
		}
	}

	// Se necesitan al menos 2 juegos en común para calcular correlación
	if len(commonGames) < 2 {
		return 0.0
	}

	// Construir vectores de valores para juegos en común
	// Vector combinando playtime_norm y rating
	var values1, values2 []float64
	for _, appID := range commonGames {
		game1 := u1.Games[appID]
		game2 := u2.Games[appID]

		// Agregar ambas características
		values1 = append(values1, game1.PlaytimeNorm, game1.Rating)
		values2 = append(values2, game2.PlaytimeNorm, game2.Rating)
	}

	n := len(values1)
	if n == 0 {
		return 0.0
	}

	// Calcular medias
	var sum1, sum2 float64
	for i := 0; i < n; i++ {
		sum1 += values1[i]
		sum2 += values2[i]
	}
	mean1 := sum1 / float64(n)
	mean2 := sum2 / float64(n)

	// Calcular correlación de Pearson
	var numerator, denom1, denom2 float64
	for i := 0; i < n; i++ {
		diff1 := values1[i] - mean1
		diff2 := values2[i] - mean2
		numerator += diff1 * diff2
		denom1 += diff1 * diff1
		denom2 += diff2 * diff2
	}

	// Evitar división por cero
	if denom1 == 0 || denom2 == 0 {
		return 0.0
	}

	denominator := math.Sqrt(denom1 * denom2)
	if denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}
