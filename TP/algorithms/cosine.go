package algorithms

import (
	"math"
	"sync"
)

// User representa un usuario con sus interacciones con juegos
type User struct {
	SteamID string
	Games   map[int]GameInteraction // key: app_id
}

// GameInteraction contiene los datos de interacción de un usuario con un juego
type GameInteraction struct {
	PlaytimeNorm float64
	Rating       float64
}

// CosineSequential calcula la similitud coseno entre todos los usuarios de forma secuencial
func CosineSequential(users []User) [][]float64 {
	n := len(users)
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
	}

	// Calcular similitud para cada par de usuarios
	for i := 0; i < n; i++ {
		similarity[i][i] = 1.0 // Un usuario es idéntico a sí mismo
		for j := i + 1; j < n; j++ {
			sim := cosineSimilarity(users[i], users[j])
			similarity[i][j] = sim
			similarity[j][i] = sim // La matriz es simétrica
		}
	}

	return similarity
}

// CosineConcurrent calcula la similitud coseno usando goroutines
func CosineConcurrent(users []User, numWorkers int) [][]float64 {
	n := len(users)
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
		similarity[i][i] = 1.0
	}

	// Calcular número total de comparaciones
	totalComparisons := (n * (n - 1)) / 2

	// Aumentar buffer basado en comparaciones
	bufferSize := min(totalComparisons, numWorkers*100)

	// Canal para enviar trabajos
	type job struct {
		i, j int
	}
	jobs := make(chan job, bufferSize)

	// Canal para recibir resultados
	type result struct {
		i, j int
		sim  float64
	}
	results := make(chan result, bufferSize)

	// Iniciar workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				sim := cosineSimilarity(users[j.i], users[j.j])
				results <- result{j.i, j.j, sim}
			}
		}()
	}

	// Enviar trabajos
	go func() {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				jobs <- job{i, j}
			}
		}
		close(jobs)
	}()

	// Cerrar canal de resultados cuando terminen todos los workers
	go func() {
		wg.Wait()
		close(results)
	}()

	// Recolectar resultados con mutex para evitar race conditions
	var mu sync.Mutex
	for res := range results {
		mu.Lock()
		similarity[res.i][res.j] = res.sim
		similarity[res.j][res.i] = res.sim
		mu.Unlock()
	}

	return similarity
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// cosineSimilarity calcula la similitud coseno entre dos usuarios
func cosineSimilarity(u1, u2 User) float64 {
	// Encontrar juegos en común
	var dotProduct, norm1, norm2 float64

	// Crear un vector combinando playtime_norm y rating
	for appID, game1 := range u1.Games {
		// Vector: [playtime_norm, rating]
		v1 := []float64{game1.PlaytimeNorm, game1.Rating}

		if game2, exists := u2.Games[appID]; exists {
			v2 := []float64{game2.PlaytimeNorm, game2.Rating}

			// Producto punto
			for k := 0; k < len(v1); k++ {
				dotProduct += v1[k] * v2[k]
			}
		}

		// Norma del usuario 1
		for k := 0; k < len(v1); k++ {
			norm1 += v1[k] * v1[k]
		}
	}

	// Norma del usuario 2
	for _, game2 := range u2.Games {
		v2 := []float64{game2.PlaytimeNorm, game2.Rating}
		for k := 0; k < len(v2); k++ {
			norm2 += v2[k] * v2[k]
		}
	}

	norm1 = math.Sqrt(norm1)
	norm2 = math.Sqrt(norm2)

	// Evitar división por cero
	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}

	return dotProduct / (norm1 * norm2)
}
