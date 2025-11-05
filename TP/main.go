package main

import (
	"TP/algorithms"
	"TP/benchmark"
	"encoding/csv"
	"fmt"
	"os"
	"runtime"
	"strconv"
)

const (
	datasetPath = "preprocessing/steam_knn_ready.csv"
	resultsPath = "results/results_benchmark.csv"
	resultsDir  = "results"
)

func main() {
	fmt.Println("==============================================")
	fmt.Println("  Sistema de Recomendación Steam - Entregable 2")
	fmt.Println("  Análisis de Algoritmos de Similitud")
	fmt.Println("==============================================")

	// Detectar número de CPUs
	numCPU := runtime.NumCPU()
	fmt.Printf("CPUs detectados: %d\n\n", numCPU)

	// Cargar dataset completo
	fmt.Println("Cargando dataset...")
	allUsers, err := loadDataset(datasetPath)
	if err != nil {
		fmt.Printf("Error al cargar dataset: %v\n", err)
		return
	}
	fmt.Printf("Dataset cargado: %d usuarios únicos\n\n", len(allUsers))

	// Crear directorio de resultados si no existe
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		fmt.Printf("Error al crear directorio de resultados: %v\n", err)
		return
	}

	// Configuración de pruebas
	testSizes := []int{5000, 7000, 9000, 11000} // Datasets más grandes
	//workerCounts := []int{2, 4, 8, 16, 32}
	workerCounts := []int{2, 4, 8}

	// Ajustar workers según CPUs disponibles
	maxWorkers := numCPU * 2
	var adjustedWorkers []int
	for _, w := range workerCounts {
		if w <= maxWorkers {
			adjustedWorkers = append(adjustedWorkers, w)
		}
	}
	adjustedWorkers = append(adjustedWorkers, maxWorkers)

	fmt.Printf("Configuración de pruebas:\n")
	fmt.Printf("  - Tamaños: %v\n", testSizes)
	fmt.Printf("  - Workers: %v\n", adjustedWorkers)
	fmt.Println()

	// Preparar archivo de resultados CSV
	resultsFile, err := os.Create(resultsPath)
	if err != nil {
		fmt.Printf("Error al crear archivo de resultados: %v\n", err)
		return
	}
	defer resultsFile.Close()

	csvWriter := csv.NewWriter(resultsFile)
	defer csvWriter.Flush()

	// Escribir encabezado
	csvWriter.Write([]string{
		"algoritmo",
		"tamaño_dataset",
		"modo",
		"num_workers",
		"tiempo_ms",
		"speedup",
		"comparaciones",
	})

	// Ejecutar pruebas para cada tamaño
	for _, size := range testSizes {
		if size > len(allUsers) {
			fmt.Printf("⚠️  Tamaño %d excede usuarios disponibles (%d). Omitiendo...\n\n", size, len(allUsers))
			continue
		}

		users := allUsers[:size]
		numComparisons := (size * (size - 1)) / 2

		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("📊 PRUEBAS CON %d USUARIOS (%d comparaciones)\n", size, numComparisons)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

		// Probar cada algoritmo
		testAlgorithm("Cosine Similarity", users, adjustedWorkers, csvWriter,
			algorithms.CosineSequential,
			algorithms.CosineConcurrent)

		testAlgorithm("Pearson Correlation", users, adjustedWorkers, csvWriter,
			algorithms.PearsonSequential,
			algorithms.PearsonConcurrent)

		testAlgorithm("Jaccard Index", users, adjustedWorkers, csvWriter,
			algorithms.JaccardSequential,
			algorithms.JaccardConcurrent)

		testAlgorithm("Jaccard Weighted", users, adjustedWorkers, csvWriter,
			algorithms.JaccardWeightedSequential,
			algorithms.JaccardWeightedConcurrent)

		fmt.Println()
	}

	fmt.Printf("✅ Resultados guardados en: %s\n", resultsPath)
	fmt.Println("\n==============================================")
	fmt.Println("  Pruebas completadas exitosamente")
	fmt.Println("==============================================")
}

// testAlgorithm ejecuta pruebas secuenciales y concurrentes para un algoritmo
func testAlgorithm(
	name string,
	users []algorithms.User,
	workerCounts []int,
	csvWriter *csv.Writer,
	seqFunc func([]algorithms.User) [][]float64,
	concFunc func([]algorithms.User, int) [][]float64,
) {
	fmt.Printf("🔍 %s\n", name)
	fmt.Println("─────────────────────────────────────────────")

	size := len(users)
	numComparisons := (size * (size - 1)) / 2

	// Ejecutar versión secuencial
	fmt.Print("   Secuencial... ")
	seqTime := benchmark.MeasureTime(func() {
		seqFunc(users)
	})
	fmt.Printf("%.2f ms\n", seqTime)

	// Guardar resultado secuencial
	csvWriter.Write([]string{
		name,
		strconv.Itoa(size),
		"secuencial",
		"1",
		fmt.Sprintf("%.2f", seqTime),
		"1.00",
		strconv.Itoa(numComparisons),
	})
	csvWriter.Flush()

	// Ejecutar versiones concurrentes
	fmt.Println("   Concurrente:")
	for _, workers := range workerCounts {
		concTime := benchmark.MeasureTime(func() {
			concFunc(users, workers)
		})
		speedup := seqTime / concTime

		fmt.Printf("     %2d workers: %.2f ms (speedup: %.2fx)\n",
			workers, concTime, speedup)

		// Guardar resultado concurrente
		csvWriter.Write([]string{
			name,
			strconv.Itoa(size),
			"concurrente",
			strconv.Itoa(workers),
			fmt.Sprintf("%.2f", concTime),
			fmt.Sprintf("%.2f", speedup),
			strconv.Itoa(numComparisons),
		})
		csvWriter.Flush()
	}

	fmt.Println()
}

// loadDataset carga el dataset CSV y construye la estructura de usuarios
func loadDataset(filepath string) ([]algorithms.User, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("no se pudo abrir el archivo: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Leer encabezado
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("error al leer encabezado: %w", err)
	}

	// Verificar columnas esperadas
	if len(header) < 4 {
		return nil, fmt.Errorf("formato incorrecto, se esperaban al menos 4 columnas")
	}

	fmt.Printf("Columnas detectadas: %v\n", header)

	// Mapa temporal para agrupar por usuario
	userMap := make(map[string]*algorithms.User)

	// Leer todas las filas
	lineNum := 1
	for {
		record, err := reader.Read()
		if err != nil {
			break // EOF o error
		}
		lineNum++

		if len(record) < 4 {
			continue // Saltar filas incompletas
		}

		// Parsear datos
		appID, err := strconv.Atoi(record[0])
		if err != nil {
			continue
		}

		steamID := record[1]

		playtimeNorm, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			continue
		}

		rating, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			continue
		}

		// Agregar o actualizar usuario
		if user, exists := userMap[steamID]; exists {
			user.Games[appID] = algorithms.GameInteraction{
				PlaytimeNorm: playtimeNorm,
				Rating:       rating,
			}
		} else {
			userMap[steamID] = &algorithms.User{
				SteamID: steamID,
				Games: map[int]algorithms.GameInteraction{
					appID: {
						PlaytimeNorm: playtimeNorm,
						Rating:       rating,
					},
				},
			}
		}
	}

	// Convertir mapa a slice
	users := make([]algorithms.User, 0, len(userMap))
	for _, user := range userMap {
		users = append(users, *user)
	}

	return users, nil
}
