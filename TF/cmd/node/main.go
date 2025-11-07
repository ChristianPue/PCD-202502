package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"time"

	"TF/internal/ml"
)

// simple logger util
func banner(title string) {
	line := strings.Repeat("=", 40)
	fmt.Println(line)
	fmt.Println(">>", title)
	fmt.Println(line)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Uso: go run cmd/node/main.go [10|20|25]")
		return
	}
	size := os.Args[1]

	var datasetPath string
	switch size {
	case "10":
		datasetPath = "dataset/10M/ratings.csv"
	case "20":
		datasetPath = "dataset/20M/ratings.csv"
	case "25":
		datasetPath = "dataset/25M/ratings.csv"
	default:
		log.Fatalf("Tamaño no válido: %s (usa 10, 20 o 25)", size)
	}

	banner("Cargando dataset")
	ds, err := ml.LoadDataset(datasetPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Dataset cargado OK: %s\n", datasetPath)
	fmt.Printf("Usuarios: %d  |  Películas: %d\n", ds.Users, ds.Movies)

	//---------------------------------------------
	// CONFIGURACIÓN EXPERIMENTO
	//---------------------------------------------
	userID := 1
	topK := 10
	neighborK := 30
	metrics := []ml.SimMetric{ml.CosineSim, ml.PearsonSim, ml.JaccardSim}

	//---------------------------------------------
	// ETAPA 3: MEDIR SPEEDUP Y SCALABILITY
	//---------------------------------------------
	fmt.Println()
	banner("Benchmark: Secuencial vs Paralelo")
	fmt.Printf("UserID=%d, TopK=%d, Vecinos=%d\n\n", userID, topK, neighborK)

	for _, metric := range metrics {
		fmt.Printf("==> Métrica: %s\n", metricName(metric))

		// SECUENCIAL
		startSeq := time.Now()
		recsSeq := ml.RecommendItemBased(ds, userID, topK, metric, neighborK)
		durSeq := time.Since(startSeq)
		fmt.Printf("  Secuencial: %v\n", durSeq)

		fmt.Println("    Ejemplo resultados (Secuencial):")
		for i := 0; i < 3 && i < len(recsSeq); i++ {
			r := recsSeq[i]
			fmt.Printf("    %02d) movie=%d  score=%.4f\n", i+1, r.MovieID, r.Score)
		}

		// PARALELO con diferentes workers
		for _, workers := range []int{2, 4, 8, runtime.NumCPU()} {
			startPar := time.Now()
			recsPar := ml.RecommendItemBasedParallel(ds, userID, topK, metric, neighborK, workers)
			durPar := time.Since(startPar)
			speedup := float64(durSeq) / float64(durPar)
			fmt.Printf("  Paralelo (%2d workers): %-10v → Speedup: %.2fx\n", workers, durPar, speedup)

			fmt.Println("    Ejemplo resultados (Paralelo):")
			for i := 0; i < 3 && i < len(recsPar); i++ {
				r := recsPar[i]
				fmt.Printf("    %02d) movie=%d  score=%.4f\n", i+1, r.MovieID, r.Score)
			}
		}
		fmt.Println()
	}
}

// convertir enum a texto
func metricName(m ml.SimMetric) string {
	switch m {
	case ml.CosineSim:
		return "Coseno"
	case ml.PearsonSim:
		return "Pearson"
	case ml.JaccardSim:
		return "Jaccard"
	default:
		return "Desconocida"
	}
}
