package main

import (
	"fmt"
	"log"
	"os"
	"strings"

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
	if len(os.Args) < 3 {
		fmt.Println("Uso: go run cmd/node/main.go [10|20|25] [puerto]")
		return
	}

	size := os.Args[1]
	port := os.Args[2]

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

	fmt.Printf("[NODE] Dataset cargado OK: %s\n", datasetPath)
	fmt.Printf("[NODE] Usuarios: %d  |  Películas: %d\n", ds.Users, ds.Movies)

	err = StartNodeServer(port, ds)
	if err != nil {
		log.Fatal(err)
	}

}
