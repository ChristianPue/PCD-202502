package ml

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type MovieInfo struct {
	ID     int    `json:"id"`
	Title  string `json:"title"`
	Genres string `json:"genres"`
}

type Dataset struct {
	UserRatings map[int]map[int]float64
	MoviesMeta  map[int]MovieInfo
	Users       int
	Movies      int
	Tags        map[int][]string
}

// ================================================
// LoadDataset: Carga optimizada
// Orden: Movies -> Ratings (filtrando) -> Tags
// ================================================
func LoadDataset(ratingsPath string) (*Dataset, error) {
	ds := &Dataset{
		UserRatings: make(map[int]map[int]float64),
		MoviesMeta:  make(map[int]MovieInfo),
		Tags:        make(map[int][]string),
	}

	dir := filepath.Dir(ratingsPath)

	// -----------------------
	// 1. CARGAR MOVIES (PRIMERO)
	// -----------------------
	// Formato: movieId::title::genres
	moviesPath := filepath.Join(dir, "movies.csv")
	fmt.Println("[DATASET] Cargando películas desde:", moviesPath)

	fm, err := os.Open(moviesPath)
	if err != nil {
		return nil, fmt.Errorf("no se pudo abrir movies.csv: %v", err)
	}
	defer fm.Close()

	scannerM := bufio.NewScanner(fm)
	for scannerM.Scan() {
		line := scannerM.Text()
		parts := strings.Split(line, "::") // Separador manual
		if len(parts) < 3 {
			continue
		}

		mid, _ := strconv.Atoi(parts[0])
		title := parts[1]
		genres := parts[2]

		ds.MoviesMeta[mid] = MovieInfo{
			ID:     mid,
			Title:  title,
			Genres: genres,
		}
		if mid > ds.Movies {
			ds.Movies = mid
		}
	}
	fmt.Printf("[DATASET] %d películas cargadas.\n", len(ds.MoviesMeta))

	// -----------------------
	// 2. CARGAR RATINGS
	// -----------------------
	// Formato: userId,movieId,rating,timestamp
	fmt.Println("[DATASET] Cargando ratings desde:", ratingsPath)

	f, err := os.Open(ratingsPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.ReuseRecord = true // OPTIMIZACIÓN: Reutilizar memoria del slice para cada fila

	// Saltar header si existe (detectando si el primer campo es texto)
	// O simplemente leemos uno y lo descartamos si sabemos que tiene header
	_, _ = r.Read()

	count := 0
	ignored := 0

	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
		if len(row) < 3 {
			continue
		}

		mid, _ := strconv.Atoi(row[1])

		// FILTRO EN TIEMPO REAL:
		// Si la película no existe en movies.csv, ignoramos el rating.
		// Esto evita cargar basura en RAM y evita el paso de "limpieza" posterior.
		if _, exists := ds.MoviesMeta[mid]; !exists {
			ignored++
			continue
		}

		uid, _ := strconv.Atoi(row[0])
		raw, _ := strconv.ParseFloat(row[2], 64)
		rating := raw / 5.0 // Normalizar

		if _, ok := ds.UserRatings[uid]; !ok {
			ds.UserRatings[uid] = make(map[int]float64)
		}
		ds.UserRatings[uid][mid] = rating

		if uid > ds.Users {
			ds.Users = uid
		}
		count++
	}
	fmt.Printf("[DATASET] %d ratings cargados. (%d ignorados por falta de metadata)\n", count, ignored)

	// -----------------------
	// 3. CARGAR TAGS (Opcional)
	// -----------------------
	// Formato: UserID::MovieID::Tag::Timestamp
	tagsPath := filepath.Join(dir, "tags.csv")
	ft, err := os.Open(tagsPath)
	if err == nil {
		defer ft.Close()

		// IMPORTANTE: Usamos bufio scanner porque el separador es "::"
		// csv.NewReader NO soporta separadores de múltiples caracteres.
		scannerT := bufio.NewScanner(ft)
		tagsCount := 0

		for scannerT.Scan() {
			line := scannerT.Text()
			parts := strings.Split(line, "::")
			if len(parts) < 3 {
				continue
			}

			mid, _ := strconv.Atoi(parts[1])
			tag := parts[2]

			// Solo guardar tags de películas que existen
			if _, exists := ds.MoviesMeta[mid]; exists {
				ds.Tags[mid] = append(ds.Tags[mid], tag)
				tagsCount++
			}
		}
		fmt.Printf("[DATASET] %d tags cargados.\n", tagsCount)
	} else {
		fmt.Println("[DATASET] tags.csv no encontrado, continuando sin tags.")
	}

	return ds, nil
}
