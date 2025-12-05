package ml

import (
	"bufio"
	"encoding/csv"
	"fmt"
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
	Tags        map[int][]string // opcional (movieID -> lista de tags)
}

// ================================================
// LoadDataset: ratings.csv + movies.csv (+ tags.csv opcional)
// ================================================
func LoadDataset(ratingsPath string) (*Dataset, error) {

	// -----------------------
	// 1. CARGAR RATINGS
	// -----------------------
	f, err := os.Open(ratingsPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = ','   // ratings.csv usa comas
	_, _ = r.Read() // skip header

	ds := &Dataset{
		UserRatings: make(map[int]map[int]float64),
		MoviesMeta:  make(map[int]MovieInfo),
		Tags:        make(map[int][]string),
	}

	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		if len(row) < 3 {
			continue
		}

		uid, _ := strconv.Atoi(row[0])
		mid, _ := strconv.Atoi(row[1])
		raw, _ := strconv.ParseFloat(row[2], 64)

		// normalizar rating 1..5 → 0..1
		rating := raw / 5.0

		if _, ok := ds.UserRatings[uid]; !ok {
			ds.UserRatings[uid] = map[int]float64{}
		}

		ds.UserRatings[uid][mid] = rating

		if uid > ds.Users {
			ds.Users = uid
		}
		if mid > ds.Movies {
			ds.Movies = mid
		}
	}

	// -----------------------
	// 2. CARGAR MOVIES
	// -----------------------
	dir := filepath.Dir(ratingsPath)
	moviesPath := filepath.Join(dir, "movies.csv")

	fm, err := os.Open(moviesPath)
	if err == nil {
		defer fm.Close()

		scanner := bufio.NewScanner(fm)

		for scanner.Scan() {
			line := scanner.Text()
			parts := strings.Split(line, "::")
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
		}

		fmt.Printf("Cargadas %d películas desde movies.csv.\n", len(ds.MoviesMeta))
	} else {
		fmt.Println("Advertencia: movies.csv no encontrado.")
	}

	// -----------------------
	// 3. CARGAR TAGS (opcional)
	// -----------------------
	tagsPath := filepath.Join(dir, "tags.csv")
	ft, err := os.Open(tagsPath)
	if err == nil {
		defer ft.Close()

		rt := csv.NewReader(ft)
		// tags también vienen separadas por ::
		for {
			rowRaw, err := rt.Read()
			if err != nil {
				break
			}
			if len(rowRaw) == 0 {
				continue
			}

			parts := strings.Split(rowRaw[0], "::")
			if len(parts) < 3 {
				continue
			}

			mid, _ := strconv.Atoi(parts[1])
			tag := parts[2]

			ds.Tags[mid] = append(ds.Tags[mid], tag)
		}

		fmt.Printf("Cargados tags para %d películas.\n", len(ds.Tags))
	}

	// ----------------------------------------------------------
	// 4. Asegurar coherencia: toda película de ratings debe
	// tener metadata válida, incluso si no aparece en movies.csv
	// ----------------------------------------------------------
	missing := 0
	for _, items := range ds.UserRatings {
		for mid := range items {
			if _, ok := ds.MoviesMeta[mid]; !ok {
				// Crear metadato sintético
				ds.MoviesMeta[mid] = MovieInfo{
					ID:     mid,
					Title:  fmt.Sprintf("Unknown Movie %d", mid),
					Genres: "Unknown",
				}
				missing++
			}
		}
	}

	fmt.Printf("Películas sin metadata corregidas: %d\n", missing)

	// ----------------------------------------------------------
	// 5. LIMPIEZA: eliminar películas sin metadata real
	// ----------------------------------------------------------

	cleaned := make(map[int]map[int]float64)
	removedCount := 0

	for uid, items := range ds.UserRatings {
		for mid, rating := range items {
			meta, ok := ds.MoviesMeta[mid]
			if !ok {
				// película no existe en movies.csv → eliminar
				removedCount++
				continue
			}
			if strings.HasPrefix(meta.Title, "Unknown") || meta.Genres == "Unknown" {
				// metadata sintética → eliminar
				removedCount++
				continue
			}

			// si pasa filtros, incluir en nuevo mapa
			if _, exists := cleaned[uid]; !exists {
				cleaned[uid] = make(map[int]float64)
			}
			cleaned[uid][mid] = rating
		}
	}

	ds.UserRatings = cleaned

	fmt.Printf("Limpieza: %d ratings eliminados por falta de metadata.\n", removedCount)

	return ds, nil
}
