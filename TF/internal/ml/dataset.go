package ml

import (
	"encoding/csv"
	"os"
	"strconv"
)

type Dataset struct {
	UserRatings map[int]map[int]float64
	Users       int
	Movies      int
}

// Etapa 1: leer → limpiar → seleccionar campos → normalizar rating
func LoadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	_, _ = r.Read() // skip header

	ds := &Dataset{
		UserRatings: make(map[int]map[int]float64),
	}

	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		if len(row) < 3 {
			continue // limpieza mínima
		}

		uid, _ := strconv.Atoi(row[0])
		mid, _ := strconv.Atoi(row[1])
		raw, _ := strconv.ParseFloat(row[2], 64)

		// normalización explícita 1..5 → [0..1]
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

	return ds, nil
}
