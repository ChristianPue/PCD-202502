package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"TF/internal/cluster"
	"TF/internal/ml"
	"TF/internal/storage"
)

var coord *cluster.Coordinator
var redisStore *storage.RedisStore
var mongoStore *storage.MongoStore

// ---------------------------
//
//	/health
//
// ---------------------------
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// ---------------------------
//
//	/recommend/:userId
//
// ---------------------------
func recommendHandler(w http.ResponseWriter, r *http.Request) {
	// obtener userId de la URL
	uidStr := r.URL.Path[len("/recommend/"):]
	uid, err := strconv.Atoi(uidStr)
	if err != nil {
		http.Error(w, "userId inválido", http.StatusBadRequest)
		return
	}

	// -------------------------------
	// 1) Revisar cache Redis primero
	// -------------------------------
	cached, _ := redisStore.GetRecommendations(uid)
	if cached != nil {
		fmt.Println("[CACHE] Resultado desde Redis")

		// convertir []int → []ItemScore
		res := make([]ml.ItemScore, len(cached))
		for i, id := range cached {
			res[i] = ml.ItemScore{MovieID: id, Score: 0} // score no necesario aquí
		}

		w.Header().Set("Content-Type", "application/json")

		// Enriquecer respuesta con metadata (title, genres)
		out := make([]map[string]interface{}, len(res))

		for i, item := range res {
			meta := coord.Dataset.MoviesMeta[item.MovieID]

			out[i] = map[string]interface{}{
				"movieId": item.MovieID,
				"title":   meta.Title,
				"genres":  meta.Genres,
				"score":   item.Score,
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(out)

		return
	}

	// -----------------------------------
	// 2) Si no hay cache → calcular normal
	// -----------------------------------
	fmt.Println("[CACHE] No existe → calculando…")

	// llamar al coordinador con parámetros por defecto
	// topK=10, metric=CosineSim (0), neighborK=20
	res := coord.ComputeRecommendations(uid, 10, ml.CosineSim, 20)

	// guardar en Mongo
	// convertimos ItemScore -> []int (solo MovieID)
	movieIDs := make([]int, len(res))
	for i, r := range res {
		movieIDs[i] = r.MovieID
	}

	// -----------------------------------
	// 3) Guardar en cache Redis
	// -----------------------------------
	redisStore.SaveRecommendations(uid, movieIDs)

	// almacena recomendación
	if mongoStore != nil {
		mongoStore.SaveRecommendation(uid, movieIDs)
	}

	// -------------------------------------------
	// Enriquecer respuesta con metadata
	// -------------------------------------------
	out := make([]map[string]interface{}, len(res))

	for i, item := range res {
		meta := coord.Dataset.MoviesMeta[item.MovieID]

		out[i] = map[string]interface{}{
			"movieId": item.MovieID,
			"title":   meta.Title,
			"genres":  meta.Genres,
			"score":   item.Score,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)

}

// ---------------------------
//
//	MAIN
//
// ---------------------------
func main() {

	// cargar dataset una sola vez
	path := os.Getenv("DATASET_PATH")
	if path == "" {
		log.Fatal("Falta variable DATASET_PATH en API")
	}

	ds, err := ml.LoadDataset(path)

	if err != nil {
		log.Fatal("Error cargando dataset: ", err)
	}

	// inicializar coordinador con nodos y dataset
	coord = cluster.NewCoordinator([]string{
		"localhost:9001",
		"localhost:9002",
	}, ds)

	// iniciar Redis
	redisStore, err = storage.NewRedisStore("localhost:6379")
	if err != nil {
		log.Fatal("Error conectando a Redis:", err)
	}

	// inicializar MongoDB
	mongoStore, err = storage.NewMongoStore("mongodb://localhost:27017", "pcd")
	if err != nil {
		log.Fatal("Error conectando a Mongo:", err)
	}

	fmt.Println("API escuchando en http://localhost:8080")

	// registrar coordinador para WebSocket
	SetWebSocketCoordinator(coord)

	// Middleware CORS
	corsMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/recommend/", recommendHandler)
	mux.HandleFunc("/ws", WebSocketHandler)

	// Nuevos endpoints
	mux.HandleFunc("/movies", func(w http.ResponseWriter, r *http.Request) {
		limitStr := r.URL.Query().Get("limit")
		offsetStr := r.URL.Query().Get("offset")

		limit := 50
		offset := 0

		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
		if o, err := strconv.Atoi(offsetStr); err == nil && o >= 0 {
			offset = o
		}

		var movies []ml.MovieInfo
		// Convertir mapa a slice (ineficiente para datasets grandes, pero funcional para demo)
		// En producción, usar una estructura de datos ordenada o base de datos
		for _, m := range coord.Dataset.MoviesMeta {
			movies = append(movies, m)
		}

		// Paginación simple (en memoria)
		start := offset
		end := offset + limit
		if start > len(movies) {
			start = len(movies)
		}
		if end > len(movies) {
			end = len(movies)
		}

		// Nota: El orden del mapa es aleatorio, así que la paginación será inconsistente
		// sin un ordenamiento previo. Para demo está bien.
		paged := movies[start:end]

		resp := map[string]interface{}{
			"data":   paged,
			"total":  len(movies),
			"limit":  limit,
			"offset": offset,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("/movies/", func(w http.ResponseWriter, r *http.Request) {
		idStr := r.URL.Path[len("/movies/"):]
		id, err := strconv.Atoi(idStr)
		if err != nil {
			http.Error(w, "ID inválido", http.StatusBadRequest)
			return
		}

		if movie, ok := coord.Dataset.MoviesMeta[id]; ok {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(movie)
		} else {
			http.Error(w, "Película no encontrada", http.StatusNotFound)
		}
	})

	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		query := r.URL.Query().Get("q")
		if query == "" {
			http.Error(w, "Query requerida", http.StatusBadRequest)
			return
		}

		var results []ml.MovieInfo
		qLower := strings.ToLower(query)
		// Búsqueda lineal simple por título (case-insensitive)
		for _, m := range coord.Dataset.MoviesMeta {
			if strings.Contains(strings.ToLower(m.Title), qLower) {
				results = append(results, m)
				if len(results) >= 20 { // limitar resultados
					break
				}
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(results)
	})

	log.Fatal(http.ListenAndServe(":8080", corsMiddleware(mux)))
}
