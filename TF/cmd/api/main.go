package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"TF/internal/cluster"
	"TF/internal/ml"
	"TF/internal/storage"
)

var (
	coord        *cluster.Coordinator
	redisStore   *storage.RedisStore
	mongoStore   *storage.MongoStore
	sortedMovies []ml.MovieInfo // Cache ordenado para paginación estable
)

// ---------------------------
//
//	Middleware CORS Global
//
// ---------------------------
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 1. Permitir origen (Frontend)
		w.Header().Set("Access-Control-Allow-Origin", "*")

		// 2. Permitir métodos
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")

		// 3. CRÍTICO: Permitir headers específicos como Content-Type
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

		// 4. Manejar preflight (OPTIONS)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// ---------------------------
//
//	Health Check
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
	// Nota: Ya no necesitamos llamar a enableCors aquí manualmente

	uidStr := strings.TrimPrefix(r.URL.Path, "/recommend/")
	uid, err := strconv.Atoi(uidStr)
	if err != nil {
		http.Error(w, "userId inválido", http.StatusBadRequest)
		return
	}

	// 1. Intentar desde Redis (Cache Hit)
	if redisStore != nil {
		cachedIDs, _ := redisStore.GetRecommendations(uid)
		if len(cachedIDs) > 0 {
			fmt.Printf("[API] Cache HIT user %d\n", uid)
			response := enrichMovies(cachedIDs, coord.Dataset)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}
	}

	// 2. Calcular (Cache Miss)
	fmt.Printf("[API] Cache MISS user %d -> Calculando en cluster...\n", uid)
	results := coord.ComputeRecommendations(uid, 10, 1, 20)

	movieIDs := make([]int, len(results))
	for i, item := range results {
		movieIDs[i] = item.MovieID
	}

	// 3. Guardar en Storage (Async)
	go func(u int, ids []int) {
		if redisStore != nil {
			redisStore.SaveRecommendations(u, ids)
		}
		if mongoStore != nil {
			mongoStore.SaveRecommendation(u, ids)
		}
	}(uid, movieIDs)

	// 4. Responder
	response := enrichMovies(movieIDs, coord.Dataset)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Helper para convertir IDs a objetos JSON completos
func enrichMovies(ids []int, ds *ml.Dataset) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(ids))
	for _, id := range ids {
		if meta, ok := ds.MoviesMeta[id]; ok {
			out = append(out, map[string]interface{}{
				"movieId": id,
				"title":   meta.Title,
				"genres":  meta.Genres,
			})
		}
	}
	return out
}

// ---------------------------
//
//	MAIN
//
// ---------------------------
func main() {
	// 1. Cargar configuración
	datasetPath := os.Getenv("DATASET_PATH")
	if datasetPath == "" {
		log.Fatal("Falta variable DATASET_PATH")
	}

	nodesEnv := os.Getenv("WORKER_NODES")
	if nodesEnv == "" {
		nodesEnv = "localhost:9001,localhost:9002,localhost:9003"
		fmt.Println("[WARN] Usando nodos por defecto (localhost).")
	}
	nodeAddrs := strings.Split(nodesEnv, ",")

	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}

	mongoAddr := os.Getenv("MONGO_ADDR")
	if mongoAddr == "" {
		mongoAddr = "mongodb://localhost:27017"
	}

	// 2. Cargar Dataset
	fmt.Println("[API] Cargando dataset...")
	ds, err := ml.LoadDataset(datasetPath)
	if err != nil {
		log.Fatal("Error cargando dataset: ", err)
	}

	// 3. Indexar
	fmt.Println("[API] Indexando películas para paginación...")
	sortedMovies = make([]ml.MovieInfo, 0, len(ds.MoviesMeta))
	for _, m := range ds.MoviesMeta {
		sortedMovies = append(sortedMovies, m)
	}
	sort.Slice(sortedMovies, func(i, j int) bool {
		return sortedMovies[i].ID < sortedMovies[j].ID
	})

	// 4. Inicializar
	coord = cluster.NewCoordinator(nodeAddrs, ds)

	// WebSocket Setup (Vital para el chat/WS)
	SetWebSocketCoordinator(coord)

	redisStore, err = storage.NewRedisStore(redisAddr)
	if err != nil {
		fmt.Println("[WARN] No se pudo conectar a Redis:", err)
	}

	mongoStore, err = storage.NewMongoStore(mongoAddr, "pcd")
	if err != nil {
		fmt.Println("[WARN] No se pudo conectar a Mongo:", err)
	}

	// 5. Rutas
	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/recommend/", recommendHandler)
	mux.HandleFunc("/ws", WebSocketHandler) // ¡No olvides registrar el WS!

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

		total := len(sortedMovies)
		if offset >= total {
			json.NewEncoder(w).Encode(map[string]interface{}{"data": []ml.MovieInfo{}, "total": total})
			return
		}

		end := offset + limit
		if end > total {
			end = total
		}

		data := sortedMovies[offset:end]
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data":   data,
			"total":  total,
			"limit":  limit,
			"offset": offset,
		})
	})

	// NUEVO ENDPOINT: Obtener película por ID (/movies/123)
	// Nota la barra al final "/movies/" para que capture todo lo que sigue
	mux.HandleFunc("/movies/", func(w http.ResponseWriter, r *http.Request) {
		// CORS Headers (ya cubierto por el middleware, pero no estorba)

		// Validar que sea GET
		if r.Method != "GET" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extraer el ID de la URL. Ej: "/movies/1" -> "1"
		idStr := strings.TrimPrefix(r.URL.Path, "/movies/")

		// Si el string está vacío (es decir, llamaron a /movies/ sin ID), ignoramos
		// para que no choque con el endpoint de paginación si hubiera conflicto,
		// aunque Go suele manejar esto bien si el otro no tiene slash final.
		if idStr == "" {
			http.Error(w, "ID requerido", http.StatusBadRequest)
			return
		}

		id, err := strconv.Atoi(idStr)
		if err != nil {
			http.Error(w, "ID inválido", http.StatusBadRequest)
			return
		}

		// Buscar en el mapa en memoria
		if movie, ok := coord.Dataset.MoviesMeta[id]; ok {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(movie)
		} else {
			http.Error(w, "Película no encontrada", http.StatusNotFound)
		}
	})

	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		query := strings.ToLower(r.URL.Query().Get("q"))
		if query == "" {
			return
		}

		var results []ml.MovieInfo
		count := 0
		for _, m := range sortedMovies {
			if strings.Contains(strings.ToLower(m.Title), query) {
				results = append(results, m)
				count++
				if count >= 20 {
					break
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(results)
	})

	fmt.Println("[API] Servidor listo en port 8080")

	// AQUI ESTÁ LA SOLUCIÓN: Envolvemos 'mux' con el middleware
	log.Fatal(http.ListenAndServe(":8080", corsMiddleware(mux)))
}
