package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"

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
		json.NewEncoder(w).Encode(res)
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

	// devolver JSON
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}

// ---------------------------
//
//	MAIN
//
// ---------------------------
func main() {

	// cargar dataset una sola vez
	ds, err := ml.LoadDataset("./dataset/10M/ratings.csv")
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

	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/recommend/", recommendHandler)
	http.HandleFunc("/ws", WebSocketHandler) // registrar ruta WS

	log.Fatal(http.ListenAndServe(":8080", nil))
}
