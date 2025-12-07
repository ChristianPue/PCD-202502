package cluster

import (
	"encoding/json"
	"fmt"
	"net"
	"time" // IMPORTANTE: Agregado para los timeouts

	"TF/internal/ml"
)

// -------------------------------
// Coordinador
// -------------------------------

type Coordinator struct {
	NodeAddresses []string    // lista de nodos ML ("node1:9001"...)
	Dataset       *ml.Dataset // dataset cargado
}

// Crear coordinador con nodos disponibles
func NewCoordinator(nodes []string, ds *ml.Dataset) *Coordinator {
	return &Coordinator{
		NodeAddresses: nodes,
		Dataset:       ds,
	}
}

// -------------------------------
// Lógica principal distribuida
// -------------------------------

// Divide lista en N partes iguales
func splitIntoChunks(items []int, n int) [][]int {
	if n <= 0 {
		return [][]int{items}
	}

	res := make([][]int, n)
	for i := 0; i < n; i++ {
		res[i] = make([]int, 0)
	}

	// Round-robin distribution
	for i, item := range items {
		idx := i % n
		res[idx] = append(res[idx], item)
	}

	return res
}

// Ejecuta item-based de forma distribuida
func (c *Coordinator) RecommendDistributed(userID int, topK int, metric int, neighborK int) map[int]float64 {

	// 1) Construir lista de items candidatos (todas las películas que user NO ha visto)
	userRatings := c.Dataset.UserRatings[userID]
	itemIndex := ml.BuildItemIndex(c.Dataset)

	candidates := make([]int, 0)
	for item := range itemIndex {
		if _, seen := userRatings[item]; !seen {
			candidates = append(candidates, item)
		}
	}

	// 2) Dividir candidatos según cantidad de nodos
	chunks := splitIntoChunks(candidates, len(c.NodeAddresses))

	// 3) Crear canal para respuestas
	resultCh := make(chan map[int]float64, len(c.NodeAddresses))

	// 4) Enviar tareas a cada nodo
	for i, nodeAddr := range c.NodeAddresses {
		// NOTA: Usamos TaskRequest directamente (sin el prefijo cluster.)
		task := TaskRequest{
			UserID:    userID,
			ItemIDs:   chunks[i],
			Metric:    metric,
			NeighborK: neighborK,
			TopK:      topK,
		}

		go func(addr string, t TaskRequest) {
			res := sendTask(addr, t)
			resultCh <- res
		}(nodeAddr, task)
	}

	// 5) Combinar resultados parciales
	scores := make(map[int]float64)

	for i := 0; i < len(c.NodeAddresses); i++ {
		part := <-resultCh
		for k, v := range part {
			scores[k] = v
		}
	}

	return scores
}

// Enviar tarea a un nodo ML vía TCP
// NOTA: Aquí también quitamos el prefijo cluster.
func sendTask(address string, task TaskRequest) map[int]float64 {
	conn, err := net.DialTimeout("tcp", address, 2*time.Second)
	if err != nil {
		fmt.Printf("[COORD] Error conectando a nodo %s: %v\n", address, err)
		return map[int]float64{}
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(1800 * time.Second))

	// enviar JSON
	encoder := json.NewEncoder(conn)
	if err := encoder.Encode(&task); err != nil {
		fmt.Printf("[COORD] Error enviando a %s: %v\n", address, err)
		return map[int]float64{}
	}

	// leer respuesta
	// NOTA: Usamos TaskResponse directamente
	var res TaskResponse
	decoder := json.NewDecoder(conn)
	if err := decoder.Decode(&res); err != nil {
		fmt.Printf("[COORD] Error leyendo de %s: %v\n", address, err)
		return map[int]float64{}
	}

	if res.Error != "" {
		fmt.Printf("[COORD] Nodo %s reportó error: %s\n", address, res.Error)
		return map[int]float64{}
	}

	return res.Scores
}

// ------------------------------------------------------------
// Método público para ser usado desde la API
// ------------------------------------------------------------
func (c *Coordinator) ComputeRecommendations(userID int, topK int, metric int, neighborK int) []ml.ItemScore {

	// 1. Obtener scores distribuidos (map[int]float64)
	scoresMap := c.RecommendDistributed(userID, topK, metric, neighborK)

	// 2. Convertir a lista usando helper de ml
	return ml.RecommendTopK(scoresMap, topK)
}
