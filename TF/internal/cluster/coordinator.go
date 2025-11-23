package cluster

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"sort"

	"TF/internal/ml"
)

// -------------------------------
// Estructuras compartidas
// -------------------------------

// Tarea enviada a los nodos ML
type Task struct {
	UserID    int   `json:"user_id"`
	Items     []int `json:"items"`
	Metric    int   `json:"metric"`
	NeighborK int   `json:"neighbor_k"`
	TopK      int   `json:"top_k"`
}

// Respuesta recibida desde cada nodo ML
type TaskResult struct {
	Partial map[int]float64 `json:"partial"`
}

// -------------------------------
// Coordinador
// -------------------------------

type Coordinator struct {
	NodeAddresses []string    // lista de nodos ML ("localhost:9001"...)
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
	res := make([][]int, 0)
	chunkSize := len(items) / n
	if chunkSize == 0 {
		chunkSize = 1
	}

	for i := 0; i < len(items); i += chunkSize {
		end := i + chunkSize
		if end > len(items) {
			end = len(items)
		}
		res = append(res, items[i:end])
	}
	return res
}

// Ejecuta item-based de forma distribuida
func (c *Coordinator) RecommendDistributed(userID int, topK int, metric ml.SimMetric, neighborK int) map[int]float64 {

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
		task := Task{
			UserID:    userID,
			Items:     chunks[i],
			Metric:    int(metric),
			NeighborK: neighborK,
			TopK:      topK,
		}

		go func(addr string, t Task) {
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
func sendTask(address string, task Task) map[int]float64 {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		fmt.Println("[COORD] Error conectando a nodo:", address, err)
		return map[int]float64{}
	}
	defer conn.Close()

	// enviar JSON
	data, _ := json.Marshal(task)
	data = append(data, '\n')
	conn.Write(data)

	// leer respuesta
	reader := bufio.NewReader(conn)
	respBytes, err := reader.ReadBytes('\n')
	if err != nil {
		fmt.Println("[COORD] Error leyendo respuesta:", err)
		return map[int]float64{}
	}

	var res TaskResult
	json.Unmarshal(respBytes, &res)
	return res.Partial
}

// ------------------------------------------------------------
// Método público para ser usado desde la API
// ------------------------------------------------------------
func (c *Coordinator) ComputeRecommendations(userID int, topK int, metric ml.SimMetric, neighborK int) []ml.ItemScore {

	// 1. Obtener scores distribuidos (map[int]float64)
	scoresMap := c.RecommendDistributed(userID, topK, metric, neighborK)

	// 2. Convertir a lista
	list := make([]ml.ItemScore, 0, len(scoresMap))
	for movieID, score := range scoresMap {
		list = append(list, ml.ItemScore{
			MovieID: movieID,
			Score:   score,
		})
	}

	// 3. Ordenar por score descendente
	sort.Slice(list, func(i, j int) bool {
		return list[i].Score > list[j].Score
	})

	// 4. Obtener top K
	if len(list) > topK {
		list = list[:topK]
	}

	return list
}
