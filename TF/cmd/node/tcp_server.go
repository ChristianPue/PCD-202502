package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"

	"TF/internal/ml"
)

// Tarea enviada por el coordinador
type Task struct {
	UserID    int   `json:"user_id"`
	Items     []int `json:"items"`
	Metric    int   `json:"metric"`
	NeighborK int   `json:"neighbor_k"`
	TopK      int   `json:"top_k"`
}

// Respuesta generada por el nodo
type TaskResult struct {
	Partial map[int]float64 `json:"partial"`
}

// Arranca el nodo ML escuchando en un puerto
func StartNodeServer(port string, ds *ml.Dataset) error {
	addr := ":" + port
	fmt.Println("[NODE] Escuchando en", addr)

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	for {
		conn, err := ln.Accept()
		if err != nil {
			fmt.Println("Error aceptando conexión:", err)
			continue
		}
		go handleConnection(conn, ds)
	}
}

// Procesa una solicitud entrante
func handleConnection(conn net.Conn, ds *ml.Dataset) {
	defer conn.Close()

	reader := bufio.NewReader(conn)
	data, err := reader.ReadBytes('\n')
	if err != nil {
		fmt.Println("Error leyendo:", err)
		return
	}

	// Decodificar tarea
	var task Task
	err = json.Unmarshal(data, &task)
	if err != nil {
		fmt.Println("Error JSON:", err)
		return
	}

	// Construir itemIndex una vez por nodo (simple)
	itemIndex := ml.BuildItemIndex(ds)
	userRatings := ds.UserRatings[task.UserID]

	// Procesar items del chunk
	partial := make(map[int]float64)
	for _, item := range task.Items {
		score := ml.RecommendScoreSingle(item, userRatings, itemIndex, ml.SimMetric(task.Metric), task.NeighborK)
		partial[item] = score
	}

	// Enviar respuesta
	res := TaskResult{Partial: partial}
	out, _ := json.Marshal(res)
	out = append(out, '\n')
	conn.Write(out)

	fmt.Println("[NODE] Tarea procesada con", len(task.Items), "items")
}
