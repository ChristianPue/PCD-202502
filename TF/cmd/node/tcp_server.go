package main

import (
	"encoding/json"
	"fmt"
	"net"

	"TF/internal/cluster" // Importamos el protocolo compartido
	"TF/internal/ml"
)

// Arranca el nodo ML escuchando en un puerto
func StartNodeServer(port string, ds *ml.Dataset) error {
	addr := ":" + port
	fmt.Println("[NODE] Escuchando en", addr)

	// OPTIMIZACIÓN: Pre-calcular índices una sola vez al inicio
	// Si lo hiciéramos por cada petición, el sistema sería lentísimo.
	fmt.Println("[NODE] Construyendo índice de items (puede tardar)...")
	itemIndex := ml.BuildItemIndex(ds)
	fmt.Println("[NODE] Índice construido. Listo para recibir tareas.")

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
		// Pasamos el índice ya calculado a la goroutine
		go handleConnection(conn, ds, itemIndex)
	}
}

// Procesa una solicitud entrante
func handleConnection(conn net.Conn, ds *ml.Dataset, itemIndex map[int]map[int]float64) {
	defer conn.Close()

	// 1. Decodificar la solicitud usando el protocolo compartido
	var req cluster.TaskRequest
	decoder := json.NewDecoder(conn)
	if err := decoder.Decode(&req); err != nil {
		fmt.Println("[NODE] Error decodificando JSON:", err)
		return
	}

	// 2. Validar o preparar datos
	// Obtenemos los ratings del usuario para compararlos con los items candidatos
	userRatings := ds.UserRatings[req.UserID]

	// Si el usuario no existe en este dataset (y no se enviaron ratings en la request),
	// userRatings será nil. La función de recomendación debe manejar esto o devolvemos vacío.

	scores := make(map[int]float64)

	// 3. Procesar items del chunk (Heavy Lifting)
	// Iteramos sobre los IDs que el coordinador nos mandó
	for _, itemID := range req.ItemIDs {
		// Llamada directa a ML. Convertimos el int del protocolo a SimMetric
		score := ml.RecommendScoreSingle(
			itemID,
			userRatings,
			itemIndex,
			ml.SimMetric(req.Metric),
			req.NeighborK,
		)

		// Solo agregamos si el score es relevante (opcional, ahorra ancho de banda)
		if score > 0 {
			scores[itemID] = score
		}
	}

	// 4. Enviar respuesta usando el protocolo compartido
	res := cluster.TaskResponse{
		Scores: scores,
	}

	encoder := json.NewEncoder(conn)
	if err := encoder.Encode(&res); err != nil {
		fmt.Println("[NODE] Error enviando respuesta:", err)
	}

	fmt.Printf("[NODE] Procesados %d items para User %d\n", len(req.ItemIDs), req.UserID)
}
