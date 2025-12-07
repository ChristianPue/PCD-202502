package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"TF/internal/cluster"
	"TF/internal/ml"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Desarrollo: permite conexiones desde cualquier origen (React/Vue/etc)
	},
}

// Variable global para acceder al coordinador
var wsCoordinator *cluster.Coordinator

func SetWebSocketCoordinator(c *cluster.Coordinator) {
	wsCoordinator = c
}

// Struct para respuesta enriquecida (JSON bonito para el frontend)
type WSRecommendationResponse struct {
	MovieID int     `json:"movieId"`
	Title   string  `json:"title"`
	Genres  string  `json:"genres"`
	Score   float64 `json:"score"`
}

func WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	// 1. Upgrade HTTP -> WebSocket
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Println("[WS] Error upgrade:", err)
		return
	}
	defer conn.Close()

	fmt.Println("[WS] Cliente conectado")

	for {
		// 2. Leer mensaje
		_, msg, err := conn.ReadMessage()
		if err != nil {
			fmt.Println("[WS] Cliente desconectado:", err)
			break
		}

		text := string(msg)
		// fmt.Println("[WS] Recibido:", text) // Descomentar para debug

		// 3. Ping-Pong (para mantener viva la conexión)
		if text == "ping" {
			conn.WriteMessage(websocket.TextMessage, []byte("pong"))
			continue
		}

		// 4. Comando "recommend <userID>"
		if strings.HasPrefix(text, "recommend") {
			parts := strings.Split(text, " ")
			if len(parts) != 2 {
				conn.WriteMessage(websocket.TextMessage, []byte(`{"error": "uso: recommend <userID>"}`))
				continue
			}

			userID, err := strconv.Atoi(parts[1])
			if err != nil {
				conn.WriteMessage(websocket.TextMessage, []byte(`{"error": "userID debe ser número"}`))
				continue
			}

			if wsCoordinator == nil {
				conn.WriteMessage(websocket.TextMessage, []byte(`{"error": "servidor no listo"}`))
				continue
			}

			// 5. Calcular recomendaciones (Distribuido)
			// Nota: Esto bloquea el loop de lectura. En producción idealmente iría en una goroutine,
			// pero gorilla/websocket no permite escrituras concurrentes sin mutex. Para este TP está bien así.
			rawRecs := wsCoordinator.ComputeRecommendations(
				userID,
				10,                // Top K
				int(ml.CosineSim), // Métrica (asegúrate que ml.CosineSim existe o usa int(1))
				20,                // Vecinos
			)

			// 6. Enriquecer datos (ID -> Título)
			response := make([]WSRecommendationResponse, 0, len(rawRecs))
			for _, item := range rawRecs {
				meta, ok := wsCoordinator.Dataset.MoviesMeta[item.MovieID]
				title := "Desconocido"
				genres := ""
				if ok {
					title = meta.Title
					genres = meta.Genres
				}

				response = append(response, WSRecommendationResponse{
					MovieID: item.MovieID,
					Title:   title,
					Genres:  genres,
					Score:   item.Score,
				})
			}

			// 7. Enviar JSON
			jsonData, _ := json.Marshal(response)
			if err := conn.WriteMessage(websocket.TextMessage, jsonData); err != nil {
				fmt.Println("[WS] Error escribiendo:", err)
				break
			}
			continue
		}

		conn.WriteMessage(websocket.TextMessage, []byte(`{"error": "comando desconocido"}`))
	}
}
