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

// Necesario para upgrade de HTTP → WS
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // permitir todas las conexiones (para prueba)
	},
}

// Coordinador global (lo inyectarás desde main.go)
var wsCoordinator *cluster.Coordinator

// Registrar coordinador para WS
func SetWebSocketCoordinator(c *cluster.Coordinator) {
	wsCoordinator = c
}

// Handler WebSocket
func WebSocketHandler(w http.ResponseWriter, r *http.Request) {

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Println("[WS] Error upgrade:", err)
		return
	}
	defer conn.Close()

	fmt.Println("[WS] Cliente conectado")

	for {
		// Leer mensaje desde el cliente
		_, msg, err := conn.ReadMessage()
		if err != nil {
			fmt.Println("[WS] Error lectura:", err)
			return
		}

		text := string(msg)
		fmt.Println("[WS] Recibido:", text)

		// -------------------------
		// Respuestas básicas
		// -------------------------
		if text == "ping" {
			conn.WriteMessage(websocket.TextMessage, []byte("pong"))
			continue
		}

		// -------------------------
		// Petición: "recommend 1"
		// -------------------------
		if strings.HasPrefix(text, "recommend") {

			parts := strings.Split(text, " ")
			if len(parts) != 2 {
				conn.WriteMessage(websocket.TextMessage,
					[]byte("uso: recommend <userID>"))
				continue
			}

			userID, err := strconv.Atoi(parts[1])
			if err != nil {
				conn.WriteMessage(websocket.TextMessage,
					[]byte("userID inválido"))
				continue
			}

			if wsCoordinator == nil {
				conn.WriteMessage(websocket.TextMessage,
					[]byte("coordinador no inicializado"))
				continue
			}

			recs := wsCoordinator.ComputeRecommendations(
				userID,
				10,           // topK
				ml.CosineSim, // métrica
				30,           // vecinos
			)

			jsonData, _ := json.Marshal(recs)
			conn.WriteMessage(websocket.TextMessage, jsonData)
			continue
		}

		// mensaje desconocido
		conn.WriteMessage(websocket.TextMessage, []byte("comando no reconocido"))
	}
}
