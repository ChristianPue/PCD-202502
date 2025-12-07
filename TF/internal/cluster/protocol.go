package cluster

// Definimos constantes para evitar errores de "dedo" (typos)
const (
	MetricCosine  = 1
	MetricPearson = 2
)

// TaskRequest: Lo que el Coordinador envía al Worker
type TaskRequest struct {
	UserID    int   `json:"user_id"`    // CRÍTICO: Necesario para que el worker sepa con qué comparar
	ItemIDs   []int `json:"item_ids"`   // El chunk de películas a evaluar
	Metric    int   `json:"metric"`     // int es más ligero que string
	NeighborK int   `json:"neighbor_k"` // Vecinos cercanos (si aplica)
	TopK      int   `json:"top_k"`      // Opcional, por si el worker filtra antes de devolver
}

// TaskResponse: Lo que el Worker responde
type TaskResponse struct {
	// Mapa de PelículaID -> Score de similitud
	Scores map[int]float64 `json:"scores"`
	Error  string          `json:"error,omitempty"` // Para reportar fallos remotos
}
