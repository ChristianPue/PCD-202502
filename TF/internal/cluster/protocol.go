package cluster

// Mensaje enviado entre coordinador y nodos
type TaskRequest struct {
	ChunkID   int
	ItemIDs   []int
	Metric    string
	NeighborK int
}

type TaskResponse struct {
	ChunkID int
	Scores  map[int]float64
}
