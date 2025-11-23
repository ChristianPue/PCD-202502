package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// Wrapper del cliente Redis
type RedisStore struct {
	client *redis.Client
	ctx    context.Context
}

// Crear nueva conexión Redis
func NewRedisStore(addr string) (*RedisStore, error) {

	opts := &redis.Options{
		Addr: addr, // "localhost:6379"
		DB:   0,
	}

	client := redis.NewClient(opts)

	ctx := context.Background()

	// comprobar conexión
	_, err := client.Ping(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("error conectando a Redis: %v", err)
	}

	fmt.Println("[Redis] Conectado OK")

	return &RedisStore{
		client: client,
		ctx:    ctx,
	}, nil
}

// Guardar lista de recomendaciones en redis por userId
func (r *RedisStore) SaveRecommendations(userID int, movies []int) error {

	key := fmt.Sprintf("recommend:%d", userID)

	// convertir a JSON
	data, _ := json.Marshal(movies)

	// TTL opcional (30 minutos por defecto)
	err := r.client.Set(r.ctx, key, data, 30*time.Minute).Err()
	if err != nil {
		return fmt.Errorf("error guardando en Redis: %v", err)
	}

	return nil
}

// Leer recomendaciones si existen
func (r *RedisStore) GetRecommendations(userID int) ([]int, error) {

	key := fmt.Sprintf("recommend:%d", userID)

	val, err := r.client.Get(r.ctx, key).Result()
	if err == redis.Nil {
		// key no existe
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	// decodificar JSON
	var movies []int
	json.Unmarshal([]byte(val), &movies)

	return movies, nil
}
