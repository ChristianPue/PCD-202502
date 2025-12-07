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
}

// Crear nueva conexión Redis
func NewRedisStore(addr string) (*RedisStore, error) {
	opts := &redis.Options{
		Addr: addr,
		DB:   0,
	}

	client := redis.NewClient(opts)

	// Contexto temporal solo para el Ping inicial
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := client.Ping(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("error conectando a Redis: %v", err)
	}

	fmt.Println("[Redis] Conectado OK")

	return &RedisStore{
		client: client,
	}, nil
}

// Guardar lista de recomendaciones en redis por userId
func (r *RedisStore) SaveRecommendations(userID int, movies []int) error {
	key := fmt.Sprintf("recommend:%d", userID)

	data, _ := json.Marshal(movies)

	// Timeout corto para operaciones de caché (2s)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := r.client.Set(ctx, key, data, 30*time.Minute).Err()
	if err != nil {
		return fmt.Errorf("error guardando en Redis: %v", err)
	}

	return nil
}

// Leer recomendaciones si existen
func (r *RedisStore) GetRecommendations(userID int) ([]int, error) {
	key := fmt.Sprintf("recommend:%d", userID)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	val, err := r.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return nil, nil // No existe, no es error
	}
	if err != nil {
		return nil, err // Error real de conexión
	}

	var movies []int
	if err := json.Unmarshal([]byte(val), &movies); err != nil {
		return nil, fmt.Errorf("error deserializando JSON de Redis: %v", err)
	}

	return movies, nil
}
