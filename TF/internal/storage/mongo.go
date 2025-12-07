package storage

import (
	"context"
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

// Struct del almacenamiento Mongo
type MongoStore struct {
	client     *mongo.Client
	collection *mongo.Collection
}

// Documento que guardaremos
type RecommendationDocument struct {
	UserID    int       `bson:"userId"`
	Movies    []int     `bson:"recommendedMovies"`
	Timestamp time.Time `bson:"timestamp"`
}

// -------------------------------------------
// Inicializar conexión a MongoDB
// -------------------------------------------
func NewMongoStore(uri string, dbName string) (*MongoStore, error) {
	// 1. Timeout de conexión inicial (10s)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientOptions := options.Client().ApplyURI(uri)
	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		return nil, fmt.Errorf("error creando cliente Mongo: %v", err)
	}

	// 2. Verificar conexión real (Ping)
	if err := client.Ping(ctx, readpref.Primary()); err != nil {
		return nil, fmt.Errorf("no se pudo hacer ping a Mongo: %v", err)
	}

	collection := client.Database(dbName).Collection("recommendations")
	fmt.Println("[Mongo] Conectado y listo en colección:", collection.Name())

	return &MongoStore{
		client:     client,
		collection: collection,
	}, nil
}

// -------------------------------------------
// Guardar un documento de recomendación
// -------------------------------------------
func (m *MongoStore) SaveRecommendation(userID int, movies []int) error {
	// 3. Timeout por operación (5s). Si Mongo está lento, fallamos rápido.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	doc := RecommendationDocument{
		UserID:    userID,
		Movies:    movies,
		Timestamp: time.Now(),
	}

	_, err := m.collection.InsertOne(ctx, doc)
	if err != nil {
		return fmt.Errorf("error insertando en MongoDB: %v", err)
	}

	// fmt.Println("[Mongo] Guardado para user:", userID) // Comentar para no ensuciar logs en prod
	return nil
}
