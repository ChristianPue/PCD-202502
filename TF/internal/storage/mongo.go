package storage

import (
	"context"
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
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

	// Crear cliente
	client, err := mongo.Connect(context.Background(), options.Client().ApplyURI(uri))
	if err != nil {
		return nil, fmt.Errorf("error conectando a MongoDB: %v", err)
	}

	// Seleccionar colección
	collection := client.Database(dbName).Collection("recommendations")

	fmt.Println("[Mongo] Conectado OK, colección lista:", collection.Name())

	return &MongoStore{
		client:     client,
		collection: collection,
	}, nil
}

// -------------------------------------------
// Guardar un documento de recomendación
// -------------------------------------------
func (m *MongoStore) SaveRecommendation(userID int, movies []int) error {

	doc := RecommendationDocument{
		UserID:    userID,
		Movies:    movies,
		Timestamp: time.Now(),
	}

	_, err := m.collection.InsertOne(context.Background(), doc)
	if err != nil {
		return fmt.Errorf("error insertando en MongoDB: %v", err)
	}

	fmt.Println("[Mongo] Recomendación guardada para user:", userID)
	return nil
}
