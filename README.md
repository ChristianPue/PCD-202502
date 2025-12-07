# TF – Sistema Distribuido de Recomendación de Películas (MovieLens + Go + TCP + MongoDB + Redis + Docker)

Este proyecto implementa un sistema distribuido de Filtrado Colaborativo usando Go.
La arquitectura contiene:
- API HTTP
- Nodos ML responsables del cálculo paralelo
- Coordinador distribuido
- MongoDB para persistencia
- Redis para caché
- Dataset MovieLens (10M, 20M, 25M)
- Docker / docker-compose para despliegue

## 🚀 1. Cómo ejecutar el sistema
### ✔️ 1.1 Ejecutar nodos ML (nodos trabajadores)
En tres terminales distintas:
```bash
go run ./cmd/node 10 9001
```
```bash
go run ./cmd/node 10 9002
```
```bash
go run ./cmd/node 10 9003
```

### ✔️ 1.2 Levantar el API
```bash
cd TF
go run ./cmd/api
```
El API estará disponible en `http://localhost:8080`

### ✔️ 1.3 Frontend
```bash
cd TF/frontend
npm install
npm run dev
```
Abrir `http://localhost:5173` (o el puerto que indique Vite)

Donde:
- `10` es el tamaño del dataset (10M, 20M, 25M)
- `9001`, `9002` y `9003` son los puertos de los nodos.

### ✔️ 1.4 Ejecutar API para Docker
Comprobar si Redis y MongoDB están corriendo:
```bash
docker ps
```

Si no están corriendo, iniciarlos:
```bash
docker-compose up -d
```
Comandos para levantar Redis y Mongo
```bash
docker-compose up -d redis mongo
```

En otra terminal:
```bash
go run ./cmd/api
```
Esto iniciará:
- Servidor HTTP en http://localhost:8080
- Coordinador que se conecta a los nodos 9001 y 9002

### ✔️ 1.3 Probar el endpoint de recomendaciones
Abrir en el navegador:
```bash
http://localhost:8080/recommend/1
```
El sistema:
1. Primero consulta Redis (cache)
2. Si NO existe, llama al cluster distribuido
3. Recibe resultados parciales de los nodos ML
4. Combina y devuelve un JSON de recomendaciones
5. Guarda el resultado en MongoDB y Redis

### 📦 2. Tecnologías utilizadas
| Componente              | Tecnología                     |
|-------------------------|--------------------------------|
| Lenguaje                | Go 1.25.1                        |
| Cálculo distribuido     | TCP sockets                    |
| Dataset                 | MovieLens 10M / 20M / 25M      |
| Caché                   | Redis                          |
| Persistencia            | MongoDB                        |
| Comunicación tiempo real| WebSockets                     |
| Contenedores            | Docker & Docker Compose        |

### 📚 3. Arquitectura del Proyecto
```
├── TF/
│   ├── cmd/
│   │   └── api/                # Código del API REST
│   ├── main.go                  # Entry point del API
│   ├── ws.go                    # WebSocket support
│   ├── node/
│   │   ├── main.go              # Entry point de los nodos workers
│   │   ├── tcp_server.go
│   │   ├── worker.go
│   │   └── dataset/             # Datasets preprocesados por nodo
│   │       ├── 10M/
│   │       │   ├── movies.csv
│   │       │   ├── ratings.csv
│   │       │   └── tags.csv
│   │       ├── 20M/
│   │       └── 25M/
│   ├── frontend/                # Frontend en React/Vite
│   ├── internal/
│   │   └── cluster/             # Lógica de coordinación y comunicación
│   ├── coordinator.go
│   ├── protocol.go
│   ├── ml/
│   │   ├── dataset.go
│   │   ├── recommender.go       # Algoritmos de recomendación
│   │   └── similarity.go        # Cálculo de similitud (coseno, etc.)
│   ├── storage/
│   │   ├── mongo.go             # Persistencia de resultados/cache
│   │   └── redis.go
│   ├── api.exe                  # Binario compilado (Windows)
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.node
│   └── go.mod
```

### ⚙️ 4. ¿Qué hace cada módulo?

#### cmd/api/
- **main.go**: servidor HTTP, endpoints `/recommend/:userId` y `/health`
- **ws.go**: WebSocket para monitoreo en tiempo real

#### cmd/node/
Cada nodo ejecuta:
- un servidor TCP  
- recibe tareas del coordinador  
- procesa *chunks* de similitud  
- envía resultados parciales  

#### internal/ml/
Lógica del sistema de recomendación:
- carga del dataset  
- filtrado colaborativo (*item-based* y *user-based*)  
- métricas: Cosine, Pearson, Jaccard  
- paralelización con goroutines  

#### internal/cluster/
- coordinador distribuido  
- balanceo por *chunks*  
- envío TCP hacia los nodos  
- combinación de resultados  

#### internal/storage/
- **mongo.go**: guarda recomendaciones  
- **redis.go**: caché de resultados  

### 🐳 5. Ejecución con Docker

#### Construir imágenes
```bash
docker compose build
```
Las imagenes son los siguientes archivos:
```
api.tar
node1.tar
node2.tar
node3.tar
```

#### Ejecutar
```bash
docker compose up
```

#### Abrir el navegador
```bash
http://localhost:8080/recommend/1
```

### 🧪 6. Validaciones
#### Redis
```bash
redis-cli
GET recommend:1
```

#### MongoDB
```bash
db.recommendations.find()
```