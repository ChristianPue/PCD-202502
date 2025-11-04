package benchmark

import (
	"time"
)

// MeasureTime mide el tiempo de ejecución de una función en milisegundos
func MeasureTime(fn func()) float64 {
	start := time.Now()
	fn()
	elapsed := time.Since(start)
	return float64(elapsed.Microseconds()) / 1000.0 // Convertir a milisegundos
}

// CalculateSpeedup calcula el speedup entre tiempo secuencial y concurrente
func CalculateSpeedup(sequentialTime, concurrentTime float64) float64 {
	if concurrentTime == 0 {
		return 0
	}
	return sequentialTime / concurrentTime
}

// CalculateEfficiency calcula la eficiencia del paralelismo
// Eficiencia = Speedup / NumWorkers
func CalculateEfficiency(speedup float64, numWorkers int) float64 {
	if numWorkers == 0 {
		return 0
	}
	return speedup / float64(numWorkers)
}