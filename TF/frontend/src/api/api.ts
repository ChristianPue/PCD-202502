// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8080/ws';

// Types
export interface Movie {
  id: number;          // Unificado
  title: string;
  genres: string[];    // Convertiremos el string "Action|Adventure" a array para el frontend
  score?: number;      // Opcional, solo viene en recomendaciones
}

// Lo que responde el Backend (Raw)
interface BackendDTO {
  id?: number;         // Viene en /movies y /search
  movieId?: number;    // Viene en /recommend
  title: string;
  genres: string;      // Viene como "Action|Adventure"
  score?: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
}

// Helper para limpiar los datos crudos del backend
const normalizeMovie = (raw: BackendDTO): Movie => ({
  id: raw.id || raw.movieId || 0,
  title: raw.title,
  genres: raw.genres ? raw.genres.split('|') : [], // "Action|Comedy" -> ["Action", "Comedy"]
  score: raw.score,
});

// Fetch wrapper genérico
async function fetchAPI<T>(endpoint: string, params: Record<string, string | number> = {}): Promise<T> {
  const url = new URL(`${API_BASE_URL}${endpoint}`);

  Object.keys(params).forEach(key => {
    if (params[key] !== undefined) {
      url.searchParams.append(key, String(params[key]));
    }
  });

  try {
    const res = await fetch(url.toString(), {
      headers: { 'Content-Type': 'application/json' }
    });

    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return await res.json();
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error);
    throw error;
  }
}

// API Methods
export const movieAPI = {
  getHealth: () => fetch(`${API_BASE_URL}/health`).then(res => res.text()),

  getWsUrl: () => WS_URL,

  // 1. Catálogo (Paginado)
  getMovies: async (limit = 50, offset = 0) => {
    const res = await fetchAPI<PaginatedResponse<BackendDTO>>('/movies', { limit, offset });
    return {
      ...res,
      data: res.data.map(normalizeMovie)
    };
  },

  // 2. Buscar por ID (El endpoint SÍ existe en tu backend actual)
  getMovieById: async (id: string | number) => {
    try {
      const res = await fetchAPI<BackendDTO>(`/movies/${id}`);
      return normalizeMovie(res);
    } catch (e) {
      return null;
    }
  },

  // 3. Buscador
  searchMovies: async (query: string) => {
    if (!query) return [];
    const res = await fetchAPI<BackendDTO[]>('/search', { q: query });
    return (res || []).map(normalizeMovie);
  },

  // 4. Recomendaciones HTTP
  getRecommendations: async (userId: string | number) => {
    const res = await fetchAPI<BackendDTO[]>(`/recommend/${userId}`);
    return (res || []).map(normalizeMovie);
  },
};