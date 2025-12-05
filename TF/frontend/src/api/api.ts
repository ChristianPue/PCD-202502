// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8080/ws';

// Types
export interface Movie {
  id: number;          // Matches MovieInfo.ID
  movieId?: number;    // Matches /recommend response
  title: string;
  genres: string;      // Backend sends pipe-separated string
  score?: number;      // For recommendations
  poster?: string;     // Optional, if we add it later
  year?: number;
  rating?: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
}

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number>;
}

// Fetch wrapper with error handling
async function fetchAPI<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;

  // Build URL with query parameters
  let url = `${API_BASE_URL}${endpoint}`;
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      url += `?${queryString}`;
    }
  }

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      headers: {
        'Content-Type': 'application/json',
        ...fetchOptions.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// HTTP Methods
const api = {
  get: <T>(endpoint: string, params?: Record<string, string | number>) =>
    fetchAPI<T>(endpoint, { method: 'GET', params }),
};

// API Functions
export const movieAPI = {
  // 0. GET /health
  getHealth: () =>
    fetch(`${API_BASE_URL}/health`).then(res => res.text()),

  // Helper for WebSocket URL
  getWsUrl: () => WS_URL,

  // 1. GET /movies?limit=&offset=
  getMovies: (limit: number = 50, offset: number = 0) =>
    api.get<PaginatedResponse<Movie>>('/movies', { limit, offset }),

  // 2. GET /movies/:id
  getMovieById: (id: string | number) =>
    api.get<Movie>(`/movies/${id}`),

  // 3. GET /search?q=
  searchMovies: (query: string) =>
    api.get<Movie[]>('/search', { q: query }),

  // 4. GET /recommend/:userId
  getRecommendations: (userId: string | number) =>
    api.get<Movie[]>(`/recommend/${userId}`),
};
