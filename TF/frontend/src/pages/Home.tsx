import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { movieAPI, type Movie } from '../api/api';
import { useAuth } from '../context/AuthContext';
import MovieCard from '../components/MovieCard';
import Loader from '../components/Loader';

const Home: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [movies, setMovies] = useState<Movie[]>([]);
  const [recommendations, setRecommendations] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [offset, setOffset] = useState(0);
  const limit = 50;

  // Ref para evitar peticiones duplicadas
  const fetchedRecsForUser = useRef<number | null>(null);

  // Helper para extraer el año
  const extractYear = (title: string): number | undefined => {
    const match = title.match(/\((\d{4})\)/);
    return match ? parseInt(match[1]) : undefined;
  };

  useEffect(() => {
    const fetchContent = async () => {
      try {
        setLoading(true);
        setError(null);

        // 1. Cargar catálogo (siempre se carga)
        const popularData = await movieAPI.getMovies(limit, offset);
        setMovies(popularData.data);

        // 2. Cargar recomendaciones (SOLO si hay usuario y no se ha pedido antes)
        if (user) {
          // Verificamos si ya pedimos datos para este ID
          if (fetchedRecsForUser.current !== user.id) {

            // CORRECCIÓN CLAVE: Marcamos como "pedida" ANTES del await.
            // Esto evita que la segunda ejecución del StrictMode lance otra petición.
            fetchedRecsForUser.current = user.id;

            try {
              console.log('Solicitando recomendaciones al cluster para:', user.id);
              const recs = await movieAPI.getRecommendations(user.id);
              setRecommendations(recs || []);
            } catch (recError) {
              console.warn('Fallo en recomendaciones:', recError);
              // Si falla, podrías querer resetear el ref para permitir reintentos:
              // fetchedRecsForUser.current = null; 
            }
          } else {
            console.log('Recomendaciones ya cargadas o en progreso para:', user.id);
          }
        }
      } catch (err) {
        setError('Error conectando con el API Gateway.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchContent();
  }, [offset, user]);

  const handleNextPage = () => {
    setOffset(prev => prev + limit);
    window.scrollTo(0, 0);
  };

  const handlePrevPage = () => {
    setOffset(prev => Math.max(0, prev - limit));
    window.scrollTo(0, 0);
  };

  if (loading) return <Loader size="large" message="Cargando catálogo distribuido..." />;

  if (error) {
    return (
      <div className="error-container text-center" style={{ padding: '2rem' }}>
        <h2>Sistema No Disponible</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()} style={btnStyle}>Reintentar</button>
      </div>
    );
  }

  return (
    <div className="home-page fade-in">

      {user && recommendations.length > 0 && (
        <section style={{ marginBottom: '4rem' }}>
          <h2 style={{ marginBottom: '1.5rem', color: '#646cff' }}>
            Recomendado para ti, {user.name}
          </h2>
          <div className="movies-grid" style={gridStyle}>
            {recommendations.map((movie) => (
              <MovieCard
                key={`rec-${movie.id}`}
                id={movie.id}
                title={movie.title}
                genres={movie.genres}
                year={extractYear(movie.title)}
                rating={movie.score ? Math.round(movie.score * 100) / 100 : undefined}
                isRecommendation={true}
                onClick={() => navigate(`/movie/${movie.id}`)}
              />
            ))}
          </div>
        </section>
      )}

      <section>
        <h1 className="text-center" style={{ marginBottom: '2rem' }}>Catálogo de Películas</h1>
        {movies.length === 0 ? (
          <p className="text-center">No hay películas disponibles.</p>
        ) : (
          <>
            <div className="movies-grid" style={gridStyle}>
              {movies.map((movie) => (
                <MovieCard
                  key={movie.id}
                  id={movie.id}
                  title={movie.title}
                  genres={movie.genres}
                  year={extractYear(movie.title)}
                  rating={undefined}
                  onClick={() => navigate(`/movie/${movie.id}`)}
                />
              ))}
            </div>

            <div className="pagination" style={{ display: 'flex', justifyContent: 'center', gap: '1rem', margin: '3rem 0' }}>
              <button onClick={handlePrevPage} disabled={offset === 0} style={btnStyle}>Anterior</button>
              <span style={{ alignSelf: 'center' }}>Página {Math.floor(offset / limit) + 1}</span>
              <button onClick={handleNextPage} disabled={movies.length < limit} style={btnStyle}>Siguiente</button>
            </div>
          </>
        )}
      </section>
    </div>
  );
};

const gridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
  gap: '2rem',
  padding: '1rem'
};

const btnStyle: React.CSSProperties = {
  padding: '0.5rem 1rem',
  cursor: 'pointer',
  background: '#333',
  color: 'white',
  border: 'none',
  borderRadius: '4px'
};

export default Home;