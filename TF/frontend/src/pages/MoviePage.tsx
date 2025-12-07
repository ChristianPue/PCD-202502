import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { movieAPI, type Movie } from '../api/api';
import Loader from '../components/Loader';

const MoviePage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [movie, setMovie] = useState<Movie | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Helper: Extraer año del título
  const extractYear = (title: string): string => {
    const match = title.match(/\((\d{4})\)/);
    return match ? match[1] : '';
  };

  // Helper: Generar color consistente con el ID (Igual que en MovieCard)
  const generateColor = (id: number) => {
    const hue = (id * 137.508) % 360;
    return `hsl(${hue}, 60%, 30%)`;
  };

  useEffect(() => {
    const fetchMovie = async () => {
      if (!id) return;

      try {
        setLoading(true);
        setError(null);

        // Ahora el backend SÍ tiene este endpoint habilitado
        const data = await movieAPI.getMovieById(id);

        if (!data) throw new Error('Movie not found');
        setMovie(data);
      } catch (err) {
        setError('No se pudo cargar la película. Puede que el ID no exista.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchMovie();
  }, [id]);

  if (loading) return <Loader size="large" message="Cargando detalles..." />;

  if (error || !movie) {
    return (
      <div className="error-container" style={{ textAlign: 'center', marginTop: '4rem' }}>
        <h2>Error</h2>
        <p>{error || 'Película no encontrada'}</p>
        <button
          onClick={() => navigate(-1)}
          style={{ padding: '0.5rem 1rem', cursor: 'pointer', marginTop: '1rem' }}
        >
          Volver
        </button>
      </div>
    );
  }

  const bgColor = generateColor(movie.id);
  const year = extractYear(movie.title);

  return (
    <div className="movie-page fade-in">
      <button
        className="back-button"
        onClick={() => navigate(-1)}
        style={{
          marginBottom: '2rem',
          background: 'transparent',
          border: '1px solid #444',
          color: 'white',
          padding: '0.5rem 1rem',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        ← Volver
      </button>

      <div className="movie-details-container" style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(300px, 1fr) 2fr',
        gap: '3rem',
        alignItems: 'start'
      }}>
        {/* COLUMNA IZQUIERDA: PÓSTER GENERADO */}
        <div className="movie-poster-large" style={{
          borderRadius: '1rem',
          overflow: 'hidden',
          boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
          aspectRatio: '2/3',
          background: `linear-gradient(135deg, ${bgColor} 0%, #111 100%)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          textAlign: 'center'
        }}>
          <h1 style={{
            color: 'rgba(255,255,255,0.9)',
            fontSize: '2.5rem',
            textShadow: '0 4px 8px rgba(0,0,0,0.6)'
          }}>
            {movie.title}
          </h1>
        </div>

        {/* COLUMNA DERECHA: INFORMACIÓN */}
        <div className="movie-info-detailed" style={{ paddingTop: '1rem' }}>
          <h1 style={{ marginBottom: '0.5rem', fontSize: '2.5rem' }}>{movie.title}</h1>

          <div className="meta-row" style={{
            display: 'flex',
            gap: '1rem',
            marginBottom: '2rem',
            color: '#aaa',
            fontSize: '1.2rem',
            alignItems: 'center'
          }}>
            {year && (
              <span style={{
                background: '#333',
                padding: '2px 8px',
                borderRadius: '4px',
                color: 'white',
                fontSize: '1rem'
              }}>
                {year}
              </span>
            )}
            <span>ID: {movie.id}</span>
          </div>

          <div className="genres-list" style={{ marginBottom: '2rem' }}>
            <h3 style={{ marginBottom: '1rem', fontSize: '1.2rem', color: '#888' }}>Géneros</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.8rem' }}>
              {movie.genres.map((genre, index) => (
                <span key={index} style={{
                  background: 'rgba(255,255,255,0.1)',
                  padding: '0.5rem 1rem',
                  borderRadius: '2rem',
                  fontSize: '0.9rem',
                  color: '#ddd',
                  border: '1px solid rgba(255,255,255,0.2)'
                }}>
                  {genre}
                </span>
              ))}
            </div>
          </div>

          <div className="description">
            <h3 style={{ marginBottom: '1rem', fontSize: '1.2rem', color: '#888' }}>Sinopsis</h3>
            <p style={{ lineHeight: '1.8', fontSize: '1.1rem', color: '#ccc' }}>
              Actualmente no disponemos de una sinopsis detallada para esta película en el dataset de MovieLens 10M.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MoviePage;