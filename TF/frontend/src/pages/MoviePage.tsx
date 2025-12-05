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

  useEffect(() => {
    const fetchMovie = async () => {
      if (!id) return;

      try {
        setLoading(true);
        setError(null);
        const data = await movieAPI.getMovieById(id);
        setMovie(data);
      } catch (err) {
        setError('Failed to load movie details.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchMovie();
  }, [id]);

  if (loading) return <Loader size="large" message="Loading details..." />;

  if (error || !movie) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error || 'Movie not found'}</p>
        <button onClick={() => navigate(-1)}>Go Back</button>
      </div>
    );
  }

  const genreList = Array.isArray(movie.genres)
    ? movie.genres
    : movie.genres
      ? movie.genres.split('|')
      : [];

  return (
    <div className="movie-page fade-in">
      <button
        className="back-button"
        onClick={() => navigate(-1)}
        style={{ marginBottom: '2rem' }}
      >
        ← Back
      </button>

      <div className="movie-details-container" style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(300px, 1fr) 2fr',
        gap: '3rem',
        alignItems: 'start'
      }}>
        <div className="movie-poster-large" style={{
          borderRadius: '1rem',
          overflow: 'hidden',
          boxShadow: 'var(--shadow-xl)'
        }}>
          {movie.poster ? (
            <img
              src={movie.poster}
              alt={movie.title}
              style={{ width: '100%', display: 'block' }}
            />
          ) : (
            <div className="poster-placeholder" style={{ aspectRatio: '2/3', fontSize: '1.5rem' }}>
              No Image
            </div>
          )}
        </div>

        <div className="movie-info-detailed">
          <h1 style={{ marginBottom: '0.5rem' }}>{movie.title}</h1>

          <div className="meta-row" style={{
            display: 'flex',
            gap: '1rem',
            marginBottom: '2rem',
            color: 'var(--text-secondary)',
            fontSize: '1.1rem'
          }}>
            {movie.year && <span>{movie.year}</span>}
            {movie.rating && (
              <span style={{ color: 'var(--accent-primary)', fontWeight: 'bold' }}>
                ⭐ {movie.rating}/10
              </span>
            )}
          </div>

          <div className="genres-list" style={{ marginBottom: '2rem' }}>
            <h3 style={{ marginBottom: '1rem', fontSize: '1.2rem' }}>Genres</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {genreList.map((genre, index) => (
                <span key={index} style={{
                  background: 'var(--bg-tertiary)',
                  padding: '0.5rem 1rem',
                  borderRadius: '2rem',
                  fontSize: '0.9rem',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)'
                }}>
                  {genre}
                </span>
              ))}
            </div>
          </div>

          {/* Placeholder for description since API doesn't provide it yet */}
          <div className="description">
            <h3 style={{ marginBottom: '1rem', fontSize: '1.2rem' }}>Overview</h3>
            <p style={{ lineHeight: '1.8', fontSize: '1.1rem' }}>
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MoviePage;
