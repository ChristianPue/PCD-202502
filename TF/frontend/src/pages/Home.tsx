import React, { useEffect, useState } from 'react';
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

  // Pagination state
  const [offset, setOffset] = useState(0);
  const limit = 50;

  // Ref to track if we already fetched recommendations for this user
  const fetchedRecsForUser = React.useRef<number | null>(null);

  useEffect(() => {
    const fetchContent = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch popular movies
        const popularData = await movieAPI.getMovies(limit, offset);
        setMovies(popularData.data);

        // Fetch recommendations if user is logged in
        if (user) {
          // Only fetch if we haven't fetched for this user ID yet
          if (fetchedRecsForUser.current !== user.id) {
            try {
              console.log('Fetching recommendations for user:', user.id);
              const recs = await movieAPI.getRecommendations(user.id);
              setRecommendations(recs || []);
              fetchedRecsForUser.current = user.id;
            } catch (recError) {
              console.error('Failed to load recommendations', recError);
              // Don't block main content if recs fail
            }
          }
        } else {
          setRecommendations([]);
          fetchedRecsForUser.current = null;
        }

      } catch (err) {
        setError('Failed to load movies. Please try again later.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchContent();
  }, [offset, user]); // Re-run when offset changes or user logs in/out

  const handleNextPage = () => {
    setOffset(prev => prev + limit);
    window.scrollTo(0, 0);
  };

  const handlePrevPage = () => {
    setOffset(prev => Math.max(0, prev - limit));
    window.scrollTo(0, 0);
  };

  if (loading) return <Loader size="large" message="Loading movies..." />;

  if (error) {
    return (
      <div className="error-container text-center">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="home-page fade-in">

      {/* Recommendations Section */}
      {user && recommendations.length > 0 && (
        <section style={{ marginBottom: '4rem' }}>
          <h2 style={{ marginBottom: '1.5rem', color: 'var(--accent-primary)' }}>
            Recommended for You, {user.name}
          </h2>
          <div className="movies-grid" style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: '2rem',
            padding: '1rem'
          }}>
            {recommendations.map((movie) => (
              <MovieCard
                key={`rec-${movie.id || movie.movieId}`}
                title={movie.title}
                genres={movie.genres}
                year={movie.year}
                poster={movie.poster}
                rating={movie.score ? Math.round(movie.score * 100) / 10 : movie.rating}
                onClick={() => navigate(`/movie/${movie.id || movie.movieId}`)}
              />
            ))}
          </div>
        </section>
      )}

      <section>
        <h1 className="text-center" style={{ marginBottom: '2rem' }}>Popular Movies</h1>

        {movies.length === 0 ? (
          <p className="text-center">No movies found.</p>
        ) : (
          <>
            <div className="movies-grid" style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: '2rem',
              padding: '1rem'
            }}>
              {movies.map((movie) => (
                <MovieCard
                  key={movie.id}
                  title={movie.title}
                  genres={movie.genres}
                  year={movie.year}
                  poster={movie.poster}
                  rating={movie.rating}
                  onClick={() => navigate(`/movie/${movie.id}`)}
                />
              ))}
            </div>

            <div className="pagination" style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '1rem',
              margin: '2rem 0'
            }}>
              <button
                onClick={handlePrevPage}
                disabled={offset === 0}
              >
                Previous
              </button>
              <span style={{ alignSelf: 'center' }}>
                Page {Math.floor(offset / limit) + 1}
              </span>
              <button
                onClick={handleNextPage}
                disabled={movies.length < limit}
              >
                Next
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
};

export default Home;
