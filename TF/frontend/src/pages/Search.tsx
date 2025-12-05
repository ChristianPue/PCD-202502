import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { movieAPI, type Movie } from '../api/api';
import MovieCard from '../components/MovieCard';
import Loader from '../components/Loader';

const Search: React.FC = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    try {
      setLoading(true);
      setSearched(true);
      const data = await movieAPI.searchMovies(query);
      setResults(data || []);
    } catch (err) {
      console.error('Search failed:', err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-page fade-in">
      <div className="search-header text-center" style={{ marginBottom: '3rem' }}>
        <h1 style={{ marginBottom: '2rem' }}>Find Movies</h1>

        <form onSubmit={handleSearch} style={{
          maxWidth: '600px',
          margin: '0 auto',
          display: 'flex',
          gap: '1rem'
        }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by title..."
            style={{
              flex: 1,
              padding: '1rem 1.5rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-color)',
              background: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              fontSize: '1.1rem',
              outline: 'none'
            }}
          />
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {loading ? (
        <Loader message="Searching database..." />
      ) : (
        <div className="search-results">
          {searched && (
            <h2 style={{ marginBottom: '2rem', fontSize: '1.5rem', color: 'var(--text-secondary)' }}>
              {results.length > 0
                ? `Found ${results.length} results for "${query}"`
                : `No results found for "${query}"`
              }
            </h2>
          )}

          <div className="movies-grid">
            {results.map((movie) => (
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
        </div>
      )}
    </div>
  );
};

export default Search;
