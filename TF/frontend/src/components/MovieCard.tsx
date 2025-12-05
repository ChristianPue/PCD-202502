import React from 'react';

interface MovieCardProps {
  title: string;
  genres?: string | string[];
  year?: number;
  poster?: string;
  rating?: number;
  onClick?: () => void;
}

const MovieCard: React.FC<MovieCardProps> = ({
  title,
  genres,
  year,
  poster,
  rating,
  onClick
}) => {
  const genreList = Array.isArray(genres)
    ? genres
    : genres
      ? genres.split('|')
      : [];

  return (
    <div className="movie-card" onClick={onClick}>
      <div className="movie-poster">
        {poster ? (
          <img src={poster} alt={title} />
        ) : (
          <div className="poster-placeholder">No Image</div>
        )}
      </div>
      <div className="movie-info">
        <h3 className="movie-title">{title}</h3>
        {year && <p className="movie-year">{year}</p>}
        {genreList.length > 0 && (
          <p className="movie-genres">{genreList.slice(0, 2).join(', ')}</p>
        )}
        {rating && <p className="movie-rating">⭐ {rating}/10</p>}
      </div>
    </div>
  );
};

export default MovieCard;
