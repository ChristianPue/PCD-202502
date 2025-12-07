import React from 'react';

interface MovieCardProps {
  id: number;
  title: string;
  genres: string[];
  year?: number;
  rating?: number;
  isRecommendation?: boolean;
  onClick: () => void;
}

const MovieCard: React.FC<MovieCardProps> = ({
  id,
  title,
  genres,
  year,
  rating,
  isRecommendation,
  onClick
}) => {

  // Generar un color aleatorio estable basado en el ID de la película
  const generateColor = (id: number) => {
    const hue = (id * 137.508) % 360; // Número áureo para buena distribución
    return `hsl(${hue}, 60%, 30%)`;
  };

  const bgColor = generateColor(id);

  return (
    <div
      onClick={onClick}
      className="movie-card"
      style={{
        cursor: 'pointer',
        borderRadius: '8px',
        overflow: 'hidden',
        boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
        transition: 'transform 0.2s',
        backgroundColor: '#1a1a1a',
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
      }}
      // Efecto hover simple (puedes moverlo a CSS)
      onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
      onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
    >
      {/* 1. Placeholder de Póster (Gradiente) */}
      <div style={{
        height: '240px',
        background: `linear-gradient(135deg, ${bgColor} 0%, #1a1a1a 100%)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
        textAlign: 'center',
        position: 'relative'
      }}>
        <span style={{
          color: 'rgba(255,255,255,0.9)',
          fontWeight: 'bold',
          fontSize: '1.1rem',
          textShadow: '0 2px 4px rgba(0,0,0,0.5)'
        }}>
          {title}
        </span>

        {/* Badge de recomendación */}
        {isRecommendation && rating && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: '#ffd700',
            color: '#000',
            fontWeight: 'bold',
            padding: '4px 8px',
            borderRadius: '12px',
            fontSize: '0.8rem',
            boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            Score: {rating.toFixed(2)}
          </div>
        )}
      </div>

      {/* 2. Información */}
      <div style={{ padding: '1rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
        <h3 style={{
          margin: '0 0 0.5rem 0',
          fontSize: '1rem',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis'
        }}>
          {title}
        </h3>

        <div style={{ fontSize: '0.85rem', color: '#aaa', marginBottom: '0.5rem' }}>
          {year || 'Unknown Year'}
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: 'auto' }}>
          {genres.slice(0, 3).map((g, i) => (
            <span key={i} style={{
              fontSize: '0.7rem',
              background: '#333',
              padding: '2px 6px',
              borderRadius: '4px',
              color: '#ddd'
            }}>
              {g}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MovieCard;