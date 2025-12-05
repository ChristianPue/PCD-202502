import React from 'react';

interface LoaderProps {
  size?: 'small' | 'medium' | 'large';
  message?: string;
}

const Loader: React.FC<LoaderProps> = ({
  size = 'medium',
  message = 'Loading...'
}) => {
  return (
    <div className={`loader-container loader-${size}`}>
      <div className="spinner"></div>
      {message && <p className="loader-message">{message}</p>}
    </div>
  );
};

export default Loader;
