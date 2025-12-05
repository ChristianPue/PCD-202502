import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Header: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="header">
      <div className="header-content">
        <h1>🎬 Movie App</h1>
        <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/search">Search</NavLink>

          {user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginLeft: '1rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Hi, {user.name}</span>
              <button
                onClick={handleLogout}
                style={{
                  padding: '0.4rem 0.8rem',
                  fontSize: '0.9rem',
                  background: 'transparent',
                  borderColor: 'var(--border-color)'
                }}
              >
                Logout
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', gap: '0.5rem', marginLeft: '1rem' }}>
              <NavLink to="/login">Login</NavLink>
              <NavLink to="/register" style={{
                background: 'var(--accent-primary)',
                color: 'white',
                padding: '0.4rem 1rem',
                borderRadius: 'var(--radius-md)'
              }}>
                Register
              </NavLink>
            </div>
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;
