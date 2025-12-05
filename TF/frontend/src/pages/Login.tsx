import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (login(username, password)) {
      navigate('/');
    } else {
      setError('User not found or invalid credentials.');
    }
  };

  return (
    <div className="auth-page fade-in" style={{
      maxWidth: '400px',
      margin: '4rem auto',
      padding: '2rem',
      background: 'var(--card-bg)',
      borderRadius: 'var(--radius-lg)',
      boxShadow: 'var(--shadow-lg)'
    }}>
      <h1 className="text-center" style={{ marginBottom: '2rem' }}>Login</h1>

      {error && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          color: '#ef4444',
          padding: '1rem',
          borderRadius: 'var(--radius-md)',
          marginBottom: '1.5rem',
          textAlign: 'center'
        }}>
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-color)',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              outline: 'none'
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-color)',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              outline: 'none'
            }}
          />
        </div>

        <button type="submit" style={{ width: '100%', marginTop: '1rem' }}>
          Login
        </button>
      </form>

      <p className="text-center" style={{ marginTop: '2rem', color: 'var(--text-secondary)' }}>
        Don't have an account? <Link to="/register">Register</Link>
      </p>
    </div>
  );
};

export default Login;
