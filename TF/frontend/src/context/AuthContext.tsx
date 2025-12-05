import React, { createContext, useContext, useState, useEffect } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  login: (username: string, password: string) => boolean;
  register: (username: string, email: string, password: string) => boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // Check for persisted user on mount
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const login = (username: string, password: string) => {
    // For simulation, we just check if the user exists in localStorage
    // In a real app, we would validate the password too
    console.log('Login attempt:', username, password ? '***' : 'no-pass');
    const users = JSON.parse(localStorage.getItem('users') || '[]');
    const foundUser = users.find((u: User) => u.name === username);

    if (foundUser) {
      setUser(foundUser);
      localStorage.setItem('currentUser', JSON.stringify(foundUser));
      return true;
    }
    return false;
  };

  const register = (username: string, email: string, password: string) => {
    console.log('Registering:', username, email, password ? '***' : 'no-pass');
    const users = JSON.parse(localStorage.getItem('users') || '[]');

    // Check if username or email already exists
    if (users.some((u: User) => u.name === username || u.email === email)) {
      return false;
    }

    // Generate random ID between 1 and 1000 for demo purposes
    const newUser: User = {
      id: Math.floor(Math.random() * 1000) + 1,
      name: username,
      email: email
    };

    users.push(newUser);
    localStorage.setItem('users', JSON.stringify(users));

    // Auto login after register
    setUser(newUser);
    localStorage.setItem('currentUser', JSON.stringify(newUser));
    return true;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('currentUser');
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
