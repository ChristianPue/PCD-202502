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
    // Recuperar sesión al cargar
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const login = (username: string, password: string) => {
    console.log('Login attempt:', username);

    // ---------------------------------------------------------
    // MODO DEMO / TESTING:
    // Si el usuario ingresa un NÚMERO (ej: "1", "42"), 
    // lo tratamos como un UserID existente del Dataset.
    // Esto es vital para ver recomendaciones reales.
    // ---------------------------------------------------------
    const parsedId = parseInt(username, 10);

    if (!isNaN(parsedId) && parsedId > 0) {
      // Crear un usuario ficticio basado en el ID del dataset
      const datasetUser: User = {
        id: parsedId,
        name: `Dataset User ${parsedId}`,
        email: `user${parsedId}@movielens.org`
      };

      setUser(datasetUser);
      localStorage.setItem('currentUser', JSON.stringify(datasetUser));
      return true;
    }

    // Lógica antigua (para usuarios registrados manualmente en el frontend)
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
    // Nota: Los usuarios nuevos NO tendrán recomendaciones hasta que el backend 
    // soporte guardar nuevos ratings.
    const users = JSON.parse(localStorage.getItem('users') || '[]');

    if (users.some((u: User) => u.name === username || u.email === email)) {
      return false;
    }

    // Generamos IDs altos para no chocar con el dataset (que suele ir de 1 a ~138,000)
    const newUser: User = {
      id: Math.floor(Math.random() * 10000) + 200000,
      name: username,
      email: email
    };

    users.push(newUser);
    localStorage.setItem('users', JSON.stringify(users));

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