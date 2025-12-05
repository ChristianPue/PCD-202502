import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import './styles/main.css'
import Header from './components/Header'
import Home from './pages/Home'
import Search from './pages/Search'
import MoviePage from './pages/MoviePage'
import Login from './pages/Login'
import Register from './pages/Register'

function App() {
  return (
    <AuthProvider>
      <Router>
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/search" element={<Search />} />
            <Route path="/movie/:id" element={<MoviePage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Routes>
        </main>
      </Router>
    </AuthProvider>
  )
}

export default App
