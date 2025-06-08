/**
 * Main App component for the Legal Intelligence Platform.
 *
 * This component sets up the application structure with routing,
 * theme configuration, and global providers.
 *
 * @author Legal Intelligence Platform Team
 * @version 1.0.0
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { HelmetProvider } from 'react-helmet-async';

// Components
import Layout from '@components/Layout/Layout';
import ProtectedRoute from '@components/Auth/ProtectedRoute';

// Pages
import LoginPage from '@pages/Auth/LoginPage';
import RegisterPage from '@pages/Auth/RegisterPage';
import DashboardPage from '@pages/Dashboard/DashboardPage';
import DocumentsPage from '@pages/Documents/DocumentsPage';
import DocumentUploadPage from '@pages/Documents/DocumentUploadPage';
import DocumentViewPage from '@pages/Documents/DocumentViewPage';
import AnalysisPage from '@pages/Analysis/AnalysisPage';
import AnalysisResultsPage from '@pages/Analysis/AnalysisResultsPage';
import AgentsPage from '@pages/Agents/AgentsPage';
import SettingsPage from '@pages/Settings/SettingsPage';
import ProfilePage from '@pages/Profile/ProfilePage';
import NotFoundPage from '@pages/NotFound/NotFoundPage';

// Hooks and utilities
import { useAuthStore } from '@store/authStore';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    text: {
      primary: '#333333',
      secondary: '#666666',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          padding: '8px 16px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          borderRadius: 12,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
});

/**
 * Main App component that provides the application structure.
 */
function App(): JSX.Element {
  const { isAuthenticated, isLoading } = useAuthStore();

  // Show loading spinner while checking authentication
  if (isLoading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="100vh"
        >
          <div>Loading...</div>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <HelmetProvider>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Router>
            <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
              <Routes>
                {/* Public routes */}
                <Route
                  path="/login"
                  element={
                    isAuthenticated ? <Navigate to="/dashboard" replace /> : <LoginPage />
                  }
                />
                <Route
                  path="/register"
                  element={
                    isAuthenticated ? <Navigate to="/dashboard" replace /> : <RegisterPage />
                  }
                />

                {/* Protected routes */}
                <Route
                  path="/"
                  element={
                    <ProtectedRoute>
                      <Layout />
                    </ProtectedRoute>
                  }
                >
                  <Route index element={<Navigate to="/dashboard" replace />} />
                  <Route path="dashboard" element={<DashboardPage />} />

                  {/* Document routes */}
                  <Route path="documents" element={<DocumentsPage />} />
                  <Route path="documents/upload" element={<DocumentUploadPage />} />
                  <Route path="documents/:id" element={<DocumentViewPage />} />

                  {/* Analysis routes */}
                  <Route path="analysis" element={<AnalysisPage />} />
                  <Route path="analysis/:id" element={<AnalysisResultsPage />} />

                  {/* Agent routes */}
                  <Route path="agents" element={<AgentsPage />} />

                  {/* User routes */}
                  <Route path="profile" element={<ProfilePage />} />
                  <Route path="settings" element={<SettingsPage />} />
                </Route>

                {/* 404 route */}
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
            </Box>
          </Router>

          {/* Global toast notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#333',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#4caf50',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#f44336',
                  secondary: '#fff',
                },
              },
            }}
          />
        </ThemeProvider>
      </QueryClientProvider>
    </HelmetProvider>
  );
}

export default App;