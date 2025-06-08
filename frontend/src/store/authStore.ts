/**
 * Authentication store using Zustand for state management.
 * 
 * This store manages user authentication state, login/logout functionality,
 * and token management for the Legal Intelligence Platform.
 * 
 * @author Legal Intelligence Platform Team
 * @version 1.0.0
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from 'axios';
import toast from 'react-hot-toast';

// Types
export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login?: string;
  organization?: string;
  profile_picture?: string;
}

export interface AuthState {
  // State
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  
  // Actions
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  register: (userData: RegisterData) => Promise<boolean>;
  refreshToken: () => Promise<boolean>;
  updateUser: (userData: Partial<User>) => void;
  checkAuth: () => Promise<void>;
}

export interface RegisterData {
  username: string;
  email: string;
  full_name: string;
  password: string;
  role?: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user_id: number;
  username: string;
  role: string;
}

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Configure axios defaults
axios.defaults.baseURL = API_BASE_URL;

// Create the auth store
export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // Initial state
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: true,

      // Login action
      login: async (username: string, password: string): Promise<boolean> => {
        try {
          set({ isLoading: true });

          const formData = new FormData();
          formData.append('username', username);
          formData.append('password', password);

          const response = await axios.post<LoginResponse>('/api/v1/auth/token', formData, {
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
          });

          const { access_token, user_id, username: responseUsername, role } = response.data;

          // Get user details
          const userResponse = await axios.get<User>(`/api/v1/users/${user_id}`, {
            headers: {
              Authorization: `Bearer ${access_token}`,
            },
          });

          const user = userResponse.data;

          // Update axios default headers
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

          // Update state
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });

          toast.success(`Welcome back, ${user.full_name}!`);
          return true;

        } catch (error: any) {
          console.error('Login error:', error);
          
          const errorMessage = error.response?.data?.detail || 'Login failed. Please try again.';
          toast.error(errorMessage);
          
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
          
          return false;
        }
      },

      // Logout action
      logout: () => {
        // Clear axios default headers
        delete axios.defaults.headers.common['Authorization'];

        // Clear state
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });

        toast.success('Logged out successfully');
      },

      // Register action
      register: async (userData: RegisterData): Promise<boolean> => {
        try {
          set({ isLoading: true });

          const response = await axios.post('/api/v1/auth/register', userData);

          toast.success('Registration successful! Please log in.');
          
          set({ isLoading: false });
          return true;

        } catch (error: any) {
          console.error('Registration error:', error);
          
          const errorMessage = error.response?.data?.detail || 'Registration failed. Please try again.';
          toast.error(errorMessage);
          
          set({ isLoading: false });
          return false;
        }
      },

      // Refresh token action
      refreshToken: async (): Promise<boolean> => {
        try {
          const { token } = get();
          
          if (!token) {
            return false;
          }

          // Check if token is still valid by making a request to /me endpoint
          const response = await axios.get<User>('/api/v1/auth/me', {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });

          const user = response.data;

          // Update state with fresh user data
          set({
            user,
            isAuthenticated: true,
            isLoading: false,
          });

          return true;

        } catch (error: any) {
          console.error('Token refresh error:', error);
          
          // Token is invalid, clear auth state
          get().logout();
          return false;
        }
      },

      // Update user action
      updateUser: (userData: Partial<User>) => {
        const { user } = get();
        
        if (user) {
          set({
            user: { ...user, ...userData },
          });
        }
      },

      // Check authentication status
      checkAuth: async () => {
        try {
          const { token } = get();
          
          if (!token) {
            set({
              user: null,
              token: null,
              isAuthenticated: false,
              isLoading: false,
            });
            return;
          }

          // Set axios header
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

          // Verify token and get user data
          const success = await get().refreshToken();
          
          if (!success) {
            set({
              user: null,
              token: null,
              isAuthenticated: false,
              isLoading: false,
            });
          }

        } catch (error) {
          console.error('Auth check error:', error);
          
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
        }
      },
    }),
    {
      name: 'legal-intelligence-auth',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// Initialize auth check on store creation
useAuthStore.getState().checkAuth();
