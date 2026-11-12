import axios from 'axios';
import { getStoredToken, logoutEverywhere } from '../utils/authStorage';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
});

// Request interceptor — attach API key and Bearer JWT token.
// In production, Nginx injects X-API-Key server-side via proxy_set_header.
// In dev (vite proxy or direct), we read it from the VITE_API_KEY env var.
// We send it here too so the dev server (which bypasses Nginx) works correctly.
api.interceptors.request.use((config) => {
  const apiKey = import.meta.env.VITE_API_KEY;
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }

  const token = getStoredToken();
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }

  return config;
});

// Response interceptor — reset auth on 401, retry once on 429, log errors.
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    // R-42: log status and URL only; response bodies can contain user data.
    console.error('API Error:', error.response?.status ?? error.message, error.config?.url);

    if (error.response?.status === 401) {
      logoutEverywhere();
    }

    const originalRequest = error.config;

    if (error.response?.status === 429 && originalRequest && !originalRequest._retry) {
      originalRequest._retry = true;
      console.warn('Rate limit hit. Retrying in 2 seconds…');
      await new Promise((resolve) => setTimeout(resolve, 2000));
      return api(originalRequest);
    }

    return Promise.reject(error);
  }
);

export default api;
