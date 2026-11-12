import React from 'react';
import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AuthProvider, useAuth } from '../AuthContext';

// Mock api
vi.mock('../../services/api', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  },
}));

import api from '../../services/api';

function TestConsumer() {
  const { token, isAuthenticated, userEmail, login, register, logout } = useAuth();
  return (
    <div>
      <div data-testid="auth-status">{isAuthenticated ? 'authenticated' : 'unauthenticated'}</div>
      <div data-testid="token">{token || 'no-token'}</div>
      <div data-testid="email">{userEmail || 'no-email'}</div>
      <button onClick={() => login('test@example.com', 'password123')}>Login</button>
      <button onClick={() => register('new@example.com', 'password123')}>Register</button>
      <button onClick={logout}>Logout</button>
    </div>
  );
}

describe('AuthContext', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('restores token from localStorage on initial load', () => {
    localStorage.setItem('access_token', 'saved-jwt-token');
    localStorage.setItem('user_email', 'persisted@example.com');

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    expect(screen.getByTestId('auth-status')).toHaveTextContent('authenticated');
    expect(screen.getByTestId('token')).toHaveTextContent('saved-jwt-token');
    expect(screen.getByTestId('email')).toHaveTextContent('persisted@example.com');
  });

  it('starts unauthenticated when no token is in localStorage', () => {
    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    expect(screen.getByTestId('auth-status')).toHaveTextContent('unauthenticated');
    expect(screen.getByTestId('token')).toHaveTextContent('no-token');
  });

  it('logs in successfully and saves token', async () => {
    (api.post as any).mockResolvedValueOnce({
      data: { access_token: 'new-jwt-token', token_type: 'bearer' },
    });

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    await act(async () => {
      screen.getByText('Login').click();
    });

    expect(api.post).toHaveBeenCalledWith('/auth/login', {
      email: 'test@example.com',
      password: 'password123',
    });
    expect(screen.getByTestId('auth-status')).toHaveTextContent('authenticated');
    expect(screen.getByTestId('token')).toHaveTextContent('new-jwt-token');
    expect(localStorage.getItem('access_token')).toBe('new-jwt-token');
  });

  it('registers successfully and saves token', async () => {
    (api.post as any).mockResolvedValueOnce({
      data: { access_token: 'registered-jwt-token', token_type: 'bearer' },
    });

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    await act(async () => {
      screen.getByText('Register').click();
    });

    expect(api.post).toHaveBeenCalledWith('/auth/register', {
      email: 'new@example.com',
      password: 'password123',
    });
    expect(screen.getByTestId('auth-status')).toHaveTextContent('authenticated');
    expect(screen.getByTestId('token')).toHaveTextContent('registered-jwt-token');
    expect(localStorage.getItem('access_token')).toBe('registered-jwt-token');
  });

  it('surfaces 401 error on invalid credentials during login', async () => {
    (api.post as any).mockRejectedValueOnce({
      response: { status: 401, data: { detail: 'Incorrect email or password' } },
    });

    let thrownError: Error | null = null;
    function ErrorConsumer() {
      const { login } = useAuth();
      return (
        <button
          onClick={async () => {
            try {
              await login('bad@example.com', 'wrong');
            } catch (err: any) {
              thrownError = err;
            }
          }}
        >
          Do Bad Login
        </button>
      );
    }

    render(
      <AuthProvider>
        <ErrorConsumer />
      </AuthProvider>
    );

    await act(async () => {
      screen.getByText('Do Bad Login').click();
    });

    expect(thrownError).not.toBeNull();
    expect((thrownError as any).message).toContain('Incorrect email or password');
    expect(localStorage.getItem('access_token')).toBeNull();
  });

  it('surfaces 409 error when registering existing email', async () => {
    (api.post as any).mockRejectedValueOnce({
      response: { status: 409, data: { detail: 'User already exists' } },
    });

    let thrownError: Error | null = null;
    function ErrorConsumer() {
      const { register } = useAuth();
      return (
        <button
          onClick={async () => {
            try {
              await register('existing@example.com', 'pass');
            } catch (err: any) {
              thrownError = err;
            }
          }}
        >
          Do Duplicate Register
        </button>
      );
    }

    render(
      <AuthProvider>
        <ErrorConsumer />
      </AuthProvider>
    );

    await act(async () => {
      screen.getByText('Do Duplicate Register').click();
    });

    expect(thrownError).not.toBeNull();
    expect((thrownError as any).message).toContain('User already exists');
  });

  it('clears token on logout', async () => {
    localStorage.setItem('access_token', 'active-token');

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    expect(screen.getByTestId('auth-status')).toHaveTextContent('authenticated');

    await act(async () => {
      screen.getByText('Logout').click();
    });

    expect(screen.getByTestId('auth-status')).toHaveTextContent('unauthenticated');
    expect(screen.getByTestId('token')).toHaveTextContent('no-token');
    expect(localStorage.getItem('access_token')).toBeNull();
  });

  it('resets auth when window dispatches auth:logout event', async () => {
    localStorage.setItem('access_token', 'active-token');

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    expect(screen.getByTestId('auth-status')).toHaveTextContent('authenticated');

    await act(async () => {
      window.dispatchEvent(new CustomEvent('auth:logout'));
    });

    expect(screen.getByTestId('auth-status')).toHaveTextContent('unauthenticated');
    expect(screen.getByTestId('token')).toHaveTextContent('no-token');
    expect(localStorage.getItem('access_token')).toBeNull();
  });
});
