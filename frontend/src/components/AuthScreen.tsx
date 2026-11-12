import React, { useState } from 'react';
import { BrainCircuit, Mail, Lock, LogIn, UserPlus, AlertCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { apiErrorMessage } from '../utils/errors';

export default function AuthScreen() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const isRegister = mode === 'register';

  const switchMode = (newMode: 'login' | 'register') => {
    setMode(newMode);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim() || isLoading) return;

    setError(null);
    setIsLoading(false); // reset
    setIsLoading(true);

    try {
      if (isRegister) {
        await register(email.trim(), password);
      } else {
        await login(email.trim(), password);
      }
    } catch (err: unknown) {
      setError(apiErrorMessage(err, 'Authentication failed. Please try again.'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
      }}
    >
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '420px',
          padding: '2.5rem',
          boxShadow: '0 12px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(157, 78, 221, 0.15)',
        }}
      >
        {/* Brand Header */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div
            style={{
              display: 'inline-flex',
              padding: '0.75rem',
              borderRadius: 'var(--radius-lg)',
              background: 'rgba(0, 245, 212, 0.08)',
              marginBottom: '1rem',
              boxShadow: '0 0 16px rgba(0, 245, 212, 0.2)',
            }}
          >
            <BrainCircuit className="brand-icon" size={36} aria-hidden="true" />
          </div>
          <h1 className="text-gradient" style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Second Brain RAG
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            {isRegister
              ? 'Create your account to start curating knowledge'
              : 'Sign in to access your AI second brain'}
          </p>
        </div>

        {/* Tab Switcher */}
        <div
          role="tablist"
          aria-label="Authentication Options"
          style={{
            display: 'flex',
            background: 'rgba(255, 255, 255, 0.04)',
            borderRadius: 'var(--radius-md)',
            padding: '4px',
            marginBottom: '1.75rem',
            border: 'var(--glass-border)',
          }}
        >
          <button
            type="button"
            role="tab"
            aria-selected={!isRegister}
            onClick={() => switchMode('login')}
            style={{
              flex: 1,
              padding: '0.6rem 0',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              fontSize: '0.875rem',
              fontWeight: !isRegister ? 600 : 400,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              background: !isRegister ? 'var(--accent-primary)' : 'transparent',
              color: !isRegister ? '#ffffff' : 'var(--text-secondary)',
              boxShadow: !isRegister ? '0 2px 8px var(--accent-glow)' : 'none',
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={isRegister}
            onClick={() => switchMode('register')}
            style={{
              flex: 1,
              padding: '0.6rem 0',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              fontSize: '0.875rem',
              fontWeight: isRegister ? 600 : 400,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              background: isRegister ? 'var(--accent-primary)' : 'transparent',
              color: isRegister ? '#ffffff' : 'var(--text-secondary)',
              boxShadow: isRegister ? '0 2px 8px var(--accent-glow)' : 'none',
            }}
          >
            Register
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div
            className="error-box"
            role="alert"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
              fontSize: '0.85rem',
              marginBottom: '1.25rem',
            }}
          >
            <AlertCircle size={16} style={{ flexShrink: 0 }} aria-hidden="true" />
            <span>{error}</span>
          </div>
        )}

        {/* Auth Form */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div>
            <label
              htmlFor="email-input"
              style={{
                display: 'block',
                fontSize: '0.85rem',
                fontWeight: 500,
                color: 'var(--text-primary)',
                marginBottom: '0.5rem',
              }}
            >
              Email Address
            </label>
            <div style={{ position: 'relative' }}>
              <Mail
                size={16}
                aria-hidden="true"
                style={{
                  position: 'absolute',
                  left: '1rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'var(--text-muted)',
                }}
              />
              <input
                id="email-input"
                type="email"
                name="email"
                autoComplete="email"
                required
                placeholder="name@example.com"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  if (error) setError(null);
                }}
                disabled={isLoading}
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem 0.75rem 2.6rem',
                  borderRadius: 'var(--radius-md)',
                  border: 'var(--glass-border)',
                  background: 'rgba(0, 0, 0, 0.3)',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem',
                  outline: 'none',
                  transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
                }}
              />
            </div>
          </div>

          <div>
            <label
              htmlFor="password-input"
              style={{
                display: 'block',
                fontSize: '0.85rem',
                fontWeight: 500,
                color: 'var(--text-primary)',
                marginBottom: '0.5rem',
              }}
            >
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock
                size={16}
                aria-hidden="true"
                style={{
                  position: 'absolute',
                  left: '1rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'var(--text-muted)',
                }}
              />
              <input
                id="password-input"
                type="password"
                name="password"
                autoComplete={isRegister ? 'new-password' : 'current-password'}
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  if (error) setError(null);
                }}
                disabled={isLoading}
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem 0.75rem 2.6rem',
                  borderRadius: 'var(--radius-md)',
                  border: 'var(--glass-border)',
                  background: 'rgba(0, 0, 0, 0.3)',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem',
                  outline: 'none',
                  transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
                }}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || !email.trim() || !password.trim()}
            style={{
              width: '100%',
              padding: '0.85rem',
              marginTop: '0.5rem',
              borderRadius: 'var(--radius-md)',
              border: 'none',
              background: 'linear-gradient(135deg, var(--accent-primary), #7b2cbf)',
              color: '#ffffff',
              fontSize: '0.95rem',
              fontWeight: 600,
              cursor: isLoading || !email.trim() || !password.trim() ? 'not-allowed' : 'pointer',
              opacity: isLoading || !email.trim() || !password.trim() ? 0.6 : 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              boxShadow: '0 4px 15px var(--accent-glow)',
              transition: 'all 0.2s ease',
            }}
          >
            {isLoading ? (
              <>
                <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} aria-hidden="true" />
                <span>{isRegister ? 'Creating Account…' : 'Signing In…'}</span>
              </>
            ) : isRegister ? (
              <>
                <UserPlus size={18} aria-hidden="true" />
                <span>Create Account</span>
              </>
            ) : (
              <>
                <LogIn size={18} aria-hidden="true" />
                <span>Sign In</span>
              </>
            )}
          </button>
        </form>

        {/* Footer Toggle */}
        <div style={{ marginTop: '1.75rem', textAlign: 'center' }}>
          <button
            type="button"
            onClick={() => switchMode(isRegister ? 'login' : 'register')}
            disabled={isLoading}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: '0.85rem',
              cursor: 'pointer',
              textDecoration: 'underline',
              padding: '0.25rem',
              transition: 'color 0.2s ease',
            }}
          >
            {isRegister ? 'Already have an account? Sign in' : "Don't have an account? Register"}
          </button>
        </div>
      </div>
    </div>
  );
}
