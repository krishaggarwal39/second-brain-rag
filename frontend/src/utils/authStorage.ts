// Single source of truth for client-side auth storage (R-42).
// The token lives under ONE key; the legacy 'token' key is migrated on first read.

export const TOKEN_KEY = 'access_token';
export const LEGACY_TOKEN_KEY = 'token';
export const USER_EMAIL_KEY = 'user_email';

export function getStoredToken(): string | null {
  let token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    const legacy = localStorage.getItem(LEGACY_TOKEN_KEY);
    if (legacy && legacy.trim()) {
      localStorage.setItem(TOKEN_KEY, legacy);
      token = legacy;
    }
  }
  localStorage.removeItem(LEGACY_TOKEN_KEY);
  return token && token.trim().length > 0 ? token : null;
}

export function storeToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.removeItem(LEGACY_TOKEN_KEY);
}

export function clearStoredAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(LEGACY_TOKEN_KEY);
  localStorage.removeItem(USER_EMAIL_KEY);
}

/** Clear auth and notify the app (AuthContext listens for this event). */
export function logoutEverywhere(): void {
  clearStoredAuth();
  window.dispatchEvent(new CustomEvent('auth:logout'));
}
