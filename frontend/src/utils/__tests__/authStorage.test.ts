import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  TOKEN_KEY,
  LEGACY_TOKEN_KEY,
  USER_EMAIL_KEY,
  getStoredToken,
  storeToken,
  clearStoredAuth,
  logoutEverywhere,
} from '../authStorage';

describe('authStorage (R-42)', () => {
  beforeEach(() => localStorage.clear());

  it('migrates the legacy token key on first read', () => {
    localStorage.setItem(LEGACY_TOKEN_KEY, 'old-jwt');
    expect(getStoredToken()).toBe('old-jwt');
    expect(localStorage.getItem(TOKEN_KEY)).toBe('old-jwt');
    expect(localStorage.getItem(LEGACY_TOKEN_KEY)).toBeNull();
  });

  it('prefers the current key and drops the legacy one', () => {
    localStorage.setItem(TOKEN_KEY, 'new-jwt');
    localStorage.setItem(LEGACY_TOKEN_KEY, 'old-jwt');
    expect(getStoredToken()).toBe('new-jwt');
    expect(localStorage.getItem(LEGACY_TOKEN_KEY)).toBeNull();
  });

  it('treats blank tokens as absent', () => {
    localStorage.setItem(TOKEN_KEY, '   ');
    expect(getStoredToken()).toBeNull();
  });

  it('stores under exactly one key', () => {
    storeToken('abc');
    expect(localStorage.getItem(TOKEN_KEY)).toBe('abc');
    expect(localStorage.getItem(LEGACY_TOKEN_KEY)).toBeNull();
  });

  it('clears everything and dispatches auth:logout', () => {
    storeToken('abc');
    localStorage.setItem(USER_EMAIL_KEY, 'a@b.c');
    const spy = vi.fn();
    window.addEventListener('auth:logout', spy);
    logoutEverywhere();
    window.removeEventListener('auth:logout', spy);
    expect(localStorage.length).toBe(0);
    expect(spy).toHaveBeenCalledTimes(1);
    clearStoredAuth();
  });
});
