// R-43: landmark and label details. Not a WCAG audit; full validation needs
// manual testing with assistive technology.
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ChatInterface from '../ChatInterface';

describe('ChatInterface accessibility details', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it('does not add page-level banner or search landmarks', () => {
    render(<ChatInterface />);
    expect(screen.queryByRole('banner')).not.toBeInTheDocument();
    expect(screen.queryByRole('search')).not.toBeInTheDocument();
    expect(screen.getByRole('form', { name: 'Send a message' })).toBeInTheDocument();
  });

  it('labels who said each message for screen readers', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({ detail: 'boom' }),
    } as unknown as Response);
    render(<ChatInterface />);
    fireEvent.change(screen.getByPlaceholderText('Ask AI Assistant…'), { target: { value: 'Hello there' } });
    fireEvent.submit(screen.getByRole('form', { name: 'Send a message' }));
    await waitFor(() => expect(screen.getByText('You said:', { exact: false })).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText('Assistant said:', { exact: false })).toBeInTheDocument());
  });
});
