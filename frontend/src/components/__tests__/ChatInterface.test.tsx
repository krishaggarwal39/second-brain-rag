import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import ChatInterface from '../ChatInterface';

describe('ChatInterface (R-07 & R-41)', () => {
  beforeEach(() => {
    localStorage.setItem('access_token', 'mock-token');
    vi.restoreAllMocks();
  });

  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('renders chat interface correctly', () => {
    render(<ChatInterface />);
    expect(screen.getByText('AI Research Assistant')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Ask AI Assistant…')).toBeInTheDocument();
  });

  it('replaces empty assistant bubble with error text on stream error instead of leaving an empty bubble', async () => {
    // Mock fetch where response.body reader throws after returning nothing
    const mockReader = {
      read: vi.fn().mockRejectedValueOnce(new Error('Network disconnected')),
    };
    const mockResponse = {
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    };
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(mockResponse as any);

    render(<ChatInterface />);

    const input = screen.getByPlaceholderText('Ask AI Assistant…');
    fireEvent.change(input, { target: { value: 'Test message' } });

    const form = screen.getByRole('form', { name: 'Send a message' });
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText(/⚠️ Error: Network disconnected/)).toBeInTheDocument();
    });

    // Verify there is no empty bubble rendered (empty assistant bubble should have been replaced)
    const bubbles = document.querySelectorAll('.message.assistant .bubble');
    expect(bubbles.length).toBe(1);
    expect(bubbles[0].textContent).toContain('⚠️ Error: Network disconnected');
  });

  it('surfaces JSON parse errors via console.warn', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    // Provide bad SSE JSON line
    const chunks = [
      new TextEncoder().encode('data: {invalid_json\n\n'),
      new TextEncoder().encode('data: [DONE]\n\n'),
    ];
    let chunkIndex = 0;
    const mockReader = {
      read: vi.fn().mockImplementation(() => {
        if (chunkIndex < chunks.length) {
          return Promise.resolve({ done: false, value: chunks[chunkIndex++] });
        }
        return Promise.resolve({ done: true, value: undefined });
      }),
    };
    const mockResponse = {
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    };
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(mockResponse as any);

    render(<ChatInterface />);

    const input = screen.getByPlaceholderText('Ask AI Assistant…');
    fireEvent.change(input, { target: { value: 'Check parse error' } });

    const form = screen.getByRole('form', { name: 'Send a message' });
    fireEvent.submit(form);

    await waitFor(() => {
      expect(warnSpy).toHaveBeenCalledWith(
        'Failed to parse SSE JSON:',
        expect.any(Error),
        '{invalid_json'
      );
    });
  });
});
