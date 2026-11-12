import { describe, it, expect } from 'vitest';
import {
  buildChatPayload,
  isErrorOrFailureMessage,
  Message,
  MAX_HISTORY_ITEMS,
  MAX_CONTENT_LENGTH,
} from '../chatPayload';

describe('chatPayload helper (R-07)', () => {
  it('caps history to the last 20 messages when given 25 messages', () => {
    const messages: Message[] = Array.from({ length: 25 }, (_, i) => ({
      role: (i % 2 === 0 ? 'user' : 'assistant') as 'user' | 'assistant',
      content: `Message #${i + 1}`,
    }));

    const payload = buildChatPayload('What is next?', messages);

    expect(payload.question).toBe('What is next?');
    expect(payload.chat_history).toHaveLength(MAX_HISTORY_ITEMS);
    // Should contain the last 20 messages (indices 5 to 24, i.e., Message #6 to Message #25)
    expect(payload.chat_history[0].content).toBe('Message #6');
    expect(payload.chat_history[payload.chat_history.length - 1].content).toBe('Message #25');
  });

  it('truncates a 5000-character message to 4000 characters', () => {
    const longContent = 'A'.repeat(5000);
    const messages: Message[] = [
      { role: 'user', content: longContent },
    ];

    const payload = buildChatPayload('Summary please', messages);

    expect(payload.chat_history).toHaveLength(1);
    expect(payload.chat_history[0].content).toHaveLength(MAX_CONTENT_LENGTH);
    expect(payload.chat_history[0].content).toBe('A'.repeat(MAX_CONTENT_LENGTH));
  });

  it('excludes error and assistant-failure bubbles', () => {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: '⚠️ Error: Failed to fetch' },
      { role: 'assistant', content: 'Error: Rate limit exceeded', isError: true },
      { role: 'assistant', content: 'Valid assistant reply' },
      { role: 'assistant', content: 'Something broke', isError: true },
    ];

    const payload = buildChatPayload('Next question', messages);

    expect(payload.chat_history).toHaveLength(2);
    expect(payload.chat_history[0]).toEqual({ role: 'user', content: 'Hello' });
    expect(payload.chat_history[1]).toEqual({ role: 'assistant', content: 'Valid assistant reply' });
  });

  it('excludes empty and whitespace-only bubbles', () => {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: '' },
      { role: 'assistant', content: '   ' },
      { role: 'user', content: '\n\t  ' },
      { role: 'assistant', content: 'Here is the answer' },
    ];

    const payload = buildChatPayload('Follow-up', messages);

    expect(payload.chat_history).toHaveLength(2);
    expect(payload.chat_history[0]).toEqual({ role: 'user', content: 'Hello' });
    expect(payload.chat_history[1]).toEqual({ role: 'assistant', content: 'Here is the answer' });
  });

  it('correctly identifies error and failure messages', () => {
    expect(isErrorOrFailureMessage({ role: 'assistant', content: '⚠️ Error: 500' })).toBe(true);
    expect(isErrorOrFailureMessage({ role: 'assistant', content: 'Error: timeout' })).toBe(true);
    expect(isErrorOrFailureMessage({ role: 'assistant', content: 'failed', isError: true })).toBe(true);
    expect(isErrorOrFailureMessage({ role: 'user', content: 'Error in my code', isError: false })).toBe(false);
    expect(isErrorOrFailureMessage({ role: 'assistant', content: 'Normal response' })).toBe(false);
  });
});
