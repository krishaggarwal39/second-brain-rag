// Contract test (R-34): the payload the UI builds must satisfy the backend's
// pydantic limits. The limits are read from the backend source, so changing
// either side without the other fails this test (the Playwright suite mocks
// every route and could not catch R-07).
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { buildChatPayload, Message } from '../chatPayload';

const chatPy = readFileSync(resolve(__dirname, '../../../../backend/app/api/routes/chat.py'), 'utf-8');

function limit(pattern: RegExp, name: string): number {
  const m = chatPy.match(pattern);
  if (!m) throw new Error(`Could not find ${name} in backend chat.py`);
  return Number(m[1]);
}

const MAX_HISTORY = limit(/chat_history:[^\n]*max_length=(\d+)/, 'chat_history max_length');
const MAX_CONTENT = limit(/content:[^\n]*max_length=(\d+)/, 'ChatMessage.content max_length');
const MAX_QUESTION = limit(/question:[^\n]*max_length=(\d+)/, 'question max_length');

describe('chat payload vs backend limits', () => {
  it('found sane limits', () => {
    expect(MAX_HISTORY).toBeGreaterThan(0);
    expect(MAX_CONTENT).toBeGreaterThan(0);
    expect(MAX_QUESTION).toBeGreaterThan(0);
  });

  it('never exceeds history and content limits', () => {
    const messages: Message[] = Array.from({ length: MAX_HISTORY * 3 }, (_, i) => ({
      role: i % 2 ? 'assistant' : 'user',
      content: 'x'.repeat(MAX_CONTENT * 2),
    }));
    const payload = buildChatPayload('question', messages);
    expect(payload.chat_history.length).toBeLessThanOrEqual(MAX_HISTORY);
    for (const m of payload.chat_history) {
      expect(m.content.length).toBeLessThanOrEqual(MAX_CONTENT);
      expect(m.content.length).toBeGreaterThan(0);
    }
  });

  it('keeps the input question within the backend limit', () => {
    // The textarea enforces the limit; the payload must not inflate it.
    const payload = buildChatPayload('q'.repeat(MAX_QUESTION), []);
    expect(payload.question.length).toBeLessThanOrEqual(MAX_QUESTION);
  });
});
