import { Citation } from '../components/Citations';

// Prefix of assistant error bubbles; used to keep them out of chat history.
export const CHAT_ERROR_PREFIX = '⚠️ Error:';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  isError?: boolean;
}

export interface ChatHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatPayload {
  question: string;
  chat_history: ChatHistoryItem[];
}

export const MAX_HISTORY_ITEMS = 20;
export const MAX_CONTENT_LENGTH = 4000;

export function isErrorOrFailureMessage(msg: Message): boolean {
  if (msg.isError) return true;
  if (msg.role === 'assistant') {
    const trimmed = msg.content.trim();
    if (trimmed.startsWith(CHAT_ERROR_PREFIX) || trimmed.startsWith('Error:')) {
      return true;
    }
  }
  return false;
}

/**
 * Builds the chat payload compliant with backend limits:
 * - At most 20 history items (last 20 messages)
 * - Each item content capped at 4000 characters
 * - Excludes error/failure bubbles and empty messages
 */
export function buildChatPayload(
  question: string,
  messages: Message[],
  maxHistory: number = MAX_HISTORY_ITEMS,
  maxContentLength: number = MAX_CONTENT_LENGTH
): ChatPayload {
  const validMessages = messages.filter((m) => {
    if (!m || typeof m.content !== 'string') return false;
    if (!m.content.trim()) return false;
    if (isErrorOrFailureMessage(m)) return false;
    return true;
  });

  const recentMessages = validMessages.slice(-maxHistory);

  const chat_history: ChatHistoryItem[] = recentMessages.map((m) => ({
    role: m.role,
    content: m.content.length > maxContentLength ? m.content.slice(0, maxContentLength) : m.content,
  }));

  return {
    question,
    chat_history,
  };
}
