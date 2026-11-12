import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, Trash2 } from 'lucide-react';
import Citations from './Citations';
import { buildChatPayload, CHAT_ERROR_PREFIX, Message } from '../utils/chatPayload';
import { getStoredToken, logoutEverywhere } from '../utils/authStorage';
import { apiErrorMessage } from '../utils/errors';

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = async (e: React.SyntheticEvent) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    const payload = buildChatPayload(userMessage.content, messages);
    let aiMessageAdded = false;

    try {
      const baseUrl = import.meta.env.VITE_API_URL || '/api';

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      // Attach the key in dev; in production Nginx injects it server-side.
      const apiKey = import.meta.env.VITE_API_KEY;
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }

      // Attach Bearer token for JWT authentication
      const token = getStoredToken();
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${baseUrl}/chat/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      // Always check HTTP status before trying to read the body as SSE.
      // A non-ok response (401, 403, 429, 500…) would otherwise silently fail
      // inside the parse loop and leave isTyping stuck as true forever.
      if (!response.ok) {
        if (response.status === 401) {
          logoutEverywhere();
        }

        let detail = `HTTP ${response.status}`;
        try {
          const errBody = await response.json();
          detail = errBody?.detail || JSON.stringify(errBody);
        } catch {
          detail = await response.text().catch(() => detail);
        }
        throw new Error(detail);
      }

      if (!response.body) throw new Error('No response body from server.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      const aiMessage: Message = { role: 'assistant', content: '' };
      setMessages((prev) => [...prev, aiMessage]);
      aiMessageAdded = true;

      let buffer = '';
      let streamDone = false;

      while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let newlineIndex: number;
        while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);

          if (!line.startsWith('data: ')) continue;

          const dataStr = line.slice(6);
          if (dataStr === '[DONE]') {
            streamDone = true;
            break;
          }

          try {
            const data = JSON.parse(dataStr);
            if (data.text) {
              aiMessage.content += data.text;
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = { ...aiMessage };
                return next;
              });
            } else if (data.citations) {
              aiMessage.citations = data.citations;
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = { ...aiMessage };
                return next;
              });
            }
          } catch (parseError) {
            console.warn('Failed to parse SSE JSON:', parseError, dataStr);
          }
        }
      }
    } catch (error: unknown) {
      console.error('Chat stream error:', error);
      const reason = apiErrorMessage(error, 'An unexpected error occurred.');
      const errorText = `${CHAT_ERROR_PREFIX} ${reason}`;
      setMessages((prev) => {
        if (aiMessageAdded) {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === 'assistant' && !last.content) {
            next[next.length - 1] = {
              role: 'assistant',
              content: errorText,
              isError: true,
            };
            return next;
          }
        }
        return [
          ...prev,
          {
            role: 'assistant',
            content: errorText,
            isError: true,
          },
        ];
      });
    } finally {
      // isTyping is always cleared — even on error — so the UI is never frozen.
      setIsTyping(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="chat-container">
      {/* A plain section header: the page-level banner landmark belongs to App. */}
      <div className="chat-header">
        <div className="chat-status" aria-hidden="true" />
        <h2 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          AI Research Assistant
        </h2>
        {messages.length > 0 && (
          <button onClick={clearChat} className="delete-btn" title="Clear Chat" aria-label="Clear chat history" style={{ marginLeft: 'auto' }}>
            <Trash2 size={16} aria-hidden="true" />
          </button>
        )}
      </div>

      <div className="chat-messages" ref={scrollRef} role="log" aria-live="polite" aria-label="Chat messages">
        {messages.length === 0 && !isTyping && (
          <div
            style={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-muted)',
            }}
          >
            <Bot size={48} style={{ opacity: 0.5, marginBottom: '1rem' }} aria-hidden="true" />
            <p>Ask a question about your documents.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="bubble">
              <span className="sr-only">{msg.role === 'user' ? 'You said: ' : 'Assistant said: '}</span>
              {msg.content}
              {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                <Citations citations={msg.citations} />
              )}
            </div>
          </div>
        ))}

        {isTyping && messages[messages.length - 1]?.role !== 'assistant' && (
          <div className="message ai" role="status" aria-live="polite" aria-label="AI is generating a response">
            <div
              className="bubble"
              style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', minHeight: '44px' }}
            >
              <span className="spinner" aria-hidden="true" />
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Synthesizing…
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="chat-input-wrapper">
        {/* A named form, not role="search": this composes a chat message. */}
        <form onSubmit={handleSend} className="input-box" aria-label="Send a message">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend(e);
              }
            }}
            placeholder="Ask AI Assistant…"
            aria-label="Type your question"
            disabled={isTyping}
            rows={1}
          />
          <button type="submit" disabled={!input.trim() || isTyping} className="btn-send" aria-label="Send message">
            <Send size={18} aria-hidden="true" />
          </button>
        </form>
      </div>
    </div>
  );
}
