import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const API_URL = "/ask";

const SUGGESTIONS = [
  "Where is the Computer Science department?",
  "What is in Room 205?",
  "Where is Physics Department",
  "What is in Room 416",
  "Where is pharmacology classroom?",
  "What is in Room 206?",
];

const THINKING_STATUSES = [
  "Searching campus information",
  "Checking available information",
  "Finding the best answer",
  "Preparing your response",
];

const MAX_TEXTAREA_HEIGHT = 200;
const SCROLL_BOTTOM_THRESHOLD = 120;
const MAX_DRAG_OFFSET_RATIO = 0.4; // composer can be dragged up to 40% of viewport height
// Instant answers (e.g. professor contact lookups) come back in a few milliseconds. Keep the
// loading indicator up for at least this long so it behaves the same as slower campus answers.
const MIN_THINKING_MS = 700; // NEW
// If the server never answers (cold start, network drop), stop waiting and show the error row
// instead of leaving the loading indicator on forever.
const REQUEST_TIMEOUT_MS = 90000; // NEW

let idCounter = 0;
const nextId = () => `m${++idCounter}`;

export default function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const [statusIndex, setStatusIndex] = useState(0);
  const [isNearBottom, setIsNearBottom] = useState(true);
  const [showNewResponse, setShowNewResponse] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  // How far the composer has been dragged UP from its default resting spot.
  // Because the composer is a normal flex child (not fixed/absolute), this
  // is just its own margin-bottom — the chat area above it resizes itself
  // automatically. No manual height math, no overlap possible.
  const [dragOffset, setDragOffset] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  const scrollRef = useRef(null);
  const textareaRef = useRef(null);
  const dragStartRef = useRef(null);
  // Id of the newest request. A reply from an older request (e.g. one sent before "New Chat")
  // must not hide the loading indicator of a newer request that is still pending.
  const requestIdRef = useRef(0); // NEW

  // Rotate the thinking status message while waiting on the API
  useEffect(() => {
    if (!isThinking) {
      setStatusIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setStatusIndex((i) => (i + 1) % THINKING_STATUSES.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [isThinking]);

  // Auto-grow the textarea up to a max height
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT) + "px";
  }, [question]);

  const clampDragOffset = useCallback((offset) => {
    const max = window.innerHeight * MAX_DRAG_OFFSET_RATIO;
    return Math.min(Math.max(offset, 0), max);
  }, []);

  // If the window is resized/rotated, keep the offset within the new bounds
  useEffect(() => {
    const onResize = () => setDragOffset((o) => clampDragOffset(o));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [clampDragOffset]);

  const handleDragPointerDown = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragStartRef.current = { y: e.clientY, offset: dragOffset };
    setIsDragging(true);
  };

  const handleDragPointerMove = (e) => {
    if (!dragStartRef.current) return;
    const delta = dragStartRef.current.y - e.clientY; // dragging up increases the offset
    setDragOffset(clampDragOffset(dragStartRef.current.offset + delta));
  };

  const handleDragPointerUp = (e) => {
    dragStartRef.current = null;
    setIsDragging(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* no-op */
    }
  };

  const scrollToBottom = useCallback((smooth = true) => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
    setShowNewResponse(false);
  }, []);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceFromBottom < SCROLL_BOTTOM_THRESHOLD;
    setIsNearBottom(nearBottom);
    if (nearBottom) setShowNewResponse(false);
  }, []);

  useEffect(() => {
    if (isNearBottom) {
      scrollToBottom(true);
    } else if (messages.length > 0 && messages[messages.length - 1].role !== "user") {
      setShowNewResponse(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, isThinking]);

  const callApi = async (userQuestion) => {
    const requestId = ++requestIdRef.current; // NEW
    // Starts now; awaited after the reply arrives. A slow reply has already outlasted it (no extra wait).
    const minVisible = new Promise((resolve) => setTimeout(resolve, MIN_THINKING_MS)); // NEW
    const controller = new AbortController(); // NEW
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS); // NEW

    // Shown immediately, on every kind of query.
    setIsThinking(true);

    let reply; // NEW
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userQuestion }),
        signal: controller.signal, // NEW
      });

      if (!response.ok) throw new Error("Server error");

      const data = await response.json();
      reply = { id: nextId(), role: "assistant", text: data.answer }; // CHANGED (was setMessages)
    } catch (err) {
      reply = { id: nextId(), role: "error", text: "Something went wrong.", question: userQuestion }; // CHANGED
    } finally {
      clearTimeout(timeoutId); // CHANGED (setIsThinking(false) moved below)
    }

    // Fast replies (e.g. contact lookups): let the indicator be seen instead of flashing.
    await minVisible; // NEW

    // A newer request (or New Chat) took over while this one was pending: leave the UI alone.
    if (requestId !== requestIdRef.current) return; // NEW

    setMessages((prev) => [...prev, reply]); // CHANGED
    setIsThinking(false); // CHANGED (moved from finally)
  };

  const sendMessage = (text) => {
    const userQuestion = (text ?? question).trim();
    if (!userQuestion || isThinking) return;

    if (!hasStarted) setHasStarted(true);
    setQuestion("");
    setIsNearBottom(true);
    setMessages((prev) => [...prev, { id: nextId(), role: "user", text: userQuestion }]);
    callApi(userQuestion);
  };

  const retryMessage = (msgId, userQuestion) => {
    setMessages((prev) => prev.filter((m) => m.id !== msgId));
    setIsNearBottom(true);
    callApi(userQuestion);
  };

  const startNewChat = () => {
    requestIdRef.current += 1; // NEW: ignore any reply that is still on its way
    setMessages([]);
    setHasStarted(false);
    setQuestion("");
    setIsThinking(false);
    setShowNewResponse(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const showWelcome = !hasStarted && messages.length === 0;

  return (
    <div className="cu-app">
      <div className="cu-aurora" aria-hidden="true">
        <span className="cu-aurora-blob cu-aurora-blob--1" />
        <span className="cu-aurora-blob cu-aurora-blob--2" />
        <span className="cu-aurora-blob cu-aurora-blob--3" />
      </div>
      <div className="cu-noise" aria-hidden="true" />

      <header className="cu-header">
        <div className="cu-brand">
          <span className="cu-brand-mark">CUPB-GPT</span>
          <span className="cu-brand-sub">Campus AI Assistant</span>
        </div>
        <div className="cu-header-right">
          <span className="cu-status-pill">
            <span className="cu-status-dot" /> CUPB AI
          </span>
          <button type="button" className="cu-btn cu-newchat" onClick={startNewChat}>
            New Chat
          </button>
        </div>
      </header>

      {/* This wrapper is the ONLY scroll container in the app. It is a flex:1
          child with min-height:0, which is what actually lets overflow-y:auto
          work inside a flex column instead of the whole page growing. */}
      <div className="cu-chat-wrap">
        <main ref={scrollRef} onScroll={handleScroll} className="cu-scroll cu-main">
          {showWelcome ? (
            <div className="cu-welcome">
              <div className="cu-sparkle">✦</div>
              <h1 className="cu-title">Welcome to CUPB-GPT</h1>
              <p className="cu-subtitle">Your AI assistant for Central University of Punjab.</p>

              <div className="cu-suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <button
                    key={s}
                    type="button"
                    className="cu-chip"
                    style={{ animationDelay: `${0.25 + i * 0.06}s` }}
                    onClick={() => sendMessage(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="cu-thread">
              {messages.map((msg) => (
                <MessageRow key={msg.id} msg={msg} onRetry={retryMessage} />
              ))}
              {isThinking && <ThinkingRow statusText={THINKING_STATUSES[statusIndex]} />}
            </div>
          )}
        </main>

        {showNewResponse && (
          <button type="button" className="cu-btn cu-new-response" onClick={() => scrollToBottom(true)}>
            ↓ New response
          </button>
        )}
      </div>

      {/* Composer: a normal flex child at the bottom of .cu-app, NOT
          position:fixed/absolute. It always occupies its own real space,
          so the chat area above it (flex:1) can never be covered by it —
          the browser's layout engine guarantees that, not JS math. */}
      <div
        className={`cu-composer-dock ${isDragging ? "cu-composer-dock--dragging" : ""}`}
        style={{ marginBottom: `${dragOffset}px` }}
      >
        <div className={`cu-composer ${isDragging ? "cu-composer--dragging" : ""}`}>
          <div
            className="cu-drag-handle"
            onPointerDown={handleDragPointerDown}
            onPointerMove={handleDragPointerMove}
            onPointerUp={handleDragPointerUp}
            onPointerCancel={handleDragPointerUp}
            role="separator"
            aria-orientation="horizontal"
            aria-label="Drag to move the message box"
            title="Drag to reposition"
          >
            <span className="cu-drag-grip" />
          </div>

          <div className="cu-inputbox">
            <textarea
              ref={textareaRef}
              className="cu-textarea"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask CUPB-GPT anything..."
              rows={1}
              aria-label="Ask CUPB-GPT anything"
              disabled={isThinking}
            />
            <button
              type="button"
              className={`cu-send ${question.trim() ? "cu-send--active" : ""}`}
              onClick={() => sendMessage()}
              disabled={!question.trim() || isThinking}
              aria-label="Send message"
            >
              ➤
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageRow({ msg, onRetry }) {
  if (msg.role === "error") {
    return (
      <div className="cu-row cu-row--assistant">
        <div className="cu-avatar">✦</div>
        <div className="cu-bubble cu-bubble--error">
          <div className="cu-error-title">Something went wrong</div>
          <div className="cu-error-text">Please try asking again.</div>
          <button type="button" className="cu-btn cu-retry" onClick={() => onRetry(msg.id, msg.question)}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`cu-row cu-row--${msg.role}`}>
      {msg.role === "assistant" && <div className="cu-avatar">✦</div>}
      <div className={`cu-bubble cu-bubble--${msg.role}`}>{msg.text}</div>
    </div>
  );
}

function ThinkingRow({ statusText }) {
  return (
    <div className="cu-row cu-row--assistant">
      <div className="cu-avatar cu-avatar--pulse">✦</div>
      <div className="cu-thinking">
        <span className="cu-thinking-text">{statusText}</span>
        <span className="cu-dots">
          <span className="cu-dot" style={{ animationDelay: "0s" }} />
          <span className="cu-dot" style={{ animationDelay: "0.2s" }} />
          <span className="cu-dot" style={{ animationDelay: "0.4s" }} />
        </span>
      </div>
    </div>
  );
}