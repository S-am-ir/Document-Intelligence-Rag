"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  streamQuery,
  getOrCreateSession,
  linkSessionToUser,
  AgentEvent,
  StreamEvent,
  UploadedFile,
  RetrievedImage,
} from "@/lib/api"
import { useAuth } from "@/lib/auth"
import FileUpload from "@/components/FileUpload"
import AuthBanner from "@/components/AuthBanner"
import AuthModal from "@/components/AuthModal"
import ConversationSidebar from "@/components/ConversationSidebar"

// ── Types ──────────────────────────────────────────────────────────────────

type Message = {
  role: "user" | "assistant"
  content: string
  events?: AgentEvent[]
  images?: RetrievedImage[]
}

// ── Prompt suggestions ──────────────────────────────────────────────────────

const SUGGESTIONS = [
  "Summarise the key findings in this document",
  "What does the chart or figure on page 2 show?",
  "Extract all tables and numerical data",
  "What are the main conclusions or recommendations?",
]

// ── Color tokens ────────────────────────────────────────────────────────────

const C = {
  bg: "#f5f5f4",
  headerBg: "#ffffff",
  sidebarBg: "#fafaf9",
  userBubble: "#ea580c",
  assistantBg: "#ffffff",
  accent: "#ea580c",
  accentHover: "#c2410c",
  accentLight: "#fff7ed",
  accentBorder: "#fed7aa",
  text: "#1c1917",
  textSecondary: "#78716c",
  textMuted: "#a8a29e",
  border: "#e7e5e4",
  borderLight: "#f5f5f4",
  inputBg: "#ffffff",
}

// ── Custom markdown components ──────────────────────────────────────────────

const markdownComponents = {
  h1: ({ children }: { children?: React.ReactNode }) => (
    <h2 className="text-base font-semibold mt-4 mb-2" style={{ color: C.text }}>{children}</h2>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h3 className="text-sm font-semibold mt-3 mb-1" style={{ color: C.text }}>{children}</h3>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h4 className="text-sm font-semibold mt-2 mb-1" style={{ color: C.text }}>{children}</h4>
  ),
  p: ({ children }: { children?: React.ReactNode }) => (
    <p className="leading-relaxed text-sm mb-2 last:mb-0" style={{ color: "#44403c" }}>{children}</p>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <ul className="list-disc pl-4 my-2 space-y-0.5 text-sm" style={{ color: "#44403c" }}>{children}</ul>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <ol className="list-decimal pl-4 my-2 space-y-0.5 text-sm" style={{ color: "#44403c" }}>{children}</ol>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <li className="leading-relaxed">{children}</li>
  ),
  code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
    const isBlock = className?.includes("language-")
    if (isBlock) {
      return (
        <pre className="rounded-lg p-3 my-2 overflow-x-auto text-xs font-mono" style={{ background: "#f5f5f4" }}>
          <code style={{ color: C.text }}>{children}</code>
        </pre>
      )
    }
    return (
      <code className="text-[0.82em] px-1.5 py-0.5 rounded font-mono" style={{ background: C.accentLight, color: C.accent }}>
        {children}
      </code>
    )
  },
  pre: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-xs border-collapse rounded-lg overflow-hidden" style={{ border: `1px solid ${C.border}` }}>
        {children}
      </table>
    </div>
  ),
  thead: ({ children }: { children?: React.ReactNode }) => (
    <thead style={{ background: C.borderLight }}>{children}</thead>
  ),
  th: ({ children }: { children?: React.ReactNode }) => (
    <th className="px-3 py-1.5 text-left font-semibold" style={{ border: `1px solid ${C.border}`, color: C.text }}>{children}</th>
  ),
  td: ({ children }: { children?: React.ReactNode }) => (
    <td className="px-3 py-1.5" style={{ border: `1px solid ${C.border}`, color: "#57534e" }}>{children}</td>
  ),
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <blockquote className="pl-3 my-2 text-sm italic" style={{ borderLeft: `2px solid ${C.accentBorder}`, color: "#78716c" }}>
      {children}
    </blockquote>
  ),
  a: ({ children, href }: { children?: React.ReactNode; href?: string }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2" style={{ color: C.accent }}>
      {children}
    </a>
  ),
  strong: ({ children }: { children?: React.ReactNode }) => (
    <strong className="font-semibold" style={{ color: C.text }}>{children}</strong>
  ),
  hr: () => <hr className="my-3" style={{ borderColor: C.border }} />,
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function Home() {
  const { user, signOut, isConfigured } = useAuth()
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessionError, setSessionError] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isRunning, setIsRunning] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [showAuth, setShowAuth] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  // Session init
  useEffect(() => {
    let cancelled = false
    getOrCreateSession(user?.id)
      .then((id) => {
        if (!cancelled) setSessionId(id)
        if (user?.id && id) linkSessionToUser(id, user.id)
      })
      .catch((err) => {
        if (!cancelled) setSessionError(err.message || "Failed to connect")
      })
    return () => { cancelled = true }
  }, [user?.id])

  // Scroll to bottom
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight
    }
  }, [messages])

  const handleSubmit = useCallback(
    async (e: React.FormEvent | React.KeyboardEvent) => {
      e.preventDefault()
      if (!input.trim() || !sessionId || isRunning) return

      const query = input.trim()
      setInput("")
      setIsRunning(true)
      setMessages((prev) => [...prev, { role: "user", content: query }])

      let finalOutput = ""
      const imagesForTurn: RetrievedImage[] = []

      try {
        await streamQuery(query, sessionId, uploadedFiles, (event: StreamEvent) => {
          if (event.type === "final") finalOutput = event.output
          if (event.type === "image") {
            imagesForTurn.push({ image_id: event.image_id, image_b64: event.image_b64, caption: event.caption })
          }
          if (event.type === "error") {
            finalOutput = finalOutput || `Error: ${event.message}`
          }
        })

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: finalOutput || "No response generated.",
            images: imagesForTurn,
          },
        ])
      } catch (err: unknown) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Something went wrong: ${err instanceof Error ? err.message : "Unknown error"}. Please try again.`,
          },
        ])
      } finally {
        setIsRunning(false)
      }
    },
    [input, sessionId, isRunning, uploadedFiles],
  )

  const handleSuggestion = useCallback((text: string) => {
    setInput(text)
    const textarea = document.querySelector("textarea") as HTMLTextAreaElement | null
    textarea?.focus()
  }, [])

  const handleNewSession = useCallback(() => {
    setMessages([])
    setUploadedFiles([])
    setSessionId(null)
    setSessionError(null)
    localStorage.removeItem("aether_session_id")
    getOrCreateSession(user?.id).then(setSessionId).catch((err) => setSessionError(err.message))
  }, [user?.id])

  const handleSelectSession = useCallback((sid: string) => {
    setSessionId(sid)
    setMessages([])
    setUploadedFiles([])
    setSessionError(null)
    localStorage.setItem("aether_session_id", sid)
  }, [])

  return (
    <div className="min-h-screen flex flex-col" style={{ background: C.bg, color: C.text }}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="px-5 py-3 flex items-center justify-between shrink-0 sticky top-0 z-10" style={{ background: C.headerBg, borderBottom: `1px solid ${C.border}` }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ background: C.accent }}>
            <svg viewBox="0 0 20 20" fill="white" className="w-4 h-4">
              <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>
            </svg>
          </div>
          <div>
            <p className="text-xs" style={{ color: C.textMuted }}>Document Intelligence RAG</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all font-medium"
            style={{
              border: `1px solid ${uploadedFiles.length > 0 ? C.accentBorder : C.border}`,
              color: uploadedFiles.length > 0 ? C.accent : C.textSecondary,
              background: uploadedFiles.length > 0 ? C.accentLight : "transparent",
            }}
          >
            <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
              <path d="M4.5 3A1.5 1.5 0 003 4.5v7A1.5 1.5 0 004.5 13h7a1.5 1.5 0 001.5-1.5V7.621a1.5 1.5 0 00-.44-1.06l-3.12-3.122A1.5 1.5 0 008.38 3H4.5zm4 1.06l3.12 3.122H9.5a1 1 0 01-1-1V4.06z"/>
            </svg>
            {uploadedFiles.length > 0 ? `${uploadedFiles.length} file${uploadedFiles.length > 1 ? "s" : ""}` : "Attach"}
          </button>

          {isConfigured && user ? (
            <div className="flex items-center gap-1.5">
              <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold" style={{ background: C.accentLight, color: C.accent }}>
                {user.email?.[0]?.toUpperCase() || "U"}
              </div>
              <button onClick={() => signOut()} className="text-xs transition-colors" style={{ color: C.textMuted }} title="Sign out">
                <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
                  <path d="M6 12.5a.5.5 0 0 0 .5.5h8a.5.5 0 0 0 .5-.5v-9a.5.5 0 0 0-.5-.5h-8a.5.5 0 0 0-.5.5v2a.5.5 0 0 1-1 0v-2A1.5 1.5 0 0 1 6.5 2h8A1.5 1.5 0 0 1 16 3.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 5 12.5v-2a.5.5 0 0 1 1 0v2z"/>
                  <path d="M.146 8.354a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L1.707 7.5H10.5a.5.5 0 0 1 0 1H1.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3z"/>
                </svg>
              </button>
            </div>
          ) : isConfigured && !user ? (
            <button onClick={() => setShowAuth(true)} className="text-xs font-medium" style={{ color: C.accent }}>Sign in</button>
          ) : null}

          <div className="flex items-center gap-1.5 text-xs" style={{ color: C.textMuted }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: sessionError ? "#ef4444" : sessionId ? "#22c55e" : "#a8a29e" }} />
          </div>
        </div>
      </header>

      {/* ── Connection error banner ─────────────────────────────────────── */}
      {sessionError && (
        <div className="px-4 py-2.5 flex items-center gap-2" style={{ background: "#fef2f2", borderBottom: "1px solid #fecaca" }}>
          <p className="text-xs" style={{ color: "#b91c1c" }}>Cannot connect: {sessionError}</p>
          <button onClick={() => { setSessionError(null); getOrCreateSession(user?.id).then(setSessionId).catch((err) => setSessionError(err.message)) }}
            className="ml-auto text-xs font-medium underline" style={{ color: "#b91c1c" }}>Retry</button>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">

        {/* ── Sidebar ─────────────────────────────────────────────────── */}
        {sidebarOpen && (
          <aside className="w-72 flex flex-col shrink-0 overflow-hidden" style={{ background: C.sidebarBg, borderRight: `1px solid ${C.border}` }}>
            {isConfigured && user && (
              <>
                <div className="px-4 py-3" style={{ borderBottom: `1px solid ${C.borderLight}` }}>
                  <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: C.textSecondary }}>Conversations</p>
                </div>
                <ConversationSidebar currentSessionId={sessionId} onSelectSession={handleSelectSession} onNewSession={handleNewSession} />
                <div style={{ borderTop: `1px solid ${C.borderLight}` }} />
              </>
            )}

            <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: `1px solid ${C.borderLight}` }}>
              <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: C.textSecondary }}>Documents</p>
              <button onClick={() => setSidebarOpen(false)} style={{ color: C.textMuted }}>
                <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
                  <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {sessionId && <FileUpload sessionId={sessionId} uploadedFiles={uploadedFiles} onFilesChange={setUploadedFiles} disabled={isRunning} />}
            </div>
          </aside>
        )}

        {/* ── Main chat ───────────────────────────────────────────────── */}
        <div className="flex-1 flex flex-col min-w-0 mx-auto w-full px-4 py-4" style={{ maxWidth: "52rem" }}>

          {/* Empty state */}
          {messages.length === 0 && !isRunning && (
            <div className="flex-1 flex flex-col items-center justify-center gap-6 text-center px-4">
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center shadow-sm" style={{ background: C.accent }}>
                <svg viewBox="0 0 24 24" fill="white" className="w-7 h-7">
                  <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2" style={{ color: C.text }}>Document Intelligence</h2>
                <p className="text-sm max-w-md leading-relaxed" style={{ color: C.textSecondary }}>
                  Upload PDFs, DOCX, CSVs, or images. Ask questions, extract data, and explore charts.
                </p>
              </div>
              <button onClick={() => setSidebarOpen(true)}
                className="flex items-center gap-2 px-5 py-2.5 text-white text-sm font-medium rounded-xl transition-colors shadow-sm"
                style={{ background: C.accent }}>
                <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
                  <path d="M8 1a.5.5 0 01.5.5V6h4.5a.5.5 0 010 1H8.5v4.5a.5.5 0 01-1 0V7H3a.5.5 0 010-1h4.5V1.5A.5.5 0 018 1z"/>
                </svg>
                Attach documents
              </button>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg text-left mt-2">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => handleSuggestion(s)}
                    className="text-left text-xs rounded-xl px-3 py-2.5 transition-all"
                    style={{ color: C.textSecondary, border: `1px solid ${C.border}`, background: C.headerBg }}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          <div ref={messagesContainerRef} className="flex-1 space-y-6 overflow-y-auto pb-4">
            {messages.map((msg, i) => (
              <div key={i}>
                {msg.role === "user" ? (
                  <div className="flex justify-end">
                    <div className="text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm leading-relaxed" style={{ background: C.userBubble, maxWidth: "80%" }}>
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {/* Images ABOVE answer */}
                    {msg.images && msg.images.length > 0 && (
                      <div className="flex gap-3 flex-wrap">
                        {msg.images.map((img) => (
                          <div key={img.image_id} className="rounded-xl overflow-hidden shadow-sm" style={{ border: `1px solid ${C.border}`, background: C.headerBg, maxWidth: "420px" }}>
                            <img src={`data:image/png;base64,${img.image_b64}`} alt="Retrieved figure" className="w-full object-contain" style={{ maxHeight: "320px" }} />
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Answer */}
                    <div className="flex gap-2.5 items-start">
                      <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5" style={{ background: C.accentLight, border: `1px solid ${C.accentBorder}` }}>
                        <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5" style={{ color: C.accent }}>
                          <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>
                        </svg>
                      </div>
                      <div className="rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm" style={{ background: C.assistantBg, border: `1px solid ${C.border}`, maxWidth: "90%" }}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Live streaming indicator */}
            {isRunning && (
              <div className="flex gap-2.5 items-start">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 animate-pulse" style={{ background: C.accentLight, border: `1px solid ${C.accentBorder}` }}>
                  <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5" style={{ color: C.accent }}>
                    <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>
                  </svg>
                </div>
                <div className="rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm" style={{ background: C.assistantBg, border: `1px solid ${C.border}` }}>
                  <div className="flex gap-1 items-center">
                    {[0, 150, 300].map((d) => (
                      <span key={d} className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: C.textMuted, animationDelay: `${d}ms` }} />
                    ))}
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* ── Input area ─────────────────────────────────────────────── */}
          <div className="shrink-0 pt-3">
            {uploadedFiles.length > 0 && !sidebarOpen && (
              <div className="flex gap-1.5 mb-2 flex-wrap">
                {uploadedFiles.map((f) => (
                  <span key={f.name} className="text-xs px-2 py-0.5 rounded-full font-medium"
                    style={{ background: C.accentLight, border: `1px solid ${C.accentBorder}`, color: C.accent }}>
                    {f.name}
                  </span>
                ))}
              </div>
            )}

            <div className="flex gap-2 items-end rounded-2xl px-3 py-2 shadow-sm" style={{ background: C.inputBg, border: `1px solid ${C.border}` }}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(e) } }}
                placeholder={uploadedFiles.length > 0 ? "Ask about your documents..." : "Ask anything, or attach documents..."}
                rows={1}
                className="flex-1 bg-transparent text-sm resize-none outline-none leading-relaxed min-h-[28px] max-h-32 py-1"
                style={{ color: C.text, fieldSizing: "content" } as React.CSSProperties}
                disabled={isRunning || !sessionId}
              />
              <button onClick={handleSubmit} disabled={isRunning || !input.trim() || !sessionId}
                className="rounded-xl w-8 h-8 flex items-center justify-center transition-colors shrink-0"
                style={{ background: C.accent, opacity: isRunning || !input.trim() || !sessionId ? 0.3 : 1 }}
                title="Send (Enter)">
                <svg viewBox="0 0 16 16" fill="white" className="w-3.5 h-3.5">
                  <path d="M8 1.5a.5.5 0 01.354.146l6 6a.5.5 0 010 .708l-6 6a.5.5 0 01-.708-.708L13.293 8H1.5a.5.5 0 010-1h11.793L7.646 2.354A.5.5 0 018 1.5z"/>
                </svg>
              </button>
            </div>

            <p className="text-xs text-center mt-2" style={{ color: C.textMuted }}>
              Enter to send · Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>

      <AuthModal open={showAuth} onClose={() => setShowAuth(false)} />
    </div>
  )
}
