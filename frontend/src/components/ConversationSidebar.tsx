"use client"

import { useState, useEffect, useCallback } from "react"
import { useAuth } from "@/lib/auth"
import { getUserSessions, UserSession } from "@/lib/api"

type Props = {
  currentSessionId: string | null
  onSelectSession: (sessionId: string) => void
  onNewSession: () => void
}

export default function ConversationSidebar({
  currentSessionId,
  onSelectSession,
  onNewSession,
}: Props) {
  const { user, isConfigured } = useAuth()
  const [sessions, setSessions] = useState<UserSession[]>([])
  const [loading, setLoading] = useState(false)

  const fetchSessions = useCallback(async () => {
    if (!user) return
    setLoading(true)
    try {
      const s = await getUserSessions(user.id)
      setSessions(s)
    } catch {
      // Silently fail
    } finally {
      setLoading(false)
    }
  }, [user])

  useEffect(() => {
    if (user && isConfigured) {
      fetchSessions()
    }
  }, [user, isConfigured, fetchSessions])

  // Don't render if not authenticated
  if (!isConfigured || !user) return null

  return (
    <div className="flex flex-col h-full">
      {/* New conversation button */}
      <div className="px-3 py-3">
        <button
          onClick={onNewSession}
          className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-violet-700 bg-violet-50 hover:bg-violet-100 border border-violet-200 rounded-xl transition-colors"
        >
          <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
            <path d="M8 1a.5.5 0 01.5.5V6h4.5a.5.5 0 010 1H8.5v4.5a.5.5 0 01-1 0V7H3a.5.5 0 010-1h4.5V1.5A.5.5 0 018 1z"/>
          </svg>
          New conversation
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-0.5">
        {loading && (
          <p className="text-xs text-gray-400 text-center py-4">Loading...</p>
        )}
        {!loading && sessions.length === 0 && (
          <p className="text-xs text-gray-400 text-center py-4">
            No saved conversations yet
          </p>
        )}
        {sessions.map((s) => (
          <button
            key={s.session_id}
            onClick={() => onSelectSession(s.session_id)}
            className={[
              "w-full text-left px-3 py-2 rounded-lg transition-colors group",
              currentSessionId === s.session_id
                ? "bg-violet-50 border border-violet-200"
                : "hover:bg-gray-50 border border-transparent",
            ].join(" ")}
          >
            <p className={[
              "text-xs font-medium truncate",
              currentSessionId === s.session_id
                ? "text-violet-700"
                : "text-gray-700",
            ].join(" ")}>
              {s.title}
            </p>
            <p className="text-[10px] text-gray-400 mt-0.5">
              {s.message_count} messages
            </p>
          </button>
        ))}
      </div>
    </div>
  )
}
