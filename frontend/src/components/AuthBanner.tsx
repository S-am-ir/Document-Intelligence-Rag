"use client"

import { useState } from "react"
import { useAuth } from "@/lib/auth"

type Props = {
  onSignInClick: () => void
}

export default function AuthBanner({ onSignInClick }: Props) {
  const { user, isConfigured, loading } = useAuth()
  const [dismissed, setDismissed] = useState(false)

  // Don't show if: auth not configured, loading, user signed in, or dismissed
  if (!isConfigured || loading || user || dismissed) return null

  return (
    <div className="bg-gradient-to-r from-violet-50 to-indigo-50 border-b border-violet-100 px-4 py-2 flex items-center gap-3">
      <div className="w-6 h-6 rounded-full bg-violet-100 flex items-center justify-center shrink-0">
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5 text-violet-600">
          <path d="M8 1a2 2 0 0 1 2 2v4H6V3a2 2 0 0 1 2-2zm3 6V3a3 3 0 0 0-6 0v4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2z"/>
        </svg>
      </div>
      <p className="text-xs text-violet-700 flex-1">
        <span className="font-medium">Sign in</span> to save your conversations and uploaded documents across sessions.
      </p>
      <button
        onClick={onSignInClick}
        className="text-xs font-medium text-violet-700 bg-white border border-violet-200 rounded-lg px-3 py-1 hover:bg-violet-50 transition-colors shrink-0"
      >
        Sign in
      </button>
      <button
        onClick={() => setDismissed(true)}
        className="text-violet-400 hover:text-violet-600 transition-colors shrink-0"
      >
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
          <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
        </svg>
      </button>
    </div>
  )
}
