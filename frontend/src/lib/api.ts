const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

export type UploadedFile = {
  name: string
  path: string
  type: string
  size: number
}

export type AgentEvent = {
  agent: string
  type: string
  message?: string
  score?: number
  next?: string
  reasoning?: string
}

export type RetrievedImage = {
  image_id: string
  image_b64: string
  caption: string
}

export type StreamEvent =
  | { type: "start"; session_id: string }
  | { type: "agent_event"; event: AgentEvent }
  | { type: "final"; output: string; images: { image_id: string; caption: string; source: string; page: number }[] }
  | { type: "image"; image_id: string; image_b64: string; caption: string }
  | { type: "error"; message: string; trace?: string }
  | { type: "done" }

export type UserSession = {
  session_id: string
  created_at: string
  updated_at: string
  title: string
  message_count: number
}


export async function getOrCreateSession(userId?: string | null): Promise<string> {
  const stored =
    typeof window !== "undefined" ? localStorage.getItem("aether_session_id") : null
  const res = await fetch(`${BACKEND_URL}/api/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: stored || null, user_id: userId || null }),
  })
  if (!res.ok) throw new Error(`Session creation failed (${res.status})`)
  const data = await res.json()
  if (typeof window !== "undefined") {
    localStorage.setItem("aether_session_id", data.session_id)
  }
  return data.session_id
}


export async function getUserSessions(userId: string): Promise<UserSession[]> {
  const res = await fetch(`${BACKEND_URL}/api/sessions/${userId}`)
  if (!res.ok) return []
  const data = await res.json()
  return data.sessions || []
}


export async function linkSessionToUser(sessionId: string, userId: string): Promise<void> {
  await fetch(`${BACKEND_URL}/api/sessions/${sessionId}/link/${userId}`, {
    method: "POST",
  })
}


export async function uploadFiles(
  files: File[],
  sessionId: string,
): Promise<UploadedFile[]> {
  const formData = new FormData()
  formData.append("session_id", sessionId)
  files.forEach((f) => formData.append("files", f))

  const res = await fetch(`${BACKEND_URL}/api/upload`, {
    method: "POST",
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }))
    throw new Error(err.detail || "Upload failed")
  }
  return (await res.json()).uploaded
}


export async function clearUploads(sessionId: string): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/api/upload/${sessionId}`, { method: "DELETE" })
  if (!res.ok) throw new Error("Failed to clear uploads")
}


export async function deleteFile(sessionId: string, filename: string): Promise<void> {
  const encoded = encodeURIComponent(filename)
  const res = await fetch(`${BACKEND_URL}/api/upload/${sessionId}/${encoded}`, {
    method: "DELETE",
  })
  if (!res.ok) throw new Error(`Failed to delete ${filename}`)
}


export async function streamQuery(
  query: string,
  sessionId: string,
  uploadedFiles: UploadedFile[],
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const formData = new FormData()
  formData.append("query", query)
  formData.append("session_id", sessionId)
  formData.append("uploaded_files", JSON.stringify(uploadedFiles))

  const res = await fetch(`${BACKEND_URL}/api/query/stream`, {
    method: "POST",
    body: formData,
  })
  if (!res.ok) throw new Error(`Backend error: ${res.status}`)

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() || ""
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          onEvent(JSON.parse(line.slice(6)) as StreamEvent)
        } catch {
          /* skip malformed */
        }
      }
    }
  }

  // Process any remaining data in the buffer
  if (buffer.trim().startsWith("data: ")) {
    try {
      onEvent(JSON.parse(buffer.trim().slice(6)) as StreamEvent)
    } catch {
      /* skip malformed */
    }
  }
}
