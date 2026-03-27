"use client"

import { useRef, useState } from "react"
import { uploadFiles, clearUploads, deleteFile, UploadedFile } from "@/lib/api"

const SUPPORTED = ".pdf,.docx,.txt,.md,.csv,.png,.jpg,.jpeg,.webp"

const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50 MB

const FILE_ICONS: Record<string, string> = {
  pdf:  "PDF",
  docx: "DOC",
  txt:  "TXT",
  md:   "MD",
  csv:  "CSV",
  png:  "IMG",
  jpg:  "IMG",
  jpeg: "IMG",
  webp: "IMG",
}

function formatSize(bytes: number): string {
  if (bytes < 1024)          return `${bytes} B`
  if (bytes < 1024 * 1024)   return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

type Props = {
  sessionId: string
  uploadedFiles: UploadedFile[]
  onFilesChange: (files: UploadedFile[]) => void
  disabled?: boolean
}

export default function FileUpload({
  sessionId,
  uploadedFiles,
  onFilesChange,
  disabled,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging]   = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError]             = useState<string | null>(null)

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0 || !sessionId) return
    setError(null)

    // Validate file sizes
    const oversized = Array.from(files).filter((f) => f.size > MAX_FILE_SIZE)
    if (oversized.length > 0) {
      setError(`File too large (max 50 MB): ${oversized.map((f) => f.name).join(", ")}`)
      return
    }

    // Check for duplicates
    const existing = new Set(uploadedFiles.map((f) => f.name))
    const newFiles = Array.from(files).filter((f) => !existing.has(f.name))
    if (newFiles.length === 0) {
      setError("All selected files are already uploaded")
      return
    }

    setIsUploading(true)
    try {
      const uploaded = await uploadFiles(newFiles, sessionId)
      onFilesChange([...uploadedFiles, ...uploaded])
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed")
    } finally {
      setIsUploading(false)
    }
  }

  async function removeFile(filename: string) {
    const updated = uploadedFiles.filter((f) => f.name !== filename)
    onFilesChange(updated)
    try {
      await deleteFile(sessionId, filename)
    } catch {
      // Silently handle — file removed from UI, server cleanup can be retried
    }
  }

  async function clearAll() {
    onFilesChange([])
    try {
      await clearUploads(sessionId)
    } catch {
      // Silently handle
    }
  }

  return (
    <div className="space-y-2">
      {/* Drop zone */}
      <div
        className={[
          "border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all",
          isDragging
            ? "border-violet-400 bg-violet-50"
            : "border-gray-200 hover:border-violet-300 hover:bg-gray-50",
          disabled ? "opacity-50 cursor-not-allowed" : "",
        ].join(" ")}
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          if (!disabled) handleFiles(e.dataTransfer.files)
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={SUPPORTED}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
          disabled={disabled}
        />

        {isUploading ? (
          <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            Processing document...
          </div>
        ) : (
          <div>
            <p className="text-sm text-gray-500">
              <span className="text-violet-600 font-medium">Drop files here</span> or{" "}
              <span className="text-gray-700 font-medium">click to browse</span>
            </p>
            <p className="text-xs text-gray-400 mt-1">
              PDF · DOCX · TXT · CSV · Images · Max 50 MB
            </p>
          </div>
        )}
      </div>

      {/* Error */}
      {error && <p className="text-xs text-red-500 px-1">{error}</p>}

      {/* Files list */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-1.5">
          {uploadedFiles.map((file) => (
            <div
              key={file.name}
              className="flex items-center gap-2.5 bg-white border border-gray-200 rounded-lg px-3 py-2 group"
            >
              <span className="text-[10px] font-bold text-violet-600 bg-violet-50 rounded px-1.5 py-0.5 shrink-0">
                {FILE_ICONS[file.type] || "FILE"}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-gray-700 truncate">{file.name}</p>
                <p className="text-xs text-gray-400">{formatSize(file.size)}</p>
              </div>
              <button
                onClick={() => removeFile(file.name)}
                disabled={disabled}
                className="text-gray-300 hover:text-red-400 transition-colors text-sm disabled:cursor-not-allowed shrink-0 opacity-60 group-hover:opacity-100"
                title="Remove file"
              >
                <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
                  <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
                </svg>
              </button>
            </div>
          ))}
          {uploadedFiles.length > 1 && (
            <button
              onClick={clearAll}
              disabled={disabled}
              className="text-xs text-gray-400 hover:text-red-500 transition-colors px-1 disabled:cursor-not-allowed"
            >
              Clear all
            </button>
          )}
        </div>
      )}
    </div>
  )
}
