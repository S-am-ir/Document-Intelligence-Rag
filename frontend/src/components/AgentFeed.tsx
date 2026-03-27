"use client"

import { useState } from "react"
import { AgentEvent } from "@/lib/api"

const AGENT_META: Record<string, { color: string; bg: string; label: string }> = {
  ingest:         { color: "text-indigo-600",  bg: "bg-indigo-50",   label: "Ingest"    },
  decompose:      { color: "text-violet-600",  bg: "bg-violet-50",   label: "Decompose" },
  retrieve:       { color: "text-sky-600",     bg: "bg-sky-50",      label: "Retrieve"  },
  rerank:         { color: "text-teal-600",    bg: "bg-teal-50",     label: "Rerank"    },
  reflect:        { color: "text-amber-600",   bg: "bg-amber-50",    label: "Reflect"   },
  generate:       { color: "text-emerald-600", bg: "bg-emerald-50",  label: "Generate"  },
  document_agent: { color: "text-blue-600",    bg: "bg-blue-50",     label: "Docs"      },
  final:          { color: "text-green-600",   bg: "bg-green-50",    label: "Done"      },
  system:         { color: "text-gray-500",    bg: "bg-gray-50",     label: "System"    },
}

function getEventText(event: AgentEvent): string {
  switch (event.type) {
    case "parsing":
    case "parsed":
    case "ingesting":
    case "ingested":
    case "searching":
    case "fetched":
    case "rewriting":
    case "decomposing":
    case "decomposed":
    case "retrieved":
    case "reranking":
    case "reranked":
    case "reflecting":
    case "reformulating":
    case "passed":
    case "generating":
    case "complete":
    case "error":
    case "info":
      return event.message || event.type
    case "verdict":
      return `Score ${event.score}/10 — ${event.message || ""}`
    default:
      return event.message || event.type
  }
}

function getTextColor(event: AgentEvent): string {
  if (event.type === "error")   return "text-red-500"
  if (event.type === "complete" || event.type === "passed") return "text-emerald-600"
  if (event.type === "rewriting") return "text-violet-500"
  return "text-gray-500"
}

type Props = {
  events: AgentEvent[]
  isRunning: boolean
}

export default function AgentFeed({ events, isRunning }: Props) {
  const [collapsed, setCollapsed] = useState(false)

  if (events.length === 0 && !isRunning) return null

  const showContent = isRunning || !collapsed

  return (
    <div className="rounded-xl border border-gray-200 bg-gray-50 overflow-hidden">
      <button
        onClick={() => !isRunning && setCollapsed(!collapsed)}
        className={[
          "w-full px-3 py-2 flex items-center gap-2",
          isRunning ? "cursor-default" : "cursor-pointer hover:bg-gray-100 transition-colors",
        ].join(" ")}
      >
        <span className="text-[10px] text-gray-400 uppercase tracking-widest font-mono">
          Agent trace
        </span>
        {isRunning && (
          <span className="flex gap-0.5 items-center">
            {[0, 120, 240].map((d) => (
              <span
                key={d}
                className="w-1 h-1 rounded-full bg-violet-400 animate-bounce"
                style={{ animationDelay: `${d}ms` }}
              />
            ))}
          </span>
        )}
        {!isRunning && (
          <svg
            viewBox="0 0 16 16"
            fill="currentColor"
            className={[
              "w-3 h-3 text-gray-400 ml-auto transition-transform",
              collapsed ? "" : "rotate-180",
            ].join(" ")}
          >
            <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
          </svg>
        )}
        <span className="text-[10px] text-gray-400 ml-1">
          {events.length} event{events.length !== 1 ? "s" : ""}
        </span>
      </button>

      {showContent && (
        <div className="px-3 pb-2.5 space-y-0.5 max-h-52 overflow-y-auto border-t border-gray-100">
          {events.map((event, i) => {
            const meta = AGENT_META[event.agent] || {
              color: "text-gray-500", bg: "bg-gray-50", label: event.agent,
            }
            return (
              <div key={i} className="flex items-start gap-2 text-xs leading-relaxed py-0.5">
                <span className={`shrink-0 font-medium ${meta.color} min-w-[68px]`}>
                  {meta.label}
                </span>
                <span className={`${getTextColor(event)} min-w-0 truncate`}>
                  {getEventText(event)}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
