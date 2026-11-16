// DSH plugin `rwkv-todo`: registers `todo_plan` and `todo_update` (Stateful
// Work List) for RWKV and teacher sessions.
//
// Why a separate tool plugin instead of DSH's stock `@deepseek-ai/dsh-tool-todo`:
//   1. `todo_write` enforces `additionalProperties: false` on `{content, status}`,
//      and DSH's session-format validator rejects extra keys on `todo/write`
//      events, so `note` cannot be stored on `todo/write` directly.
//   2. `todo_write` is whole-list replacement and returns only status counts
//      (`Updated todo list: ...`), never echoing the list back into context.
//   3. Rewriting the whole list on every step wastes tokens and causes 7B models
//      to paraphrase or drop earlier line numbers/paths from `note`.
//
// `rwkv-todo` stores the full `{id, content, status, note}` state per turn in an
// ignorable `plugin:rwkv-todo/state` session record (surviving process-per-turn
// runners), echoes the full formatted list in every tool result, and mirrors a
// valid `{content, status}` projection onto `todo/write` so DSH's native Desktop
// TodoPanel renders live progress without modifying `deepseek-harness`.

import type { Context } from '@deepseek-ai/cordis'
import { appendPluginRecord, pluginRecordOf } from '@deepseek-ai/dsh-session'
import { defineTool, ToolArgsError } from '@deepseek-ai/dsh-tools'
import {
  applyTodoPlan,
  applyTodoUpdate,
  formatWorklist,
  isWorklistTool,
  normalizeWorklistStatus,
  projectNativeTodos,
  TODO_PLAN_TOOL,
  TODO_UPDATE_TOOL,
  type WorklistItem,
  type WorklistStatus,
} from './core/worklist.ts'
import { TODO_PLAN_DESCRIPTION, TODO_UPDATE_DESCRIPTION } from './core/tools.ts'

export { TODO_PLAN_DESCRIPTION, TODO_UPDATE_DESCRIPTION }

export const name = 'rwkv-todo'
export const inject = ['tools']

const STATE_RECORD = 'plugin:rwkv-todo/state'

interface PersistedState {
  readonly turn: number
  readonly items: readonly WorklistItem[]
  readonly completed?: boolean
}

interface SessionEventLike {
  readonly type?: string
  readonly data?: unknown
}

interface SessionLike {
  snapshotEvents?(): readonly SessionEventLike[]
  append?(type: 'todo/write', data: { todos: { content: string; status: 'pending' | 'in_progress' | 'completed' }[] }): unknown
}

interface ExecLike {
  readonly agent?: { readonly session?: SessionLike }
}

interface ContextLike {
  readonly tools: { register(definition: unknown): () => void }
  on?(event: string, listener: (session: SessionLike, ev: SessionEventLike) => void): () => void
}

const appendRecord = appendPluginRecord as unknown as (session: object, type: string, data: unknown) => unknown
const recordOf = pluginRecordOf as unknown as (event: unknown) => { type: string; data: unknown } | undefined

const VALID_STATUSES = new Set<string>(['pending', 'in_progress', 'completed', 'dropped'])

export { projectNativeTodos }

function decodeItems(raw: unknown): WorklistItem[] | undefined {
  if (!Array.isArray(raw)) return undefined
  const items: WorklistItem[] = []
  for (const entry of raw) {
    if (typeof entry !== 'object' || entry === null) return undefined
    const { id, content, status, note } = entry as Record<string, unknown>
    if (typeof id !== 'number' || !Number.isInteger(id) || id < 1) return undefined
    if (typeof content !== 'string' || content === '') return undefined
    if (typeof status !== 'string' || !VALID_STATUSES.has(status)) return undefined
    if (note !== undefined && typeof note !== 'string') return undefined
    items.push({
      id,
      content,
      status: status as WorklistStatus,
      ...(typeof note === 'string' && note !== '' ? { note } : {}),
    })
  }
  return items
}

function turnIndexOf(events: readonly SessionEventLike[] | undefined): number {
  if (events === undefined) return 1
  let turns = 0
  for (const event of events) {
    if (event.type === 'turn/start') turns++
  }
  return Math.max(1, turns)
}

/**
 * True when an earlier `todo_update` in the current turn already marked an item
 * `completed` and NO workspace tool (`read`, `edit_lines`, `write`, `bash`, ...)
 * has executed since then — preventing empty-window hallucinated closeouts.
 */
export function blockedBackToBackComplete(events: readonly SessionEventLike[] | undefined): boolean {
  if (events === undefined || events.length === 0) return false
  let skippedCurrentCall = false
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i]!
    if (ev.type === 'turn/start') return false
    if (ev.type === 'tool/call') {
      if (!skippedCurrentCall) {
        skippedCurrentCall = true
        continue
      }
      const toolName = typeof ev.data === 'object' && ev.data !== null
        ? (ev.data as { name?: unknown }).name
        : undefined
      if (typeof toolName === 'string' && !isWorklistTool(toolName) && toolName !== 'todo_write') {
        return false
      }
      continue
    }
    const rec = recordOf(ev)
    if (rec?.type === STATE_RECORD && typeof rec.data === 'object' && rec.data !== null) {
      if ((rec.data as { completed?: unknown }).completed === true) {
        return true
      }
    }
  }
  return false
}

/** One active worklist per agent session and user turn ("one turn, one list"). */
class WorklistStore {
  private readonly bySession = new WeakMap<object, { turn: number; items: WorklistItem[] }>()
  private readonly hydrated = new WeakSet<object>()
  private fallback: WorklistItem[] = []

  get(session: SessionLike | undefined): { turn: number; items: WorklistItem[] } {
    if (session === undefined) return { turn: 1, items: this.fallback }
    const events = session.snapshotEvents?.()
    const currentTurn = turnIndexOf(events)
    if (!this.hydrated.has(session)) {
      this.hydrated.add(session)
      let items: WorklistItem[] = []
      if (events !== undefined) {
        let logTurn = 0
        for (const event of events) {
          if (event.type === 'turn/start') {
            logTurn++
            items = []
            continue
          }
          const rec = recordOf(event)
          if (rec?.type !== STATE_RECORD || typeof rec.data !== 'object' || rec.data === null) continue
          const decoded = decodeItems((rec.data as { items?: unknown }).items)
          const recTurn = (rec.data as { turn?: unknown }).turn
          if (decoded !== undefined && (typeof recTurn !== 'number' || recTurn === Math.max(1, logTurn))) {
            items = decoded
          }
        }
      }
      const entry = { turn: currentTurn, items }
      this.bySession.set(session, entry)
      return entry
    }
    const existing = this.bySession.get(session)
    if (existing === undefined || existing.turn !== currentTurn) {
      const fresh = { turn: currentTurn, items: [] as WorklistItem[] }
      this.bySession.set(session, fresh)
      return fresh
    }
    return existing
  }

  save(session: SessionLike | undefined, turn: number, items: WorklistItem[], completed = false): void {
    if (session === undefined) {
      this.fallback = items
      return
    }
    this.bySession.set(session, { turn, items })
    const payload: PersistedState = { turn, items, ...(completed ? { completed: true } : {}) }
    try {
      appendRecord(session, STATE_RECORD, payload)
    } catch {
      // Outside a live DSH session store (unit tests): keep in memory.
    }
    // Mirror onto DSH's native `todo/write` projection so the Desktop TodoPanel
    // renders clean single-line checklist titles (`[id] title`, without stuffing
    // multi-line technical notes into the compact dock row).
    try {
      session.append?.('todo/write', { todos: projectNativeTodos(items) })
    } catch {
      // Ignore if session does not accept todo/write in minimal test harnesses.
    }
  }

  /**
   * Settle any lingering `in_progress` indicator in the Desktop `TodoPanel` once
   * `turn/end` commits so an unfinished item does not keep spinning while idle.
   */
  settleTurnEnd(session: SessionLike | undefined): void {
    if (session === undefined) return
    const { items } = this.get(session)
    if (!items.some(item => item.status === 'in_progress')) return
    try {
      session.append?.('todo/write', { todos: projectNativeTodos(items, { turnEnded: true }) })
    } catch {
      // Ignore if session rejects post-turn UI projection append.
    }
  }
}

const STATUS_CN: Record<string, string> = {
  completed: '已完成',
  in_progress: '记录进展',
  dropped: '已跳过',
}

export function apply(ctx: Context): void {
  const host = ctx as unknown as ContextLike
  const store = new WorklistStore()

  host.on?.('session/event', (session, ev) => {
    if (ev?.type === 'turn/end') store.settleTurnEnd(session)
  })

  host.tools.register(defineTool({
    name: TODO_PLAN_TOOL,
    description: TODO_PLAN_DESCRIPTION,
    parameters: {
      items: {
        type: 'array',
        required: true,
        description: 'Ordered list of concrete subgoals (2-10 short action lines, without numbers). Mid-turn calls keep completed/dropped items and their notes and replace only unfinished items.',
        items: { type: 'string' },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          summary: { type: 'string', required: true },
        },
      },
      render: (_args: unknown, value: { summary: string }) => [{ type: 'text', text: value.summary }],
    },
    execute(args: { items: string[] }, exec: ExecLike) {
      const session = exec.agent?.session
      const current = store.get(session)
      const updated = applyTodoPlan(current.items, args.items)
      if (typeof updated === 'string') throw new ToolArgsError([updated])
      store.save(session, current.turn, updated)
      return Promise.resolve({ summary: formatWorklist(updated) })
    },
    presentCall: (args: { items?: unknown }) => ({
      card: 'generic',
      title: `规划工单（${Array.isArray(args.items) ? args.items.length : 0} 项）`,
      kind: 'other',
      rawInput: args.items,
    }),
  } as never))

  host.tools.register(defineTool({
    name: TODO_UPDATE_TOOL,
    description: TODO_UPDATE_DESCRIPTION,
    parameters: {
      id: {
        type: 'number',
        required: true,
        description: '1-based item number from the current todo list.',
      },
      status: {
        type: 'string',
        required: true,
        enum: ['in_progress', 'completed', 'dropped'],
        description: 'in_progress | completed | dropped. Marking completed or dropped automatically sets the next pending item to in_progress.',
      },
      note: {
        type: 'string',
        description: 'Concrete findings, exact file paths/lines, applied edits, or test outcomes. Required when status is completed or dropped; also used on in_progress to checkpoint progress.',
      },
      content: {
        type: 'string',
        description: 'Optional refined title for this item if the initial wording was inaccurate.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          summary: { type: 'string', required: true },
        },
      },
      render: (_args: unknown, value: { summary: string }) => [{ type: 'text', text: value.summary }],
    },
    execute(args: { id: number; status: string; note?: string; content?: string }, exec: ExecLike) {
      const session = exec.agent?.session
      const statusNorm = normalizeWorklistStatus(args.status)
      if (statusNorm === 'completed' && blockedBackToBackComplete(session?.snapshotEvents?.())) {
        throw new ToolArgsError([
          'Cannot mark another item completed immediately after a previous completion with no workspace tool call in between. Execute or verify this item with a workspace tool first (read, edit_lines, write, or bash), or use status="in_progress" / "dropped".',
        ])
      }
      const current = store.get(session)
      const updated = applyTodoUpdate(current.items, args)
      if (typeof updated === 'string') throw new ToolArgsError([updated])
      store.save(session, current.turn, updated.items, statusNorm === 'completed')
      return Promise.resolve({ summary: formatWorklist(updated.items) })
    },
    presentCall: (args: Record<string, unknown>) => {
      const st = typeof args.status === 'string' ? (STATUS_CN[args.status] ?? args.status) : '更新'
      return {
        card: 'generic',
        title: `更新工单 #${String(args.id ?? '?')} · ${st}`,
        kind: 'other',
        rawInput: args,
      }
    },
  } as never))
}
