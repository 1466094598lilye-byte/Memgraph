/**
 * MemGraph Plugin for OpenClaw — Embedding-driven recall
 *
 * Behavior:
 *   - Auto ingest: after each LLM output, POST user+assistant to /ingest
 *   - Smart recall: every turn, check if query embedding points to turns
 *     not in current context. Only recall when needed (compressed/lost turns).
 *   - Context tracking: maintain set of turn_ids in current context window.
 *     Clear on session_start and after_compaction.
 *   - Manual recall: memgraph_recall tool always available.
 *
 * Key insight: embedding similarity is free (local model). LLM recall only
 * fires when matches point to compressed/lost content. No wasted recalls.
 */

const MEMGRAPH_URL = process.env.MEMGRAPH_URL || "http://localhost:18821";

// ── State ──
// Track turn_ids that are in the current context window per session
const contextTurnIds = new Map<string, Set<number>>();
// Cache user prompt from llm_input to pair with llm_output
const pendingPrompt = new Map<string, string>();

function getContextSet(sessionKey: string): Set<number> {
  if (!contextTurnIds.has(sessionKey)) {
    contextTurnIds.set(sessionKey, new Set());
  }
  return contextTurnIds.get(sessionKey)!;
}

// ── HTTP helper ──

async function memgraphFetch(
  path: string,
  body?: any,
  timeoutMs = 10000,
): Promise<any> {
  const url = `${MEMGRAPH_URL}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!resp.ok) {
      throw new Error(`MemGraph ${resp.status} ${resp.statusText}`);
    }
    return resp.json();
  } finally {
    clearTimeout(timer);
  }
}

async function isServerUp(): Promise<boolean> {
  try {
    await memgraphFetch("/health", undefined, 2000);
    return true;
  } catch {
    return false;
  }
}

// ── Plugin ──

export default function register(api: any) {
  const log = api.logger;

  // ── Tool: memgraph_recall (manual, always available) ──

  api.registerTool(
    (ctx: any) => {
      return {
        name: "memgraph_recall",
        description:
          "Query long-term memory from previous sessions. Use to recall facts, preferences, decisions, or conversations from past sessions no longer in context. Requires MemGraph server running.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description:
                "Natural language query for what to recall from past sessions.",
            },
            top_k: {
              type: "number",
              description: "Number of relevant turns to retrieve (default: 10)",
              minimum: 1,
              maximum: 50,
            },
          },
          required: ["query"],
        },
        async handler(params: { query: string; top_k?: number }) {
          try {
            const result = await memgraphFetch("/recall", {
              query: params.query,
              top_k: params.top_k ?? 10,
            });
            return {
              content: [
                {
                  type: "text" as const,
                  text: result.result_text || "(no memories found)",
                },
              ],
            };
          } catch (err: any) {
            const msg = err?.message || String(err);
            if (msg.includes("ECONNREFUSED") || msg.includes("fetch failed") || msg.includes("aborted")) {
              return {
                content: [
                  {
                    type: "text" as const,
                    text: "MemGraph server is not running. Start it with: cd /root/memgraph && python3 -m memgraph.server",
                  },
                ],
              };
            }
            return {
              content: [{ type: "text" as const, text: `MemGraph error: ${msg}` }],
            };
          }
        },
      };
    },
    { names: ["memgraph_recall"], optional: true },
  );

  // ── Hook: cache user prompt from llm_input ──

  api.on("llm_input", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || event.sessionId || "default";
    if (event.prompt) {
      pendingPrompt.set(key, event.prompt);
    }
  });

  // ── Hook: auto ingest after each LLM output ──

  api.on("llm_output", async (event: any, ctx: any) => {
    if (!(await isServerUp())) return;

    // Get the user prompt cached from llm_input
    const key = ctx.sessionKey || event.sessionId || "default";
    const userText = pendingPrompt.get(key) || "";
    pendingPrompt.delete(key);

    const assistantTexts: string[] = event.assistantTexts || [];
    const assistantText = assistantTexts.join("\n");

    if (!userText && !assistantText) return;
    if (assistantTexts.every((t: string) => !t.trim())) return;
    // Also skip if user text is empty — can't extract meaningful memo
    if (!userText.trim()) return;

    try {
      const result = await memgraphFetch("/ingest", {
        user_text: userText || "",
        assistant_text: assistantText || "",
        session_id: ctx.sessionKey || ctx.sessionId,
      });

      // Track the new turn_id in context
      if (result.total_turns) {
        const key = ctx.sessionKey || "default";
        const turnId = result.total_turns - 1; // 0-indexed
        getContextSet(key).add(turnId);
      }

      log.info(`[memgraph] ingested turn, total=${result.total_turns}`);
    } catch (err: any) {
      log.warn(`[memgraph] ingest failed: ${err?.message || err}`);
    }
  });

  // ── Hook: clear context window on new session ──

  api.on("session_start", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || event.sessionKey || "default";
    contextTurnIds.set(key, new Set());
    log.info(`[memgraph] session_start → cleared context window for ${key}`);
  });

  // ── Hook: clear context window after compaction ──

  api.on("after_compaction", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || "default";
    contextTurnIds.set(key, new Set());
    log.info(`[memgraph] after_compaction → cleared context window for ${key}`);
  });

  // ── Hook: embedding-driven smart recall ──
  // Every turn: check if query points to turns not in context.
  // Only inject recall when there are missing matches.

  api.on("before_prompt_build", async (event: any, ctx: any) => {
    if (!(await isServerUp())) return;

    const key = ctx.sessionKey || "default";
    const query = event.prompt || "";
    if (!query) return;

    const ctxSet = getContextSet(key);

    try {
      // Ask server: does this query point to turns outside our context?
      const check = await memgraphFetch("/check_context", {
        query,
        context_turn_ids: Array.from(ctxSet),
        top_k: 5,
      });

      if (!check.needs_recall) {
        // All top matches are in context, no recall needed
        return;
      }

      log.info(
        `[memgraph] Smart recall triggered: ${check.reason}`,
      );

      // Recall needed — fetch full recall result
      const result = await memgraphFetch("/recall", {
        query,
        top_k: 10,
      });

      const text = result.result_text;
      if (!text || text === "(no memories found)") return;

      log.info(
        `[memgraph] Injected ${text.length} chars of long-term memory`,
      );

      return {
        prependContext: `<memgraph_long_term_memory>\n${text}\n</memgraph_long_term_memory>`,
      };
    } catch (err: any) {
      log.warn(`[memgraph] smart recall check failed: ${err?.message || err}`);
    }
  });
}
