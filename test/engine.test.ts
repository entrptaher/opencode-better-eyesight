import { describe, expect, test } from "bun:test"
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { EyesightEngine, parseOptions } from "../src/vision.ts"

// 1x1 PNGs with distinct bytes so image hashes differ
const PNG_A =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
const PNG_B =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

function stubClient(messagesData: Array<{ info: { role: string }; parts: Array<Record<string, string>> }> = []) {
  const prompts: Array<{ path: { id: string }; body: { parts: Array<{ type: string; url?: string }> } }> = []
  const created: string[] = []
  let n = 0
  const client = {
    session: {
      messages: async () => ({ data: messagesData }),
      create: async () => {
        const id = `child_${++n}`
        created.push(id)
        return { data: { id } }
      },
      prompt: async (opts: never) => {
        prompts.push(opts)
        return { data: { parts: [{ type: "text", text: `answer ${prompts.length}` }] } }
      },
      abort: async () => ({}),
    },
  }
  return { client: client as never, prompts, created }
}

function makeEngine(client: never, extra: Record<string, unknown> = {}) {
  const opts = parseOptions({ model: "openrouter/test-model", cacheDir: mkdtempSync(join(tmpdir(), "eyesight-engine-")), ...extra })
  return new EyesightEngine(
    {
      client,
      project: {} as never,
      directory: "/tmp",
      worktree: "/tmp",
      serverUrl: new URL("http://localhost:1"),
      $: {} as never,
      experimental_workspace: {} as never,
    },
    opts,
  )
}

describe("engine vision-call hygiene (regression: runaway 100k-token calls)", () => {
  test("history=false (default): every call is one-shot — fresh session, image re-sent", async () => {
    const { client, prompts, created } = stubClient()
    const engine = makeEngine(client)
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "q1" })
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "q2" })

    expect(created).toEqual(["child_1", "child_2"])
    expect(prompts.every((p) => p.body.parts.some((x) => x.type === "file"))).toBe(true)
    expect(prompts[0].body.parts.some((p) => p.type === "text")).toBe(true)
  })

  test("history=true: image is uploaded once; follow-up questions are text-only in the same child session", async () => {
    const { client, prompts } = stubClient()
    const engine = makeEngine(client, { history: true })
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "q1" })
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "q2" })
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "q3" })

    expect(prompts.length).toBe(3)
    expect(prompts[0].body.parts.some((p) => p.type === "file")).toBe(true)
    expect(prompts[1].body.parts.some((p) => p.type === "file")).toBe(false)
    expect(prompts[2].body.parts.some((p) => p.type === "file")).toBe(false)
    expect(new Set(prompts.map((p) => p.path.id)).size).toBe(1)
  })

  test("a different image gets a fresh child session and re-uploads the image", async () => {
    const { client, prompts, created } = stubClient()
    const engine = makeEngine(client, { history: true })
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "q1" })
    await engine.analyze({ parentSessionID: "p1", source: PNG_B, question: "q1" })

    expect(created).toEqual(["child_1", "child_2"])
    expect(prompts[1].body.parts.some((p) => p.type === "file")).toBe(true)
  })

  test("conversation-context drift no longer defeats the cache (context excluded from key)", async () => {
    const { client, prompts } = stubClient()
    const engine = makeEngine(client)
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "same", contextText: "context A" })
    const second = await engine.analyze({
      parentSessionID: "p1",
      source: PNG_A,
      question: "same",
      contextText: "completely different context B",
    })
    expect(second.cached).toBe(true)
    expect(prompts.length).toBe(1)
  })

  test("identical ask is served from cache without a model call", async () => {
    const { client, prompts } = stubClient()
    const engine = makeEngine(client)
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "same" })
    const second = await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "same" })
    expect(second.cached).toBe(true)
    expect(prompts.length).toBe(1)
  })

  test("a bare attached filename resolves to the session image (models pass display names)", async () => {
    const { client, prompts } = stubClient([
      {
        info: { role: "user" },
        parts: [{ type: "file", mime: "image/png", url: PNG_A, filename: "Screenshot 2026-08-17 at 02-17-08.png" }],
      },
    ])
    const engine = makeEngine(client)
    const result = await engine.analyze({
      parentSessionID: "p1",
      source: "Screenshot 2026-08-17 at 02-17-08.png",
      question: "what icons?",
    })
    expect(result.image.sourceKind).toBe("session")
    expect(prompts[0].body.parts.some((p) => p.type === "file" && p.url === PNG_A)).toBe(true)
  })

  test("a source that matches nothing fails honestly", async () => {
    const { client } = stubClient()
    const engine = makeEngine(client)
    await expect(engine.analyze({ parentSessionID: "p1", source: "does-not-exist.png", question: "q" })).rejects.toThrow()
  })
})

describe("chat.message self-injection guard (regression: pointer blocks inside vision calls)", () => {
  test("never injects into our own vision child session, but still injects into user sessions", async () => {
    const { client, created } = stubClient()
    const engine = makeEngine(client)
    await engine.analyze({ parentSessionID: "p1", source: PNG_A, question: "q" })

    const childOut = { message: { id: "m1", sessionID: created[0] }, parts: [{ type: "file", mime: "image/png", url: PNG_A }] }
    await engine.injectAnalysis({ sessionID: created[0], agent: "eyesight" }, childOut as never)
    expect(childOut.parts.length).toBe(1)

    const parentOut = { message: { id: "m2", sessionID: "p1" }, parts: [{ type: "file", mime: "image/png", url: PNG_A }] }
    await engine.injectAnalysis({ sessionID: "p1" }, parentOut as never)
    expect(parentOut.parts.length).toBe(2)
    const text = (parentOut.parts[1] as { text?: string }).text ?? ""
    expect(text).toContain("<eyesight>")
    expect(text).toContain('source="attached"')
    expect(text).toContain("Do not search the filesystem")
  })
})
