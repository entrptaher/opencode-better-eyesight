import { describe, expect, test } from "bun:test"
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { cacheKey, ResponseCache, type CacheEntry } from "../src/cache.ts"
import { extForMime, sniffMime } from "../src/image.ts"
import {
  buildInjectedPart,
  buildVisionMessage,
  detailDirective,
  distillUserTurns,
  effectiveModel,
  parseModel,
  parseOptions,
} from "../src/vision.ts"

describe("parseModel", () => {
  test("splits on the first slash only (openrouter model ids contain slashes)", () => {
    const m = parseModel("openrouter/dots-studio/dots-3-note-preview:free")
    expect(m).toEqual({
      providerID: "openrouter",
      modelID: "dots-studio/dots-3-note-preview:free",
      raw: "openrouter/dots-studio/dots-3-note-preview:free",
    })
  })

  test("rejects strings without a provider", () => {
    expect(parseModel("noprovider")).toBeUndefined()
    expect(parseModel("/leading")).toBeUndefined()
    expect(parseModel("trailing/")).toBeUndefined()
  })
})

describe("effectiveModel (per-call override)", () => {
  const configured = parseModel("openrouter/configured-model")

  test("falls back to the configured model when nothing is requested", () => {
    expect(effectiveModel(configured).model).toBe(configured)
    expect(effectiveModel(configured, "   ").model).toBe(configured)
    expect(effectiveModel(undefined, "  ").model).toBeUndefined()
  })

  test("a requested model wins over the configured default", () => {
    expect(effectiveModel(configured, "anthropic/claude-sonnet-4-5").model).toEqual(parseModel("anthropic/claude-sonnet-4-5"))
    expect(effectiveModel(undefined, "openrouter/a/b:free").model?.modelID).toBe("a/b:free")
  })

  test("an invalid requested model reports an error and keeps the default", () => {
    const r = effectiveModel(configured, "broken")
    expect(r.error).toContain('"broken"')
    expect(r.model).toBe(configured)
  })
})

describe("parseOptions", () => {
  const clean = { EYESIGHT_MODEL: undefined, EYESIGHT_DISABLE: undefined, EYESIGHT_AUTO_ANALYZE: undefined, EYESIGHT_CACHE_TTL_HOURS: undefined, EYESIGHT_CACHE_DIR: undefined, EYESIGHT_HISTORY: undefined }

  test("defaults", () => {
    const o = parseOptions({}, clean)
    expect(o.enabled).toBe(true)
    expect(o.autoAnalyze).toBe(true)
    expect(o.history).toBe(false)
    expect(o.agentName).toBe("eyesight")
    expect(o.cacheTtlMs).toBe(168 * 3_600_000)
    expect(o.model).toBeUndefined()
    expect(o.modelError).toBeUndefined()
  })

  test("model option and invalid model error", () => {
    expect(parseOptions({ model: "openrouter/a/b:free" }, clean).model?.modelID).toBe("a/b:free")
    expect(parseOptions({ model: "broken" }, clean).modelError).toBe("broken")
  })

  test("env overrides", () => {
    expect(parseOptions({}, { ...clean, EYESIGHT_MODEL: "anthropic/claude-sonnet-4-5" }).model?.providerID).toBe("anthropic")
    expect(parseOptions({}, { ...clean, EYESIGHT_AUTO_ANALYZE: "0" }).autoAnalyze).toBe(false)
    expect(parseOptions({}, { ...clean, EYESIGHT_CACHE_TTL_HOURS: "0" }).cacheTtlMs).toBe(0)
    expect(parseOptions({}, { ...clean, EYESIGHT_HISTORY: "1" }).history).toBe(true)
    expect(parseOptions({ history: true }, clean).history).toBe(true)
  })

  test("cacheTtlHours 0 disables the cache", () => {
    const cache = new ResponseCache({ dir: join(mkdtempSync(join(tmpdir(), "eyesight-"))), ttlMs: 0 })
    expect(cache.enabled).toBe(false)
  })
})

describe("cache", () => {
  test("cacheKey is deterministic and separates inputs", () => {
    expect(cacheKey(["m", "q", "h"])).toBe(cacheKey(["m", "q", "h"]))
    expect(cacheKey(["m", "q", "h"])).not.toBe(cacheKey(["m", "q2", "h"]))
  })

  test("set/get roundtrip, ttl expiry, disabled", async () => {
    const dir = mkdtempSync(join(tmpdir(), "eyesight-"))
    let now = 1_000_000
    const cache = new ResponseCache({ dir, ttlMs: 3_600_000, now: () => now })
    const entry: CacheEntry = {
      v: 1,
      answer: "a button says Submit",
      model: "openrouter/x",
      question: "what does the button say",
      imageHash: "abc",
      contextHash: "def",
      detail: "normal",
      created: now,
    }
    await cache.set("k", entry)
    expect((await cache.get("k"))?.answer).toBe("a button says Submit")

    now += 3_600_001
    expect(await cache.get("k")).toBeNull()

    const disabled = new ResponseCache({ dir, ttlMs: 0 })
    await disabled.set("k2", entry)
    expect(await disabled.get("k2")).toBeNull()
  })
})

describe("mime sniffing", () => {
  test("magic bytes", () => {
    expect(sniffMime(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0]))).toBe("image/png")
    expect(sniffMime(Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 16, 74, 70, 73, 70, 0, 1]))).toBe("image/jpeg")
    expect(sniffMime(Buffer.from("RIFF\u0000\u0000\u0000\u0000WEBPVP8 ", "latin1"))).toBe("image/webp")
    expect(sniffMime(Buffer.alloc(32, 7))).toBeNull()
    expect(sniffMime(Buffer.alloc(8))).toBeNull()
  })

  test("ext mapping", () => {
    expect(extForMime("image/png")).toBe("png")
    expect(extForMime("image/jpeg")).toBe("jpg")
    expect(extForMime("image/unknown")).toBeNull()
  })
})

describe("parent prompt distillation", () => {
  const messages = [
    { info: { role: "user" }, parts: [{ type: "text", text: "hey" }] },
    { info: { role: "assistant" }, parts: [{ type: "text", text: "hello" }] },
    {
      info: { role: "user" },
      parts: [
        { type: "text", text: "real ask", synthetic: false },
        { type: "text", text: "synthetic noise", synthetic: true },
      ],
    },
  ] as never

  test("keeps user text only, drops synthetic parts, joins turns", () => {
    expect(distillUserTurns(messages, 4, 1000)).toBe("hey\n---\nreal ask")
  })

  test("caps to the most recent characters", () => {
    const out = distillUserTurns(messages, 4, 8)
    expect(out.startsWith("\u2026")).toBe(true)
    expect(out.endsWith("real ask")).toBe(true)
  })

  test("respects the message cap", () => {
    expect(distillUserTurns(messages, 1, 1000)).toBe("real ask")
  })

  test("strips injected <eyesight> pointer blocks from forwarded context", () => {
    const withBlock = [
      {
        info: { role: "user" },
        parts: [{ type: "text", text: "what is this? [Image 1]\n<eyesight>\nnoise\nmore noise\n</eyesight>" }],
      },
    ] as never
    expect(distillUserTurns(withBlock, 4, 1000)).toBe("what is this? [Image 1]")
  })
})

describe("vision message and injected part", () => {
  test("buildVisionMessage carries question, context and injection guardrails", () => {
    const text = buildVisionMessage({ question: "why did the login fail?", contextText: "user: my login screen errors", detail: "brief" })
    expect(text).toContain("<question>\nwhy did the login fail?")
    expect(text).toContain("<user_conversation>\nuser: my login screen errors")
    expect(text).toContain("untrusted data")
    expect(text).toContain(detailDirective("brief"))
  })

  test("injected part matches the shape opencode accepts", () => {
    const part = buildInjectedPart("msg_1", "ses_1", "<eyesight>hi</eyesight>") as { id: string; sessionID: string; messageID: string; type: string }
    expect(part.id.startsWith("prt_")).toBe(true)
    expect(part.id.length).toBeLessThanOrEqual(30)
    expect(part.sessionID).toBe("ses_1")
    expect(part.messageID).toBe("msg_1")
    expect(part.type).toBe("text")
  })
})
