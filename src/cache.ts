import { createHash } from "node:crypto"
import { mkdir, readFile, rename, writeFile } from "node:fs/promises"
import { join } from "node:path"

export function cacheKey(parts: string[]): string {
  return createHash("sha256").update(parts.join("\u0000")).digest("hex")
}

export type CacheEntry = {
  v: 1
  answer: string
  model: string
  question: string
  imageHash: string
  contextHash: string
  detail: string
  created: number
}

/**
 * Content-addressed response cache. Memory first, then a JSON file per key.
 * `ttlMs <= 0` disables both read and write.
 */
export class ResponseCache {
  private mem = new Map<string, CacheEntry>()
  readonly enabled: boolean

  constructor(
    private cfg: { dir: string; ttlMs: number; now?: () => number },
  ) {
    this.enabled = cfg.ttlMs > 0
  }

  private now() {
    return this.cfg.now?.() ?? Date.now()
  }

  async get(key: string): Promise<CacheEntry | null> {
    if (!this.enabled) return null
    const mem = this.mem.get(key)
    if (mem) return this.fresh(mem) ? mem : null
    try {
      const entry = JSON.parse(await readFile(this.fileFor(key), "utf8")) as CacheEntry
      this.mem.set(key, entry)
      return this.fresh(entry) ? entry : null
    } catch {
      return null
    }
  }

  async set(key: string, entry: CacheEntry): Promise<void> {
    if (!this.enabled) return
    this.mem.set(key, entry)
    if (this.mem.size > 128) this.mem.delete(this.mem.keys().next().value as string)
    const file = this.fileFor(key)
    const tmp = `${file}.tmp`
    try {
      await mkdir(this.cfg.dir, { recursive: true })
      await writeFile(tmp, JSON.stringify(entry))
      await rename(tmp, file)
    } catch {
      // A failed cache write must never fail the vision call itself.
    }
  }

  private fresh(entry: CacheEntry) {
    return this.now() - entry.created < this.cfg.ttlMs
  }

  private fileFor(key: string) {
    return join(this.cfg.dir, `${key}.json`)
  }
}
