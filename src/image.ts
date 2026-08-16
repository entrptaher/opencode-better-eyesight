import { createHash } from "node:crypto"
import { existsSync } from "node:fs"
import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { execFile } from "node:child_process"
import { homedir } from "node:os"
import { join, dirname, basename, extname, resolve } from "node:path"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

export type ResolvedImage = {
  mime: string
  hash: string
  /** Stable on-disk path (original for file sources, cache dir otherwise). */
  path: string
  /** data: URL built from the raw bytes — what gets sent to the vision model. */
  dataUrl: string
  bytes: number
  sourceKind: "clipboard" | "file" | "url" | "data" | "session"
  filename: string
}

export type ImageDeps = {
  worktree: string
  cacheDir: string
  fetchTimeoutMs: number
  maxImageBytes: number
}

const MAGIC: Array<{ mime: string; test(b: Buffer): boolean }> = [
  {
    mime: "image/png",
    test: (b) => b.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])),
  },
  { mime: "image/jpeg", test: (b) => b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff },
  { mime: "image/gif", test: (b) => b.subarray(0, 4).toString("latin1") === "GIF8" },
  {
    mime: "image/webp",
    test: (b) => b.subarray(0, 4).toString("latin1") === "RIFF" && b.subarray(8, 12).toString("latin1") === "WEBP",
  },
  { mime: "image/bmp", test: (b) => b.subarray(0, 2).toString("latin1") === "BM" },
  {
    mime: "image/avif",
    test: (b) => b.subarray(4, 8).toString("latin1") === "ftyp" && b.subarray(8, 16).toString("latin1").includes("avif"),
  },
]

export function sniffMime(bytes: Buffer): string | null {
  if (bytes.length < 12) return null
  for (const m of MAGIC) {
    try {
      if (m.test(bytes)) return m.mime
    } catch {
      // fall through
    }
  }
  return null
}

const EXT_FOR_MIME: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/bmp": "bmp",
  "image/avif": "avif",
}

export function extForMime(mime: string): string | null {
  return EXT_FOR_MIME[mime] ?? null
}

function mimeForExt(name: string): string | undefined {
  const ext = extname(name).toLowerCase().slice(1)
  return (
    Object.entries(EXT_FOR_MIME).find(([, e]) => e === ext || (ext === "jpeg" && e === "jpg"))?.[0] ?? undefined
  )
}

export function isImageFilePart(part: unknown): boolean {
  const p = part as { type?: string; mime?: string } | undefined
  return p?.type === "file" && typeof p.mime === "string" && p.mime.startsWith("image/")
}

async function readClipboard(cacheDir: string, maxImageBytes: number): Promise<Buffer> {
  if (process.platform === "darwin") {
    const out = join(cacheDir, `clipboard-${Date.now()}.png`)
    // «class PNGf» is the only clipboard image class most vision inputs need; write raw bytes via AppleScript.
    const script = [
      "on run",
      "  set pngData to (the clipboard as \u00ABclass PNGf\u00BB)",
      `  set fp to open for access POSIX file "${out}" with write permission`,
      "  set eof of fp to 0",
      "  write pngData to fp",
      "  close access fp",
      "end run",
    ].join("\n")
    await mkdir(cacheDir, { recursive: true })
    try {
      await execFileAsync("osascript", ["-e", script])
      const bytes = await readFile(out)
      await rm(out, { force: true })
      if (bytes.length > 0) return bytes
    } catch {
      // fall through to the friendly error below
    }
    throw new Error(
      "No image on the macOS clipboard. Copy an image first (e.g. ctrl+shift+cmd+4 for a screenshot into the clipboard), or pass a file path / URL as the source.",
    )
  }
  if (process.platform === "linux") {
    const tries: Array<[string, string[]]> = [
      ["wl-paste", ["--type", "image/png"]],
      ["xclip", ["-selection", "clipboard", "-t", "image/png", "-o"]],
    ]
    for (const [bin, args] of tries) {
      try {
        const { stdout } = await execFileAsync(bin, args, {
          encoding: "buffer",
          maxBuffer: maxImageBytes + 1024,
        })
        if (stdout.length > 0) return Buffer.from(stdout)
      } catch {
        // try the next tool
      }
    }
    throw new Error("No image on the clipboard (tried wl-paste and xclip). Copy an image first or pass a file path / URL.")
  }
  // ponytail: no Windows clipboard reader; pass a file path or https URL instead.
  throw new Error(`Clipboard capture is not supported on ${process.platform}. Pass a file path or https URL as the source.`)
}

function parseDataUrl(url: string): { bytes: Buffer; mimeHint?: string } {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url)
  if (!m) throw new Error("Invalid data: URL (expected data:<mime>;base64,<payload>)")
  const [, mime, isBase64, payload] = m
  const bytes = isBase64 ? Buffer.from(payload, "base64") : Buffer.from(decodeURIComponent(payload), "utf8")
  return { bytes, mimeHint: mime || undefined }
}

async function fetchImage(url: string, deps: ImageDeps): Promise<{ bytes: Buffer; mimeHint?: string; filenameHint?: string }> {
  const res = await fetch(url, { signal: AbortSignal.timeout(deps.fetchTimeoutMs) })
  if (!res.ok) throw new Error(`Fetching ${url} failed with HTTP ${res.status}`)
  const declared = Number(res.headers.get("content-length") ?? 0)
  if (declared && declared > deps.maxImageBytes) {
    throw new Error(`Image at ${url} is ${(declared / 1048576).toFixed(1)}MB, over the ${(deps.maxImageBytes / 1048576).toFixed(0)}MB limit`)
  }
  const mimeHint = (res.headers.get("content-type") ?? "").split(";")[0].trim() || undefined
  let pathname: string | undefined
  try {
    pathname = decodeURIComponent(new URL(url).pathname)
  } catch {
    pathname = undefined
  }
  return {
    bytes: Buffer.from(await res.arrayBuffer()),
    mimeHint,
    filenameHint: pathname ? basename(pathname) : undefined,
  }
}

/**
 * Resolve any accepted source into raw bytes plus stable metadata.
 * Sources: "clipboard", data: URL, http(s) URL, or a file path (relative paths resolve against the worktree).
 */
export async function resolveImageSource(source: string, deps: ImageDeps): Promise<ResolvedImage> {
  const s = source.trim()
  let bytes: Buffer
  let kind: ResolvedImage["sourceKind"]
  let mimeHint: string | undefined
  let filenameHint: string | undefined
  let originalPath: string | undefined

  if (/^(clipboard|clip|paste)$/i.test(s)) {
    kind = "clipboard"
    bytes = await readClipboard(deps.cacheDir, deps.maxImageBytes)
    filenameHint = `clipboard-${new Date().toISOString().replace(/[:.]/g, "-")}.png`
  } else if (/^data:/i.test(s)) {
    kind = "data"
    ;({ bytes, mimeHint } = parseDataUrl(s))
  } else if (/^https?:\/\//i.test(s)) {
    kind = "url"
    ;({ bytes, mimeHint, filenameHint } = await fetchImage(s, deps))
  } else {
    kind = "file"
    originalPath = resolve(deps.worktree, s)
    try {
      bytes = await readFile(originalPath)
    } catch (err) {
      throw new Error(`Cannot read image file "${originalPath}": ${(err as Error).message}`)
    }
    filenameHint = basename(originalPath)
  }

  if (bytes.length === 0) throw new Error(`Image source "${s}" is empty`)
  // ponytail: size is checked after download rather than streamed; upgrade path is a capped stream reader.
  if (bytes.length > deps.maxImageBytes) {
    throw new Error(
      `Image is ${(bytes.length / 1048576).toFixed(1)}MB, over the ${(deps.maxImageBytes / 1048576).toFixed(0)}MB limit (option "maxImageBytes")`,
    )
  }

  const mime = sniffMime(bytes) ?? mimeHint ?? mimeForExt(originalPath ?? filenameHint ?? "")
  if (!mime || !mime.startsWith("image/")) {
    throw new Error(`"${s}" does not look like a supported image (detected ${mime ?? "unknown"}). Supported: png, jpeg, webp, gif, bmp, avif.`)
  }

  const hash = createHash("sha256").update(bytes).digest("hex")
  const ext = extForMime(mime) ?? "png"
  let path: string
  if (kind === "file" && originalPath) {
    path = originalPath
  } else {
    path = join(deps.cacheDir, "images", `${hash}.${ext}`)
    await mkdir(dirname(path), { recursive: true })
    if (!existsSync(path)) await writeFile(path, bytes)
  }

  return {
    mime,
    hash,
    path,
    dataUrl: `data:${mime};base64,${bytes.toString("base64")}`,
    bytes: bytes.length,
    sourceKind: kind,
    filename: filenameHint ?? basename(path),
  }
}

export function defaultCacheDir(): string {
  return join(homedir(), ".cache", "opencode-better-eyesight")
}

export type InjectionRef = { ref: string; mime: string; kind: "local" | "remote"; filename: string }

/**
 * Fast, local-only preparation for the chat.message hook: materializes data-URL pastes
 * to stable cache paths and verifies local files, but never fetches remote URLs and
 * never calls a model. Remote URLs are passed through untouched for the tool to fetch.
 */
export async function resolveForInjection(url: string, deps: ImageDeps): Promise<InjectionRef | null> {
  try {
    if (/^data:/i.test(url)) {
      const { bytes, mimeHint } = parseDataUrl(url)
      if (bytes.length === 0 || bytes.length > deps.maxImageBytes) return null
      const mime = sniffMime(bytes) ?? mimeHint
      if (!mime || !mime.startsWith("image/")) return null
      const hash = createHash("sha256").update(bytes).digest("hex")
      const ext = extForMime(mime) ?? "png"
      const path = join(deps.cacheDir, "images", `${hash}.${ext}`)
      await mkdir(dirname(path), { recursive: true })
      if (!existsSync(path)) await writeFile(path, bytes)
      return { ref: path, mime, kind: "local", filename: `${hash.slice(0, 12)}.${ext}` }
    }
    if (/^https?:\/\//i.test(url)) {
      let name = "image"
      try {
        name = basename(decodeURIComponent(new URL(url).pathname)) || "image"
      } catch {
        // keep the fallback name
      }
      return { ref: url, mime: "remote image", kind: "remote", filename: name }
    }
    const path = resolve(deps.worktree, url)
    if (!existsSync(path)) return null
    return { ref: path, mime: mimeForExt(path) ?? "image/*", kind: "local", filename: basename(path) }
  } catch {
    return null
  }
}
