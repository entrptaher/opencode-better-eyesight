import type { Config, Hooks, Plugin, PluginOptions, ToolResult } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
import type { AgentConfig } from "@opencode-ai/sdk"
import { EyesightEngine, parseOptions, AGENT_PROMPT } from "./src/vision.ts"

const z = tool.schema

export const EyesightPlugin: Plugin = async (input, options) => {
  const opts = parseOptions((options ?? {}) as PluginOptions)
  if (!opts.enabled) return {}

  const engine = new EyesightEngine(input, opts)

  const hooks: Hooks = {
    config: async (config: Config) => {
      // Register a tool-less vision agent so sub-calls can never recurse into tools.
      // Never overwrite a user-defined agent with the same name.
      if (config.agent?.[opts.agentName]) return
      const agent = {
        description: "Vision analyst - answers questions about images for agents that cannot see them.",
        mode: "subagent",
        prompt: AGENT_PROMPT,
        tools: { "*": false },
        permission: { "*": "deny" },
        maxSteps: 2,
      } as AgentConfig
      if (opts.model) agent.model = opts.model.raw
      config.agent = { ...config.agent, [opts.agentName]: agent }
    },

    "chat.message": async (hookInput, output) => {
      if (!opts.autoAnalyze) return
      try {
        await engine.injectAnalysis(hookInput, output)
      } catch (err) {
        // Auto-analysis must never break the chat pipeline; the eyesight tool remains available.
        const app = (input.client as { app?: { log?: (body: unknown) => Promise<unknown> } }).app
        void app?.log?.({ service: "eyesight", level: "warn", message: `auto-analyze failed: ${(err as Error).message}` }).catch(() => {})
      }
    },

    tool: {
      eyesight: tool({
        description:
          "Ask a dedicated vision model (routed through opencode, so any provider/model opencode can use) about an image. " +
          "Use this whenever you need to see, read, or understand an image: screenshots, UI mocks, photos, diagrams, charts, scanned text. " +
          "The user's current request in this conversation is forwarded to the vision model automatically when no explicit question is given. " +
          "Each call is normally stateless: one image, one question, one answer. " +
          "An optional model argument ('provider/model-id') overrides the vision model for a call — use it when the user asks for a specific model.",
        args: {
          source: z
            .string()
            .describe(
              'Where the image is: "attached" (easiest — the most recent image in this conversation), "clipboard", a file path (e.g. a path from an <eyesight> block or an attached image\'s filename), an https URL, or a data: URL.',
            ),
          question: z
            .string()
            .optional()
            .describe(
              "What to ask about the image. Optional: when omitted, the user's current request from this conversation is forwarded as context and the vision model answers in that context.",
            ),
          model: z
            .string()
            .optional()
            .describe(
              "Optional vision model override for this call, 'provider/model-id' format (e.g. 'openrouter/google/gemini-2.5-flash:free', 'anthropic/claude-sonnet-4-5'). Use when the user asks for a specific model; otherwise the plugin's configured default is used.",
            ),
          detail: z
            .enum(["brief", "normal", "exhaustive"])
            .optional()
            .describe("Answer depth. Default 'normal'; use 'exhaustive' to transcribe all text and enumerate every element."),
          fresh: z
            .boolean()
            .optional()
            .describe(
              "Bypass the cache and ask again. Only use when a previous answer for the same image and question seems stale or wrong; identical asks are served instantly from cache otherwise.",
            ),
        },
        async execute(args, context): Promise<ToolResult> {
          const modelName = opts.model?.raw ?? "opencode default model"
          context.metadata({ title: `eyesight: resolving ${args.source}${args.model ? ` (${args.model})` : ""}` })
          try {
            const result = await engine.analyze({
              parentSessionID: context.sessionID,
              source: args.source,
              question: args.question,
              detail: args.detail,
              fresh: args.fresh,
              model: args.model,
              signal: context.abort,
            })
            const label = `eyesight: ${result.model}${result.cached ? " (cached)" : ""}`
            context.metadata({ title: label })
            return {
              title: label,
              output: result.answer,
              metadata: {
                cached: result.cached,
                model: result.model,
                source: result.image.sourceKind,
                image: result.image.path,
                imageHash: result.image.hash,
                visionSession: result.childSessionID,
              },
              attachments: [
                {
                  type: "file",
                  mime: result.image.mime,
                  url: result.image.path,
                  filename: result.image.filename,
                },
              ],
            }
          } catch (err) {
            return {
              title: "eyesight: failed",
              output:
                `eyesight failed: ${(err as Error).message}\n\n` +
                `If this is a model problem, check the "model" option on the opencode-better-eyesight plugin (currently: ${opts.modelError ? `invalid: ${opts.modelError}` : modelName}) and that the provider is authenticated in opencode. ` +
                `If the same request fails twice, stop retrying and report the error to the user.`,
              metadata: { error: true, model: opts.model?.raw },
            }
          }
        },
      }),
    },

    dispose: async () => {
      engine.dispose()
    },
  }

  return hooks
}

export default EyesightPlugin
