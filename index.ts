import { AnthropicProviderSettings, createAnthropic } from "@ai-sdk/anthropic";
import { DeepInfraProviderSettings, createDeepInfra } from "@ai-sdk/deepinfra";
import { DeepSeekProviderSettings, createDeepSeek } from "@ai-sdk/deepseek";
import { GoogleGenerativeAIProviderSettings, createGoogleGenerativeAI } from "@ai-sdk/google";
import { GroqProviderSettings, createGroq } from "@ai-sdk/groq";
import { OpenAIProviderSettings, createOpenAI } from "@ai-sdk/openai";
import {
  CoreMessage,
  CoreTool,
  CoreUserMessage,
  GenerateObjectResult,
  GenerateTextResult,
  LanguageModelV1,
  Output,
  StreamTextResult,
  generateObject,
  generateText,
  streamText,
  tool,
} from "ai";
import { JigsawStack } from "jigsawstack";
import { z } from "zod";

export interface GeneratePromptObj {
  role: CoreUserMessage["role"];
  content:
    | string
    | {
        type: "text" | "image" | "file";
        data: string;
        mimeType?: string;
      }[];
}

export interface GenerateParams {
  stream?: boolean;
  reasoning?: boolean;
  multiLLM?: boolean;
  system?: string;
  prompt: string | GeneratePromptObj[];
  schema?: z.ZodSchema;
  contextTool?: {
    web?: boolean;
  };
  autoTool?: boolean;
  temperature?: number;
  topK?: number;
  topP?: number;
}

interface ToolUsed {
  name: string;
  result: any;
  args: any;
  type: "tool-call" | "tool-context-use";
}

type Files = {
  type: "image" | "file";
  data: Blob | string;
  mimeType?: string;
};

type GenerateBase = {
  toolUsed?: ToolUsed[];
  reasoningText?: string;
  object?: Awaited<ReturnType<typeof generateText>>["experimental_output"];
  partialObjectStream?: ReturnType<typeof streamText>["experimental_partialOutputStream"];
  textStream?: ReturnType<typeof streamText>["textStream"];
  files?: Files[];
};

export type GenerateResponse = GenerateTextResult<Record<string, CoreTool<any, any>>, any> &
  StreamTextResult<Record<string, CoreTool<any, any>>, any> &
  GenerateBase;

type EmbeddingParamsOriginal = Omit<Parameters<ReturnType<typeof JigsawStack>["embedding"]>["0"], "token_overflow_mode" | "file_store_key">;
export type EmbeddingResponse = Awaited<ReturnType<ReturnType<typeof JigsawStack>["embedding"]>>;

export interface EmbeddingParams {
  type: EmbeddingParamsOriginal["type"];
  text: EmbeddingParamsOriginal["text"];
  fileContent: EmbeddingParamsOriginal["file_content"];
  url: EmbeddingParamsOriginal["url"];
}

const messageMap = (prompts: GeneratePromptObj[], removeMedia: boolean = false) => {
  const messageMapped: CoreMessage[] = prompts.map((m) => ({
    content:
      typeof m.content === "string"
        ? m.content
        : (
            m.content.map((c) => ({
              type: c.type,
              image: c.type === "image" ? c.data : undefined,
              text: c.type === "text" ? c.data : undefined,
              data: c.type === "file" ? c.data : undefined,
              mimeType: c.type !== "text" ? c?.mimeType : undefined,
            })) as any
          ).filter((c) => (removeMedia ? !["image", "file"].includes(c.type) : true)),
    role: m.role,
  }));

  return messageMapped;
};

const fallback = async <T, H, G>(key: string, item: T[], genFunc: (args?: H) => Promise<G>) => {
  let index = -1;
  do {
    try {
      const resp = await genFunc(index >= 0 ? ({ [key]: item[index] } as H) : undefined);
      return resp;
    } catch (error) {
      index++;
      if (index >= item.length) {
        throw error;
      }
    }
  } while (index < item.length);
};

export const createOmiAI = (config?: {
  groqProviderConfig?: GroqProviderSettings;
  googleProviderConfig?: GoogleGenerativeAIProviderSettings;
  openaiProviderConfig?: OpenAIProviderSettings;
  anthropicProviderConfig?: AnthropicProviderSettings;
  deepseekProviderConfig?: DeepSeekProviderSettings;
  deepinfraProviderConfig?: DeepInfraProviderSettings;
  jigsawProviderConfig?: NonNullable<Parameters<typeof JigsawStack>["0"]>;
}) => {
  const groq = createGroq({
    apiKey: process.env?.GROQ_API_KEY || undefined,
    ...config?.groqProviderConfig,
  });

  const google = createGoogleGenerativeAI({
    apiKey: process.env?.GOOGLE_GENERATIVE_AI_API_KEY || undefined,
    ...config?.googleProviderConfig,
  });

  const openai = createOpenAI({
    apiKey: process.env?.OPENAI_API_KEY || undefined,
    compatibility: "strict",
    ...config?.openaiProviderConfig,
  });

  const anthropic = createAnthropic({
    apiKey: process.env?.ANTHROPIC_API_KEY || undefined,
    ...config?.anthropicProviderConfig,
  });

  const deepseek = createDeepSeek({
    apiKey: process.env?.DEEPSEEK_API_KEY || undefined,
    ...config?.deepseekProviderConfig,
  });

  const deepinfra = createDeepInfra({
    apiKey: process.env?.DEEPINFRA_API_KEY || undefined,
    ...config?.deepinfraProviderConfig,
  });

  const jigsaw = JigsawStack({
    apiKey: process.env?.JIGSAWSTACK_API_KEY || undefined,
    ...config?.jigsawProviderConfig,
  });

  const modelList: {
    [key: string]: {
      id: string;
      speed: number;
      smarts: number;
      context_window: number;
      file_type_support: string[];
      description: string;
      specialty: string[];
      fallback: string | null;
      modelProvider: any;
    };
  } = {
    "gemini-1.5-flash-8b": {
      id: "gemini-1.5-flash-8b",
      speed: 4,
      smarts: 3,
      context_window: 1000000,
      file_type_support: ["audio", "image", "video", "pdf", "text"],
      description: "Great for simple tasks that may require a large context. Also great for all file types.",
      specialty: ["large-files"],
      fallback: null,
      modelProvider: google("gemini-1.5-flash-8b", {
        structuredOutputs: false,
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold: "BLOCK_NONE",
          },
        ],
      }),
    },
    "gemini-1.5-flash": {
      id: "gemini-1.5-flash",
      speed: 3,
      smarts: 4,
      context_window: 1000000,
      file_type_support: ["audio", "image", "video", "pdf", "text"],
      description: "Great for most common tasks and questions that may require a large context. Also great for all file types.",
      specialty: ["large-files"],
      fallback: null,
      modelProvider: google("gemini-1.5-flash", {
        structuredOutputs: false,
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold: "BLOCK_NONE",
          },
        ],
      }),
    },
    "gpt-4o": {
      id: "gpt-4o",
      speed: 2,
      smarts: 4,
      context_window: 128000,
      file_type_support: ["image", "text"],
      description: "Great for highly complex tasks.",
      specialty: ["general"],
      fallback: null,
      modelProvider: openai("gpt-4o"),
    },
    "claude-3-5-sonnet-latest": {
      id: "claude-3-5-sonnet-latest",
      speed: 2,
      smarts: 4,
      context_window: 200000,
      file_type_support: ["text"],
      description: "Great for highly complex coding tasks.",
      specialty: ["code"],
      fallback: null,
      modelProvider: anthropic("claude-3-5-sonnet-latest"),
    },
    "llama-3.3-70b-specdec": {
      id: "llama-3.3-70b-specdec",
      speed: 5,
      smarts: 2,
      context_window: 8192,
      file_type_support: ["text"],
      description: "Great for simple tasks that require speed and tool use.",
      specialty: ["small-tasks"],
      fallback: "gemini-1.5-flash-8b",
      modelProvider: groq("llama-3.3-70b-specdec"),
    },
  };

  const modelKeys = Object.keys(modelList);

  const modelListForLLM = modelKeys
    .map((key) => {
      return {
        ...modelList[key],
        modelProvider: undefined,
      };
    })
    .reduce((acc, curr, index) => {
      acc[modelKeys[index]] = curr;
      return acc;
    }, {});

  const generate = async ({
    prompt,
    schema,
    system,
    contextTool,
    reasoning,
    multiLLM,
    stream,
    autoTool = true,
    temperature = 0,
    topK,
    topP,
  }: GenerateParams) => {
    try {
      const toolUsed: ToolUsed[] = [];

      const prompts: GeneratePromptObj[] = Array.isArray(prompt) ? prompt : [{ role: "user", content: prompt }];
      const lastPromptContent = prompts[prompts.length - 1].content;
      // const allFiles = prompts
      //   .filter((p) => Array.isArray(p.content))
      //   .map((p) => p.content)
      //   .flat()
      //   .filter((c) => (c as any).type != "text");

      const latestPromptFiles = (Array.isArray(lastPromptContent) ? lastPromptContent.filter((c) => (c as any).type != "text") : []).map((f, i) => ({
        ref: `https://onellmref.com/${i}`,
        ...f,
      }));

      const genFiles: Files[] = [];

      const tools = {
        web_search: tool({
          description: "Search the web for the given query",
          parameters: z.object({
            query: z.string().describe("The query to search the web for"),
          }),
          execute: async ({ query }) => {
            const searchResult = await jigsaw.web.search({
              query,
              ai_overview: false,
            });
            return searchResult.results.map((c) => ({
              title: c.title,
              content: c?.content || c?.description,
              snippets: c.snippets,
              url: c.url,
            }));
          },
        }),
        ai_scraper: tool({
          description: "Scrape a website given the url and fields to scrape",
          parameters: z.object({
            url: z.string().describe("The url to scrape"),
            fields: z.array(z.string()).describe("The fields to scrape"),
          }),
          execute: async ({ url, fields }) => {
            const scrapedData = await jigsaw.web.ai_scrape({
              url: url.includes("https://onellmref.com") ? latestPromptFiles.find((f) => f.ref == url)?.data || url : url,
              element_prompts: fields,
            });
            return scrapedData.context;
          },
        }),
        ocr: tool({
          description: "Performs OCR on an image or PDF URL",
          parameters: z.object({
            url: z.string().describe("The image or PDF URL to OCR"),
            fields: z.array(z.string()).describe("fields to extract from the image or PDF"),
          }),
          execute: async ({ url, fields }) => {
            const scrapedData = await jigsaw.vision.vocr({
              url: url.includes("https://onellmref.com") ? latestPromptFiles.find((f) => f.ref == url)?.data || url : url,
              prompt: fields,
            });
            return scrapedData.context;
          },
        }),
        ai_image_generation: tool({
          description: "Generate an image given the prompt",
          parameters: z.object({
            prompt: z.string().describe("The prompt to generate an image for"),
          }),
          execute: async ({ prompt }) => {
            const image = await jigsaw.image_generation({ prompt: prompt });
            const blob = await image.blob();
            genFiles.push({ type: "image", data: blob, mimeType: "image/png" });
            return "image generated";
          },
        }),
        speech_to_text: tool({
          description: "Convert speech to text",
          parameters: z.object({
            url: z.string().describe("The audio URL to convert to text"),
          }),
          execute: async ({ url }) => {
            const text = await jigsaw.audio.speech_to_text({
              url: url.includes("https://onellmref.com") ? latestPromptFiles.find((f) => f.ref == url)?.data || url : url,
            });
            return text;
          },
        }),
        text_to_speech: tool({
          description: "Convert text to speech",
          parameters: z.object({
            text: z.string().describe("The text to convert to speech"),
          }),
          execute: async ({ text }) => {
            const speech = await jigsaw.audio.text_to_speech({ text: text });
            const blob = await speech.blob();
            genFiles.push({ type: "file", data: blob, mimeType: "audio/mpeg" });
            return "audio generated";
          },
        }),
      };

      const preConfigModel = (
        await fallback<LanguageModelV1, any, GenerateObjectResult<any>>("model", [modelList["gemini-1.5-flash-8b"].modelProvider], (args) =>
          generateObject({
            model: modelList["llama-3.3-70b-specdec"].modelProvider,
            prompt: `List of models:${JSON.stringify(modelListForLLM)}\nList of tools:${Object.keys(tools).join(",")}\nPrompt: ${JSON.stringify(prompts[prompts.length - 1])}`,
            schema: z.object({
              model: z.string().describe("The best AI model based on the prompt, context and files"),
              web_search: z.boolean().describe("Whether to use web search for the prompt"),
              reasoning: z
                .boolean()
                .describe(
                  "Whether to use reasoning which requires a longer thinking process and more depth before answering. Only use this for the more complex tasks like maths, counting characters, counting, science & research"
                ),
              use_tool: z.boolean().describe("Is there a possibility that the task requires a tool to achieve it's goal from the list of tool?"),
            }),
            temperature: 0,
            ...args,
          })
        )
      )?.object;

      console.log("preConfigModel: ", preConfigModel);

      const selectedModelID = preConfigModel.model;
      const modelConfig = modelList[selectedModelID];

      let latestTextPrompt = Array.isArray(lastPromptContent) ? lastPromptContent?.find((c) => c.type == "text")?.data : lastPromptContent;
      let searchResult: Awaited<ReturnType<typeof jigsaw.web.search>> | undefined = undefined;

      if (latestTextPrompt) {
        if ((contextTool?.web || preConfigModel.web_search) && contextTool?.web !== false) {
          searchResult = await jigsaw.web.search({
            query: latestTextPrompt,
            ai_overview: false,
          });

          toolUsed.push({
            name: "web_search",
            result: searchResult,
            type: "tool-context-use",
            args: {
              query: latestTextPrompt,
              ai_overview: false,
            },
          });

          const webSearchResultMap = searchResult.results
            .map((c) => ({
              title: c.title,
              content: c?.content || c?.description,
              snippets: c.snippets,
              url: c.url,
            }))
            .slice(0, 3);

          prompts.splice(prompts.length - 1, 0, {
            role: "user",
            content: [{ type: "text", data: `Web search result: ${JSON.stringify(webSearchResultMap)}` }],
          });
        }
      }

      if (preConfigModel.use_tool && autoTool) {
        const toolResult = await fallback<LanguageModelV1, any, GenerateTextResult<any, any>>(
          "model",
          [groq("llama-3.3-70b-versatile"), openai("gpt-4o-mini")],
          (args) =>
            generateText({
              model: modelList["gpt-4o"].modelProvider,
              prompt:
                latestTextPrompt && latestPromptFiles
                  ? `${latestTextPrompt}\nfiles: ${JSON.stringify(latestPromptFiles.map((f) => f.ref))}`
                  : latestTextPrompt,
              system,
              tools: tools,
              toolChoice: "auto",
              maxSteps: 5,
              temperature,
              topK,
              topP,
              ...args,
            })
        );

        const toolText = toolResult?.text;

        const allToolCalls =
          toolResult?.steps
            .map((s) =>
              s.toolCalls.map((t, i) => ({
                ...t,
                result: s.toolResults[i].result,
              }))
            )
            .flat() || [];

        toolUsed.push(
          ...allToolCalls.map((t) => ({
            name: t.toolName,
            result: t.result,
            type: "tool-call" as const,
            args: t.args,
          }))
        );

        prompts.splice(prompts.length - 1, 0, {
          role: "user",
          content: [{ type: "text", data: `Tool result: ${toolText}` }],
        });
      }

      let reasoningText: string | undefined = undefined;

      if ((reasoning || preConfigModel.reasoning) && reasoning !== false) {
        const reasoningResult = await fallback<LanguageModelV1, any, GenerateTextResult<any, any>>(
          "model",
          [deepinfra("deepseek-ai/DeepSeek-R1"), deepseek("deepseek-reasoner")],
          (args) =>
            generateText({
              model: groq("deepseek-r1-distill-llama-70b"),
              system,
              messages: messageMap(prompts, true),
              temperature,
              topK,
              topP,
              maxTokens: undefined,
              ...args,
            })
        );

        reasoningText = reasoningResult?.reasoning || reasoningResult?.text;

        if (reasoningText?.includes("<think>")) {
          reasoningText = reasoningText.split("<think>")[1].split("</think>")[0]?.trim();
        }

        prompts.splice(prompts.length - 1, 0, {
          role: "user",
          content: [{ type: "text", data: `Reasoning context: ${reasoningText}` }],
        });
      }

      const generateFunction = stream ? streamText : generateText;

      if (multiLLM && !stream) {
        const multiLLMPromiseResult = await Promise.allSettled(
          Object.keys(modelList).map((m) =>
            generateText({
              model: modelList[m].modelProvider,
              system,
              messages: messageMap(prompts, true),
              temperature,
              topK,
              topP,
            })
          )
        );

        const multiLLMResults = multiLLMPromiseResult.filter((r) => r.status == "fulfilled").map((r) => r.value);

        prompts.splice(prompts.length - 1, 0, {
          role: "user",
          content: [{ type: "text", data: `Context from multiple LLMs: ${JSON.stringify(multiLLMResults.map((r) => r.text))}` }],
        });
      }

      const llmResp = await fallback<LanguageModelV1, any, any>(
        "model",
        [
          modelConfig?.fallback && modelList[modelConfig?.fallback]
            ? modelList[modelConfig?.fallback].modelProvider
            : modelConfig.id == "gpt-4o"
              ? google("gemini-1.5-flash")
              : openai("gpt-4o"),
        ],
        (args) =>
          generateFunction({
            model: modelConfig.modelProvider,
            system,
            messages: messageMap(prompts),
            experimental_output: schema
              ? Output.object({
                  schema,
                })
              : undefined,
            temperature,
            topK,
            topP,
            ...args,
          }) as any
      );

      let llmResult = llmResp as GenerateResponse;

      llmResult["toolUsed"] = toolUsed;
      llmResult["files"] = genFiles;
      if (reasoningText) {
        llmResult["reasoningText"] = reasoningText;
      }

      if (schema) {
        if (stream) {
          llmResult.partialObjectStream = (llmResp as StreamTextResult<Record<string, CoreTool<any, any>>, any>).experimental_partialOutputStream;
        } else {
          llmResult.object = (llmResp as GenerateTextResult<Record<string, CoreTool<any, any>>, any>).experimental_output;
        }
      }

      return llmResult;
    } catch (error: any) {
      throw error;
    }
  };

  const embedding = async (params: EmbeddingParams) => {
    const resp = await jigsaw.embedding({
      type: params.type,
      text: params.text,
      file_content: params.fileContent,
      url: params.url,
      token_overflow_mode: "error",
    });

    return {
      embeddings: resp.embeddings,
      chunks: resp.chunks,
    };
  };

  return {
    generate,
    embedding,
  };
};
