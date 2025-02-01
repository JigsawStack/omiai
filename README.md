# OmiAI SDK

![OmiAI](/examples/how_it_works.png)

OmiAI is an opinionated AI SDK for Typescript that auto-picks the best model from a suite of curated models depending on the prompt. It includes built-in o3-like reasoning, curated tools, internet access and full multi-modal support with almost all media types.

The idea is for OmiAI to be the last framework you need for LLMs where it feels like you're just using one LLM that's good at everything!

- ⭐ Curated list of models based on model quality, speed and cost
- 🧠 Automatically picks the best model for each task
- 🔗 Automatically chains models for complex tasks
- 🤔 Built-in reasoning (o3-mini, DeepSeek r1)
- 🔨 Built-in tools and tool calling (Image generation, OCR, SST, etc)
- 🌐 Access to the internet in real-time
- 🔁 Model rate-limiting fallback, retries
- 📁 Multimodal LLM support (PDF, Images, Files, Audio, CSV, JSON)
- 🧮 Multimodal embedding model

Powered by [Vercel's AI SDK](https://sdk.vercel.ai/docs/introduction) for orchestration & [JigsawStack](https://jigsawstack.com/) for tools and embeddings.

## ⚡️ Quick Install

```bash
npm i omiai
```

Some benefits include:

- Never having to pick to model or deal with over configuration
- Plug > Prompt > Play
- Always having full multimodal support regardless of the model
- Kept up to date with the latest models and features
- Use reasoning mixed with other models to solve complex tasks

## Usage

### Set up

You'll have to set up all API keys for the LLM providers used in the SDK. Check out the `.env.example` file for all API keys needed.

Either create a `.env` file in the root of your project based on the `.env.example` or pass your API keys in `createOmiAI` like this:

```ts
const omi = createOmiAI({
  openaiProviderConfig: {
    apiKey: process.env.OPENAI_API_KEY,
  },
  anthropicProviderConfig: {
    apiKey: process.env.ANTHROPIC_API_KEY,
  },
  ...
});
```

### Basic

```ts
import { createOmiAI } from "omiai";

const omi = createOmiAI();

const result = await omi.generate({
  prompt: "What is the meaning of life?",
});

console.log(result?.text);
```

### Structured output with Zod

```ts
import { z } from "zod";

const result = await omi.generate({
  prompt: "How many r's are there in the word 'strawberries'?",
  schema: z.object({
    answer: z.number().describe("The answer to the question"),
  }),
});

console.log(result?.object);
```

### Streaming text

```ts
const result = await omi.generate({
  prompt: "Tell me a story of a person who discovered the meaning of life.",
  stream: true,
});

let text = "";
for await (const chunk of result?.textStream) {
  text += chunk;
  console.clear();
  console.log(text);
}
```

### Streaming object

```ts
const result = await omi.generate({
  prompt: "Tell me a story of a person who discovered the meaning of life.",
  schema: z.object({
    story: z.string().max(1000).describe("The story"),
    character_names: z
      .array(z.string())
      .describe("The names of the characters in the story"),
  }),
  stream: true,
});

for await (const chunk of result?.partialObjectStream ?? []) {
  console.log(chunk);
}
```

### Messages

```ts
const result = await omi.generate({
  prompt: [{ role: "user", content: "What is the meaning of life?" }],
});

console.log(result?.text);
```

### Attach images/files

```ts
const result = await omi.generate({
  prompt: [
    {
      role: "user",
      content: [
        {
          type: "text",
          data: "Extract the total price of the items in the image", //will tool call OCR tool
        },
        {
          type: "image",
          data: "https://media.snopes.com/2021/08/239918331_10228097135359041_3825446756894757753_n.jpg",
          mimeType: "image/jpg",
        },
      ],
    },
  ],
  schema: z.object({
    total_price: z
      .number()
      .describe("The total price of the items in the image"),
  }),
});

console.log(result?.object);
```

### Reasoning

Reasoning is automated so you don't have to explicitly call it. Based on the complexity of the prompt, it will automatically decide if it needs to use reasoning or not.

If you want to force reasoning, you can set the `reasoning` parameter to `true` or if you want to disable it permanently, set it to `false`. Removing the key will set it to auto.

```ts
const result = await omi.generate({
  prompt: "How many r's are there in the text: 'strawberry'?",
  reasoning: true,
  schema: z.object({
    answer: z.number(),
  }),
});

console.log("reason: ", result?.reasoningText);
console.log("result: ", result?.object);
```

Get the reasoning text from `result?.reasoningText`

### Multi-LLM

Multi-LLM is a technique that runs your prompts across multiple LLMs and merges the results. This is useful if you want to get a more accurate allowing models to cross-check each other.

Note: This can shoot up your costs as it would run it across ~5-6 LLMs in parallel.

```ts
const result = await omi.generate({
  prompt: "What is the meaning of life?",
  multiLLM: true,
});
```

### Web search

Web search is automated so you don't have to explicitly turn it on. It will automatically decide if it needs to use web search or not based on the prompt. You can also force it to run by setting the `contextTool.web` parameter to `true` or if you want to disable it permanently, set it to `false`. Removing the key will set it to auto.

```ts
const result = await omi.generate({
  prompt: "What won the US presidential election in 2025?",
  contextTool: {
    web: true,
  },
});
```

### Embeddings

The embedding model is powered by [JigsawStack](https://jigsawstack.com) and you can view the [full docs here](https://jigsawstack.com/embedding)

```ts
const result = await omi.embedding({
  type: "text",
  text: "Hello, world!",
});

console.log(result.embeddings);
```

```ts
const result = await omi.embedding({
  type: "pdf",
  url: "https://example.com/file.pdf",
});

console.log(result.embeddings);
```

### Tool calling

You can pass your own tools to the SDK by using the `tools` parameter. This is tool function params is based on the Vercel's AI SDK. [Check out full docs for tools here](https://sdk.vercel.ai/docs/ai-sdk-core/tools-and-tool-calling)

```ts
import { createOmiAI, tool } from "omiai";

const omi = createOmiAI();

const result = await omi.generate({
  prompt: "What is the weather in San Francisco?",
  tools: {
    weather: tool({
      description: "Get the weather in a location",
      parameters: z.object({
        location: z.string().describe("The location to get the weather for"),
      }),
      execute: async ({ location }) => ({
        location,
        temperature: 72 + Math.floor(Math.random() * 21) - 10,
      }),
    }),
  },
});
```

## SDK Params

### `omi.generate`

`reasoning`, `contextTool`, `autoTool` and the actual LLM that will execute your prompt are all automatically decided based on your prompt. You can turn off auto decisions for any of these by setting the relevant to `false`. You can also force them to run by setting them to `true`. If the field is `undefined` or not provided, it will be set to auto.

```ts
interface GenerateParams {
  stream?: boolean;
  reasoning?: boolean; // Auto turns on depending on the prompt. Set to true to force reasoning. Set to false to disable auto-reasoning.
  multiLLM?: boolean; // Turn on if you want to run your prompt across all models then merge the results.
  system?: string;
  prompt: string | GeneratePromptObj[]; // String prompt or array which will treated as messages.
  schema?: z.ZodSchema; // Schema to use for structured output.
  contextTool?: {
    web?: boolean; //Auto turns on depending on the prompt. Set to true to force web-search. Set to false to disable web search.
  };
  autoTool?: boolean; // Auto turns on depending on the prompt. Set to true to force tool-calling. Set to false to disable tool-calling.
  temperature?: number;
  topK?: number;
  topP?: number;
  tools?: {
    [key: string]: ReturnType<typeof tool>;
  };
}

interface GeneratePromptObj {
  role: CoreMessage["role"];
  content:
    | string
    | {
        type: "text" | "image" | "file";
        data: string; //url or base64
        mimeType?: string; //mimeType of the file
      }[];
}
```

### `omi.embedding`

View [full docs here](https://jigsawstack.com/embedding)

```ts
interface EmbeddingParams {
  type: "audio" | "image" | "pdf" | "text" | "text-other";
  text?: string;
  url?: string;
  fileContent?: string;
}
```

## Contributing

Contributions are welcome! Please feel free to submit a PR :)
