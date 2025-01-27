# OmiAI SDK

OmiAI is a LLM SDK for Typescript. It is a highly opinionated AI SDK that that's where you don't pick the model, it auto-picks the best model from a suite of curated models depending on your prompt/messages. It includes built-in o1-like reasoning, curated tools, internet access and full multi-modal support.

- Pre-selected list of models based on model quality, speed and cost
- Automatically picks the best model for each task
- Automatically chains models for complex tasks
- Built-in reasoning (DeepSeek r1)
- Curated built-in tools and tool calling
- Dynamic web search
- Simple and easy to use
- Built on-top of Vercel's AI SDK
- Built in model rate-limiting fallback, retries
- Multi-modal LLM support (PDF, Images, Files, Audio, CSV, JSON)
- Multi-modal embedding model powered by [JigsawStack Embedding](https://jigsawstack.com/embedding)

## ⚡️ Quick Install

```bash
npm i omiai
```

## Usage

### Set up

You'll have to set up all api keys for the LLM providers used in the SDK. Check out the `.env.example` file for all api keys needed.

Either create a `.env` file in the root of your project based on the `.env.example` or pass your api keys in `createOmiAI` like this:

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
  prompt: "Tell me a story of a person that discovered the meaning of life.",
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
  prompt: "Tell me a story of a person that discovered the meaning of life.",
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

## SDK Params

### `omi.generate`

`reasoning`, `context_tool`, `auto_tool` and the actual LLM that will execute your prompt are all automatically decided based on your prompt. You can turn off auto decisions for any of these by setting the relevant to `false`. You can also force them to run by setting them to `true`. If the field is `undefined` or not provided, it will be set to auto.

```ts
interface GenerateParams {
  stream?: boolean;
  reasoning?: boolean; // Auto turns on depending on prompt. Set to true to force reasoning. Set to false to disable auto-reasoning.
  multi_llm?: boolean; // Turn on if you want to run your prompt across all models then merge the results.
  system?: string;
  prompt: string | GeneratePromptObj[]; // String prompt or array which will treated as messages.
  schema?: z.ZodSchema; // Schema to use for structured output.
  context_tool?: {
    web?: boolean; //Auto turns on depending on prompt. Set to true to force web-search. Set to false to disable web-search.
  };
  auto_tool?: boolean; // Auto turns on depending on prompt. Set to true to force tool-calling. Set to false to disable tool-calling.
  temperature?: number;
  topK?: number;
  topP?: number;
}

interface GeneratePromptObj {
  role: CoreUserMessage["role"];
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
  file_content?: string;
}
```
