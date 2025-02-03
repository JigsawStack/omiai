import dotenv from "dotenv";
dotenv.config();

import { z } from "zod";
import { createOmiAI } from "../index";

const omi = createOmiAI();

const run = async () => {
  const result = await omi.generate({
    prompt: "how many r's are there in the text: 'strrawberry'?",
    schema: z.object({
      answer: z.string(),
    }),
  });

  console.log("reason: ", result?.reasoningText);
  console.log("result: ", result?.text);
};

run();
