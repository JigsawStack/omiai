import dotenv from "dotenv";
dotenv.config();

import { createOmiAI } from "../index";
import { z } from "zod";

const omi = createOmiAI();

const run = async () => {
  const result = await omi.generate({
    prompt: "how many r's are there in the text: 'strawberry'?",
    schema: z.object({
      answer: z.number(),
    }),
  });

  console.log("reason: ", result?.reasoningText);
  console.log("result: ", result?.object);
};

run();
