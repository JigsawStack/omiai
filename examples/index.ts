import dotenv from "dotenv";
dotenv.config();

import { createOmiAI } from "../index";

const omi = createOmiAI();

const run = async () => {
  const result = await omi.generate({
    prompt: "how many r's are there in the text: 'strrawberry'?",
  });

  console.log("reason: ", result?.text);
};

run();
