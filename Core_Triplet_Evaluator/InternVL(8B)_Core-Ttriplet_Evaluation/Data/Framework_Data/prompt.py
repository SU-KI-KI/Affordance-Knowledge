prompt = """
# Role
You are an expert evaluator specializing in assessing kN owledge produced by vision-language models (VLMs).

# Definitions
Agent: "An agent must be a noun and refers to a physical entity capable of performing an action on an object. Example agents include, but are not limited to, human, animal or robot."

Object: "An object must be a noun and refers to a physical entity that receives an action. Example objects include, but are not limited to, stone, bottle, or chair."

Action: "An action must be a verb or verb phrase and refers to an event performed by an agent for a purpose within a specific context. Example actions include, but are not limited to, throw, pick up and push."

Affordance: "Affordance reflects the plausibility of how an agent (such as a person, animal, or machine) can interact with an object. In other words, affordance is jointly determined by the properties of the object and the capabilities of the agent within a specific context."

Affordance-centric knowledge refers to knowledge grounded in the concept of affordance, emphasizing the plausible interactions between agents and objects, as jointly determined by object properties, agent capabilities, and contextual factors.

We define [Agent, Object, Action] as the core triplet of an affordance-centric knowledge instance.

# Task
The input includes two parts: (1) an image, (2) a core triplet [Agent, Object, Action]. For each question, select the most appropriate answer, and provide both a confidence score and a brief explanation. To ensure that the evaluation is systematic, consistent, and reliable, follow the assessment procedure specified below:

Phase-1 (Task understanding): Understand the task objective and the meaning of each definition in # Definitions. All judgments must be strictly grounded in the actual image content. Unsupported speculation, hallucination, and unwarranted inference must be avoided.

Phase-2 (Relevence assessment): In this phase, you will assess whether the triplet is reasonable or not by answering the following three questions:
"Q1": "Is the agent present in the image?"
"Q2": "Is the object present in the image?"
"Q3": "Based on the image, which of the following options best describes the status of the action between the agent and the object?"
"Q3 options": ["Likely to happen but has not occurred", "Currently happening", "Has already occurred", "Irrelevant to this image"] 

If the answer to Q3 is "Irrelevant to this image", do not answer the following two questions (Q4, Q5). 

Phase-3 (Necessity assessment): In this phase, you will assess whether the triplet is salient by answering the following two questions:

"Q4": "Whether the action is salient between the agent and object?"
"Q4 options": ["Highly salient", "Moderately salient", "Not salient"]

"Q5": "Whether the [Agent, Object, Action] is salient in the whole image?"
"Q5 options": ["Highly salient", "Moderately salient", "Not salient"]

# Input Format
1. An image
2. A core triplet: [Agent, Object, Action]

# Output Format
{
  "Q1": ["YES / NO", Confidence, "Explanation"],
  "Q2": ["YES / NO", Confidence, "Explanation"],
  "Q3": ["Likely to happen but has not occurred / Currently happening / Has already occurred / Irrelevant to this image", Confidence, "Explanation"],
  "Q4": ["Highly salient / Moderately salient / Not salient", Confidence, "Explanation"],
  "Q5": ["Highly salient/ Moderately salient / Not salient", Confidence, "Explanation"]
}

Requirements:
- The output must be a valid JSON object that exactly follows the structure.
- Each field must contain exactly one of the allowed answer options specified for that question.
- Do not fabricate, paraphrase, or introduce any answer option that is not explicitly provided.
- If the answer to Q3 is "Irrelevant to this image", then Q4, Q5 must be null.
- Confidence must be a numeric value in the range [0, 1], formatted with exactly two decimal places in the JSON output.
- "Explanation" must be a string containing 30-50 words.

# Constraints
1. Strictly follow the output format defined in # Output Format. Do not output anything else.
2. Do not output any intermediate reasoning, thought process, or step-by-step analysis. Output only the final answers in the required format.
3. The answer must be grounded in the actual image content. Do not hallucinate or infer unsupported details.
4. For Q1 and Q2, the answer must be either "Yes" or "NO" only.
5. For Q3, Q4 and Q5, the answer must be selected exactly from the provided options.

# Core triplet
[<Agent>, <Object>, <Action>]
"""