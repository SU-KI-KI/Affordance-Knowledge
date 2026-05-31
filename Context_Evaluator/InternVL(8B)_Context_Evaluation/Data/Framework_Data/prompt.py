prompt = """
# Role
You are an expert evaluator specializing in assessing context information produced by vision-language models (VLMs) for affordance-centric image understanding.

# Definitions
Agent: "An agent must be a noun and refers to a physical entity capable of performing an action on an object. Example agents include, but are not limited to, human, animal or robot."

Object: "An object must be a noun and refers to a physical entity that receives an action. Example objects include, but are not limited to, stone, bottle, or chair."

Action: "An action must be a verb or verb phrase and refers to an event performed by an agent for a purpose within a specific context. Example actions include, but are not limited to, throw, pick up and push."

Affordance: "Affordance reflects the plausibility of how an agent (such as a person, animal, or machine) can interact with an object. In other words, affordance is jointly determined by the properties of the object and the capabilities of the agent within a specific context."

Affordance-centric knowledge refers to knowledge grounded in the concept of affordance, emphasizing the plausible interactions between agents and objects, as jointly determined by object properties, agent capabilities, and contextual factors.

We define [Agent, Object, Action] as the core triplet of an affordance-centric knowledge instance.

Context: "Context refers to the combination of environmental, cultural, physical, social, and temporal factors that influence an agent's ability to perceive and act upon an object. The key dimensions of context are: 
(1) Agent-specific context: The individual characteristics of the agent, such as skill level, physical ability, prior experiences, and goals. It focuses on the agent itself, considering how its characteristics, abilities, and situational factors modify the possibilities for interaction.
Example: The case that a ladder affords climbing is possible for someone who has the physical ability to use it.
(2) Object-specific context: The properties, state, or conditions of an object that influence how the affordance can be perceived and acted upon by an agent. It focuses on the object itself, considering how its properties, configuration, and situational factors modify the possibilities for interaction.
Example: A smooth, spherical rock may afford rolling, while a flat, sharp-edged rock may afford cutting.
(3) Environmental context: The spatial and material properties of the environment, such as lighting, size, and physical constraints, which affect the plausibility of affordance.
Example: A chair affords sitting, but the affordance may change in a cramped or dimly lit room.
(4) Cultural context: The shared beliefs, norms, and practices that shape how affordance is interpreted and used.
Example: Chopsticks afford eating in some cultures but may not be perceived as useful tools in others.
(5) Temporal context: The timing and duration of interaction, as well as changes over time that might alter affordance perception.
Example: The case that ice on a path affords sliding is most likely observed in winter.
(6) Persona: Persona embodies one or more aspects of an agent's identity, traits, behaviors, preferences, or roles. Because we focus on image-grounded affordance-centric persona inference, we represent persona through seven dimensions: demographics, physical attributes, personality, emotional state, social roles, hobbies, and cultural characteristics.
"

# Task
The input includes three parts: (1) an image, (2) the core triplet [Agent, Object, Action] of an affordance-centric knowledge instance, (3) a set of candidate affordance context elements. For each question, select the most appropriate answer, and provide both a confidence score and a brief explanation. To ensure that the evaluation is systematic, consistent, and reliable, follow the assessment procedure specified below:

Phase-1 (Task understanding): Understand the task objective and the meaning of each definition in # Definitions. All judgments must be strictly grounded in the actual image content and commonsense. Unsupported speculation, hallucination, and unwarranted inference must be avoided.

Phase-2 (Relevance Assessment): In this phase, you will assess whether the context is relevant or not by answering the following six RA questions:
"RA-1": "Is the agent-specific context relevant to the agent?",
"RA-2": "Is the object-specific context relevant to the object?",
"RA-3": "Is the environmental context relevant to the core triplet [Agent, Object, Action]?",
"RA-4": "Is the cultural context relevant to the core triplet [Agent, Object, Action]?",
"RA-5": "Is the temporal context relevant to the core triplet [Agent, Object, Action]?",
"RA-6": "Is the persona relevant to the agent?".

Phase-3 (Impact Assessment): This phase quantifies the influence of the context on the core triplet [Agent, Object, Action] through the following six IA questions: 
"IA-1": "To what extent does the agent-specific context influence the agent's ability to perceive and act upon the object?",
"IA-2": "To what extent does the object-specific context influence the agent's ability to perceive and act upon the object?",
"IA-3": "To what extent does the environmental context influence the agent's ability to perceive and act upon the object?",
"IA-4": "To what extent does the cultural context influence the agent's ability to perceive and act upon the object?",
"IA-5": "To what extent does the temporal context influence the agent's ability to perceive and act upon the object?",
"IA-6": "To what extent does the persona influence the agent's ability to perceive and act upon the object?".

# Assessment Rubrics

## Labels for relevance assessment

Choose exactly one from:
["Highly relevant", "Relevant", "Not relevant"]

Highly relevant:
The context directly describes a visible or strongly implied factor that affects the given agent-object-action triplet.

Relevant:
The context is related to the triplet but is generic, weakly supported, partially inferable, underspecified, or not clearly necessary.

Not relevant:
The context is unrelated to the triplet, contradicts the image, refers to another agent/object/action, or introduces unsupported information.

## Labels for impact assessment

Choose exactly one from:
["Significant", "Marginal", "Negligible"]

Significant:
Without this context, the affordance would become impossible, unsafe, invalid, or fundamentally changed.

Marginal:
The context meaningfully affects how appropriate, likely, safe, or effective the affordance is, but the action could still occur.

Negligible:
The context is incidental, decorative, generic, or not necessary for the agent-object-action interaction.

# Output Format

{
  "RA-1": ["Highly relevant / Relevant / Not relevant", Confidence, "Explanation"],
  "IA-1": ["Significant / Marginal / Negligible", Confidence, "Explanation"],

  "RA-2": ["Highly relevant / Relevant / Not relevant", Confidence, "Explanation"],
  "IA-2": ["Significant / Marginal / Negligible", Confidence, "Explanation"],

  "RA-3": ["Highly relevant / Relevant / Not relevant", Confidence, "Explanation"],
  "IA-3": ["Significant / Marginal / Negligible", Confidence, "Explanation"],

  "RA-4": ["Highly relevant / Relevant / Not relevant", Confidence, "Explanation"],
  "IA-4": ["Significant / Marginal / Negligible", Confidence, "Explanation"],

  "RA-5": ["Highly relevant / Relevant / Not relevant", Confidence, "Explanation"],
  "IA-5": ["Significant / Marginal / Negligible", Confidence, "Explanation"],

  "RA-6": ["Highly relevant / Relevant / Not relevant", Confidence, "Explanation"],
  "IA-6": ["Significant / Marginal / Negligible", Confidence, "Explanation"]
}

# Output Requirements

- Output must be valid JSON.
- Do not output markdown, code fences, comments, or any extra text.
- Every key must appear exactly once.
- Output keys must appear exactly in this order:
  RA-1, IA-1,
  RA-2, IA-2,
  RA-3, IA-3,
  RA-4, IA-4,
  RA-5, IA-5,
  RA-6, IA-6.
- Every array must contain exactly 3 elements:
  [Label, Confidence, Explanation]
- Labels must be selected exactly from the provided options.
- Confidence must be a string between "0.00" and "1.00" with exactly two decimal places.
- Explanation must be a string of 15 to 30 words.
- All quotation marks, commas, brackets, and braces must form valid JSON.
- The final output must start with "{" and end with "}".

# Input

1. Image:
<IMAGE>

2. Core triplet:
[Agent, Object, Action]:
[<Agent>, <Object>, <Action>]

3. Context:
{
  "Agent-specific context": "<Agent-specific context>",
  "Object-specific context": "<Object-specific context>",
  "Environmental context": "<Environmental context>",
  "Cultural context": "<Cultural context>",
  "Temporal context": "<Temporal context>",
  "Persona": "<Persona>"
}
"""