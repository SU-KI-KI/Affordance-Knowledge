prompt = """
# Role：
You are an excellent image analyst.

# Task: 
Given an image and a list of objects detected within it, your task is to perform image understanding, extract affordance knowledge and conduct affordance-centric inferences.

# Definition:
1. Affordance: Affordance reflects the plausibility of how an agent (such as a person, animal, or machine) can interact with an object. In other words, affordance is jointly determined by the properties of the object and the capabilities of the agent within a specific context.

2. Agent: An agent must be a noun and refers to a physical entity capable of performing an action on an object. Example agents include, but are not limited to, human, animal or robot.

3. Object: An object must be a noun and refers to a physical entity that receives an action. Example objects include, but are not limited to, stone, bottle, or chair.

4. Action: An action must be a verb or verb phrase and refers to an event performed by an agent for a purpose within a specific context. Example actions include, but are not limited to, throw, pick up and push.

5. Context: Context refers to the combination of environmental, cultural, physical, social, and temporal factors that influence an agent's ability to perceive and act upon an object. The key dimensions of context are: 
(1) Agent-specific context: The individual characteristics of the agent, such as skill level, physical ability, prior experiences, and goals. It focuses on the agent itself, considering how its characteristics, ability, and situational factors modify the possibilities for interaction.
Example: The case that a ladder affords climbing is possible for someone who has the physical ability to use it.
(2) Object-specific context: The properties, state, or conditions of an object that influence how the affordance can be perceived and acted upon by an agent. It focuses on the object itself, considering how its properties, configuration, and situational factors modify the possibilities for interaction.
Example: A smooth, spherical rock may afford rolling, while a flat, sharp-edged rock may afford cutting.
(3) Environmental context: The spatial and material properties of the environment, such as lighting, size, and physical constraints, which affect the plausibility of affordance.
Example: A chair affords sitting, but the affordance may change in a cramped or dimly lit room.
(4) Cultural context: The shared beliefs, norms, and practices that shape how affordance is interpreted and used.
Example: Chopsticks afford eating in some cultures but may not be perceived as useful tools in others.
(5) Temporal context: The timing and duration of interaction, as well as changes over time that might alter affordance perception.
Example: The case that ice on a path affords sliding is most likely observed in winter.

6. Affordance-centric inference refers to a series of inferences that are associated with specific affordance knowledge. The key dimensions of affordance-centric inference are:
(1) Action purpose: Action Purpose refers to the underlying reason or intent that motivates an action, explaining why it is performed and what it aims to achieve.
(2) Action duration: Action duration refers to the length of time that an action takes to be performed, encompassing the period from its initiation to its completion. We define five types of action duration, namely seconds, minutes, hours, days, and longer. For instance, turning off a light switch has an effect duration that might be described as minutes or hours, while planting a tree has an effect that could be classified as days or longer.
(3) Action occurrence type: According to the regularity, we define four types of action occurrence. Habitual: Repeated actions that are performed regularly (e.g., daily exercise). Spontaneous: Actions performed without prior planning or thought (e.g., laughing at a joke). Planned: Actions deliberately planned in advance (e.g., attending a meeting). Reactive: Immediate responses to stimuli or situations (e.g., dodging a ball).
(4) Action effect: Action effect refers to the outcome or impact resulting from an action.
(5) Action effect duration: Action effect duration refers to the persistence of the consequences or outcomes of an action, namely how long the impact of an action remains after its completion. We define five types of action effect duration, namely seconds, minutes, hours, days, and longer. For example, turning off a light switch has an effect duration that might be described as minutes or hours, while planting a tree has an effect that could be classified as days or longer.
(6) Affordance knowledge type: According to the level of generality and the prerequisite expertise required to grasp the affordance, we define two types of affordance knowledge. Commonsense affordance knowledge: This type of affordance knowledge is understood by the majority of people without requiring specialized knowledge. These knowledge are often intuitive or derived from everyday experience. The main characteristics are universal accessibility (namely, knowledge widely shared across cultures and demographics) and practicality (namely, relates to basic, day-to-day interactions with objects and environments). For example, a chair affords sitting, and a door handle affords pulling or turning. Non-commonsense affordance knowledge: This type of affordance knowledge requires specialized training, experience, or domain-specific expertise to understand or utilize. The main characteristics are specialized accessibility (namely, the knowledge is restricted to those with specific training, background, or exposure) and domain-specificity (namely, often tied to professional, technical, or niche activities). For example, a surgical tool affords precise incision (requires medical training). 

7.Persona: Persona embodies one or more aspects of an agent's identity, traits, behaviors, preferences, or roles. Because we focus on image-grounded affordance-centric persona inference, we represent persona through seven dimensions: demographics, physical attributes, personality, emotional state, social roles, hobbies, and cultural characteristics.

# Input:
1. An image
2. Object names: A list of objects detected within the input image.

# Structured Output:
The affordance knowledge must be presented in a JSON format.
[
  {
    "Agent": "agent",
    "Object": "object",
    "Action": "action",
    "Context": {
      "Agent-specific context": "agent-specific context",
      "Object-specific context": "object-specific context",
      "Environmental context": "environmental context",
      "Cultural context": "cultural context",
      "Temporal context": "temporal context"
    },
    "Affordance-centric inference": {
      "Action purpose": "action purpose",
      "Action duration": "seconds, minutes, hours, days, longer" (select one or multiple labels),
      "Action occurrence type": "habitual, spontaneous, planned, reactive" (select one or multiple labels),
      "Action effect": "action effect",
      "Action effect duration": "seconds, minutes, hours, days, longer" (select one or multiple labels),
      "Affordance knowledge type": "commonsense affordance knowledge, non-commonsense affordance knowledge" (binary classification)
      },
      "Persona": {
        "Demographics": "demographics",
        "Physical attributes": "physical attributes",
        "Personality": "personality",
        "Emotional state": "emotional state",
        "Social roles": "social roles",
        "Hobbies": "hobbies",
        "Cultural characteristics": "cultural characteristics"
    }
  }
]

# Constraints
1. Extract as much affordance knowledge as possible from the given image.
2. Adhere to the provided definitions of all concepts. 
3. The provided list of objects should be used as a reference; it does not indicate that we are solely interested in these objects.
4. The output should follow the specified output structure in a JSON format.
"""

