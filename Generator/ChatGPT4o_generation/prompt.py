zeroshot = """
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

fewshot = """
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
(7) Persona: Persona embodies one or more aspects of an agent's identity, traits, behaviors, preferences, or roles. Because we focus on image-grounded affordance-centric persona inference, we represent persona through seven dimensions: demographics, physical attributes, personality, emotional state, social roles, hobbies, and cultural characteristics.

# Input:
1. An image
2. Object names: A list of objects detected within the input image.

# Structured Output:
The affordance knowledge must be presented in a JSON format: 
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


# Example:
Input:
Image,
Object list: ["man", "stone"],
Out put:
[
  {
    "Agent": "man",
    "Object": "stone",
    "Action": "throw",
    "Context": {
      "Agent-specific context": "junior",
      "Object-specific context": "lightweight",
      "Environmental context": "outdoor",
      "Temporal context": "daytime during a break or recreational time"
    },
    "Affordance-centric inference": {
      "Action purpose": "the person throws the stone as a form of entertainment or playful interaction",
      "Action duration": "seconds",
      "Action occurrence type": "spontaneous",
      "Action effect": "a thrown stone has the potential to damage objects or cause injury to someone",
      "Action effect duration": "seconds or hours",
      "Affordance knowledge type": "commonsense affordance knowledge"，
      "Persona": {
        "Demographics": "young male, likely in his early 20s",
        "Physical attributes": "wearing a coat suitable for outdoor activities",
        "Personality": "playful and active",
        "Emotional state": "relaxed and cheerful",
        "Social roles": "individual engaging in personal or social recreation",
        "Hobbies": "enjoying physical activities outdoors",
        "Cultural characteristics": "recreational outdoor behavior commonly observed in young adults"
      }
    }
  },
  {
    "Agent": "man",
    "Object": "chair",
    "Action": "sit",
    "Context": {
      "Agent-specific context": "tired",
      "Object-specific context": "sturdy chair designed for comfortable seating",
      "Environmental context": "outdoor setting, in a shaded or quiet area",
      "Temporal context": "after engaging in a physically demanding task or during a break"
    },
    "Affordance-centric inference": {
      "Action purpose": "the person sits on the chair to rest and alleviate fatigue",
      "Action duration": "minutes or hours",
      "Action occurrence type": "spontaneous or planned",
      "Action effect": "provides comfort and relief to the individual",
      "Action effect duration": "minutes or hours",
      "Affordance knowledge type": "commonsense affordance knowledge",
      "Persona": {
        "Demographics": "young male, likely in his early 20s",
        "Physical attributes": "wearing casual outdoor attire, seated posture indicating relaxation",
        "Personality": "pragmatic and focused on comfort",
        "Emotional state": "calm and relaxed",
        "Social roles": "individual engaging in solitary rest or social downtime",
        "Hobbies": "spending time in nature",
        "Cultural characteristics": "recreational outdoor behavior commonly observed in young adults"
      }
    }
  }
]

Input:
Image,
Object list: ["woman", "food"],
Out put:  
[
  {
    "Agent": "woman",
    "Object": "chopsticks",
    "Action": "use",
    "Context": {
      "Agent-specific context": "hungry",
      "Object-specific context": "slender utensils designed for precision handling of food",
      "Environmental context": "indoor dining area",
      "Cultural context": "eastern cultural eating habit where chopsticks are common utensils",
      "Temporal context": "during a regular mealtime"
    },
    "Affordance-centric inference": {
      "Action purpose": "the person uses chopsticks to consume food in a culturally appropriate manner",
      "Action duration": "minutes",
      "Action occurrence type": "habitual",
      "Action effect": "effective and comfortable consumption of the meal",
      "Action effect duration": "minutes",
      "Affordance knowledge type": "commonsense affordance knowledge",
      "Persona": {
        "Demographics": "young female, likely in her late 20s",
        "Physical attributes": "wearing traditional Eastern attire",
        "Personality": "focused and culturally mindful",
        "Emotional state": "satisfied or content",
        "Social roles": "individual participating in a cultural dining tradition",
        "Hobbies": "appreciating traditional meals",
        "Cultural characteristics": "people with Eastern cultural traits or customary practices"
      }
    }
  },
  {
    "Agent": "woman",
    "Object": "food",
    "Action": "dine",
    "Context": {
      "Agent-specific context": "hungry individual engaging in a meal",
      "Object-specific context": "prepared food served in a plate",
      "Environmental context": "indoor table setting, conducive for dining",
      "Cultural context": "eastern cultural eating habit where chopsticks are common utensils",
      "Temporal context": "during a regular mealtime"
    },
    "Affordance-centric inference": {
      "Action purpose": "the person consumes food to satisfy hunger and participate in a cultural dining tradition",
      "Action duration": "minutes or hours",
      "Action occurrence type": "habitual or planned",
      "Action effect": "satisfaction of hunger, nourishment, and adherence to cultural practices",
      "Action effect duration": "hours",
      "Affordance knowledge type": "commonsense affordance knowledge",
      "Persona": {
        "Demographics": "young female, likely in her late 20s",
        "Physical attributes": "seated in a comfortable dining position",
        "Personality": "mindful and engaged in traditional activities",
        "Emotional state": "content and focused",
        "Social roles": "diner participating in a social or cultural mealtime setting",
        "Hobbies": "exploring traditional cuisine",
        "Cultural characteristics": "people with Eastern cultural traits or customary practices"
      }
    }
  }
]

Input:
Image,
Object list: ["pilot", "airplane"],
Out put:
[
  {
    "Agent": "pilot",
    "Object": "airplane",
    "Action": "drive",
    "Context": {
      "Agent-specific context": "licensed and trained pilot with expertise in operating and navigating airplanes",
      "Object-specific context": "a functional airplane designed for air travel, equipped with complex controls for navigation and flight management",
      "Environmental context": "on the runway for takeoff",
      "Temporal context": "maneuvering procedures such as takeoff"
    },
    "Affordance-centric inference": {
      "Action purpose": "to transport passengers, cargo, or oneself safely from one location to another",
      "Action duration": "hours",
      "Action occurrence type": "planned",
      "Action effect": "the airplane moves safely and efficiently to its intended destination",
      "Action effect duration": "hours",
      "Affordance knowledge type": "non-commonsense affordance knowledge",
       "Persona": {
        "Demographics": "adult male or female, likely in their 30s or 40s",
        "Physical attributes": "wearing professional pilot attire",
        "Personality": "calm, responsible, and detail-oriented",
        "Emotional state": "focused and alert",
        "Social roles": "professional pilot responsible for passengers and crew",
        "Hobbies": "interest in aviation or technology",
        "Cultural characteristics": "embodying the values of precision and safety in professional aviation"
      }
    }
  }
]

# Constraints
1. Extract as much affordance knowledge as possible from the given image.
2. Adhere to the provided definitions of all concepts. 
3. The provided list of objects should be used as a reference; it does not indicate that we are solely interested in these objects.
4. The output should follow the specified output structure in a JSON format.
"""
