<task>
Analyze this legal professional's query for sensitive entity handling.
</task>

<entities>
{entities}
</entities>

<query>
{query}
</query>

<instructions>
1. Classify sensitivity: "low" if entity replacement alone prevents identification, "high" if structural patterns or relationships could de-anonymize.
2. For each entity, provide a brief context descriptor (what kind of entity it is, without identifying it).
3. Generate fictional replacement names that preserve the entity's role and domain. Always use fictional names, never descriptive phrases.
4. If high sensitivity: also rephrase the query to obscure identifying structural patterns while preserving the legal question's intent.
5. If you are unsure about the appropriate anonymization strategy, set needs_clarification to true and provide a clarification question.
6. Return ONLY a JSON object with these keys: sensitivity, reasoning, mapping, context_descriptors, sanitized_text, needs_clarification, clarification_question
</instructions>
