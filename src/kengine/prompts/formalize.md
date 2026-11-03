You translate a task's statements into a small formal language, as JSON matching the given schema. Translate what each statement says, nothing more, and don't solve the task.

For each numbered statement, write {"given": <its number>, "statements": [...]}. Use an empty list for a statement that doesn't fit the language: a question, an instruction, a calculation, or something only loosely logical.

{formal_syntax}

Use one vocabulary for the whole task: the same name for the same individual, property or relation everywhere. Choose names that also fit the task's question or conclusion, because later steps reuse yours.

Example. Statements:
1. All dogs are mammals.
2. Rex is a dog or a cat.
3. Rex is not a cat.
4. What is Rex?

Output:

{example}
