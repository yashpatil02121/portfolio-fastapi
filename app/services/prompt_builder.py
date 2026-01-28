# app/services/prompt_builder.py

def build_prompt(schema, resume_text, user_query):
    constraints_text = "\n".join(f"- {c}" for c in schema["constraints"])

    return f"""
Role:
{schema['role']}

Task:
{schema['task']}

Context:
Resume Data:
{resume_text}

Constraints:
{constraints_text}

User Question:
{user_query}

Output Format:
{schema['output_format']}
"""
