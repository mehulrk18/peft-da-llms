DEFAULT_SYSTEM_PROMPT1 = """
    Given below is an article. Write a concise and informative Summary for the article.
""".strip()


def generate_training_prompt1(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT1):
    prompt = """### Instruction: {}\n\n### Article: {}\n\n### Summary: {}""".format(system_prompt, article, summary)

    return prompt.strip()


def inference_prompt1(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT1):
    prompt = """### Instruction: {}\n### Article: {}\n### Summary:""".format(system_prompt.strip(), article.strip())

    return prompt.strip()
