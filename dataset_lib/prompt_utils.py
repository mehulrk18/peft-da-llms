DEFAULT_SYSTEM_PROMPT1 = """
    Given below is an article. Write a concise and informative Summary for the article.
""".strip()


def generate_training_prompt1(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT1):
    prompt = """### Instruction: {}\n\n### Article: {}\n\n### Summary: {}""".format(system_prompt, article, summary)

    return prompt.strip()


def inference_prompt1(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT1):
    prompt = """### Instruction: {}\n### Article: {}\n### Summary:""".format(system_prompt.strip(), article.strip())

    return prompt.strip()



# TODO: Define chat template
# First, define a tool
def get_summary_from_ai(content: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!


# Next, create a chat and apply the chat template
def get_chat_template_prompt(content: str):
    messages = [
        {"role": "system", "content": "You are a bot that generates precise and concise summary of the articles presented."},
        {"role": "user", "content": "Summarize: \n{}".format(content)}
    ]

    return messages