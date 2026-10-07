"""LLM provider abstractions with lazy concrete-provider imports."""

from importlib import import_module

__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "OpenAICodexProvider", "OpenAIResponsesProvider", "AzureOpenAIProvider"]

_EXPORTS = {
    "LLMProvider": ("PhyAgentOS.providers.base", "LLMProvider"),
    "LLMResponse": ("PhyAgentOS.providers.base", "LLMResponse"),
    "LiteLLMProvider": ("PhyAgentOS.providers.litellm_provider", "LiteLLMProvider"),
    "OpenAICodexProvider": (
        "PhyAgentOS.providers.openai_codex_provider",
        "OpenAICodexProvider",
    ),
    "OpenAIResponsesProvider": (
        "PhyAgentOS.providers.openai_responses_provider",
        "OpenAIResponsesProvider",
    ),
    "AzureOpenAIProvider": (
        "PhyAgentOS.providers.azure_openai_provider",
        "AzureOpenAIProvider",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
