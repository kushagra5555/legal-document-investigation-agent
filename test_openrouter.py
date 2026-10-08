"""Small smoke test for Gemini -> LangChain -> model."""

from __future__ import annotations

from app.llm import ConfigurationError, DependencyError, build_llm


def main() -> int:
    try:
        llm = build_llm()
        response = llm.invoke("Explain in one sentence what an AI agent is.")
        content = response.content
        if not isinstance(content, str) or not content.strip():
            print("Connection failed: the model returned an empty or malformed response.")
            return 1
        print("Gemini connection succeeded.")
        print(f"Response: {content.strip()}")
        return 0
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 2
    except DependencyError as exc:
        print(f"Environment error: {exc}")
        return 2
    except Exception:
        # Do not print raw provider exceptions: they can contain request details.
        print(
            "Gemini request failed. Check the model name, API key, network, "
            "rate limits, and provider status. The secret was not printed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
