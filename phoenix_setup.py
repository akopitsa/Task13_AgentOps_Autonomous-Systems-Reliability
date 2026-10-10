"""Реєструє "тіньову" ціну локальної моделі в Arize Phoenix (Settings → Models).

Phoenix рахує cost для кожного LLM-спану під час інжесту за llm.model_name.
llama3.1:8b у вбудованому прайсі немає, тому додаємо custom-модель з цінами з
agentops/pricing.py (за замовчуванням = gpt-4.1-mini). Запускати ДО run_scenarios.py.

    python phoenix_setup.py                 # профіль з PRICE_PROFILE
    python phoenix_setup.py gpt-4.1         # інший профіль
"""
import re
import sys

import httpx

from agentops import config, pricing


def gql(query: str, variables: dict | None = None) -> dict:
    r = httpx.post(f"{config.PHOENIX_URL}/graphql", json={"query": query, "variables": variables or {}}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(data["errors"])
    return data["data"]


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else config.PRICE_PROFILE
    price = pricing.get_price(profile)
    costs = [
        {"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": price.input_per_m},
        {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": price.output_per_m},
    ]
    pattern = re.escape(config.LLM_MODEL)
    name = config.LLM_MODEL

    models = gql("{ generativeModels(first: 1000) { edges { node { id name kind } } } }")
    existing = [e["node"] for e in models["generativeModels"]["edges"]
                if e["node"]["name"] == name and e["node"]["kind"] == "CUSTOM"]
    if existing:
        gql("mutation($input: UpdateModelMutationInput!) { updateModel(input: $input) { model { id } } }",
            {"input": {"id": existing[0]["id"], "name": name, "provider": None, "namePattern": pattern, "costs": costs}})
        action = "updated"
    else:
        gql("mutation($input: CreateModelMutationInput!) { createModel(input: $input) { model { id } } }",
            {"input": {"name": name, "provider": None, "namePattern": pattern, "costs": costs}})
        action = "created"
    print(f"[PHOENIX] model '{name}' {action}: ${price.input_per_m}/1M in, ${price.output_per_m}/1M out "
          f"(shadow price = {profile})")


if __name__ == "__main__":
    main()
