"""Check that the Gemini API key in .env works, WITHOUT sending any image or client data.

Makes two metadata-only calls (list models, get the configured model). The key is never printed.
It does not need (and does not touch) the NGO_ALLOW_CLOUD_INFERENCE transfer gate because no photos are sent.

    python scripts/check_gemini_key.py [--model gemini-3.5-flash]
"""

from __future__ import annotations

import argparse
import os
import sys

from ngo_photo_approval.config import load_model_config, load_project_config
from ngo_photo_approval.models.cloud_client import API_KEY_ENV


def scrub(text: str, key: str) -> str:
    return text.replace(key, "<key>")[:300] if key else text[:300]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model id to look up (default: models.primary in model.yaml)")
    args = ap.parse_args()

    project = load_project_config()  # also loads .env
    key = os.environ.get(API_KEY_ENV, "").strip()
    model_id = args.model or load_model_config().models["primary"].id
    print(f"auth_mode={project.cloud.auth_mode}  {API_KEY_ENV}={'present' if key else 'MISSING/blank'}")
    if not key:
        print("No key found: paste it after GEMINI_API_KEY= in .env")
        return 2

    from google import genai

    client = genai.Client(api_key=key)
    try:
        names = [m.name for m in client.models.list()]
    except Exception as exc:  # noqa: BLE001 - report any failure without leaking the key
        code = getattr(exc, "code", None)
        print(f"KEY CHECK FAILED ({type(exc).__name__}, HTTP {code}): {scrub(str(exc), key)}")
        return 1
    print(f"KEY ACCEPTED: the API returned {len(names)} models.")

    wanted = {f"models/{model_id}", model_id}
    if wanted & set(names):
        print(f"model '{model_id}': available to this key")
    else:
        try:
            client.models.get(model=model_id)
            print(f"model '{model_id}': available to this key")
        except Exception as exc:  # noqa: BLE001
            print(f"model '{model_id}': NOT available to this key (HTTP {getattr(exc, 'code', None)})")
            gem = sorted(n for n in names if "gemini" in n and ("flash" in n or "pro" in n))[:15]
            print("some Gemini models this key can see:", ", ".join(gem) or "(none)")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
