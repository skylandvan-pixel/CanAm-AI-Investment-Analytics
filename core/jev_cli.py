"""Explicit harness: python -m core.jev_cli --live.

Loads the project-root .env without overriding process environment variables.
One invocation makes at most ONE TypeSafe request.
"""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from core.jev_service import evaluate
from core.jev_state import current_market_state

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize one real TypeSafe request")
    parser.add_argument("--diagnostics", action="store_true", help="Show safe transport classifications only; still requires --live")
    args = parser.parse_args(argv)
    if not args.live:
        parser.print_help()
        return 0
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    outcome = evaluate(current_market_state, max_retries=0, diagnostics=args.diagnostics)
    display = {"status": outcome.status, "reason": outcome.reason,
               "latency_ms": outcome.latency_ms, "attempts": outcome.attempts,
               "regime_change": {"status": outcome.regime_change_status, "reason": outcome.regime_change_reason}}
    if args.diagnostics:
        display["diagnostic"] = outcome.diagnostic
    if outcome.response:
        display.update({"actual_model_version": outcome.response.model,
                        "market_bullish_true_probability": outcome.response.answers.market_bullish.noul,
                        "market_bullish_false_probability": 1.0 - outcome.response.answers.market_bullish.noul,
                        "false_probability_derivation": "1 - API market_bullish.noul",
                        "usage": outcome.response.usage.model_dump(),
                        "market_regime_probabilities": outcome.response.answers.market_regime.probabilities.model_dump()})
    print(json.dumps(display, ensure_ascii=False, indent=2))
    return 0 if outcome.status in ("ok", "disabled") else 1

if __name__ == "__main__":
    raise SystemExit(main())
