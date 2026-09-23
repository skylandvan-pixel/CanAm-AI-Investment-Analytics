"""Phase 1 questions only; no portfolio or committee interpretation."""

QUESTION_VERSION = "market-regime-v1"


def market_questions() -> dict:
    return {
        "market_bullish": {
            "type": "noul",
            "instructions": "Based only on the supplied market state, is the U.S. equity market currently in a bullish market regime?",
            "criteria": {
                "true": "Current daily trend evidence supports a bullish regime; this is not a forecast of tomorrow's return.",
                "false": "Current daily trend evidence does not support a bullish regime.",
            },
        },
        "market_regime": {
            "type": "choice",
            "instructions": "Based only on the supplied market state, what is the current U.S. equity market regime?",
            "criteria": {
                "BULL": "Predominantly positive current daily trend evidence.",
                "NEUTRAL": "Mixed or transitional current daily trend evidence.",
                "BEAR": "Predominantly negative current daily trend evidence.",
            },
        },
    }
