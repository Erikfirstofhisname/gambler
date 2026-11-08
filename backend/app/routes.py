from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .features import bias_overview, participant_bias
from .model_store import get_store
from .schemas import (
    ParticipantDetail,
    ParticipantSummary,
    PredictResponse,
    SituationInput,
)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(body: SituationInput):
    store = get_store()
    X = store.situation_frame(body.model_dump())
    deck_name = store.resolve_name(body.model, "deck")
    switch_name = store.resolve_name(body.model, "switch")
    deck_model = store.deck_model(body.model)
    switch_model = store.switch_model(body.model)

    deck_probs = deck_model.predict_proba(X)[0]
    deck_classes = [int(c) for c in deck_model.classes_]
    predicted_deck = int(deck_model.predict(X)[0])
    switch_prob = float(switch_model.predict_proba(X)[0][1])
    predicted_switch = switch_prob >= 0.5
    suboptimal = store.is_suboptimal_deck(predicted_deck)

    probs = [
        {
            "deck": cls,
            "probability": float(prob),
            "is_suboptimal": store.is_suboptimal_deck(cls),
        }
        for cls, prob in sorted(
            zip(deck_classes, deck_probs), key=lambda x: x[1], reverse=True
        )
    ]

    mode = "best held-out models" if body.model in ("best", "auto") else body.model
    if suboptimal:
        explanation = (
            f"Using {mode} (deck={deck_name}, switch={switch_name}): predicts deck "
            f"{predicted_deck}, a disadvantageous (suboptimal) choice. "
            f"Switch probability is {switch_prob:.0%}."
        )
    else:
        explanation = (
            f"Using {mode} (deck={deck_name}, switch={switch_name}): predicts deck "
            f"{predicted_deck}, an advantageous choice. "
            f"Switch probability is {switch_prob:.0%}."
        )

    if body.previous_net_outcome < 0 and suboptimal:
        explanation += " Prior loss raises risk of chasing bad decks."
    if body.trial_number >= 51 and suboptimal:
        explanation += " Late-trial bad-deck picks are especially costly."

    return PredictResponse(
        model=body.model,
        deck_model=deck_name,
        switch_model=switch_name,
        predicted_deck=predicted_deck,
        predicted_deck_suboptimal=suboptimal,
        deck_probabilities=probs,
        switch_probability=switch_prob,
        predicted_switch=predicted_switch,
        explanation=explanation,
    )


@router.get("/participants", response_model=list[ParticipantSummary])
def list_participants(tier: str | None = Query(default=None)):
    store = get_store()
    profiles = list(store.meta["profiles"].values())
    if tier:
        profiles = [p for p in profiles if p["tier"] == tier.upper()]
    profiles.sort(key=lambda p: (p["tier"], -p["late_advantageous_rate"]))
    return profiles


@router.get("/participants/{participant_id}", response_model=ParticipantDetail)
def get_participant(participant_id: str):
    store = get_store()
    profiles = store.meta["profiles"]
    if participant_id not in profiles:
        raise HTTPException(status_code=404, detail="Participant not found")

    profile = profiles[participant_id]
    same_tier = [
        p
        for pid, p in profiles.items()
        if p["tier"] == profile["tier"] and pid != participant_id
    ]
    same_tier.sort(
        key=lambda p: abs(p["late_bad_deck_rate"] - profile["late_bad_deck_rate"])
    )
    similar = same_tier[:5]

    try:
        bias = participant_bias(store.trials, participant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Participant not found") from exc

    return {**profile, "similar_players": similar, "bias": bias}


@router.get("/bias/overview")
def get_bias_overview():
    store = get_store()
    return bias_overview(store.trials)


@router.get("/bias/participant/{participant_id}")
def get_bias_participant(participant_id: str):
    store = get_store()
    try:
        return participant_bias(store.trials, participant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Participant not found") from exc


@router.get("/tiers")
def get_tiers():
    store = get_store()
    return {"tiers": store.meta["tiers_summary"]}


@router.get("/models/metrics")
def get_model_metrics():
    store = get_store()
    return {
        **store.meta["metrics"],
        "best_models": store.meta.get("best_models", {}),
    }
