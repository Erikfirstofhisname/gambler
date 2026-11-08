from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SituationInput(BaseModel):
    trial_number: int = Field(ge=1, le=100)
    previous_choice: float = Field(ge=0, le=4)
    previous_net_outcome: float
    cumulative_net_outcome_before_trial: float
    recent_loss_count_5: float = Field(ge=0, le=5)
    switched_deck_lag: float = Field(ge=0, le=1)
    model: Literal["best", "svm", "pnn", "kl_pnn"] = "best"


class DeckProb(BaseModel):
    deck: int
    probability: float
    is_suboptimal: bool


class PredictResponse(BaseModel):
    model: str
    deck_model: str
    switch_model: str
    predicted_deck: int
    predicted_deck_suboptimal: bool
    deck_probabilities: list[DeckProb]
    switch_probability: float
    predicted_switch: bool
    explanation: str


class ParticipantSummary(BaseModel):
    participant_id: str
    cluster: int
    tier: str
    switch_rate: float
    bad_deck_rate: float
    late_bad_deck_rate: float
    late_advantageous_rate: float
    mean_net_outcome: float
    final_cumulative: float
    loss_sensitivity: float
    deck_prefs: dict[str, float]


class ParticipantDetail(ParticipantSummary):
    similar_players: list[ParticipantSummary]
    bias: dict
