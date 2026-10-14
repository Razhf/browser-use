"""Data models for the per-step action guard."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskTier(str, Enum):
	"""How much damage an action can do if it is wrong.

	Assigned by deterministic rules (never by a model), so page content cannot talk an
	action down into a lower tier.
	"""

	READ = 'read'  # scroll, extract, find, wait, ...
	WRITE = 'write'  # typing into fields, selecting options, clicking non-committing elements, navigation
	IRREVERSIBLE = 'irreversible'  # submit / pay / send / delete style commits


class GuardDecision(str, Enum):
	ALLOW = 'allow'
	BLOCK_REDECIDE = 'block_redecide'  # bounced back to the agent LLM with feedback
	BLOCK_CONFIRM_DENIED = 'block_confirm_denied'  # irreversible action whose confirmation was refused or unavailable


class ActionContext(BaseModel):
	"""Everything the guard is allowed to see about one pending action."""

	model_config = ConfigDict(extra='forbid')

	task: str
	action_name: str
	params: dict[str, Any] = Field(default_factory=dict)
	target_text: str = ''  # visible text / label of the element the action targets, if any
	next_goal: str = ''  # what the agent LLM says it is trying to do this step
	recent_steps: list[str] = Field(default_factory=list)  # short summaries of the last few steps
	page_excerpt: str = ''  # bounded slice of the serialized page the agent saw


class GuardVerdict(BaseModel):
	model_config = ConfigDict(extra='forbid')

	decision: GuardDecision
	tier: RiskTier
	p_aligned: float | None = None  # decision-model P(action serves the user task)
	p_induced: float | None = None  # decision-model P(action was steered by page text)
	escalated: bool = False  # True if the LLM reviewer was consulted
	reason: str = ''


class GuardEvent(BaseModel):
	"""One guard evaluation, kept for offline evaluation (cost / latency / decisions)."""

	model_config = ConfigDict(extra='forbid')

	step: int
	action_name: str
	tier: RiskTier
	decision: GuardDecision
	p_aligned: float | None = None
	p_induced: float | None = None
	escalated: bool = False
	decision_model_calls: int = 0
	reviewer_calls: int = 0
	confirm_requests: int = 0
	latency_ms: float = 0.0


class GuardStats(BaseModel):
	model_config = ConfigDict(extra='forbid')

	events: list[GuardEvent] = Field(default_factory=list)
	no_effect_warnings: int = 0

	@property
	def decision_model_calls(self) -> int:
		return sum(e.decision_model_calls for e in self.events)

	@property
	def reviewer_calls(self) -> int:
		return sum(e.reviewer_calls for e in self.events)

	@property
	def blocked(self) -> int:
		return sum(1 for e in self.events if e.decision != GuardDecision.ALLOW)


ConfirmCallback = Callable[[ActionContext], Awaitable[bool]]
"""Asks a human whether an irreversible action may run. Must return True to allow it."""


DEFAULT_IRREVERSIBLE_KEYWORDS: tuple[str, ...] = (
	'submit',
	'pay',
	'purchase',
	'buy',
	'checkout',
	'place order',
	'confirm',
	'send',
	'delete',
	'remove',
	'transfer',
	'subscribe',
	'publish',
	'提交',
	'支付',
	'付款',
	'购买',
	'下单',
	'结算',
	'确认',
	'发送',
	'删除',
	'转账',
	'订阅',
	'发布',
)


class GuardSettings(BaseModel):
	"""Tunables for ActionGuard.

	Thresholds apply to Noul probabilities returned by the decision model. A probability within
	`uncertainty_band` of 0.5 is treated as "not sure" and escalated instead of trusted.
	"""

	model_config = ConfigDict(extra='forbid')

	enabled: bool = True
	aligned_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
	induced_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
	uncertainty_band: float = Field(default=0.15, ge=0.0, le=0.5)
	page_excerpt_chars: int = Field(default=4000, ge=0)
	recent_steps: int = Field(default=3, ge=0)
	check_no_effect: bool = True
	irreversible_keywords: list[str] = Field(default_factory=lambda: list(DEFAULT_IRREVERSIBLE_KEYWORDS))
