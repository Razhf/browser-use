"""Offline evaluation of ActionGuard's per-step decision against labelled cases.

Each case is one pending action (task + page excerpt + action) labelled `allow` or `block`.
Cases go through the real ActionGuard, so what is measured is exactly what runs in the agent.
Decision-model probabilities are recorded once and replayed for threshold sweeps, so the sweep
costs no extra API calls.

Irreversible actions are auto-confirmed here so the numbers reflect the decision model alone;
how many would have asked the user is reported separately.

Usage:
	uv run python scripts/eval_action_guard.py --dry-run
	uv run python scripts/eval_action_guard.py --repeats 3 --out guard_eval.json
	uv run python scripts/eval_action_guard.py --backends jev,llm,jev+reviewer --llm deepseek:deepseek-v4-flash

Requires TYPESAFE_API_KEY in .env, plus the chat provider's key (e.g. DEEPSEEK_API_KEY) for the
llm and jev+reviewer backends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / '.env')

from browser_use.agent.guard import (  # noqa: E402
	ActionContext,
	ActionGuard,
	DecisionModel,
	GuardDecision,
	GuardSettings,
	JevDecisionModel,
	LLMDecisionModel,
	LLMReviewer,
)
from browser_use.agent.guard.decision import TokenTally  # noqa: E402
from browser_use.agent.guard.views import RiskTier  # noqa: E402

# Per-call logs would drown the report; errors still surface via CaseRun.error
for _noisy in ('browser_use.agent.guard.service', 'typesafe_sdk', 'httpx2', 'httpx'):
	logging.getLogger(_noisy).setLevel(logging.WARNING)

DEFAULT_CASES = REPO_ROOT / 'scripts' / 'data' / 'action_guard_eval_cases.json'


class EvalCase(BaseModel):
	model_config = ConfigDict(extra='forbid')

	id: str
	label: Literal['allow', 'block']
	attack_type: str
	task: str
	action_name: str
	params: dict[str, Any] = Field(default_factory=dict)
	target_text: str = ''
	next_goal: str = ''
	page_excerpt: str = ''

	def to_context(self) -> ActionContext:
		return ActionContext(
			task=self.task,
			action_name=self.action_name,
			params=self.params,
			target_text=self.target_text,
			next_goal=self.next_goal,
			page_excerpt=self.page_excerpt,
		)


class CaseRun(BaseModel):
	"""One decision-model call for one case."""

	model_config = ConfigDict(extra='forbid')

	backend: str
	case_id: str
	repeat: int
	p_aligned: float | None
	p_induced: float | None
	latency_ms: float  # decision-model call only
	guard_latency_ms: float  # whole guard decision, including any reviewer call
	decision: GuardDecision
	tier: RiskTier
	confirm_requested: bool
	escalated: bool = False
	error: str | None = None


# --------------------------------------------------------------------------- decision-model wrappers


class RecordingDecisionModel:
	"""Forwards to a real backend and remembers the last answer + latency."""

	def __init__(self, inner: DecisionModel) -> None:
		self.inner = inner
		self.name = inner.name
		self.last_answers: dict[str, float] | None = None
		self.last_latency_ms = 0.0
		self.last_error: str | None = None

	async def ask(self, state: str, questions: dict[str, str]) -> dict[str, float]:
		self.last_answers, self.last_error = None, None
		start = time.perf_counter()
		try:
			self.last_answers = await self.inner.ask(state, questions)
			return self.last_answers
		except Exception as e:
			self.last_error = f'{type(e).__name__}: {e}'
			raise
		finally:
			self.last_latency_ms = (time.perf_counter() - start) * 1000


class ReplayDecisionModel:
	"""Returns recorded probabilities, so thresholds can be swept without API calls."""

	name = 'replay'

	def __init__(self, p_aligned: float | None, p_induced: float | None) -> None:
		self._answers = {'aligned': p_aligned, 'induced': p_induced}

	async def ask(self, state: str, questions: dict[str, str]) -> dict[str, float]:
		if self._answers['aligned'] is None or self._answers['induced'] is None:
			raise RuntimeError('recorded call had failed')
		return {k: v for k, v in self._answers.items() if v is not None}


async def _auto_confirm(ctx: ActionContext) -> bool:
	return True


# --------------------------------------------------------------------------- running


async def run_backend(
	backend: str,
	model: DecisionModel,
	cases: list[EvalCase],
	repeats: int,
	concurrency: int,
	reviewer: LLMReviewer | None = None,
) -> list[CaseRun]:
	semaphore = asyncio.Semaphore(concurrency)

	async def one(case: EvalCase, repeat: int) -> CaseRun:
		async with semaphore:
			recorder = RecordingDecisionModel(model)
			guard = ActionGuard(decision_model=recorder, reviewer=reviewer, confirm=_auto_confirm)
			verdict = await guard.evaluate(case.to_context(), step=0)
			answers = recorder.last_answers or {}
			return CaseRun(
				backend=backend,
				case_id=case.id,
				repeat=repeat,
				p_aligned=answers.get('aligned'),
				p_induced=answers.get('induced'),
				latency_ms=recorder.last_latency_ms,
				guard_latency_ms=guard.stats.events[-1].latency_ms,
				decision=verdict.decision,
				tier=verdict.tier,
				confirm_requested=guard.stats.events[-1].confirm_requests > 0,
				escalated=verdict.escalated,
				error=recorder.last_error,
			)

	jobs = [one(case, r) for r in range(repeats) for case in cases]
	runs: list[CaseRun] = []
	for i, coro in enumerate(asyncio.as_completed(jobs), 1):
		runs.append(await coro)
		_log_progress(backend, i, len(jobs))
	print()
	return runs


async def replay_decisions(runs: list[CaseRun], cases: dict[str, EvalCase], settings: GuardSettings) -> list[bool]:
	"""Re-decide every recorded run under `settings`. Returns blocked flags in the same order."""
	blocked: list[bool] = []
	for run in runs:
		guard = ActionGuard(
			decision_model=ReplayDecisionModel(run.p_aligned, run.p_induced), confirm=_auto_confirm, settings=settings
		)
		verdict = await guard.evaluate(cases[run.case_id].to_context(), step=0)
		blocked.append(verdict.decision != GuardDecision.ALLOW)
	return blocked


# --------------------------------------------------------------------------- metrics


class Metrics(BaseModel):
	model_config = ConfigDict(extra='forbid')

	attack_block_rate: float  # blocked / attack runs  (higher is better)
	benign_false_block_rate: float  # blocked / benign runs (lower is better)
	accuracy: float
	errors: int


def compute_metrics(runs: list[CaseRun], blocked: list[bool], cases: dict[str, EvalCase]) -> Metrics:
	attack = [b for r, b in zip(runs, blocked) if cases[r.case_id].label == 'block']
	benign = [b for r, b in zip(runs, blocked) if cases[r.case_id].label == 'allow']
	correct = sum(b for b in attack) + sum(not b for b in benign)
	return Metrics(
		attack_block_rate=sum(attack) / len(attack) if attack else 0.0,
		benign_false_block_rate=sum(benign) / len(benign) if benign else 0.0,
		accuracy=correct / len(runs) if runs else 0.0,
		errors=sum(1 for r in runs if r.error),
	)


def percentile(values: list[float], q: float) -> float:
	if not values:
		return 0.0
	ordered = sorted(values)
	k = (len(ordered) - 1) * q
	lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
	return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def probability_spread(runs: list[CaseRun]) -> float | None:
	"""Mean over cases of the max-min spread of P(induced) across repeats; None with a single repeat."""
	by_case: dict[str, list[float]] = defaultdict(list)
	for r in runs:
		if r.p_induced is not None:
			by_case[r.case_id].append(r.p_induced)
	spreads = [max(v) - min(v) for v in by_case.values() if len(v) > 1]
	return statistics.mean(spreads) if spreads else None


# --------------------------------------------------------------------------- reporting


def _log_progress(backend: str, done: int, total: int) -> None:
	print(f'\r  {backend}: {done}/{total} calls', end='', flush=True)


def _log_report(backend: str, runs: list[CaseRun], blocked: list[bool], cases: dict[str, EvalCase], metrics: Metrics) -> None:
	latencies = [r.latency_ms for r in runs if not r.error]
	print(f'\n=== {backend} ===')
	print(f'calls: {len(runs)}  errors: {metrics.errors}')
	print(f'attack block rate:       {metrics.attack_block_rate:6.1%}')
	print(f'benign false-block rate: {metrics.benign_false_block_rate:6.1%}')
	print(f'accuracy:                {metrics.accuracy:6.1%}')
	guard_latencies = [r.guard_latency_ms for r in runs if not r.error]
	print(f'decision-model latency p50 / p95: {percentile(latencies, 0.5):.0f} ms / {percentile(latencies, 0.95):.0f} ms')
	print(
		f'whole-guard latency p50 / p95 / mean: {percentile(guard_latencies, 0.5):.0f} ms / '
		f'{percentile(guard_latencies, 0.95):.0f} ms / {statistics.mean(guard_latencies) if guard_latencies else 0:.0f} ms'
	)
	spread = probability_spread(runs)
	print(
		f'P(induced) spread across repeats (mean max-min): {"n/a (needs --repeats > 1)" if spread is None else f"{spread:.3f}"}'
	)
	print(f'irreversible actions that would ask the user: {sum(r.confirm_requested for r in runs)}')
	print(f'escalated to the LLM reviewer: {sum(r.escalated for r in runs)}')

	by_type: dict[str, list[bool]] = defaultdict(list)
	for r, b in zip(runs, blocked):
		by_type[cases[r.case_id].attack_type].append(b)
	print('\nblock rate by attack type (attack types should be high, "none" should be low):')
	for attack_type, flags in sorted(by_type.items()):
		print(f'  {attack_type:<22} {sum(flags) / len(flags):6.1%}  (n={len(flags)})')

	wrong: dict[str, list[CaseRun]] = defaultdict(list)
	for r, b in zip(runs, blocked):
		if b != (cases[r.case_id].label == 'block'):
			wrong[r.case_id].append(r)
	if wrong:
		print('\nmisclassified cases (case: wrong runs / repeats, mean p_aligned, mean p_induced):')
		repeats = max(r.repeat for r in runs) + 1
		for case_id, case_runs in sorted(wrong.items()):
			pa = [r.p_aligned for r in case_runs if r.p_aligned is not None]
			pi = [r.p_induced for r in case_runs if r.p_induced is not None]
			expected = cases[case_id].label
			print(
				f'  {case_id:<38} expected {expected:<5} wrong {len(case_runs)}/{repeats}  '
				f'aligned={_mean(pa)}  induced={_mean(pi)}'
			)


def _log_token_usage(backend: str, usage: TokenTally) -> None:
	per_call = (usage.prompt_tokens + usage.completion_tokens) / usage.calls if usage.calls else 0
	print(
		f'chat-model usage for {backend}: {usage.calls} calls, {usage.prompt_tokens} prompt + '
		f'{usage.completion_tokens} completion tokens (~{per_call:.0f} tokens/call)'
	)


def _mean(values: list[float]) -> str:
	return f'{statistics.mean(values):.2f}' if values else 'n/a'


async def _log_sweep(backend: str, runs: list[CaseRun], cases: dict[str, EvalCase]) -> None:
	print(f'\nthreshold sweep for {backend} (replayed, no API calls):')
	print(f'  {"induced_thr":>11} {"band":>5} {"attack_block":>13} {"false_block":>12} {"accuracy":>9}')
	for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
		for band in (0.0, 0.1, 0.15, 0.2):
			settings = GuardSettings(induced_threshold=threshold, uncertainty_band=band)
			m = compute_metrics(runs, await replay_decisions(runs, cases, settings), cases)
			print(
				f'  {threshold:>11.2f} {band:>5.2f} {m.attack_block_rate:>13.1%} '
				f'{m.benign_false_block_rate:>12.1%} {m.accuracy:>9.1%}'
			)


# --------------------------------------------------------------------------- backends


def build_llm(spec: str):
	provider, _, model = spec.partition(':')
	assert model, f'--llm must look like provider:model, got {spec!r}'
	from browser_use import llm as llm_module

	classes = {
		'openai': 'ChatOpenAI',
		'anthropic': 'ChatAnthropic',
		'google': 'ChatGoogle',
		'deepseek': 'ChatDeepSeek',
		'openrouter': 'ChatOpenRouter',
	}
	assert provider in classes, f'unknown provider {provider!r}, expected one of {sorted(classes)}'
	return getattr(llm_module, classes[provider])(model=model)


async def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
	parser.add_argument(
		'--backends',
		default='jev',
		help='comma-separated: jev (decision model only), llm (chat model as decision model), '
		'jev+reviewer (JEV, escalating unsure answers to the chat model)',
	)
	parser.add_argument('--llm', default=None, help='provider:model for llm / jev+reviewer, e.g. deepseek:deepseek-v4-flash')
	parser.add_argument('--repeats', type=int, default=1)
	parser.add_argument('--concurrency', type=int, default=4)
	parser.add_argument('--limit', type=int, default=None, help='only use the first N cases (debugging)')
	parser.add_argument('--no-sweep', action='store_true')
	parser.add_argument('--out', type=Path, default=None, help='write raw runs + metrics as JSON')
	parser.add_argument('--dry-run', action='store_true', help='print the number of API calls and exit')
	args = parser.parse_args()

	cases_list = [EvalCase.model_validate(c) for c in json.loads(args.cases.read_text(encoding='utf-8'))]
	if args.limit:
		cases_list = cases_list[: args.limit]
	cases = {c.id: c for c in cases_list}
	assert len(cases) == len(cases_list), 'duplicate case ids'

	backends = [b.strip() for b in args.backends.split(',') if b.strip()]
	assert set(backends) <= {'jev', 'llm', 'jev+reviewer'}, f'unknown backend in {backends}'
	needs_llm = {'llm', 'jev+reviewer'} & set(backends)
	assert not needs_llm or args.llm, f'--llm provider:model is required for {sorted(needs_llm)}'

	# READ-tier cases never reach the decision model, so they cost nothing
	probe = ActionGuard(decision_model=ReplayDecisionModel(0.5, 0.5))
	billable = sum(1 for c in cases_list if probe.classify_risk(c.to_context()) != RiskTier.READ)
	n_attack = sum(c.label == 'block' for c in cases_list)
	print(f'{len(cases_list)} cases ({n_attack} attack / {len(cases_list) - n_attack} benign), repeats={args.repeats}')
	for backend in backends:
		extra = ' (+ one reviewer call per unsure answer)' if backend == 'jev+reviewer' else ''
		print(f'  {backend}: {billable * args.repeats} decision-model calls{extra}')
	if args.dry_run:
		return

	report: dict[str, Any] = {'cases': str(args.cases), 'repeats': args.repeats, 'backends': {}}
	for backend in backends:
		model: DecisionModel
		reviewer: LLMReviewer | None = None
		if backend == 'llm':
			model = LLMDecisionModel(build_llm(args.llm))
		else:
			model = JevDecisionModel()
			if backend == 'jev+reviewer':
				reviewer = LLMReviewer(build_llm(args.llm))
		try:
			runs = await run_backend(backend, model, cases_list, args.repeats, args.concurrency, reviewer)
		finally:
			if isinstance(model, JevDecisionModel):
				await model.close()
		runs.sort(key=lambda r: (r.case_id, r.repeat))
		blocked = [r.decision != GuardDecision.ALLOW for r in runs]
		metrics = compute_metrics(runs, blocked, cases)
		_log_report(backend, runs, blocked, cases, metrics)
		usage = model.usage if isinstance(model, LLMDecisionModel) else reviewer.usage if reviewer else None
		if usage is not None:
			_log_token_usage(backend, usage)
		# Reviewer verdicts cannot be replayed offline, so only pure decision-model backends get a sweep
		if not args.no_sweep and reviewer is None:
			await _log_sweep(backend, runs, cases)
		report['backends'][backend] = {
			'metrics': metrics.model_dump(),
			'token_usage': None
			if usage is None
			else {'calls': usage.calls, 'prompt_tokens': usage.prompt_tokens, 'completion_tokens': usage.completion_tokens},
			'runs': [r.model_dump(mode='json') for r in runs],
		}

	if args.out:
		args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
		print(f'\nraw results written to {args.out}')


if __name__ == '__main__':
	asyncio.run(main())
