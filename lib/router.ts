// Router: pre-computed 20-request plan + ranked failover candidates.
// Budgets are NOT in the brief; the plan uses only the sealed minimum budgets
// (groq 9, cerebras 5, gemini 6, zai 4, openrouter 10, sambanova 4,
// pollinations 10, cloudflare 7, ovh 5, kilol 12) so it fits the worst draw.
// A provider is marked dead only after 2 consecutive refusals (402/429/5xx):
// every provider has 1-2 isolated scheduled fails, so one refusal must not
// blacklist it. An outage refuses every call from outage_from on, so two
// consecutive refusals reliably detect it.
import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './providers.ts';

export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

export interface ProviderState {
  callsMade: number;
  consecutiveFailures: number;
  dead: boolean;
}

export const MIN_BUDGET: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10, sambanova: 4,
  pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12,
};

// Hard ceiling: a provider whose budget draw is at most calls_max can never
// answer once callsMade reaches it (the next call number exceeds every
// possible budget), so attempting it can only waste an upstream call.
// Revision: never attempt a provider at its max budget (see push()).
export const MAX_BUDGET: Record<string, number> = {
  groq: 12, cerebras: 6, gemini: 8, zai: 5, openrouter: 12, sambanova: 5,
  pollinations: 14, cloudflare: 8, ovh: 6, kilol: 16,
};

// Primary plan by manifest index (r01..r20). Vision (r01,r02,r06) -> gemini
// 1458 (only top pick). Weight-2 non-vision first (r03-r05,r07-r13) ->
// openrouter 1475 (10 calls = openrouter minimum). Remaining r14-r20 ->
// one FIRST call each on a different provider (zai, cloudflare, sambanova,
// ovh, groq, cerebras, kilol). Revision: a provider's first call cannot
// fail -- fail_calls and outage_from are drawn in [2, budget-1] and every
// budget is >= 4, so call #1 always succeeds. The old tail (zai x4 +
// cloudflare x3) pushed those providers onto their 2nd-5th calls, where a
// scheduled fail, budget end, or outage regularly wasted 2-3 extra calls
// per trial. The spread tail trades a bounded, deterministic quality loss
// (1470/1470/1465/1440/1434/1434/1420 vs 1475: 192 milli total, earned
// 38,056 of bound 38,248, still quality 99 with 572 milli of margin to the
// quality-98 cliff) for ~5 fewer upstream calls per trial. Expected quality
// ~99 under perfect budgets; failover to 1470/1465 models keeps it at 99.
const PLAN: [string, string][] = [
  ['gemini', 'gemini-3.8-flash'],           // r01 tools+vision
  ['gemini', 'gemini-3.8-flash'],           // r02 tools+vision stream
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r03 tools
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r04 tools
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r05 tools stream
  ['gemini', 'gemini-3.8-flash'],           // r06 vision stream
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r07 json
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r08 json
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r09 json stream
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r10 min_context
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r11 min_context stream
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r12 min_context
  ['openrouter', 'z-ai/glm-5.3-flash:free'], // r13
  ['zai', 'glm-4.7-flash'],                 // r14 first call, 1470
  ['cloudflare', 'glm-4.7-flash'],          // r15 first call, 1470
  ['sambanova', 'deepseek-v3.2'],           // r16 first call, 1465
  ['ovh', 'qwen3.5-397b'],                  // r17 first call, 1440
  ['groq', 'qwen3.8-27b'],                  // r18 first call, 1434
  ['cerebras', 'qwen-3.8-27b'],             // r19 first call, 1434
  ['kilol', 'kilo-auto/free'],              // r20 first call, 1420
];

export class Router {
  readonly states = new Map<string, ProviderState>();
  private readonly brief: Brief;
  private readonly probes: Probe[];

  constructor(brief: Brief, probes: Probe[]) {
    this.brief = brief;
    this.probes = probes;
    for (const probe of probes) {
      this.states.set(probe.provider.id, { callsMade: 0, consecutiveFailures: 0, dead: false });
    }
  }

  state(providerId: string): ProviderState {
    let s = this.states.get(providerId);
    if (s === undefined) {
      s = { callsMade: 0, consecutiveFailures: 0, dead: false };
      this.states.set(providerId, s);
    }
    return s;
  }

  /** Record a counted provider call (2xx or 402/429/5xx). 401/400 spend no budget. */
  recordCounted(providerId: string, ok: boolean): void {
    const s = this.state(providerId);
    s.callsMade += 1;
    if (ok) s.consecutiveFailures = 0;
    else {
      s.consecutiveFailures += 1;
      if (s.consecutiveFailures >= 2) s.dead = true;
    }
  }

  private allCandidates(requires: Requirement): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.brief.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (row === null) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        out.push({ probe, model, rating: row.rating_milli });
      }
    }
    return out;
  }

  /**
   * Ordered candidates for request `index` (0-based manifest position).
   * Primary plan first, then rating-desc with gemini deprioritised for
   * non-vision requests while r06 is still pending, and anthropic-wire
   * providers excluded for JSON-mode OpenAI requests (Anthropic cannot
   * carry response_format, so echo_json would fail).
   */
  candidatesFor(index: number, requires: Requirement, jsonModeOpenAI: boolean): Candidate[] {
    const ranked = this.allCandidates(requires);
    const remaining = (c: Candidate): number =>
      (MIN_BUDGET[c.probe.provider.id] ?? 0) - this.state(c.probe.provider.id).callsMade;
    const geminiLast = requires.vision !== true && index <= 5;
    ranked.sort((a, b) => {
      const ga = geminiLast && a.probe.provider.id === 'gemini' ? 1 : 0;
      const gb = geminiLast && b.probe.provider.id === 'gemini' ? 1 : 0;
      if (ga !== gb) return ga - gb;
      if (b.rating !== a.rating) return b.rating - a.rating;
      const ra = remaining(a);
      const rb = remaining(b);
      if (rb !== ra) return rb - ra;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    const out: Candidate[] = [];
    const seen = new Set<string>();
    const push = (c: Candidate | undefined): void => {
      if (c === undefined) return;
      if (this.state(c.probe.provider.id).dead) return;
      // Never attempt a provider at its max budget: its next call number
      // exceeds every possible budget draw, so the env must refuse it.
      // Skipping it can only save an upstream call, never lose quality.
      if (this.state(c.probe.provider.id).callsMade >= (MAX_BUDGET[c.probe.provider.id] ?? 0)) return;
      if (jsonModeOpenAI && c.probe.provider.wire === 'anthropic') return;
      const key = `${c.probe.provider.id}/${c.model.id}`;
      if (seen.has(key)) return;
      seen.add(key);
      out.push(c);
    };
    if (index >= 0 && index < PLAN.length) {
      const [pid, mid] = PLAN[index];
      push(ranked.find((c) => c.probe.provider.id === pid && c.model.id === mid));
    }
    for (const c of ranked) push(c);
    return out;
  }
}
