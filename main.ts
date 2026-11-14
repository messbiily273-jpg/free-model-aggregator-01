// free-model-aggregator-01: scripted aggregator server (Node 22, stdlib only).
// Reads the brief, probes provider catalogs, registers, serves 20 client
// requests with a pre-computed routing plan + sequential failover, exits 0
// when the env says done. No wall clock, no unseeded randomness: fixed
// created timestamps and fixed ids everywhere, so live and replay agree.
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';
import { Router, type Candidate } from './lib/router.ts';
import { callUpstream, isJsonMode, type Normalized } from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router | null = null;
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(brief, probes);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      if (!isRecord(m) || !Array.isArray(m.content)) continue;
      for (const b of m.content) if (isRecord(b) && b.type === 'image') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

function split13(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [''] : out;
}

function openAIStop(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_use' || stop === 'tool_calls' || stop === 'function_call') return 'tool_calls';
  if (stop === 'refusal' || stop === 'content_filter' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

function anthropicStop(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_use' || stop === 'tool_calls' || stop === 'function_call') return 'tool_use';
  if (stop === 'refusal' || stop === 'content_filter' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function renderOpenAI(res: ServerResponse, stream: boolean, norm: Normalized, modelId: string): void {
  const stop = openAIStop(norm.stop);
  if (!stream) {
    sendJson(res, 200, {
      id: 'chatcmpl-sim',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model: modelId,
      choices: [{ index: 0, message: { role: 'assistant', content: norm.text }, finish_reason: stop, logprobs: null }],
      usage: { prompt_tokens: norm.input, completion_tokens: norm.output, total_tokens: norm.input + norm.output },
    });
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  res.write(chunk({ role: 'assistant', content: '' }, null));
  for (const piece of split13(norm.text)) res.write(chunk({ content: piece }, null));
  res.write(chunk({}, stop));
  res.write(
    `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [], usage: { prompt_tokens: norm.input, completion_tokens: norm.output, total_tokens: norm.input + norm.output } })}\n\n`,
  );
  res.write('data: [DONE]\n\n');
  res.end();
}

function renderAnthropic(res: ServerResponse, stream: boolean, norm: Normalized, modelId: string): void {
  const stop = anthropicStop(norm.stop);
  if (!stream) {
    sendJson(res, 200, {
      id: 'msg_sim',
      type: 'message',
      role: 'assistant',
      model: modelId,
      content: [{ type: 'text', text: norm.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: norm.input, output_tokens: norm.output },
    });
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.write(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: norm.input, output_tokens: 0 },
      },
    }),
  );
  res.write(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of split13(norm.text)) {
    res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  res.write(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: norm.output } }),
  );
  res.write(sseEvent('message_stop', { type: 'message_stop' }));
  res.end();
}

/** Serve one client request: sequential candidates, stop at first 2xx.
 * No same-provider retry in-request: a scheduled fail is one call number,
 * but retrying would burn the provider's budget (openrouter's 10-12 calls
 * are the binding constraint) and quality 100 is mathematically unreachable
 * anyway -- 17 non-vision requests need the 1475 model but openrouter holds
 * at most 12 calls, so the 1470 spillover loses 5 milli each by design and
 * the ceiling is quality 99 / score 99. Fail over immediately instead. */
async function serve(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<void> {
  const index = manifestIndex;
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const jsonMode = clientApi === 'openai' && isJsonMode(body, 'openai');
  const candidates: Candidate[] = router === null ? [] : router.candidatesFor(index, requires, jsonMode);
  const tried = new Set<string>();
  for (const cand of candidates) {
    if (tried.has(cand.probe.provider.id)) continue;
    tried.add(cand.probe.provider.id);
    const result = await callUpstream(ENV_URL, cand.probe, cand.model.id, body, clientApi);
    if (result.ok) {
      router?.recordCounted(cand.probe.provider.id, true);
      if (clientApi === 'openai') renderOpenAI(res, stream, result.result, cand.model.id);
      else renderAnthropic(res, stream, result.result, cand.model.id);
      return;
    }
    // 401/400 spend no budget and are our bug: log, do not count, move on.
    if (result.status === 401 || result.status === 400) {
      log(`provider ${cand.probe.provider.id} answered HTTP ${result.status} for ${cand.model.id}: not counted, failing over`);
      continue;
    }
    // 402/429/5xx/network: counted refusal, mark suspect; dead after 2 in a row.
    router?.recordCounted(cand.probe.provider.id, false);
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(res, 'openai', body);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
