// Upstream calls: build the provider-wire request, call it, and normalise
// the answer to {text, input, output, stop} -- collected whole before any
// client rendering, because usage arrives last (OpenAI usage chunk, Gemini
// final frame, Anthropic message_delta) and Anthropic message_start needs
// input tokens up front.
import {
  authHeaders,
  isRecord,
  joinBase,
  type Json,
  type Probe,
} from './providers.ts';
import { anthropicToOpenAI } from './wire.ts';
import { geminiBody, isJsonMode, stopsOf } from './gemini.ts';

export interface Normalized {
  text: string;
  input: number;
  output: number;
  stop: string;
}

export type UpstreamResult =
  | { ok: true; status: 200; result: Normalized }
  | { ok: false; status: number };

const CALL_TIMEOUT_MS = 10000;

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textFromOpenAIContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

function textFromAnthropicContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** OpenAI client body -> Anthropic upstream body (zai). JSON mode never reaches here (router excludes it). */
export function openAIToAnthropicUpstream(body: Json, model: string): Json {
  const out: Json = { model };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const outMessages: Json[] = [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') {
      const t = textFromOpenAIContent(m.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (m.role === 'user') {
      if (typeof m.content === 'string') {
        outMessages.push({ role: 'user', content: m.content });
        continue;
      }
      if (Array.isArray(m.content)) {
        const blocks: Json[] = [];
        for (const b of m.content) {
          if (!isRecord(b)) continue;
          if (b.type === 'text' && typeof b.text === 'string') blocks.push({ type: 'text', text: b.text });
          else if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
            const match = /^data:([^;]+);base64,(.*)$/s.exec(b.image_url.url);
            if (match !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } });
            else blocks.push({ type: 'image', source: { type: 'url', url: b.image_url.url } });
          }
        }
        if (blocks.length > 0) outMessages.push({ role: 'user', content: blocks });
        continue;
      }
    }
    if (m.role === 'assistant' && typeof m.content === 'string') outMessages.push({ role: 'assistant', content: m.content });
  }
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  out.messages = outMessages;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body, 'openai');
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((t) => {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') return [];
      return [{ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', input_schema: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

interface UpstreamRequest {
  url: string;
  headers: Record<string, string>;
  body: Json;
  stream: boolean;
}

function buildUpstream(envUrl: string, probe: Probe, modelId: string, clientBody: Json, clientApi: 'openai' | 'anthropic'): UpstreamRequest {
  const base = joinBase(envUrl, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  const stream = clientBody.stream === true;
  if (probe.provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    const body: Json =
      clientApi === 'openai'
        ? { ...clientBody, model: modelId, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(clientBody, modelId);
    return { url: `${base}/v1/chat/completions`, headers, body, stream };
  }
  if (probe.provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    const body: Json = clientApi === 'anthropic' ? { ...clientBody, model: modelId } : openAIToAnthropicUpstream(clientBody, modelId);
    return { url: `${base}/v1/messages`, headers, body, stream };
  }
  // gemini
  const streamPath = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return {
    url: `${base}/v1beta/models/${encodeURIComponent(modelId)}:${streamPath}`,
    headers,
    body: geminiBody(clientBody, clientApi),
    stream,
  };
}

/** Split SSE text into frames: each frame is {event, data} pairs. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const dataParts: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataParts.push(line.slice(5).trim());
    }
    if (dataParts.length > 0) frames.push({ event, data: dataParts.join('\n') });
  }
  return frames;
}

function parseJsonSafe(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function normalizeOpenAI(body: unknown, stream: boolean): Normalized | null {
  if (stream) {
    let text = '';
    let input = 0;
    let output = 0;
    let stop = 'stop';
    if (typeof body !== 'string') return null;
    for (const frame of sseFrames(body)) {
      if (frame.data === '[DONE]') continue;
      const chunk = parseJsonSafe(frame.data);
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = num(chunk.usage.prompt_tokens);
        output = num(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') stop = choice.finish_reason;
    }
    return { text, input, output, stop };
  }
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textFromOpenAIContent(message.content),
    input: num(usage.prompt_tokens),
    output: num(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

function normalizeAnthropic(body: unknown, stream: boolean): Normalized | null {
  if (stream) {
    let text = '';
    let input = 0;
    let output = 0;
    let stop = 'end_turn';
    if (typeof body !== 'string') return null;
    for (const frame of sseFrames(body)) {
      const chunk = parseJsonSafe(frame.data);
      if (!isRecord(chunk)) continue;
      if (frame.event === 'message_start' && isRecord(chunk.message) && isRecord(chunk.message.usage)) {
        input = num(chunk.message.usage.input_tokens);
      }
      if (frame.event === 'content_block_delta' && isRecord(chunk.delta) && chunk.delta.type === 'text_delta' && typeof chunk.delta.text === 'string') {
        text += chunk.delta.text;
      }
      if (frame.event === 'message_delta') {
        if (isRecord(chunk.delta) && typeof chunk.delta.stop_reason === 'string') stop = chunk.delta.stop_reason;
        if (isRecord(chunk.usage)) output = num(chunk.usage.output_tokens);
      }
    }
    return { text, input, output, stop };
  }
  if (!isRecord(body)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textFromAnthropicContent(body.content),
    input: num(usage.input_tokens),
    output: num(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

function normalizeGemini(body: unknown, stream: boolean): Normalized | null {
  if (stream) {
    let text = '';
    let input = 0;
    let output = 0;
    let stop = 'STOP';
    if (typeof body !== 'string') return null;
    for (const frame of sseFrames(body)) {
      const chunk = parseJsonSafe(frame.data);
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (cand !== null) {
        const content = isRecord(cand.content) ? cand.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const p of parts) if (isRecord(p) && typeof p.text === 'string') text += p.text;
        if (typeof cand.finishReason === 'string') stop = cand.finishReason;
      }
      if (isRecord(chunk.usageMetadata)) {
        input = num(chunk.usageMetadata.promptTokenCount);
        output = num(chunk.usageMetadata.candidatesTokenCount);
      }
    }
    return { text, input, output, stop };
  }
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const p of parts) if (isRecord(p) && typeof p.text === 'string') text += p.text;
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    input: num(meta.promptTokenCount),
    output: num(meta.candidatesTokenCount),
    stop: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP',
  };
}

export async function callUpstream(
  envUrl: string,
  probe: Probe,
  modelId: string,
  clientBody: Json,
  clientApi: 'openai' | 'anthropic',
): Promise<UpstreamResult> {
  const req = buildUpstream(envUrl, probe, modelId, clientBody, clientApi);
  let res: Response;
  try {
    res = await fetch(req.url, {
      method: 'POST',
      headers: req.headers,
      body: JSON.stringify(req.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!res.ok) return { ok: false, status: res.status };
  const raw = await res.text();
  const parsed: unknown = req.stream ? raw : parseJsonSafe(raw);
  const norm =
    probe.provider.wire === 'openai'
      ? normalizeOpenAI(parsed, req.stream)
      : probe.provider.wire === 'anthropic'
        ? normalizeAnthropic(parsed, req.stream)
        : normalizeGemini(parsed, req.stream);
  if (norm === null) return { ok: false, status: 502 };
  // A 2xx that is not a well-formed answer is treated as a failure so the
  // request can fail over instead of serving a broken echo.
  if (norm.text === '') return { ok: false, status: 502 };
  return { ok: true, status: 200, result: norm };
}

export { isJsonMode };
