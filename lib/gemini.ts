// Gemini wire: client body (OpenAI or Anthropic) -> generateContent request.
// The env's geminiParsed reads: systemInstruction.parts text (joined "\n"),
// the last non-model content's parts text (joined "\n") and inlineData image
// bytes, functionDeclarations names in order, generationConfig
// maxOutputTokens / stopSequences / responseMimeType. So this translation
// must carry exactly the last user message's text and image bytes, the
// system text, tool names in order, max_tokens, stops and JSON mode.
import { isRecord, type Json } from './providers.ts';

export function textOfBlocks(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

export function systemTextOf(body: Json, clientApi: 'openai' | 'anthropic'): string {
  if (clientApi === 'anthropic') return textOfBlocks(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const m of messages) {
    if (!isRecord(m) || m.role !== 'system') continue;
    const t = textOfBlocks(m.content);
    if (t !== '') parts.push(t);
  }
  return parts.join('\n');
}

interface UserContent {
  text: string;
  images: { mime: string; data: string; url: string | null }[];
}

/** The last user message's text and images (the env overwrites per user message, so last wins). */
export function lastUserContent(body: Json, clientApi: 'openai' | 'anthropic'): UserContent {
  const out: UserContent = { text: '', images: [] };
  if (clientApi === 'anthropic') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) {
      if (!isRecord(m) || m.role !== 'user') continue;
      out.text = textOfBlocks(m.content);
      out.images = [];
      if (Array.isArray(m.content)) {
        for (const b of m.content) {
          if (!isRecord(b) || b.type !== 'image' || !isRecord(b.source)) continue;
          if (b.source.type === 'base64' && typeof b.source.data === 'string') {
            out.images.push({ mime: typeof b.source.media_type === 'string' ? b.source.media_type : 'image/png', data: b.source.data, url: null });
          } else if (b.source.type === 'url' && typeof b.source.url === 'string') {
            out.images.push({ mime: 'image/png', data: '', url: b.source.url });
          }
        }
      }
    }
    return out;
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m) || m.role !== 'user') continue;
    out.text = textOfBlocks(m.content);
    out.images = [];
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (!isRecord(b) || b.type !== 'image_url' || !isRecord(b.image_url) || typeof b.image_url.url !== 'string') continue;
        const url: string = b.image_url.url;
        const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
        if (match !== null) out.images.push({ mime: match[1], data: match[2], url: null });
        else out.images.push({ mime: 'image/png', data: '', url });
      }
    }
  }
  return out;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

export function toolDefsOf(body: Json, clientApi: 'openai' | 'anthropic'): ToolDef[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const out: ToolDef[] = [];
  for (const t of tools) {
    if (!isRecord(t)) continue;
    if (clientApi === 'openai') {
      const fn = isRecord(t.function) ? t.function : {};
      if (typeof fn.name !== 'string') continue;
      out.push({
        name: fn.name,
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      });
    } else {
      if (typeof t.name !== 'string') continue;
      out.push({
        name: t.name,
        description: typeof t.description === 'string' ? t.description : '',
        parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return out;
}

export function stopsOf(body: Json, clientApi: 'openai' | 'anthropic'): string[] {
  const raw = clientApi === 'openai' ? body.stop : body.stop_sequences;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((s) => (typeof s === 'string' ? [s] : []));
}

export function isJsonMode(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** Build the Gemini generateContent body for one client request. */
export function geminiBody(body: Json, clientApi: 'openai' | 'anthropic'): Json {
  const out: Json = {};
  const system = systemTextOf(body, clientApi);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const user = lastUserContent(body, clientApi);
  const parts: Json[] = [];
  if (user.text !== '') parts.push({ text: user.text });
  for (const img of user.images) {
    if (img.url !== null) parts.push({ fileData: { mimeType: img.mime, fileUri: img.url } });
    else parts.push({ inlineData: { mimeType: img.mime, data: img.data } });
  }
  if (parts.length === 0) parts.push({ text: '' });
  out.contents = [{ role: 'user', parts }];
  const tools = toolDefsOf(body, clientApi);
  if (tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  const stops = stopsOf(body, clientApi);
  if (stops.length > 0) config.stopSequences = stops;
  if (isJsonMode(body, clientApi)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}
