# free-model-aggregator-01

Scripted aggregator server (Node 22, stdlib only). Reads the brief, probes provider catalogs, routes the 20-request workload with a pre-computed plan (gemini for vision, openrouter for weight-2, zai/cloudflare for the rest) and sequential failover across all three provider wires (OpenAI, Anthropic, Gemini).

Run: `node main.ts` (trial) or `node main.ts --list` (print catalog).
