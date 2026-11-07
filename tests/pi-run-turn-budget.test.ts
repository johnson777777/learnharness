/**
 * runSubagent's turn budget counts MODEL CALLS, not agent turns. The
 * claude-code provider admits a response's parallel tool calls one MCP
 * request at a time, so one real model call surfaces as several turns —
 * the follow-ups carry zero usage. Once a run has seen a real (usage-
 * bearing) call, those must not burn maxTurns; a provider that never
 * reports usage keeps counting every turn; and a hard backstop
 * (maxTurns × 4 raw turns) still ends a run that goes all-zero later.
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Type } from "typebox";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import {
	createAssistantMessageEventStream,
	createFauxCore,
	type AssistantMessage,
	type Usage,
} from "@earendil-works/pi-ai";
import { runSubagent } from "../src/agent/subagents/pi-run.ts";
import type { PiEngine } from "../src/agent/models.ts";

const ZERO: Usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const REAL: Usage = { ...ZERO, input: 10, output: 20, totalTokens: 30 };

type Scripted = { kind: "tool" | "text"; usage: Usage };

/** A scripted engine: each call answers the next step (a lone tool call or
 *  a final text) with the given usage; a counter records calls made. */
function scriptedEngine(steps: Scripted[]): { engine: PiEngine; calls: () => number } {
	const model = createFauxCore({}).getModel();
	let calls = 0;
	const streamFn: PiEngine["streamFn"] = () => {
		const step = steps[calls++] ?? { kind: "text", usage: REAL };
		const stream = createAssistantMessageEventStream();
		const message: AssistantMessage = {
			role: "assistant",
			content:
				step.kind === "tool"
					? [{ type: "toolCall", id: `call-${calls}`, name: "noop", arguments: {} }]
					: [{ type: "text", text: "FINAL" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: step.usage,
			stopReason: step.kind === "tool" ? "toolUse" : "stop",
			timestamp: Date.now(),
		};
		queueMicrotask(() => {
			stream.push({ type: "start", partial: message });
			stream.push({ type: "done", reason: message.stopReason as "toolUse" | "stop", message });
			stream.end();
		});
		return stream;
	};
	return { engine: { model, streamFn } as PiEngine, calls: () => calls };
}

const noopInput = Type.Object({});
const noopTool: AgentTool<typeof noopInput> = {
	name: "noop",
	label: "noop",
	description: "Does nothing.",
	parameters: noopInput,
	execute: async () => ({ content: [{ type: "text", text: "ok" }], details: undefined }),
};

const run = (engine: PiEngine, maxTurns: number) =>
	runSubagent([noopTool], { role: "researcher", engine, systemPrompt: "test", prompt: "go", maxTurns });

describe("runSubagent turn budget", () => {
	it("zero-usage follow-up turns (split parallel tool calls) do not count toward maxTurns", async () => {
		// 3 model calls, each followed by 2 zero-usage admission turns = 9
		// raw turns, then the final brief on model call 4.
		const steps: Scripted[] = [];
		for (let i = 0; i < 3; i++) steps.push({ kind: "tool", usage: REAL }, { kind: "tool", usage: ZERO }, { kind: "tool", usage: ZERO });
		steps.push({ kind: "text", usage: REAL });
		const scripted = scriptedEngine(steps);

		const result = await run(scripted.engine, 4);

		assert.equal(scripted.calls(), 10);
		assert.equal(result.text, "FINAL");
		assert.equal(result.endedAtTurnCap, false);
	});

	it("still ends at maxTurns real model calls", async () => {
		const steps: Scripted[] = Array.from({ length: 10 }, () => ({ kind: "tool", usage: REAL }));
		const scripted = scriptedEngine(steps);

		const result = await run(scripted.engine, 3);

		assert.equal(scripted.calls(), 3);
		assert.equal(result.endedAtTurnCap, true);
	});

	it("a provider that never reports usage counts every turn (old behavior)", async () => {
		const steps: Scripted[] = Array.from({ length: 50 }, () => ({ kind: "tool", usage: ZERO }));
		const scripted = scriptedEngine(steps);

		const result = await run(scripted.engine, 3);

		assert.equal(scripted.calls(), 3);
		assert.equal(result.endedAtTurnCap, true);
	});

	it("zero-usage turns after a real call stop at the maxTurns × 4 raw-turn backstop", async () => {
		const steps: Scripted[] = [{ kind: "tool", usage: REAL }, ...Array.from({ length: 50 }, (): Scripted => ({ kind: "tool", usage: ZERO }))];
		const scripted = scriptedEngine(steps);

		const result = await run(scripted.engine, 3);

		assert.equal(scripted.calls(), 12);
		assert.equal(result.endedAtTurnCap, true);
	});
});
