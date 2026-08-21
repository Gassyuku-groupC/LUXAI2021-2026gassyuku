#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { create, Logger } = require(
  path.join(__dirname, "..", "node_modules", ".pnpm", "node_modules", "dimensions-ai")
);
const { LuxDesign } = require("@lux-ai/2021-challenge/lib/es5/design");

async function main() {
  const input = process.argv[2];
  const output = process.argv[3];
  if (!input || !output) {
    throw new Error("Usage: convert_replay_stateful.js INPUT.json OUTPUT.json");
  }

  const replay = JSON.parse(fs.readFileSync(input, "utf8"));
  const design = new LuxDesign("lux_ai_2021", {
    engineOptions: { noStdErr: false, timeout: { max: 1200 } },
  });
  const dimension = create(design, {
    name: "Lux AI 2021 replay converter",
    loggingLevel: Logger.LEVEL.NONE,
    activateStation: false,
    observe: false,
    createBotDirectories: false,
  });
  const configs = {
    detached: true,
    agentOptions: { detached: true },
    storeReplay: false,
    storeErrorLogs: false,
    statefulReplay: true,
    seed: Number.parseInt(replay.seed, 10),
    mapType: replay.mapType,
    width: Number(replay.width),
    height: Number(replay.height),
  };
  const match = await dimension.createMatch(
    [{ file: "blank", name: "bot1" }, { file: "blank", name: "bot2" }],
    configs
  );
  match.agents.forEach((agent) => { agent.messages = []; });
  for (const commands of replay.allCommands) {
    await match.step(commands);
    match.agents.forEach((agent) => { agent.messages = []; });
  }
  const rendered = match.state.game.replay.data;
  if (Number(rendered.width) !== Number(replay.width) ||
      Number(rendered.height) !== Number(replay.height)) {
    throw new Error(`Replay size changed during conversion: ${rendered.width}x${rendered.height}`);
  }
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(rendered));
  process.stdout.write(`Converted ${input} -> ${output}\n`);
}

main().then(() => process.exit(0)).catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
