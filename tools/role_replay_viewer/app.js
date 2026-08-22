"use strict";

const ROLE_COLORS = {
  Harvester: "#38bdf8",
  Builder: "#facc15",
  Attacker: "#fb7185",
  Firefighter: "#f97316",
  FuelDepot: "#22c55e",
  FuelStation: "#2dd4bf",
  ResearchStation: "#a78bfa",
  ManufacturingPoint: "#f472b6",
  SacrificialDecay: "#94a3b8",
};
const TEAM_COLORS = ["#42a5f5", "#ef5350"];
const RESOURCE_COLORS = { wood: "#2f9e44", coal: "#6c757d", uranium: "#22b8cf" };

const canvas = document.querySelector("#board");
const ctx = canvas.getContext("2d");
const statusText = document.querySelector("#status");
const turnInput = document.querySelector("#turn");
const turnLabel = document.querySelector("#turn-label");
const selection = document.querySelector("#selection");
const hoverDetail = document.querySelector("#hover-detail");
let replay = null;
let roles = null;
let roleFrames = new Map();
let turn = 0;
let timer = null;

function readJson(file, callback) {
  const reader = new FileReader();
  reader.onload = () => {
    try { callback(JSON.parse(reader.result)); }
    catch (error) { statusText.textContent = `Invalid JSON: ${error.message}`; }
  };
  reader.readAsText(file);
}

async function readJsonUrl(url, callback) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  callback(await response.json());
}

document.querySelector("#replay-file").addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) readJson(file, value => { replay = value; turn = 0; refreshData(); });
});
document.querySelector("#roles-file").addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) readJson(file, value => {
    roles = value;
    roleFrames = new Map((roles.frames || []).map(frame => [Number(frame.turn), frame]));
    refreshData();
  });
});

function refreshData() {
  const count = replay?.stateful?.length || 0;
  turnInput.max = Math.max(count - 1, 0);
  turn = Math.min(turn, Math.max(count - 1, 0));
  const compatible = replay && roles && Number(replay.seed) === Number(roles.seed);
  statusText.textContent = !replay ? "Load a stateful replay and its role sidecar."
    : !roles ? `Replay loaded: ${replay.width}x${replay.height}, seed ${replay.seed}. Load roles.`
    : compatible ? `Ready: ${replay.width}x${replay.height}, seed ${replay.seed}, player ${roles.player}.`
    : "Replay and role sidecar seeds do not match.";
  draw();
}

function setTurn(value) {
  if (!replay?.stateful?.length) return;
  turn = Math.max(0, Math.min(Number(value), replay.stateful.length - 1));
  turnInput.value = turn;
  draw();
}

turnInput.addEventListener("input", event => setTurn(event.target.value));
document.querySelector("#previous").addEventListener("click", () => setTurn(turn - 1));
document.querySelector("#next").addEventListener("click", () => setTurn(turn + 1));
document.querySelector("#play").addEventListener("click", togglePlay);
["show-units", "show-cities", "show-team-0", "show-team-1"].forEach(id => {
  document.querySelector(`#${id}`).addEventListener("change", draw);
});

function togglePlay() {
  const button = document.querySelector("#play");
  if (timer) {
    clearInterval(timer); timer = null; button.innerHTML = "&#9654;"; return;
  }
  button.innerHTML = "&#10074;&#10074;";
  timer = setInterval(() => {
    if (turn >= replay.stateful.length - 1) setTurn(0); else setTurn(turn + 1);
  }, Number(document.querySelector("#speed").value));
}
document.querySelector("#speed").addEventListener("change", () => {
  if (timer) { clearInterval(timer); timer = null; togglePlay(); }
});

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!replay?.stateful?.length) {
    ctx.fillStyle = "#9ba5ae"; ctx.font = "22px Segoe UI"; ctx.textAlign = "center";
    ctx.fillText("Load replay JSON", canvas.width / 2, canvas.height / 2); return;
  }
  const state = replay.stateful[turn];
  const width = Number(replay.width);
  const height = Number(replay.height);
  const cell = Math.min(canvas.width / width, canvas.height / height);
  const ox = (canvas.width - cell * width) / 2;
  const oy = (canvas.height - cell * height) / 2;
  drawGrid(state, width, height, cell, ox, oy);
  drawCities(state, cell, ox, oy);
  drawUnits(state, cell, ox, oy);
  drawRoleOverlay(roleFrames.get(Number(state.turn ?? turn)), cell, ox, oy);
  turnLabel.textContent = `Turn ${state.turn ?? turn} / ${replay.stateful.length - 1}`;
}

function drawGrid(state, width, height, cell, ox, oy) {
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const tile = state.map[y][x];
    ctx.fillStyle = (state.turn % 40) >= 30 ? "#171b23" : "#273036";
    ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
    if (tile.road > 0) {
      ctx.fillStyle = `rgba(180, 190, 195, ${Math.min(tile.road / 12, 0.45)})`;
      ctx.fillRect(ox + x * cell + cell * .31, oy + y * cell + cell * .31, cell * .38, cell * .38);
    }
    if (tile.resource && tile.resource.amount > 0) {
      ctx.fillStyle = RESOURCE_COLORS[tile.resource.type] || "#ffffff";
      ctx.beginPath(); ctx.arc(ox + (x + .5) * cell, oy + (y + .5) * cell, cell * .23, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = "#ffffff12"; ctx.strokeRect(ox + x * cell, oy + y * cell, cell, cell);
  }
}

function drawCities(state, cell, ox, oy) {
  Object.values(state.cities || {}).forEach(city => city.cityCells.forEach(tile => {
    ctx.fillStyle = `${TEAM_COLORS[city.team]}aa`;
    ctx.fillRect(ox + tile.x * cell + 2, oy + tile.y * cell + 2, cell - 4, cell - 4);
  }));
}

function drawUnits(state, cell, ox, oy) {
  Object.entries(state.teamStates || {}).forEach(([team, teamState]) => {
    Object.values(teamState.units || {}).forEach(unit => {
      ctx.fillStyle = TEAM_COLORS[Number(team)];
      const radius = unit.type === 0 ? cell * .19 : cell * .25;
      ctx.beginPath(); ctx.arc(ox + (unit.x + .5) * cell, oy + (unit.y + .5) * cell, radius, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#f8fafc"; ctx.lineWidth = Math.max(1, cell * .035); ctx.stroke();
    });
  });
}

function drawRoleOverlay(frame, cell, ox, oy) {
  if (!frame) return;
  const teamVisible = document.querySelector(`#show-team-${frame.player}`)?.checked ?? true;
  if (!teamVisible) return;
  if (document.querySelector("#show-cities").checked) frame.cities.forEach(city => city.tiles.forEach(tile => {
    ctx.fillStyle = `${ROLE_COLORS[city.role] || "#ffffff"}70`;
    ctx.fillRect(ox + tile.x * cell + 1, oy + tile.y * cell + 1, cell - 2, cell - 2);
  }));
  if (document.querySelector("#show-units").checked) frame.units.forEach(unit => {
    ctx.strokeStyle = ROLE_COLORS[unit.role] || "#ffffff";
    ctx.lineWidth = Math.max(3, cell * .11);
    ctx.beginPath(); ctx.arc(ox + (unit.x + .5) * cell, oy + (unit.y + .5) * cell, cell * .34, 0, Math.PI * 2); ctx.stroke();
    if (unit.changed) {
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(ox + (unit.x + .78) * cell, oy + (unit.y + .22) * cell, cell * .09, 0, Math.PI * 2); ctx.fill();
    }
  });
}

function cellDetail(x, y) {
  if (!replay) return [];
  const state = replay.stateful[turn];
  const frame = roleFrames.get(Number(state.turn ?? turn));
  const lines = [["Cell", `${x}, ${y}`]];
  const resource = state.map[y]?.[x]?.resource;
  if (resource?.amount > 0) lines.push(["Resource", `${resource.type} ${resource.amount}`]);
  const unit = frame?.units?.find(item => item.x === x && item.y === y);
  if (unit) {
    lines.push(["Unit", unit.id], ["Role", unit.role], ["Desired", unit.desired_role], ["Cooldown", unit.cooldown], ["Reason", unit.reason], ["Bias", frame.bias_active ? "active" : "observe only"]);
  }
  const city = frame?.cities?.find(item => item.tiles.some(tile => tile.x === x && tile.y === y));
  if (city) lines.push(["City", city.id], ["City role", city.role], ["Reason", city.reason], ["Fuel nights", Number(city.nights_of_fuel).toFixed(2)]);
  return lines;
}

canvas.addEventListener("mousemove", event => {
  if (!replay) return;
  const rect = canvas.getBoundingClientRect();
  const width = Number(replay.width), height = Number(replay.height);
  const cssCell = Math.min(rect.width / width, rect.height / height);
  const ox = (rect.width - cssCell * width) / 2, oy = (rect.height - cssCell * height) / 2;
  const x = Math.floor((event.clientX - rect.left - ox) / cssCell);
  const y = Math.floor((event.clientY - rect.top - oy) / cssCell);
  if (x < 0 || y < 0 || x >= width || y >= height) { hoverDetail.hidden = true; return; }
  const detail = cellDetail(x, y);
  selection.innerHTML = detail.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
  hoverDetail.textContent = detail.slice(0, 3).map(item => item.join(": ")).join(" | ");
  hoverDetail.hidden = false;
  hoverDetail.style.left = `${event.clientX - rect.left + 18}px`;
  hoverDetail.style.top = `${event.clientY - rect.top + 18}px`;
});
canvas.addEventListener("mouseleave", () => { hoverDetail.hidden = true; });

document.querySelector("#legend").innerHTML = Object.entries(ROLE_COLORS).map(([role, color]) =>
  `<div class="legend-item"><span class="swatch" style="background:${color}"></span><span>${role}</span></div>`
).join("");
draw();

const query = new URLSearchParams(window.location.search);
Promise.all([
  query.get("replay") ? readJsonUrl(query.get("replay"), value => { replay = value; turn = 0; }) : null,
  query.get("roles") ? readJsonUrl(query.get("roles"), value => {
    roles = value;
    roleFrames = new Map((roles.frames || []).map(frame => [Number(frame.turn), frame]));
  }) : null,
]).then(refreshData).catch(error => { statusText.textContent = error.message; });
