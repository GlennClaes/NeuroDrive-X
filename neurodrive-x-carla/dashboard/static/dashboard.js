const refreshSeconds = Number(window.NEURODRIVE_REFRESH_SECONDS || 2);
const ids = {
  episode: document.getElementById("episode"),
  reward: document.getElementById("reward"),
  speed: document.getElementById("speed"),
  collisions: document.getElementById("collisions"),
  laneInvasions: document.getElementById("laneInvasions"),
  distance: document.getElementById("distance"),
  successRate: document.getElementById("successRate"),
  averageReward: document.getElementById("averageReward"),
  healthStatus: document.getElementById("healthStatus"),
  lastUpdated: document.getElementById("lastUpdated"),
  leaderboard: document.getElementById("leaderboard"),
  plotList: document.getElementById("plotList"),
  rewardChart: document.getElementById("rewardChart"),
};

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

async function refreshDashboard() {
  try {
    const [latest, history, leaderboard, plots] = await Promise.all([
      fetchJson("/api/metrics/latest"),
      fetchJson("/api/metrics/history?limit=160"),
      fetchJson("/api/leaderboard"),
      fetchJson("/api/plots"),
    ]);
    renderLatest(latest);
    renderChart(history.items || []);
    renderLeaderboard(leaderboard.items || []);
    renderPlots(plots.items || []);
    ids.healthStatus.textContent = "Live";
    ids.healthStatus.classList.add("ok");
  } catch (error) {
    ids.healthStatus.textContent = "Offline";
    ids.healthStatus.classList.remove("ok");
    console.error(error);
  }
}

function renderLatest(payload) {
  const latest = payload.latest || {};
  const summary = payload.summary || {};
  ids.episode.textContent = latest.episode ?? summary.latest_episode ?? 0;
  ids.reward.textContent = formatNumber(latest.reward ?? 0, 2);
  ids.speed.textContent = formatNumber(latest.speed_kmh ?? 0, 1);
  ids.collisions.textContent = latest.collision_count ?? 0;
  ids.laneInvasions.textContent = latest.lane_invasion_count ?? 0;
  ids.distance.textContent = formatNumber(latest.distance_driven_m ?? summary.total_distance_m ?? 0, 1);
  ids.successRate.textContent = formatNumber((summary.success_rate ?? 0) * 100, 0);
  ids.averageReward.textContent = formatNumber(summary.average_reward_100 ?? summary.average_reward ?? 0, 2);
  ids.lastUpdated.textContent = payload.updated_at ? `Updated ${new Date(payload.updated_at).toLocaleTimeString()}` : "Waiting for metrics";
}

function renderLeaderboard(items) {
  ids.leaderboard.innerHTML = "";
  if (!items.length) {
    ids.leaderboard.innerHTML = "<tr><td colspan=\"5\">No completed runs yet</td></tr>";
    return;
  }
  for (const item of items.slice(0, 8)) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(item.model_name)}</td>
      <td>${escapeHtml(item.town)}</td>
      <td>${escapeHtml(item.weather)}</td>
      <td>${formatNumber(item.success_rate * 100, 0)}%</td>
      <td>${formatNumber(item.average_reward, 1)}</td>
    `;
    ids.leaderboard.appendChild(row);
  }
}

function renderPlots(items) {
  ids.plotList.innerHTML = "";
  if (!items.length) {
    ids.plotList.innerHTML = "<span>No graph files generated yet</span>";
    return;
  }
  for (const item of items) {
    const link = document.createElement("a");
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.innerHTML = `<span>${escapeHtml(item.name)}</span><strong>Open</strong>`;
    ids.plotList.appendChild(link);
  }
}

function renderChart(items) {
  const canvas = ids.rewardChart;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#171b23";
  ctx.fillRect(0, 0, width, height);

  const padding = { left: 58, right: 24, top: 24, bottom: 42 };
  const rewards = items.map((item) => Number(item.reward || 0));
  if (!rewards.length) {
    ctx.fillStyle = "#99a7b7";
    ctx.font = "24px Inter, sans-serif";
    ctx.fillText("Waiting for training metrics", padding.left, height / 2);
    return;
  }

  const min = Math.min(...rewards, 0);
  const max = Math.max(...rewards, 1);
  drawGrid(ctx, width, height, padding, min, max);
  drawLine(ctx, items, rewards, width, height, padding, min, max, "#31c6a6", 3);
  const average = rollingAverage(rewards, Math.min(20, rewards.length));
  drawLine(ctx, items, average, width, height, padding, Math.min(...average, min), Math.max(...average, max), "#f0b35b", 2);
}

function drawGrid(ctx, width, height, padding, min, max) {
  ctx.strokeStyle = "#2c3544";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#99a7b7";
  ctx.font = "18px Inter, sans-serif";
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + ((height - padding.top - padding.bottom) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    const label = max - ((max - min) * i) / 4;
    ctx.fillText(formatNumber(label, 0), 12, y + 6);
  }
}

function drawLine(ctx, items, values, width, height, padding, min, max, color, lineWidth) {
  const graphWidth = width - padding.left - padding.right;
  const graphHeight = height - padding.top - padding.bottom;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = padding.left + (graphWidth * index) / Math.max(values.length - 1, 1);
    const y = padding.top + graphHeight - ((value - min) / Math.max(max - min, 1)) * graphHeight;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function rollingAverage(values, windowSize) {
  return values.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1);
    const window = values.slice(start, index + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  });
}

function formatNumber(value, digits) {
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

refreshDashboard();
setInterval(refreshDashboard, refreshSeconds * 1000);

