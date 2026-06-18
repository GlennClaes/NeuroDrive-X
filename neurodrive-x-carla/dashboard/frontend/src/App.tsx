import { Activity, AlertTriangle, Car, Gauge, GitBranch, Radar, RefreshCw, Route, Trophy } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { useEffect, useMemo, useState, type ReactNode } from "react";

type EpisodeMetric = {
  episode: number;
  reward: number;
  speed_kmh: number;
  collision_count: number;
  lane_invasion_count: number;
  distance_driven_m: number;
  success: boolean;
  town: string;
  weather: string;
  detection_count?: number;
  route_completed_pct?: number;
  average_reward_100?: number;
};

type Summary = {
  episodes: number;
  success_rate: number;
  average_reward: number;
  average_reward_100?: number;
  total_collisions: number;
  total_lane_invasions: number;
  total_distance_m: number;
  best_reward?: number;
};

type LeaderboardItem = {
  model_name: string;
  town: string;
  weather: string;
  episodes: number;
  success_rate: number;
  average_reward: number;
  average_distance_m: number;
  collisions_per_episode: number;
};

type PlotItem = {
  name: string;
  url: string;
};

const emptySummary: Summary = {
  episodes: 0,
  success_rate: 0,
  average_reward: 0,
  total_collisions: 0,
  total_lane_invasions: 0,
  total_distance_m: 0
};

export function App() {
  const [latest, setLatest] = useState<EpisodeMetric | null>(null);
  const [summary, setSummary] = useState<Summary>(emptySummary);
  const [history, setHistory] = useState<EpisodeMetric[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardItem[]>([]);
  const [plots, setPlots] = useState<PlotItem[]>([]);
  const [lastUpdated, setLastUpdated] = useState<string>("Waiting for metrics");
  const [online, setOnline] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      try {
        const [latestPayload, historyPayload, leaderboardPayload, plotPayload] = await Promise.all([
          fetchJson<{ latest: EpisodeMetric | null; summary: Summary; updated_at?: string }>("/api/metrics/latest"),
          fetchJson<{ items: EpisodeMetric[]; summary: Summary }>("/api/metrics/history?limit=180"),
          fetchJson<{ items: LeaderboardItem[] }>("/api/leaderboard"),
          fetchJson<{ items: PlotItem[] }>("/api/plots")
        ]);
        if (cancelled) return;
        setLatest(latestPayload.latest);
        setSummary(latestPayload.summary ?? historyPayload.summary ?? emptySummary);
        setHistory(historyPayload.items ?? []);
        setLeaderboard(leaderboardPayload.items ?? []);
        setPlots(plotPayload.items ?? []);
        setLastUpdated(latestPayload.updated_at ? new Date(latestPayload.updated_at).toLocaleTimeString() : "Live API");
        setOnline(true);
      } catch (error) {
        if (!cancelled) setOnline(false);
        console.error(error);
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const rewardData = useMemo(
    () =>
      history.map((item, index) => ({
        episode: item.episode,
        reward: item.reward,
        average: rollingAverage(history.map((row) => row.reward), 20)[index],
        distance: item.distance_driven_m,
        collisions: item.collision_count
      })),
    [history]
  );

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">CARLA Autonomous Driving Lab</span>
          <h1>NeuroDrive X</h1>
        </div>
        <div className={online ? "status online" : "status"}>
          <Activity size={18} />
          {online ? "Live" : "Offline"}
        </div>
      </header>

      <section className="metrics" aria-label="Live metrics">
        <Metric icon={<Route />} label="Episode" value={latest?.episode ?? summary.episodes ?? 0} />
        <Metric icon={<Gauge />} label="Live Reward" value={format(latest?.reward ?? 0, 2)} />
        <Metric icon={<Car />} label="Speed" value={`${format(latest?.speed_kmh ?? 0, 1)} km/h`} />
        <Metric icon={<AlertTriangle />} label="Collisions" value={latest?.collision_count ?? summary.total_collisions ?? 0} />
        <Metric icon={<GitBranch />} label="Lane Invasions" value={latest?.lane_invasion_count ?? summary.total_lane_invasions ?? 0} />
        <Metric icon={<RefreshCw />} label="Distance" value={`${format(latest?.distance_driven_m ?? summary.total_distance_m ?? 0, 1)} m`} />
        <Metric icon={<Trophy />} label="Success Rate" value={`${format(summary.success_rate * 100, 0)}%`} />
        <Metric icon={<Radar />} label="Detections" value={latest?.detection_count ?? 0} />
      </section>

      <section className="grid">
        <article className="panel wide">
          <PanelHeader title="Training Reward" detail={`Updated ${lastUpdated}`} />
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rewardData}>
                <defs>
                  <linearGradient id="reward" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#31c6a6" stopOpacity={0.45} />
                    <stop offset="95%" stopColor="#31c6a6" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#2c3544" strokeDasharray="3 3" />
                <XAxis dataKey="episode" stroke="#99a7b7" />
                <YAxis stroke="#99a7b7" />
                <Tooltip contentStyle={{ background: "#171b23", border: "1px solid #2c3544" }} />
                <Area type="monotone" dataKey="reward" stroke="#31c6a6" fill="url(#reward)" strokeWidth={2} />
                <Line type="monotone" dataKey="average" stroke="#f0b35b" dot={false} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <PanelHeader title="Route Progress" detail={`${format((latest?.route_completed_pct ?? 0) * 100, 0)}%`} />
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${Math.min((latest?.route_completed_pct ?? 0) * 100, 100)}%` }} />
          </div>
          <div className="mini-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rewardData}>
                <CartesianGrid stroke="#2c3544" strokeDasharray="3 3" />
                <XAxis dataKey="episode" hide />
                <YAxis hide />
                <Tooltip contentStyle={{ background: "#171b23", border: "1px solid #2c3544" }} />
                <Line type="monotone" dataKey="distance" stroke="#7aa2ff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <PanelHeader title="Leaderboard" detail={`${leaderboard.length} runs`} />
          <div className="table">
            {leaderboard.length === 0 ? (
              <p className="empty">No completed runs yet</p>
            ) : (
              leaderboard.slice(0, 6).map((item) => (
                <div className="table-row" key={`${item.model_name}-${item.town}-${item.weather}`}>
                  <span>{item.town}</span>
                  <span>{item.weather}</span>
                  <strong>{format(item.success_rate * 100, 0)}%</strong>
                </div>
              ))
            )}
          </div>
        </article>

        <article className="panel">
          <PanelHeader title="Generated Plots" detail={`${plots.length} files`} />
          <div className="plot-links">
            {plots.length === 0 ? (
              <p className="empty">No graph files generated yet</p>
            ) : (
              plots.map((plot) => (
                <a href={plot.url} target="_blank" rel="noreferrer" key={plot.url}>
                  {plot.name}
                </a>
              ))
            )}
          </div>
        </article>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: ReactNode }) {
  return (
    <article className="metric">
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function PanelHeader({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="panel-header">
      <h2>{title}</h2>
      <span>{detail}</span>
    </div>
  );
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

function rollingAverage(values: number[], windowSize: number): number[] {
  return values.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1);
    const window = values.slice(start, index + 1);
    return window.reduce((sum, value) => sum + value, 0) / Math.max(window.length, 1);
  });
}

function format(value: number, digits: number): string {
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}
