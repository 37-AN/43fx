import React, { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function EquityChartPanel({ points }) {
  const [range, setRange] = useState('daily');

  const filtered = useMemo(() => {
    if (!points?.length) return [];
    const now = Date.now();
    const horizonMs = range === 'daily' ? 24 * 3600 * 1000 : 7 * 24 * 3600 * 1000;

    const recent = points.filter((p) => {
      const t = Date.parse(p.timestamp);
      return Number.isFinite(t) ? now - t <= horizonMs : true;
    });
    return recent.length ? recent : points;
  }, [points, range]);

  return (
    <section className="panel">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-neon">EQUITY CURVE</h2>
        <div className="space-x-2">
          <button
            onClick={() => setRange('daily')}
            className={`rounded px-2 py-1 text-xs ${range === 'daily' ? 'bg-neon text-black' : 'bg-zinc-800 text-zinc-300'}`}
          >
            Daily
          </button>
          <button
            onClick={() => setRange('weekly')}
            className={`rounded px-2 py-1 text-xs ${range === 'weekly' ? 'bg-neon text-black' : 'bg-zinc-800 text-zinc-300'}`}
          >
            Weekly
          </button>
        </div>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={filtered}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="timestamp" hide />
            <YAxis yAxisId="eq" stroke="#39FF14" />
            <YAxis yAxisId="dd" orientation="right" stroke="#FF0000" domain={[0, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#111', border: '1px solid #333', color: '#e5e7eb' }}
            />
            <Legend />
            <Line yAxisId="eq" type="monotone" dataKey="equity" stroke="#39FF14" dot={false} strokeWidth={2} />
            <Line yAxisId="dd" type="monotone" dataKey="drawdown_pct" stroke="#FF0000" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
