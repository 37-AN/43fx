import React from 'react';

export default function SystemStatusPanel({ status, onStartStop, onToggleAI }) {
  const liveOn = Boolean(status?.live_mode_running);
  const aiOn = Boolean(status?.ai_enabled);

  return (
    <section className="panel">
      <h2 className="mb-3 text-lg font-bold text-neon">SYSTEM STATUS</h2>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div>
          <p className="text-xs text-zinc-500">LIVE</p>
          <button
            onClick={onStartStop}
            className={`mt-1 w-full rounded border px-3 py-2 text-sm font-bold ${
              liveOn ? 'border-neon text-neon' : 'border-danger text-danger'
            }`}
          >
            {liveOn ? 'ON' : 'OFF'}
          </button>
        </div>
        <div>
          <p className="text-xs text-zinc-500">AI</p>
          <button
            onClick={onToggleAI}
            className={`mt-1 w-full rounded border px-3 py-2 text-sm font-bold ${
              aiOn ? 'border-neon text-neon' : 'border-danger text-danger'
            }`}
          >
            {aiOn ? 'ENABLED' : 'DISABLED'}
          </button>
        </div>
        <div>
          <p className="text-xs text-zinc-500">BROKER</p>
          <p className="metric mt-3 text-base">{status?.broker_type || 'unknown'}</p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">UPTIME</p>
          <p className="metric mt-3 text-base">{status?.system_uptime || '0:00:00'}</p>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <span
          className={`inline-block h-3 w-3 rounded-full ${liveOn ? 'bg-neon' : 'bg-danger'}`}
          aria-label="health-indicator"
        />
        <span className="text-xs text-zinc-400">{liveOn ? 'HEALTHY' : 'STOPPED'}</span>
      </div>
    </section>
  );
}
