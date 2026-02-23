import React, { useState } from 'react';

const today = new Date().toISOString().slice(0, 10);

export default function BacktestPanel({ result, onRun }) {
  const [instrument, setInstrument] = useState('EURUSD');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState(today);
  const [synthetic, setSynthetic] = useState(false);
  const [running, setRunning] = useState(false);

  const submit = async () => {
    setRunning(true);
    try {
      await onRun({
        instrument,
        start_date: startDate,
        end_date: endDate,
        use_synthetic: synthetic,
      });
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="panel">
      <h2 className="mb-3 text-lg font-bold text-neon">BACKTEST</h2>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <input
          className="rounded border border-zinc-700 bg-black px-2 py-2"
          value={instrument}
          onChange={(e) => setInstrument(e.target.value)}
          placeholder="Instrument"
        />
        <label className="flex items-center gap-2 rounded border border-zinc-700 px-2 py-2">
          <input type="checkbox" checked={synthetic} onChange={(e) => setSynthetic(e.target.checked)} />
          Synthetic
        </label>
        <input
          type="date"
          className="rounded border border-zinc-700 bg-black px-2 py-2"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
        />
        <input
          type="date"
          className="rounded border border-zinc-700 bg-black px-2 py-2"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
        />
      </div>
      <button
        onClick={submit}
        disabled={running}
        className="mt-3 w-full rounded bg-neon px-3 py-2 text-sm font-bold text-black disabled:opacity-60"
      >
        {running ? 'RUNNING...' : 'RUN BACKTEST'}
      </button>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <div className="metric">Total Return: {(Number(result?.total_return || 0) * 100).toFixed(2)}%</div>
        <div className="metric">Win Rate: {(Number(result?.win_rate || 0) * 100).toFixed(2)}%</div>
        <div className="metric text-danger">Max Drawdown: {(Number(result?.max_drawdown || 0) * 100).toFixed(2)}%</div>
        <div className="metric">Profit Factor: {Number(result?.profit_factor || 0).toFixed(2)}</div>
      </div>
    </section>
  );
}
