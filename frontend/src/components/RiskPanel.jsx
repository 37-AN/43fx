import React, { useEffect, useState } from 'react';

export default function RiskPanel({ risk, aiStatus, onSave }) {
  const [riskPerTrade, setRiskPerTrade] = useState(risk?.risk_per_trade ?? 0.01);
  const [aiThreshold, setAiThreshold] = useState(aiStatus?.probability_threshold ?? 0.55);

  useEffect(() => {
    setRiskPerTrade(risk?.risk_per_trade ?? 0.01);
  }, [risk?.risk_per_trade]);

  useEffect(() => {
    setAiThreshold(aiStatus?.probability_threshold ?? 0.55);
  }, [aiStatus?.probability_threshold]);

  const submit = () => {
    onSave({
      risk_per_trade: Number(riskPerTrade),
      ai_probability_threshold: Number(aiThreshold),
    });
  };

  return (
    <section className="panel">
      <h2 className="mb-3 text-lg font-bold text-neon">RISK PANEL</h2>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="metric">Equity: {Number(risk?.equity || 0).toFixed(2)}</div>
        <div className="metric">Peak Equity: {Number(risk?.peak_equity || 0).toFixed(2)}</div>
        <div className="metric text-danger">Drawdown: {(Number(risk?.current_drawdown_pct || 0) * 100).toFixed(2)}%</div>
        <div className="metric text-danger">Daily Loss: {(Number(risk?.daily_realized_loss_pct || 0) * 100).toFixed(2)}%</div>
        <div className="metric">Risk/Trade: {(Number(risk?.risk_per_trade || 0) * 100).toFixed(2)}%</div>
        <div className="metric">Max Daily Loss: {(Number(risk?.max_daily_loss_pct || 0) * 100).toFixed(2)}%</div>
      </div>

      <div className="mt-4 space-y-4">
        <div>
          <label className="text-xs text-zinc-400">Risk Per Trade: {(riskPerTrade * 100).toFixed(2)}%</label>
          <input
            type="range"
            min="0.001"
            max="0.05"
            step="0.001"
            value={riskPerTrade}
            onChange={(e) => setRiskPerTrade(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-xs text-zinc-400">AI Probability Threshold: {aiThreshold.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.01"
            value={aiThreshold}
            onChange={(e) => setAiThreshold(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <button className="w-full rounded bg-neon px-3 py-2 text-sm font-bold text-black" onClick={submit}>
          APPLY SAFE UPDATE
        </button>
      </div>
    </section>
  );
}
