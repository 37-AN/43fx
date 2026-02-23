import React, { useState } from 'react';

export default function AIPanel({ aiStatus, onTrain }) {
  const [running, setRunning] = useState(false);

  const handleTrain = async () => {
    setRunning(true);
    try {
      await onTrain();
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="panel">
      <h2 className="mb-3 text-lg font-bold text-neon">AI PANEL</h2>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="metric">Model Status: {aiStatus?.model_loaded ? 'READY' : 'MISSING'}</div>
        <div className="metric">Enabled: {aiStatus?.enabled ? 'YES' : 'NO'}</div>
        <div className="metric">Training Samples: {aiStatus?.training_sample_size ?? 0}</div>
        <div className="metric">
          Class Balance: {aiStatus?.class_balance?.positive ?? 0} / {aiStatus?.class_balance?.negative ?? 0}
        </div>
        <div className="metric">Threshold: {Number(aiStatus?.probability_threshold || 0).toFixed(2)}</div>
      </div>
      <button
        onClick={handleTrain}
        disabled={running}
        className="mt-3 w-full rounded border border-neon px-3 py-2 text-sm font-bold text-neon disabled:opacity-60"
      >
        {running ? 'TRAINING...' : 'TRAIN MODEL'}
      </button>
    </section>
  );
}
