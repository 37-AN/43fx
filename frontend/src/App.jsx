import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import AIPanel from './components/AIPanel';
import BacktestPanel from './components/BacktestPanel';
import EquityChartPanel from './components/EquityChartPanel';
import NotificationStack from './components/NotificationStack';
import RiskPanel from './components/RiskPanel';
import SystemStatusPanel from './components/SystemStatusPanel';
import TradeLogTable from './components/TradeLogTable';

export default function App() {
  const [status, setStatus] = useState(null);
  const [risk, setRisk] = useState(null);
  const [equity, setEquity] = useState([]);
  const [trades, setTrades] = useState([]);
  const [aiStatus, setAiStatus] = useState(null);
  const [backtestSummary, setBacktestSummary] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const shownAlerts = useRef(new Set());

  const pushNotification = useCallback((id, message, level = 'danger') => {
    if (shownAlerts.current.has(id)) return;
    shownAlerts.current.add(id);

    const item = { id, message, level };
    setNotifications((prev) => [item, ...prev].slice(0, 5));

    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
      shownAlerts.current.delete(id);
    }, 8000);
  }, []);

  const loadAll = useCallback(async () => {
    const [statusData, riskData, equityData, tradeData, aiData] = await Promise.all([
      api.getStatus(),
      api.getRisk(),
      api.getEquity(),
      api.getTrades(100),
      api.getAIStatus(),
    ]);

    setStatus(statusData);
    setRisk(riskData);
    setEquity(equityData.points || []);
    setTrades(tradeData.trades || []);
    setAiStatus(aiData);

    if (riskData.current_drawdown_pct > riskData.max_drawdown_pct) {
      pushNotification('drawdown-alert', 'Drawdown exceeded max threshold', 'danger');
    }
    if (!statusData.live_mode_running) {
      pushNotification('live-off', 'Live trading is currently stopped', 'danger');
    }
    if (!aiData.model_loaded) {
      pushNotification('ai-missing', 'AI disabled due to missing model', 'danger');
    }
  }, [pushNotification]);

  useEffect(() => {
    loadAll().catch(() => undefined);
    const timer = setInterval(() => loadAll().catch(() => undefined), 5000);
    return () => clearInterval(timer);
  }, [loadAll]);

  const handleStartStop = async () => {
    if (status?.live_mode_running) {
      await api.stopStrategy();
    } else {
      await api.startStrategy();
    }
    await loadAll();
  };

  const handleToggleAI = async () => {
    if (status?.ai_enabled) {
      await api.disableAI();
    } else {
      await api.enableAI();
    }
    await loadAll();
  };

  const handleConfigSave = async (payload) => {
    await api.updateConfig(payload);
    await loadAll();
  };

  const handleBacktest = async (payload) => {
    const response = await api.runBacktest(payload);
    setBacktestSummary(response.summary_metrics || null);
    await loadAll();
  };

  const handleAITrain = async () => {
    await api.trainAI();
    await loadAll();
  };

  return (
    <div className="min-h-screen bg-black bg-grid bg-[size:24px_24px] px-4 py-4 text-zinc-100 md:px-8">
      <NotificationStack notifications={notifications} />
      <h1 className="mb-4 border-b border-zinc-800 pb-2 text-2xl font-bold text-neon">FOREX CONTROL CENTER</h1>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <SystemStatusPanel status={status} onStartStop={handleStartStop} onToggleAI={handleToggleAI} />
          <EquityChartPanel points={equity} />
          <TradeLogTable trades={trades} />
        </div>

        <div className="space-y-4">
          <RiskPanel risk={risk} aiStatus={aiStatus} onSave={handleConfigSave} />
          <BacktestPanel result={backtestSummary} onRun={handleBacktest} />
          <AIPanel aiStatus={aiStatus} onTrain={handleAITrain} />
        </div>
      </div>
    </div>
  );
}
