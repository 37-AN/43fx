import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 60000,
});

export const api = {
  getStatus: () => client.get('/status').then((r) => r.data),
  getRisk: () => client.get('/risk').then((r) => r.data),
  getTrades: (limit = 100) => client.get('/trades', { params: { limit } }).then((r) => r.data),
  getEquity: () => client.get('/equity').then((r) => r.data),
  startStrategy: () => client.post('/strategy/start').then((r) => r.data),
  stopStrategy: () => client.post('/strategy/stop').then((r) => r.data),
  restartStrategy: () => client.post('/strategy/restart').then((r) => r.data),
  trainAI: () => client.post('/ai/train').then((r) => r.data),
  enableAI: () => client.post('/ai/enable').then((r) => r.data),
  disableAI: () => client.post('/ai/disable').then((r) => r.data),
  getAIStatus: () => client.get('/ai/status').then((r) => r.data),
  runBacktest: (payload) => client.post('/backtest/run', payload).then((r) => r.data),
  updateConfig: (payload) => client.post('/config/update', payload).then((r) => r.data),
};
