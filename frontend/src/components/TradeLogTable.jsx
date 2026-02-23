import React from 'react';

export default function TradeLogTable({ trades }) {
  return (
    <section className="panel">
      <h2 className="mb-3 text-lg font-bold text-neon">TRADE LOG</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-xs text-zinc-300">
          <thead>
            <tr className="border-b border-zinc-700 text-zinc-400">
              <th className="px-2 py-2">Time</th>
              <th className="px-2 py-2">Direction</th>
              <th className="px-2 py-2">Size</th>
              <th className="px-2 py-2">Entry</th>
              <th className="px-2 py-2">Exit</th>
              <th className="px-2 py-2">PnL</th>
              <th className="px-2 py-2">R-Multiple</th>
              <th className="px-2 py-2">AI Prob</th>
            </tr>
          </thead>
          <tbody>
            {trades?.map((trade, idx) => (
              <tr key={`${trade.entry_time || 't'}-${idx}`} className="border-b border-zinc-900">
                <td className="px-2 py-2">{trade.exit_time || trade.entry_time || '-'}</td>
                <td className="px-2 py-2">{trade.direction}</td>
                <td className="px-2 py-2">{trade.size?.toFixed?.(4) ?? '-'}</td>
                <td className="px-2 py-2">{trade.entry?.toFixed?.(5) ?? '-'}</td>
                <td className="px-2 py-2">{trade.exit?.toFixed?.(5) ?? '-'}</td>
                <td className={`px-2 py-2 ${Number(trade.pnl) >= 0 ? 'text-neon' : 'text-danger'}`}>
                  {Number(trade.pnl || 0).toFixed(2)}
                </td>
                <td className="px-2 py-2">{Number(trade.r_multiple || 0).toFixed(2)}</td>
                <td className="px-2 py-2">{trade.ai_probability == null ? '-' : Number(trade.ai_probability).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
