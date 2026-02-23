import React from 'react';

export default function NotificationStack({ notifications }) {
  return (
    <div className="fixed right-4 top-4 z-50 space-y-2">
      {notifications.map((n) => (
        <div
          key={n.id}
          className={`rounded border px-3 py-2 text-xs shadow-lg ${
            n.level === 'danger' ? 'border-danger bg-[#220000] text-red-200' : 'border-neon bg-[#042000] text-green-200'
          }`}
        >
          {n.message}
        </div>
      ))}
    </div>
  );
}
