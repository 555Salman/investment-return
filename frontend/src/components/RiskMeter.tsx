import React from "react";

type RiskLevel = "low" | "medium" | "high";

interface RiskMeterProps {
  level: RiskLevel;
  volatilityPct?: number;
  label?: string;
}

const CONFIG: Record<RiskLevel, { color: string; bg: string; fill: number; text: string }> = {
  low:    { color: "#10b981", bg: "#d1fae5", fill: 33,  text: "Low Risk"    },
  medium: { color: "#f59e0b", bg: "#fef3c7", fill: 66,  text: "Medium Risk" },
  high:   { color: "#ef4444", bg: "#fee2e2", fill: 100, text: "High Risk"   },
};

export const RiskMeter: React.FC<RiskMeterProps> = ({
  level,
  volatilityPct,
  label,
}) => {
  const { color, bg, fill, text } = CONFIG[level] ?? CONFIG.medium;

  // Arc parameters for SVG gauge (semi-circle, 180 degrees)
  const radius = 48;
  const cx = 60;
  const cy = 60;
  const circumference = Math.PI * radius;          // half-circle arc length
  const dashOffset = circumference * (1 - fill / 100);

  return (
    <div
      className="flex flex-col items-center gap-2 rounded-2xl p-4 shadow-sm"
      style={{ backgroundColor: bg, minWidth: 140 }}
      role="meter"
      aria-label={`Risk level: ${text}`}
      aria-valuenow={fill}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      {/* Semi-circle gauge */}
      <svg viewBox="0 0 120 70" width={120} height={70} aria-hidden="true">
        {/* Track */}
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={10}
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={`${circumference}`}
          strokeDashoffset={dashOffset}
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
        {/* Needle dot */}
        <circle cx={cx} cy={cy} r={5} fill={color} />
      </svg>

      {/* Labels */}
      <span className="text-sm font-semibold" style={{ color }}>
        {text}
      </span>

      {volatilityPct !== undefined && (
        <span className="text-xs text-gray-500">
          Vol: {volatilityPct.toFixed(2)}%
        </span>
      )}

      {label && (
        <span className="text-xs text-gray-400 text-center">{label}</span>
      )}
    </div>
  );
};

export default RiskMeter;
