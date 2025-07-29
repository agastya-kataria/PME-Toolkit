"use client"

import { Line, LineChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

const data = [
  { month: "Jan", portfolio: 10.5, benchmark: 10.5 },
  { month: "Feb", portfolio: 10.8, benchmark: 10.7 },
  { month: "Mar", portfolio: 11.2, benchmark: 10.9 },
  { month: "Apr", portfolio: 11.4, benchmark: 11.0 },
  { month: "May", portfolio: 11.6, benchmark: 11.2 },
  { month: "Jun", portfolio: 11.9, benchmark: 11.4 },
  { month: "Jul", portfolio: 12.1, benchmark: 11.6 },
  { month: "Aug", portfolio: 12.3, benchmark: 11.7 },
  { month: "Sep", portfolio: 12.7, benchmark: 11.9 },
  { month: "Oct", portfolio: 12.5, benchmark: 12.0 },
  { month: "Nov", portfolio: 12.6, benchmark: 12.1 },
  { month: "Dec", portfolio: 12.7, benchmark: 12.2 },
]

const chartConfig = {
  portfolio: {
    label: "Portfolio",
    color: "#d97706",
  },
  benchmark: {
    label: "Benchmark",
    color: "#64748b",
  },
}

export function PerformanceChart() {
  return (
    <ChartContainer config={chartConfig} className="h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="month" stroke="#9ca3af" fontSize={12} />
          <YAxis stroke="#9ca3af" fontSize={12} tickFormatter={(value) => `$${value}M`} />
          <ChartTooltip
            content={<ChartTooltipContent />}
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #d97706",
              borderRadius: "8px",
            }}
          />
          <Line
            type="monotone"
            dataKey="portfolio"
            stroke="#d97706"
            strokeWidth={3}
            dot={{ fill: "#d97706", strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6, stroke: "#d97706", strokeWidth: 2 }}
          />
          <Line
            type="monotone"
            dataKey="benchmark"
            stroke="#64748b"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}
