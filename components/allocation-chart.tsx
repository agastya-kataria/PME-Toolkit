"use client"

import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from "recharts"
import { ChartContainer } from "@/components/ui/chart"

const data = [
  { name: "Private Equity", value: 35, color: "#d97706" },
  { name: "Real Estate", value: 25, color: "#92400e" },
  { name: "Venture Capital", value: 20, color: "#f59e0b" },
  { name: "Private Debt", value: 12, color: "#374151" },
  { name: "Infrastructure", value: 8, color: "#1f2937" },
]

const chartConfig = {
  allocation: {
    label: "Allocation",
  },
}

export function AllocationChart() {
  return (
    <ChartContainer config={chartConfig} className="h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={2} dataKey="value">
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Legend
            verticalAlign="bottom"
            height={36}
            formatter={(value) => <span style={{ color: "#e2e8f0" }}>{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}
