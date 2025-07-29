"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  ComposedChart,
} from "recharts"
import { TrendingUp } from "lucide-react"

interface PerformanceChartsProps {
  timeframe: string
}

export function PerformanceCharts({ timeframe }: PerformanceChartsProps) {
  const performanceData = [
    { quarter: "Q1 2023", portfolio: 8.2, benchmark: 6.5, cashFlow: 125000 },
    { quarter: "Q2 2023", portfolio: 12.1, benchmark: 8.3, cashFlow: -85000 },
    { quarter: "Q3 2023", portfolio: 15.4, benchmark: 11.2, cashFlow: 95000 },
    { quarter: "Q4 2023", portfolio: 18.7, benchmark: 13.8, cashFlow: -120000 },
    { quarter: "Q1 2024", portfolio: 22.3, benchmark: 16.1, cashFlow: 150000 },
  ]

  const vintageData = [
    { year: "2019", irr: 24.5, multiple: 2.1, deployed: 850000 },
    { year: "2020", irr: 18.2, multiple: 1.8, deployed: 1200000 },
    { year: "2021", irr: 15.7, multiple: 1.5, deployed: 950000 },
    { year: "2022", irr: 12.3, multiple: 1.2, deployed: 750000 },
    { year: "2023", irr: 8.9, multiple: 1.1, deployed: 600000 },
  ]

  const sectorPerformance = [
    { sector: "Technology", allocation: 35, irr: 28.5, risk: "Medium" },
    { sector: "Healthcare", allocation: 25, irr: 22.1, risk: "Low" },
    { sector: "Financial Services", allocation: 20, irr: 18.7, risk: "Medium" },
    { sector: "Consumer", allocation: 15, irr: 15.3, risk: "High" },
    { sector: "Industrial", allocation: 5, irr: 12.8, risk: "Low" },
  ]

  return (
    <div className="space-y-6">
      {/* Performance vs Benchmark */}
      <Card className="bg-white border-stone-200">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-serif text-slate-800">Performance vs Benchmark</CardTitle>
            <div className="flex gap-2">
              <Select defaultValue="quarterly">
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="monthly">Monthly</SelectItem>
                  <SelectItem value="quarterly">Quarterly</SelectItem>
                  <SelectItem value="annual">Annual</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="quarter" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} tickFormatter={(value) => `${value}%`} />
                <Tooltip
                  formatter={(value: number, name: string) => [
                    `${value}%`,
                    name === "portfolio" ? "Portfolio IRR" : "Benchmark IRR",
                  ]}
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #d6d3d1",
                    borderRadius: "8px",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="portfolio"
                  stroke="#b45309"
                  strokeWidth={3}
                  name="Portfolio IRR"
                  dot={{ fill: "#b45309", strokeWidth: 2, r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="benchmark"
                  stroke="#64748b"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Benchmark IRR"
                  dot={{ fill: "#64748b", strokeWidth: 2, r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Vintage Year Analysis */}
        <Card className="bg-white border-stone-200">
          <CardHeader>
            <CardTitle className="text-xl font-serif text-slate-800">Vintage Year Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={vintageData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="year" stroke="#64748b" fontSize={12} />
                  <YAxis yAxisId="left" stroke="#64748b" fontSize={12} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    stroke="#64748b"
                    fontSize={12}
                    tickFormatter={(value) => `${value}x`}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => {
                      if (name === "irr") return [`${value}%`, "IRR"]
                      if (name === "multiple") return [`${value}x`, "Multiple"]
                      return [value, name]
                    }}
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid #d6d3d1",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar yAxisId="left" dataKey="irr" fill="#b45309" name="irr" />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="multiple"
                    stroke="#d97706"
                    strokeWidth={2}
                    name="multiple"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Cash Flow Analysis */}
        <Card className="bg-white border-stone-200">
          <CardHeader>
            <CardTitle className="text-xl font-serif text-slate-800">Cash Flow Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="quarter" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`} />
                  <Tooltip
                    formatter={(value: number) => [
                      `$${(value / 1000).toFixed(0)}K`,
                      value > 0 ? "Capital Call" : "Distribution",
                    ]}
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid #d6d3d1",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar dataKey="cashFlow" fill={(entry) => (entry > 0 ? "#dc2626" : "#16a34a")} name="Cash Flow" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sector Performance */}
      <Card className="bg-white border-stone-200">
        <CardHeader>
          <CardTitle className="text-xl font-serif text-slate-800">Sector Performance Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-stone-200">
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Sector</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Allocation</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">IRR</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Risk Level</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Performance</th>
                </tr>
              </thead>
              <tbody>
                {sectorPerformance.map((sector, index) => (
                  <tr key={index} className="border-b border-stone-100 hover:bg-stone-50 transition-colors">
                    <td className="py-3 px-4 font-medium text-slate-800">{sector.sector}</td>
                    <td className="py-3 px-4 text-slate-600">{sector.allocation}%</td>
                    <td className="py-3 px-4 font-medium text-green-600">{sector.irr}%</td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-1 rounded-full text-xs ${
                          sector.risk === "Low"
                            ? "bg-green-100 text-green-800"
                            : sector.risk === "Medium"
                              ? "bg-yellow-100 text-yellow-800"
                              : "bg-red-100 text-red-800"
                        }`}
                      >
                        {sector.risk}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-green-600" />
                        <span className="text-sm text-green-600">Outperforming</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
