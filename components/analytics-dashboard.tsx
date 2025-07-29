"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts"
import { AlertTriangle, TrendingUp, Shield, Activity } from "lucide-react"

interface AnalyticsDashboardProps {
  timeframe: string
}

export function AnalyticsDashboard({ timeframe }: AnalyticsDashboardProps) {
  const riskReturnData = [
    { name: "Fund A", risk: 12, return: 24, size: 500000 },
    { name: "Fund B", risk: 8, return: 18, size: 750000 },
    { name: "Fund C", risk: 15, return: 28, size: 400000 },
    { name: "Fund D", risk: 10, return: 22, size: 600000 },
    { name: "Fund E", risk: 6, return: 15, size: 300000 },
  ]

  const portfolioMetrics = [
    { metric: "Diversification", value: 85, benchmark: 75 },
    { metric: "Liquidity", value: 65, benchmark: 70 },
    { metric: "Risk-Adjusted Return", value: 92, benchmark: 80 },
    { metric: "Vintage Diversification", value: 78, benchmark: 85 },
    { metric: "Geographic Spread", value: 70, benchmark: 75 },
    { metric: "Sector Balance", value: 88, benchmark: 80 },
  ]

  const concentrationData = [
    { name: "Top 5 Holdings", value: 45, color: "#b45309" },
    { name: "Next 10 Holdings", value: 30, color: "#d97706" },
    { name: "Remaining Holdings", value: 25, color: "#f59e0b" },
  ]

  const stressTestScenarios = [
    {
      scenario: "Market Downturn (-20%)",
      portfolioImpact: -15.2,
      probability: "Medium",
      timeToRecover: "18 months",
    },
    {
      scenario: "Interest Rate Spike (+300bp)",
      portfolioImpact: -8.7,
      probability: "High",
      timeToRecover: "12 months",
    },
    {
      scenario: "Credit Crisis",
      portfolioImpact: -22.1,
      probability: "Low",
      timeToRecover: "24 months",
    },
    {
      scenario: "Sector Rotation",
      portfolioImpact: -5.3,
      probability: "Medium",
      timeToRecover: "6 months",
    },
  ]

  const getProbabilityColor = (probability: string) => {
    switch (probability) {
      case "High":
        return "bg-red-100 text-red-800"
      case "Medium":
        return "bg-yellow-100 text-yellow-800"
      case "Low":
        return "bg-green-100 text-green-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      {/* Risk Analysis Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-white border-stone-200">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-slate-600">Portfolio VaR</CardTitle>
            <Shield className="w-5 h-5 text-amber-700" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-slate-800 mb-1">-$285K</div>
            <div className="flex items-center gap-1 text-sm text-red-600">
              <AlertTriangle className="w-3 h-3" />
              <span>95% confidence, 1-day</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white border-stone-200">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-slate-600">Sharpe Ratio</CardTitle>
            <TrendingUp className="w-5 h-5 text-amber-700" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-slate-800 mb-1">1.84</div>
            <div className="flex items-center gap-1 text-sm text-green-600">
              <TrendingUp className="w-3 h-3" />
              <span>Above benchmark (1.52)</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white border-stone-200">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-slate-600">Beta</CardTitle>
            <Activity className="w-5 h-5 text-amber-700" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-slate-800 mb-1">0.92</div>
            <div className="flex items-center gap-1 text-sm text-slate-600">
              <span>Lower volatility than market</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk-Return Scatter */}
        <Card className="bg-white border-stone-200">
          <CardHeader>
            <CardTitle className="text-xl font-serif text-slate-800">Risk-Return Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" dataKey="risk" name="Risk" unit="%" stroke="#64748b" fontSize={12} />
                  <YAxis type="number" dataKey="return" name="Return" unit="%" stroke="#64748b" fontSize={12} />
                  <Tooltip
                    formatter={(value, name) => [`${value}%`, name === "risk" ? "Risk" : "Return"]}
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid #d6d3d1",
                      borderRadius: "8px",
                    }}
                  />
                  <Scatter data={riskReturnData} fill="#b45309" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Portfolio Metrics Radar */}
        <Card className="bg-white border-stone-200">
          <CardHeader>
            <CardTitle className="text-xl font-serif text-slate-800">Portfolio Health</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={portfolioMetrics}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} tickCount={5} />
                  <Radar
                    name="Portfolio"
                    dataKey="value"
                    stroke="#b45309"
                    fill="#b45309"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                  <Radar
                    name="Benchmark"
                    dataKey="benchmark"
                    stroke="#64748b"
                    fill="transparent"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                  />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Concentration Analysis */}
      <Card className="bg-white border-stone-200">
        <CardHeader>
          <CardTitle className="text-xl font-serif text-slate-800">Concentration Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium text-slate-700">Top Holdings Concentration</h4>
              {concentrationData.map((item, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-600">{item.name}</span>
                    <span className="text-sm font-medium text-slate-800">{item.value}%</span>
                  </div>
                  <Progress value={item.value} className="h-2" />
                </div>
              ))}
            </div>
            <div className="space-y-4">
              <h4 className="font-medium text-slate-700">Risk Metrics</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600">Herfindahl Index</span>
                  <span className="text-sm font-medium text-slate-800">0.18</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600">Effective Number of Holdings</span>
                  <span className="text-sm font-medium text-slate-800">5.6</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600">Concentration Risk Score</span>
                  <Badge className="bg-yellow-100 text-yellow-800">Medium</Badge>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stress Testing */}
      <Card className="bg-white border-stone-200">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-serif text-slate-800">Stress Test Results</CardTitle>
            <Button size="sm" className="bg-amber-700 hover:bg-amber-800 text-white">
              Run New Test
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-stone-200">
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Scenario</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Impact</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Probability</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Recovery Time</th>
                </tr>
              </thead>
              <tbody>
                {stressTestScenarios.map((scenario, index) => (
                  <tr key={index} className="border-b border-stone-100 hover:bg-stone-50 transition-colors">
                    <td className="py-3 px-4 font-medium text-slate-800">{scenario.scenario}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`font-medium ${scenario.portfolioImpact < 0 ? "text-red-600" : "text-green-600"}`}
                      >
                        {scenario.portfolioImpact > 0 ? "+" : ""}
                        {scenario.portfolioImpact}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className={getProbabilityColor(scenario.probability)}>{scenario.probability}</Badge>
                    </td>
                    <td className="py-3 px-4 text-slate-600">{scenario.timeToRecover}</td>
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
