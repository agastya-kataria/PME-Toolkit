"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts"
import { Activity, Plus } from "lucide-react"

interface PortfolioOverviewProps {
  searchTerm: string
  timeframe: string
}

export function PortfolioOverview({ searchTerm, timeframe }: PortfolioOverviewProps) {
  const [selectedAllocation, setSelectedAllocation] = useState("strategy")

  const allocationData = [
    { name: "Private Equity", value: 45, color: "#b45309" },
    { name: "Real Estate", value: 25, color: "#d97706" },
    { name: "Venture Capital", value: 15, color: "#f59e0b" },
    { name: "Private Debt", value: 10, color: "#fbbf24" },
    { name: "Infrastructure", value: 5, color: "#fcd34d" },
  ]

  const performanceData = [
    { month: "Jan", value: 2100000 },
    { month: "Feb", value: 2150000 },
    { month: "Mar", value: 2200000 },
    { month: "Apr", value: 2180000 },
    { month: "May", value: 2250000 },
    { month: "Jun", value: 2300000 },
    { month: "Jul", value: 2350000 },
    { month: "Aug", value: 2320000 },
    { month: "Sep", value: 2380000 },
    { month: "Oct", value: 2400000 },
  ]

  const recentActivity = [
    {
      id: 1,
      type: "Capital Call",
      fund: "Blackstone Growth Fund IV",
      amount: 125000,
      date: "2024-01-15",
      status: "Completed",
    },
    {
      id: 2,
      type: "Distribution",
      fund: "KKR North America Fund XIII",
      amount: 85000,
      date: "2024-01-12",
      status: "Received",
    },
    {
      id: 3,
      type: "Valuation Update",
      fund: "Apollo Strategic Fund III",
      amount: 0,
      date: "2024-01-10",
      status: "Updated",
    },
    {
      id: 4,
      type: "Capital Call",
      fund: "Carlyle Partners VIII",
      amount: 95000,
      date: "2024-01-08",
      status: "Pending",
    },
  ]

  const filteredActivity = recentActivity.filter(
    (activity) =>
      activity.fund.toLowerCase().includes(searchTerm.toLowerCase()) ||
      activity.type.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Completed":
      case "Received":
        return "bg-green-100 text-green-800"
      case "Pending":
        return "bg-yellow-100 text-yellow-800"
      case "Updated":
        return "bg-blue-100 text-blue-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Performance Chart */}
      <Card className="bg-white border-stone-200">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-serif text-slate-800">Portfolio Performance</CardTitle>
            <div className="flex gap-2">
              <Button variant={timeframe === "1M" ? "default" : "outline"} size="sm" className="text-xs">
                1M
              </Button>
              <Button variant={timeframe === "3M" ? "default" : "outline"} size="sm" className="text-xs">
                3M
              </Button>
              <Button
                variant={timeframe === "1Y" ? "default" : "outline"}
                size="sm"
                className="text-xs bg-amber-700 text-white"
              >
                1Y
              </Button>
              <Button variant={timeframe === "ALL" ? "default" : "outline"} size="sm" className="text-xs">
                All
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="month" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} tickFormatter={(value) => `$${(value / 1000000).toFixed(1)}M`} />
                <Tooltip
                  formatter={(value: number) => [formatCurrency(value), "Portfolio Value"]}
                  labelStyle={{ color: "#1e293b" }}
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #d6d3d1",
                    borderRadius: "8px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#b45309"
                  strokeWidth={3}
                  dot={{ fill: "#b45309", strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: "#b45309", strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Asset Allocation */}
        <Card className="bg-white border-stone-200">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-xl font-serif text-slate-800">Asset Allocation</CardTitle>
              <div className="flex gap-2">
                <Button
                  variant={selectedAllocation === "strategy" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedAllocation("strategy")}
                  className="text-xs bg-amber-700 text-white"
                >
                  By Strategy
                </Button>
                <Button
                  variant={selectedAllocation === "geography" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedAllocation("geography")}
                  className="text-xs"
                >
                  By Geography
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={allocationData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {allocationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number) => [`${value}%`, "Allocation"]}
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid #d6d3d1",
                      borderRadius: "8px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
              {allocationData.map((item, index) => (
                <div key={index} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-sm text-slate-600">{item.name}</span>
                  <span className="text-sm font-medium text-slate-800 ml-auto">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Fund Commitments */}
        <Card className="bg-white border-stone-200">
          <CardHeader>
            <CardTitle className="text-xl font-serif text-slate-800">Fund Commitments</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-700">Blackstone Growth IV</span>
                <span className="text-sm text-slate-600">$500K</span>
              </div>
              <Progress value={75} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>$375K called</span>
                <span>75% deployed</span>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-700">KKR North America XIII</span>
                <span className="text-sm text-slate-600">$750K</span>
              </div>
              <Progress value={60} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>$450K called</span>
                <span>60% deployed</span>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-700">Apollo Strategic III</span>
                <span className="text-sm text-slate-600">$400K</span>
              </div>
              <Progress value={90} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>$360K called</span>
                <span>90% deployed</span>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-700">Carlyle Partners VIII</span>
                <span className="text-sm text-slate-600">$600K</span>
              </div>
              <Progress value={45} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>$270K called</span>
                <span>45% deployed</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card className="bg-white border-stone-200">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-serif text-slate-800">Recent Activity</CardTitle>
            <Button size="sm" className="bg-amber-700 hover:bg-amber-800 text-white">
              <Plus className="w-4 h-4 mr-2" />
              Add Transaction
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-stone-200">
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Type</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Fund</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Amount</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Date</th>
                  <th className="text-left py-3 px-4 font-medium text-slate-700">Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredActivity.map((activity) => (
                  <tr key={activity.id} className="border-b border-stone-100 hover:bg-stone-50 transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-slate-500" />
                        <span className="text-sm font-medium text-slate-800">{activity.type}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm text-slate-600">{activity.fund}</td>
                    <td className="py-3 px-4 text-sm font-medium text-slate-800">
                      {activity.amount > 0 ? formatCurrency(activity.amount) : "—"}
                    </td>
                    <td className="py-3 px-4 text-sm text-slate-600">{new Date(activity.date).toLocaleDateString()}</td>
                    <td className="py-3 px-4">
                      <Badge className={`text-xs ${getStatusColor(activity.status)}`}>{activity.status}</Badge>
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
