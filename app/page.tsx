"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Search,
  Crown,
  User,
  TrendingUp,
  DollarSign,
  BarChart3,
  Bell,
  Download,
  ArrowUp,
  ArrowDown,
  Briefcase,
} from "lucide-react"
import { PortfolioOverview } from "@/components/portfolio-overview"
import { PerformanceCharts } from "@/components/performance-charts"
import { AnalyticsDashboard } from "@/components/analytics-dashboard"
import { ReportsSection } from "@/components/reports-section"
import { SettingsPanel } from "@/components/settings-panel"

interface PortfolioMetric {
  id: string
  title: string
  value: string
  change: string
  changeType: "positive" | "negative" | "neutral"
  icon: any
  description: string
}

export default function PrivateMarketsAnalyzer() {
  const [activeTab, setActiveTab] = useState("overview")
  const [searchTerm, setSearchTerm] = useState("")
  const [timeframe, setTimeframe] = useState("1Y")
  const [isConnected, setIsConnected] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(new Date())

  const portfolioMetrics: PortfolioMetric[] = [
    {
      id: "portfolio-value",
      title: "Portfolio Value",
      value: "$2.4M",
      change: "+12.3%",
      changeType: "positive",
      icon: TrendingUp,
      description: "Current market value of all holdings",
    },
    {
      id: "committed-capital",
      title: "Total Committed",
      value: "$3.2M",
      change: "75% deployed",
      changeType: "neutral",
      icon: DollarSign,
      description: "Total capital committed across all funds",
    },
    {
      id: "unrealized-gains",
      title: "Unrealized Gains",
      value: "$800K",
      change: "+18.7%",
      changeType: "positive",
      icon: BarChart3,
      description: "Unrealized appreciation in portfolio",
    },
    {
      id: "distributions",
      title: "Distributions Received",
      value: "$1.2M",
      change: "+$240K YTD",
      changeType: "positive",
      icon: Briefcase,
      description: "Total distributions received to date",
    },
  ]

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdated(new Date())
    }, 30000) // Update every 30 seconds

    return () => clearInterval(interval)
  }, [])

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  }

  return (
    <div className="min-h-screen bg-stone-50">
      {/* Header */}
      <header className="bg-white border-b border-stone-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo and Firm Name */}
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-amber-600 to-amber-700 rounded-full flex items-center justify-center shadow-lg">
                <Crown className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-serif font-bold text-slate-800">Kensington Capital</h1>
                <p className="text-xs text-slate-500 tracking-wider uppercase">Private Wealth Management</p>
              </div>
            </div>

            {/* Navigation */}
            <nav className="hidden lg:flex items-center gap-8">
              <button
                onClick={() => setActiveTab("overview")}
                className={`font-medium transition-colors ${
                  activeTab === "overview"
                    ? "text-amber-700 border-b-2 border-amber-700 pb-1"
                    : "text-slate-600 hover:text-amber-700"
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab("portfolio")}
                className={`font-medium transition-colors ${
                  activeTab === "portfolio"
                    ? "text-amber-700 border-b-2 border-amber-700 pb-1"
                    : "text-slate-600 hover:text-amber-700"
                }`}
              >
                Portfolio
              </button>
              <button
                onClick={() => setActiveTab("analytics")}
                className={`font-medium transition-colors ${
                  activeTab === "analytics"
                    ? "text-amber-700 border-b-2 border-amber-700 pb-1"
                    : "text-slate-600 hover:text-amber-700"
                }`}
              >
                Analytics
              </button>
              <button
                onClick={() => setActiveTab("reports")}
                className={`font-medium transition-colors ${
                  activeTab === "reports"
                    ? "text-amber-700 border-b-2 border-amber-700 pb-1"
                    : "text-slate-600 hover:text-amber-700"
                }`}
              >
                Reports
              </button>
              <button
                onClick={() => setActiveTab("settings")}
                className={`font-medium transition-colors ${
                  activeTab === "settings"
                    ? "text-amber-700 border-b-2 border-amber-700 pb-1"
                    : "text-slate-600 hover:text-amber-700"
                }`}
              >
                Settings
              </button>
            </nav>

            {/* User Actions */}
            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`} />
                <span className="text-xs text-slate-500">{isConnected ? "Connected" : "Disconnected"}</span>
              </div>

              {/* Last Updated */}
              <div className="text-xs text-slate-500">Updated: {formatTime(lastUpdated)}</div>

              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                <Input
                  placeholder="Search investments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64 bg-stone-50 border-stone-300 focus:border-amber-500"
                />
              </div>

              {/* Notifications */}
              <Button variant="ghost" size="sm" className="relative">
                <Bell className="w-5 h-5 text-slate-600" />
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full text-xs"></span>
              </Button>

              {/* User Profile */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-slate-200 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-slate-600" />
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-slate-800">W. Kensington</p>
                  <p className="text-xs text-slate-500">Managing Partner</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Dashboard Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-serif font-bold text-slate-800 mb-2">Private Markets Dashboard</h2>
            <p className="text-slate-600">Comprehensive portfolio insights and performance analytics</p>
          </div>
          <div className="flex items-center gap-4">
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger className="w-32 bg-white border-stone-300">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1M">1 Month</SelectItem>
                <SelectItem value="3M">3 Months</SelectItem>
                <SelectItem value="1Y">1 Year</SelectItem>
                <SelectItem value="3Y">3 Years</SelectItem>
                <SelectItem value="5Y">5 Years</SelectItem>
                <SelectItem value="ALL">All Time</SelectItem>
              </SelectContent>
            </Select>
            <Button className="bg-amber-700 hover:bg-amber-800 text-white">
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </Button>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {portfolioMetrics.map((metric) => (
            <Card key={metric.id} className="bg-white border-stone-200 hover:shadow-lg transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-slate-600">{metric.title}</CardTitle>
                <div className="w-10 h-10 bg-amber-50 rounded-lg flex items-center justify-center">
                  <metric.icon className="w-5 h-5 text-amber-700" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-slate-800 mb-1">{metric.value}</div>
                <div
                  className={`flex items-center gap-1 text-sm ${
                    metric.changeType === "positive"
                      ? "text-green-600"
                      : metric.changeType === "negative"
                        ? "text-red-600"
                        : "text-slate-500"
                  }`}
                >
                  {metric.changeType === "positive" && <ArrowUp className="w-3 h-3" />}
                  {metric.changeType === "negative" && <ArrowDown className="w-3 h-3" />}
                  <span>{metric.change}</span>
                </div>
                <p className="text-xs text-slate-500 mt-2">{metric.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="hidden">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <PortfolioOverview searchTerm={searchTerm} timeframe={timeframe} />
          </TabsContent>

          <TabsContent value="portfolio" className="space-y-6">
            <PerformanceCharts timeframe={timeframe} />
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <AnalyticsDashboard timeframe={timeframe} />
          </TabsContent>

          <TabsContent value="reports" className="space-y-6">
            <ReportsSection />
          </TabsContent>

          <TabsContent value="settings" className="space-y-6">
            <SettingsPanel />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-stone-200 mt-16 py-8 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex justify-center gap-8 mb-4">
            <a href="#" className="text-slate-500 hover:text-amber-700 transition-colors text-sm">
              Privacy Policy
            </a>
            <a href="#" className="text-slate-500 hover:text-amber-700 transition-colors text-sm">
              Terms of Service
            </a>
            <a href="#" className="text-slate-500 hover:text-amber-700 transition-colors text-sm">
              Compliance
            </a>
            <a href="#" className="text-slate-500 hover:text-amber-700 transition-colors text-sm">
              Contact
            </a>
          </div>
          <p className="text-center text-slate-400 text-sm">
            © 2024 Kensington Capital. All rights reserved. For accredited investors only.
          </p>
        </div>
      </footer>
    </div>
  )
}
