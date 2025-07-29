"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import {
  Search,
  Crown,
  User,
  LineChartIcon as ChartLine,
  PercentIcon as Percentage,
  DollarSign,
  Building,
  ArrowUp,
  ArrowDown,
  Download,
  Plus,
} from "lucide-react"
import { PerformanceChart } from "@/components/performance-chart"
import { AllocationChart } from "@/components/allocation-chart"
import { InvestmentModal } from "@/components/investment-modal"

interface Investment {
  id: string
  name: string
  sector: string
  date: string
  amount: number
  stage: string
  status: "Active" | "Pending" | "Closed"
}

interface Stat {
  id: string
  title: string
  value: string
  change: string
  changeType: "positive" | "negative" | "neutral"
  icon: any
}

export default function PrivateMarketsAnalyzer() {
  const [timeframe, setTimeframe] = useState("30-days")
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [investments, setInvestments] = useState<Investment[]>([])
  const [stats, setStats] = useState<Stat[]>([])

  // Initialize data
  useEffect(() => {
    const initialStats: Stat[] = [
      {
        id: "totalValue",
        title: "Total Portfolio Value",
        value: "$12.7M",
        change: "+4.2% this month",
        changeType: "positive",
        icon: ChartLine,
      },
      {
        id: "ytdReturn",
        title: "YTD Return",
        value: "8.6%",
        change: "+1.8% above benchmark",
        changeType: "positive",
        icon: Percentage,
      },
      {
        id: "newInvestments",
        title: "New Investments",
        value: "3",
        change: "$2.1M deployed",
        changeType: "neutral",
        icon: DollarSign,
      },
      {
        id: "activeHoldings",
        title: "Active Holdings",
        value: "17",
        change: "+2 added this quarter",
        changeType: "positive",
        icon: Building,
      },
    ]

    const initialInvestments: Investment[] = [
      {
        id: "1",
        name: "Horizon Capital Fund V",
        sector: "Private Equity",
        date: "2023-09-15",
        amount: 850000,
        stage: "Growth",
        status: "Active",
      },
      {
        id: "2",
        name: "Veridian Real Estate",
        sector: "Real Estate",
        date: "2023-09-02",
        amount: 1200000,
        stage: "Income",
        status: "Active",
      },
      {
        id: "3",
        name: "Nexus Venture Partners",
        sector: "Venture Capital",
        date: "2023-08-22",
        amount: 500000,
        stage: "Series B",
        status: "Pending",
      },
      {
        id: "4",
        name: "Atlas Infrastructure Fund",
        sector: "Infrastructure",
        date: "2023-08-10",
        amount: 750000,
        stage: "Mature",
        status: "Active",
      },
      {
        id: "5",
        name: "Pinnacle Credit Opportunities",
        sector: "Private Debt",
        date: "2023-07-28",
        amount: 600000,
        stage: "Senior Secured",
        status: "Closed",
      },
    ]

    setStats(initialStats)
    setInvestments(initialInvestments)
  }, [])

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount)
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
  }

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "Active":
        return "default"
      case "Pending":
        return "secondary"
      case "Closed":
        return "outline"
      default:
        return "default"
    }
  }

  const filteredInvestments = investments.filter(
    (investment) =>
      investment.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      investment.sector.toLowerCase().includes(searchTerm.toLowerCase()) ||
      investment.stage.toLowerCase().includes(searchTerm.toLowerCase()) ||
      investment.status.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const handleAddInvestment = (newInvestment: Omit<Investment, "id">) => {
    const investment: Investment = {
      ...newInvestment,
      id: Date.now().toString(),
    }
    setInvestments((prev) => [investment, ...prev])
    setIsModalOpen(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-amber-600/20 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-amber-400 to-amber-600 rounded-full flex items-center justify-center">
                <Crown className="w-6 h-6 text-slate-900" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-amber-200 to-amber-400 bg-clip-text text-transparent">
                  Vestige Capital
                </h1>
                <p className="text-xs text-slate-400 tracking-wider">PRIVATE WEALTH MANAGEMENT</p>
              </div>
            </div>

            {/* Navigation */}
            <nav className="hidden md:flex items-center gap-8">
              <a href="#" className="text-amber-400 font-medium border-b border-amber-400">
                Dashboard
              </a>
              <a href="#" className="text-slate-300 hover:text-amber-400 transition-colors">
                Portfolio
              </a>
              <a href="#" className="text-slate-300 hover:text-amber-400 transition-colors">
                Markets
              </a>
              <a href="#" className="text-slate-300 hover:text-amber-400 transition-colors">
                Reports
              </a>
              <a href="#" className="text-slate-300 hover:text-amber-400 transition-colors">
                Documents
              </a>
            </nav>

            {/* User Actions */}
            <div className="flex items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                <Input
                  placeholder="Search investments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64 bg-slate-800/50 border-amber-600/30 text-slate-200 placeholder:text-slate-400"
                />
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-slate-700 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-slate-300" />
                </div>
                <span className="text-slate-200 font-medium">W. Kensington</span>
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
            <h2 className="text-3xl font-bold text-slate-100 mb-2">Private Markets Analyzer</h2>
            <p className="text-slate-400">Performance insights and portfolio analytics</p>
          </div>
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-48 bg-slate-800/50 border-amber-600/30 text-slate-200">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-amber-600/30">
              <SelectItem value="7-days">Last 7 Days</SelectItem>
              <SelectItem value="30-days">Last 30 Days</SelectItem>
              <SelectItem value="quarter">Last Quarter</SelectItem>
              <SelectItem value="ytd">Year to Date</SelectItem>
              <SelectItem value="all">All Time</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <Card
              key={index}
              className="bg-slate-800/50 border-amber-600/20 hover:bg-slate-800/70 transition-all duration-300 hover:scale-105"
            >
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-slate-300">{stat.title}</CardTitle>
                <div className="w-10 h-10 bg-amber-600/20 rounded-lg flex items-center justify-center">
                  <stat.icon className="w-5 h-5 text-amber-400" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-slate-100 mb-1">{stat.value}</div>
                <div
                  className={`flex items-center gap-1 text-sm ${
                    stat.changeType === "positive"
                      ? "text-green-400"
                      : stat.changeType === "negative"
                        ? "text-red-400"
                        : "text-slate-400"
                  }`}
                >
                  {stat.changeType === "positive" && <ArrowUp className="w-3 h-3" />}
                  {stat.changeType === "negative" && <ArrowDown className="w-3 h-3" />}
                  <span>{stat.change}</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <Card className="lg:col-span-2 bg-slate-800/50 border-amber-600/20">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-slate-100">Portfolio Performance</CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-amber-600/30 text-slate-300 bg-transparent hover:bg-amber-600/10"
                  >
                    1M
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-amber-600/30 text-slate-300 bg-transparent hover:bg-amber-600/10"
                  >
                    3M
                  </Button>
                  <Button size="sm" className="bg-amber-600 hover:bg-amber-700 text-slate-900">
                    1Y
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-amber-600/30 text-slate-300 bg-transparent hover:bg-amber-600/10"
                  >
                    All
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <PerformanceChart />
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-amber-600/20">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-slate-100">Asset Allocation</CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-amber-600/30 text-slate-300 bg-transparent hover:bg-amber-600/10"
                  >
                    Value
                  </Button>
                  <Button size="sm" className="bg-amber-600 hover:bg-amber-700 text-slate-900">
                    Sector
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <AllocationChart />
            </CardContent>
          </Card>
        </div>

        {/* Investments Table */}
        <Card className="bg-slate-800/50 border-amber-600/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-slate-100">Recent Investments</CardTitle>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-amber-600/30 text-slate-300 bg-transparent hover:bg-amber-600/10"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
                <Button
                  size="sm"
                  className="bg-amber-600 hover:bg-amber-700 text-slate-900"
                  onClick={() => setIsModalOpen(true)}
                >
                  <Plus className="w-4 h-4 mr-2" />
                  New
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-amber-600/20">
                    <th className="text-left py-3 px-4 text-amber-400 font-medium">Investment</th>
                    <th className="text-left py-3 px-4 text-amber-400 font-medium">Sector</th>
                    <th className="text-left py-3 px-4 text-amber-400 font-medium">Date</th>
                    <th className="text-left py-3 px-4 text-amber-400 font-medium">Amount</th>
                    <th className="text-left py-3 px-4 text-amber-400 font-medium">Stage</th>
                    <th className="text-left py-3 px-4 text-amber-400 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredInvestments.map((investment) => (
                    <tr
                      key={investment.id}
                      className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                    >
                      <td className="py-3 px-4 text-slate-200 font-medium">{investment.name}</td>
                      <td className="py-3 px-4 text-slate-300">{investment.sector}</td>
                      <td className="py-3 px-4 text-slate-300">{formatDate(investment.date)}</td>
                      <td className="py-3 px-4 text-slate-200 font-medium">{formatCurrency(investment.amount)}</td>
                      <td className="py-3 px-4 text-slate-300">{investment.stage}</td>
                      <td className="py-3 px-4">
                        <Badge
                          variant={getStatusVariant(investment.status)}
                          className="bg-amber-600/20 text-amber-200 border-amber-600/30"
                        >
                          {investment.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="border-t border-amber-600/20 mt-16 py-8">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex justify-center gap-8 mb-4">
            <a href="#" className="text-slate-400 hover:text-amber-400 transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="text-slate-400 hover:text-amber-400 transition-colors">
              Terms of Service
            </a>
            <a href="#" className="text-slate-400 hover:text-amber-400 transition-colors">
              Compliance
            </a>
            <a href="#" className="text-slate-400 hover:text-amber-400 transition-colors">
              Contact
            </a>
          </div>
          <p className="text-center text-slate-400 text-sm">
            © 2023 Vestige Capital. All rights reserved. For accredited investors only.
          </p>
        </div>
      </footer>

      <InvestmentModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} onSubmit={handleAddInvestment} />
    </div>
  )
}
