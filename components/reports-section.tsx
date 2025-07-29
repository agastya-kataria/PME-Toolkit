"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { FileText, Download, Calendar, Mail, Settings, Eye, Share, Clock, CheckCircle } from "lucide-react"

export function ReportsSection() {
  const [selectedTemplate, setSelectedTemplate] = useState("quarterly")
  const [reportFrequency, setReportFrequency] = useState("quarterly")

  const reportTemplates = [
    {
      id: "quarterly",
      name: "Quarterly Performance Report",
      description: "Comprehensive quarterly portfolio performance and analytics",
      lastGenerated: "2024-01-15",
      status: "Ready",
    },
    {
      id: "annual",
      name: "Annual Summary Report",
      description: "Year-end portfolio summary with tax implications",
      lastGenerated: "2023-12-31",
      status: "Ready",
    },
    {
      id: "capital-account",
      name: "Capital Account Statement",
      description: "Detailed capital account activity and balances",
      lastGenerated: "2024-01-10",
      status: "Ready",
    },
    {
      id: "distribution",
      name: "Distribution Notice",
      description: "Distribution details and tax reporting information",
      lastGenerated: "2024-01-08",
      status: "Pending",
    },
  ]

  const scheduledReports = [
    {
      id: 1,
      name: "Monthly Portfolio Summary",
      frequency: "Monthly",
      nextRun: "2024-02-01",
      recipients: ["investor@email.com", "advisor@email.com"],
      status: "Active",
    },
    {
      id: 2,
      name: "Quarterly Performance Review",
      frequency: "Quarterly",
      nextRun: "2024-04-01",
      recipients: ["investor@email.com"],
      status: "Active",
    },
    {
      id: 3,
      name: "Annual Tax Report",
      frequency: "Annual",
      nextRun: "2024-12-31",
      recipients: ["tax-advisor@email.com"],
      status: "Active",
    },
  ]

  const recentReports = [
    {
      id: 1,
      name: "Q4 2023 Performance Report",
      type: "Quarterly",
      generated: "2024-01-15",
      size: "2.4 MB",
      status: "Completed",
    },
    {
      id: 2,
      name: "December 2023 Capital Account",
      type: "Monthly",
      generated: "2024-01-10",
      size: "1.8 MB",
      status: "Completed",
    },
    {
      id: 3,
      name: "Year-End Tax Summary 2023",
      type: "Annual",
      generated: "2024-01-05",
      size: "3.2 MB",
      status: "Completed",
    },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Ready":
      case "Completed":
      case "Active":
        return "bg-green-100 text-green-800"
      case "Pending":
        return "bg-yellow-100 text-yellow-800"
      case "Error":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="generate" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="generate">Generate Reports</TabsTrigger>
          <TabsTrigger value="scheduled">Scheduled Reports</TabsTrigger>
          <TabsTrigger value="history">Report History</TabsTrigger>
        </TabsList>

        <TabsContent value="generate" className="space-y-6">
          {/* Report Generation */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800">Generate New Report</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="template">Report Template</Label>
                  <Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {reportTemplates.map((template) => (
                        <SelectItem key={template.id} value={template.id}>
                          {template.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="period">Reporting Period</Label>
                  <Select defaultValue="q4-2023">
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="q4-2023">Q4 2023</SelectItem>
                      <SelectItem value="q3-2023">Q3 2023</SelectItem>
                      <SelectItem value="q2-2023">Q2 2023</SelectItem>
                      <SelectItem value="q1-2023">Q1 2023</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="format">Output Format</Label>
                  <Select defaultValue="pdf">
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pdf">PDF</SelectItem>
                      <SelectItem value="excel">Excel</SelectItem>
                      <SelectItem value="powerpoint">PowerPoint</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex gap-2 pt-4">
                  <Button className="flex-1 bg-amber-700 hover:bg-amber-800 text-white">
                    <FileText className="w-4 h-4 mr-2" />
                    Generate Report
                  </Button>
                  <Button variant="outline" className="flex-1 bg-transparent">
                    <Eye className="w-4 h-4 mr-2" />
                    Preview
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Report Templates */}
            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800">Available Templates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {reportTemplates.map((template) => (
                    <div
                      key={template.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedTemplate === template.id
                          ? "border-amber-500 bg-amber-50"
                          : "border-stone-200 hover:border-stone-300"
                      }`}
                      onClick={() => setSelectedTemplate(template.id)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-slate-800">{template.name}</h4>
                        <Badge className={getStatusColor(template.status)}>{template.status}</Badge>
                      </div>
                      <p className="text-sm text-slate-600 mb-2">{template.description}</p>
                      <p className="text-xs text-slate-500">
                        Last generated: {new Date(template.lastGenerated).toLocaleDateString()}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="scheduled" className="space-y-6">
          {/* Scheduled Reports */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800">Schedule New Report</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="report-name">Report Name</Label>
                  <Input id="report-name" placeholder="Enter report name" className="bg-stone-50 border-stone-300" />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="frequency">Frequency</Label>
                  <Select value={reportFrequency} onValueChange={setReportFrequency}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                      <SelectItem value="quarterly">Quarterly</SelectItem>
                      <SelectItem value="annual">Annual</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="recipients">Recipients</Label>
                  <Input id="recipients" placeholder="Enter email addresses" className="bg-stone-50 border-stone-300" />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="start-date">Start Date</Label>
                  <Input id="start-date" type="date" className="bg-stone-50 border-stone-300" />
                </div>

                <Button className="w-full bg-amber-700 hover:bg-amber-800 text-white">
                  <Calendar className="w-4 h-4 mr-2" />
                  Schedule Report
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800">Active Schedules</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {scheduledReports.map((report) => (
                    <div key={report.id} className="p-4 border border-stone-200 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-slate-800">{report.name}</h4>
                        <Badge className={getStatusColor(report.status)}>{report.status}</Badge>
                      </div>
                      <div className="space-y-1 text-sm text-slate-600">
                        <div className="flex items-center gap-2">
                          <Clock className="w-3 h-3" />
                          <span>{report.frequency}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Calendar className="w-3 h-3" />
                          <span>Next: {new Date(report.nextRun).toLocaleDateString()}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Mail className="w-3 h-3" />
                          <span>{report.recipients.length} recipient(s)</span>
                        </div>
                      </div>
                      <div className="flex gap-2 mt-3">
                        <Button variant="outline" size="sm">
                          <Settings className="w-3 h-3 mr-1" />
                          Edit
                        </Button>
                        <Button variant="outline" size="sm">
                          <Share className="w-3 h-3 mr-1" />
                          Share
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          {/* Report History */}
          <Card className="bg-white border-stone-200">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl font-serif text-slate-800">Report History</CardTitle>
                <div className="flex gap-2">
                  <Select defaultValue="all">
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Reports</SelectItem>
                      <SelectItem value="quarterly">Quarterly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                      <SelectItem value="annual">Annual</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Export List
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-stone-200">
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Report Name</th>
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Type</th>
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Generated</th>
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Size</th>
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Status</th>
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentReports.map((report) => (
                      <tr key={report.id} className="border-b border-stone-100 hover:bg-stone-50 transition-colors">
                        <td className="py-3 px-4 font-medium text-slate-800">{report.name}</td>
                        <td className="py-3 px-4 text-slate-600">{report.type}</td>
                        <td className="py-3 px-4 text-slate-600">{new Date(report.generated).toLocaleDateString()}</td>
                        <td className="py-3 px-4 text-slate-600">{report.size}</td>
                        <td className="py-3 px-4">
                          <Badge className={getStatusColor(report.status)}>
                            <CheckCircle className="w-3 h-3 mr-1" />
                            {report.status}
                          </Badge>
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex gap-2">
                            <Button variant="outline" size="sm">
                              <Download className="w-3 h-3 mr-1" />
                              Download
                            </Button>
                            <Button variant="outline" size="sm">
                              <Eye className="w-3 h-3 mr-1" />
                              View
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
