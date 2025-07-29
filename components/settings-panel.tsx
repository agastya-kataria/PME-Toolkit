"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { User, Bell, Shield, Palette, Mail, Phone, Key, Activity, Save, RefreshCw } from "lucide-react"

export function SettingsPanel() {
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    sms: false,
    weeklyReport: true,
    monthlyReport: true,
    alerts: true,
  })

  const [preferences, setPreferences] = useState({
    theme: "light",
    currency: "USD",
    dateFormat: "MM/DD/YYYY",
    timezone: "EST",
    refreshInterval: "30",
  })

  const [profile, setProfile] = useState({
    firstName: "William",
    lastName: "Kensington",
    email: "w.kensington@vestigecapital.com",
    phone: "+1 (555) 123-4567",
    title: "Managing Partner",
    company: "Kensington Capital",
  })

  const activityLog = [
    {
      id: 1,
      action: "Portfolio report generated",
      timestamp: "2024-01-15 14:30:00",
      ip: "192.168.1.100",
    },
    {
      id: 2,
      action: "Login successful",
      timestamp: "2024-01-15 09:15:00",
      ip: "192.168.1.100",
    },
    {
      id: 3,
      action: "Settings updated",
      timestamp: "2024-01-14 16:45:00",
      ip: "192.168.1.100",
    },
    {
      id: 4,
      action: "New investment added",
      timestamp: "2024-01-14 11:20:00",
      ip: "192.168.1.100",
    },
  ]

  const handleNotificationChange = (key: string, value: boolean) => {
    setNotifications((prev) => ({ ...prev, [key]: value }))
  }

  const handlePreferenceChange = (key: string, value: string) => {
    setPreferences((prev) => ({ ...prev, [key]: value }))
  }

  const handleProfileChange = (key: string, value: string) => {
    setProfile((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="profile">Profile</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="preferences">Preferences</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="activity">Activity</TabsTrigger>
        </TabsList>

        <TabsContent value="profile" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800 flex items-center gap-2">
                  <User className="w-5 h-5" />
                  Personal Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="firstName">First Name</Label>
                    <Input
                      id="firstName"
                      value={profile.firstName}
                      onChange={(e) => handleProfileChange("firstName", e.target.value)}
                      className="bg-stone-50 border-stone-300"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lastName">Last Name</Label>
                    <Input
                      id="lastName"
                      value={profile.lastName}
                      onChange={(e) => handleProfileChange("lastName", e.target.value)}
                      className="bg-stone-50 border-stone-300"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email Address</Label>
                  <Input
                    id="email"
                    type="email"
                    value={profile.email}
                    onChange={(e) => handleProfileChange("email", e.target.value)}
                    className="bg-stone-50 border-stone-300"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number</Label>
                  <Input
                    id="phone"
                    value={profile.phone}
                    onChange={(e) => handleProfileChange("phone", e.target.value)}
                    className="bg-stone-50 border-stone-300"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="title">Title</Label>
                  <Input
                    id="title"
                    value={profile.title}
                    onChange={(e) => handleProfileChange("title", e.target.value)}
                    className="bg-stone-50 border-stone-300"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="company">Company</Label>
                  <Input
                    id="company"
                    value={profile.company}
                    onChange={(e) => handleProfileChange("company", e.target.value)}
                    className="bg-stone-50 border-stone-300"
                  />
                </div>

                <Button className="w-full bg-amber-700 hover:bg-amber-800 text-white">
                  <Save className="w-4 h-4 mr-2" />
                  Save Changes
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800">Profile Picture</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-col items-center space-y-4">
                  <div className="w-32 h-32 bg-slate-200 rounded-full flex items-center justify-center">
                    <User className="w-16 h-16 text-slate-400" />
                  </div>
                  <div className="text-center">
                    <Button variant="outline" className="mb-2 bg-transparent">
                      Upload New Photo
                    </Button>
                    <p className="text-sm text-slate-500">JPG, PNG or GIF. Max size 2MB.</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-6">
          <Card className="bg-white border-stone-200">
            <CardHeader>
              <CardTitle className="text-xl font-serif text-slate-800 flex items-center gap-2">
                <Bell className="w-5 h-5" />
                Notification Preferences
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h4 className="font-medium text-slate-700">Communication Channels</h4>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Mail className="w-4 h-4 text-slate-500" />
                      <div>
                        <p className="font-medium text-slate-800">Email Notifications</p>
                        <p className="text-sm text-slate-500">Receive notifications via email</p>
                      </div>
                    </div>
                    <Switch
                      checked={notifications.email}
                      onCheckedChange={(value) => handleNotificationChange("email", value)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Bell className="w-4 h-4 text-slate-500" />
                      <div>
                        <p className="font-medium text-slate-800">Push Notifications</p>
                        <p className="text-sm text-slate-500">Browser push notifications</p>
                      </div>
                    </div>
                    <Switch
                      checked={notifications.push}
                      onCheckedChange={(value) => handleNotificationChange("push", value)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Phone className="w-4 h-4 text-slate-500" />
                      <div>
                        <p className="font-medium text-slate-800">SMS Notifications</p>
                        <p className="text-sm text-slate-500">Text message alerts</p>
                      </div>
                    </div>
                    <Switch
                      checked={notifications.sms}
                      onCheckedChange={(value) => handleNotificationChange("sms", value)}
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-medium text-slate-700">Report Notifications</h4>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-slate-800">Weekly Reports</p>
                      <p className="text-sm text-slate-500">Weekly portfolio summaries</p>
                    </div>
                    <Switch
                      checked={notifications.weeklyReport}
                      onCheckedChange={(value) => handleNotificationChange("weeklyReport", value)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-slate-800">Monthly Reports</p>
                      <p className="text-sm text-slate-500">Monthly performance reports</p>
                    </div>
                    <Switch
                      checked={notifications.monthlyReport}
                      onCheckedChange={(value) => handleNotificationChange("monthlyReport", value)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-slate-800">Portfolio Alerts</p>
                      <p className="text-sm text-slate-500">Important portfolio changes</p>
                    </div>
                    <Switch
                      checked={notifications.alerts}
                      onCheckedChange={(value) => handleNotificationChange("alerts", value)}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preferences" className="space-y-6">
          <Card className="bg-white border-stone-200">
            <CardHeader>
              <CardTitle className="text-xl font-serif text-slate-800 flex items-center gap-2">
                <Palette className="w-5 h-5" />
                Display & Data Preferences
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="theme">Theme</Label>
                    <Select value={preferences.theme} onValueChange={(value) => handlePreferenceChange("theme", value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="light">Light</SelectItem>
                        <SelectItem value="dark">Dark</SelectItem>
                        <SelectItem value="auto">Auto</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="currency">Default Currency</Label>
                    <Select
                      value={preferences.currency}
                      onValueChange={(value) => handlePreferenceChange("currency", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="USD">USD ($)</SelectItem>
                        <SelectItem value="EUR">EUR (€)</SelectItem>
                        <SelectItem value="GBP">GBP (£)</SelectItem>
                        <SelectItem value="JPY">JPY (¥)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="dateFormat">Date Format</Label>
                    <Select
                      value={preferences.dateFormat}
                      onValueChange={(value) => handlePreferenceChange("dateFormat", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="MM/DD/YYYY">MM/DD/YYYY</SelectItem>
                        <SelectItem value="DD/MM/YYYY">DD/MM/YYYY</SelectItem>
                        <SelectItem value="YYYY-MM-DD">YYYY-MM-DD</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="timezone">Timezone</Label>
                    <Select
                      value={preferences.timezone}
                      onValueChange={(value) => handlePreferenceChange("timezone", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="EST">Eastern (EST)</SelectItem>
                        <SelectItem value="CST">Central (CST)</SelectItem>
                        <SelectItem value="MST">Mountain (MST)</SelectItem>
                        <SelectItem value="PST">Pacific (PST)</SelectItem>
                        <SelectItem value="UTC">UTC</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="refreshInterval">Data Refresh Interval</Label>
                    <Select
                      value={preferences.refreshInterval}
                      onValueChange={(value) => handlePreferenceChange("refreshInterval", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="15">15 seconds</SelectItem>
                        <SelectItem value="30">30 seconds</SelectItem>
                        <SelectItem value="60">1 minute</SelectItem>
                        <SelectItem value="300">5 minutes</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <Button className="bg-amber-700 hover:bg-amber-800 text-white">
                <Save className="w-4 h-4 mr-2" />
                Save Preferences
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800 flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  Security Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-slate-700 mb-2">Change Password</h4>
                    <div className="space-y-3">
                      <Input type="password" placeholder="Current password" className="bg-stone-50 border-stone-300" />
                      <Input type="password" placeholder="New password" className="bg-stone-50 border-stone-300" />
                      <Input
                        type="password"
                        placeholder="Confirm new password"
                        className="bg-stone-50 border-stone-300"
                      />
                      <Button variant="outline" className="w-full bg-transparent">
                        <Key className="w-4 h-4 mr-2" />
                        Update Password
                      </Button>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-slate-700 mb-2">Two-Factor Authentication</h4>
                    <div className="flex items-center justify-between p-3 border border-stone-200 rounded-lg">
                      <div>
                        <p className="font-medium text-slate-800">2FA Status</p>
                        <p className="text-sm text-slate-500">
                          <Badge className="bg-green-100 text-green-800">Enabled</Badge>
                        </p>
                      </div>
                      <Button variant="outline" size="sm">
                        Configure
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white border-stone-200">
              <CardHeader>
                <CardTitle className="text-xl font-serif text-slate-800">Active Sessions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="p-3 border border-stone-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-slate-800">Current Session</p>
                        <p className="text-sm text-slate-500">Chrome on macOS</p>
                        <p className="text-xs text-slate-400">192.168.1.100</p>
                      </div>
                      <Badge className="bg-green-100 text-green-800">Active</Badge>
                    </div>
                  </div>
                  <div className="p-3 border border-stone-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-slate-800">Mobile App</p>
                        <p className="text-sm text-slate-500">iOS Safari</p>
                        <p className="text-xs text-slate-400">Last seen 2 hours ago</p>
                      </div>
                      <Button variant="outline" size="sm">
                        Revoke
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="activity" className="space-y-6">
          <Card className="bg-white border-stone-200">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl font-serif text-slate-800 flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  Activity Log
                </CardTitle>
                <Button variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-stone-200">
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Action</th>
                      <th className="text-left py-3 px-4 font-medium text-slate-700">Timestamp</th>
                      <th className="text-left py-3 px-4 font-medium text-slate-700">IP Address</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activityLog.map((activity) => (
                      <tr key={activity.id} className="border-b border-stone-100 hover:bg-stone-50 transition-colors">
                        <td className="py-3 px-4 font-medium text-slate-800">{activity.action}</td>
                        <td className="py-3 px-4 text-slate-600">{new Date(activity.timestamp).toLocaleString()}</td>
                        <td className="py-3 px-4 text-slate-600">{activity.ip}</td>
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
