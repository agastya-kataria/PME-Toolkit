"use client"

import type React from "react"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface Investment {
  name: string
  sector: string
  date: string
  amount: number
  stage: string
  status: "Active" | "Pending" | "Closed"
}

interface InvestmentModalProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (investment: Investment) => void
}

export function InvestmentModal({ isOpen, onClose, onSubmit }: InvestmentModalProps) {
  const [formData, setFormData] = useState<Investment>({
    name: "",
    sector: "",
    date: "",
    amount: 0,
    stage: "",
    status: "Active",
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
    setFormData({
      name: "",
      sector: "",
      date: "",
      amount: 0,
      stage: "",
      status: "Active",
    })
  }

  const handleChange = (field: keyof Investment, value: string | number) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-slate-800 border-amber-600/30 text-slate-100 max-w-md">
        <DialogHeader>
          <DialogTitle className="text-amber-400 text-xl">Add New Investment</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name" className="text-slate-300">
              Investment Name
            </Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => handleChange("name", e.target.value)}
              className="bg-slate-700 border-amber-600/30 text-slate-100"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="sector" className="text-slate-300">
              Sector
            </Label>
            <Select value={formData.sector} onValueChange={(value) => handleChange("sector", value)}>
              <SelectTrigger className="bg-slate-700 border-amber-600/30 text-slate-100">
                <SelectValue placeholder="Select sector" />
              </SelectTrigger>
              <SelectContent className="bg-slate-700 border-amber-600/30">
                <SelectItem value="Private Equity">Private Equity</SelectItem>
                <SelectItem value="Real Estate">Real Estate</SelectItem>
                <SelectItem value="Venture Capital">Venture Capital</SelectItem>
                <SelectItem value="Private Debt">Private Debt</SelectItem>
                <SelectItem value="Infrastructure">Infrastructure</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="date" className="text-slate-300">
              Date
            </Label>
            <Input
              id="date"
              type="date"
              value={formData.date}
              onChange={(e) => handleChange("date", e.target.value)}
              className="bg-slate-700 border-amber-600/30 text-slate-100"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="amount" className="text-slate-300">
              Amount ($)
            </Label>
            <Input
              id="amount"
              type="number"
              value={formData.amount || ""}
              onChange={(e) => handleChange("amount", Number.parseFloat(e.target.value) || 0)}
              className="bg-slate-700 border-amber-600/30 text-slate-100"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="stage" className="text-slate-300">
              Stage
            </Label>
            <Input
              id="stage"
              value={formData.stage}
              onChange={(e) => handleChange("stage", e.target.value)}
              className="bg-slate-700 border-amber-600/30 text-slate-100"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="status" className="text-slate-300">
              Status
            </Label>
            <Select
              value={formData.status}
              onValueChange={(value: "Active" | "Pending" | "Closed") => handleChange("status", value)}
            >
              <SelectTrigger className="bg-slate-700 border-amber-600/30 text-slate-100">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-700 border-amber-600/30">
                <SelectItem value="Active">Active</SelectItem>
                <SelectItem value="Pending">Pending</SelectItem>
                <SelectItem value="Closed">Closed</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              className="flex-1 border-amber-600/30 text-slate-300 hover:bg-amber-600/10 bg-transparent"
            >
              Cancel
            </Button>
            <Button type="submit" className="flex-1 bg-amber-600 hover:bg-amber-700 text-slate-900">
              Add Investment
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
