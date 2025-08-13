import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import SessionLogs from "./components/SessionLogs";
import SubmitForm from "./components/SubmitForm";
import MainLayout from "./pages/MainLayout";
import AnalyticsDashboard from "./components/AnalyticsDashboard";
import BoothDesigner from "./components/BoothDesigner";
import Dashboard from "./components/Dashboard";
import ApprovalFlow from "./components/ApprovalFlow";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Main booth request form as the default */}
          <Route path="/" element={<SubmitForm />} />
          
          {/* Dashboard and other pages with sidebar */}
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/logs" element={<SessionLogs />} />
          <Route path="/analytics" element={<AnalyticsDashboard />} />
          
          {/* Unified AI Assistant for chat and design */}
          <Route path="/chat" element={<BoothDesigner />} />
          <Route path="/chat/:session_id" element={<BoothDesigner />} />
          
          {/* Approval Flow */}
          <Route path="/approval/:sessionId" element={<ApprovalFlow />} />
          
          {/* Legacy routes */}
          <Route path="/form" element={<SubmitForm />} />
          <Route path="/design/:session_id" element={<BoothDesigner />} />
          <Route path="/index" element={<Index />} />
          
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
