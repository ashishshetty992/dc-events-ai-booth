import React, { useEffect, useState } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, Legend, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Loader2, AlertTriangle, Bot } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { API_ENDPOINTS, getApiUrl } from '@/lib/api';
import Sidebar from './Sidebar';

const tabConfigs = [
  { key: 'resource', label: 'Resource Calendar' },
  { key: 'competitors', label: 'Saudi Competitors' },
  { key: 'vendor', label: 'Vendor Outcomes' },
  { key: 'audience', label: 'Audience Profiles' },
  { key: 'trends', label: 'Global Trends' },
  { key: 'events', label: 'Saudi Events' },
  { key: 'market', label: 'Market Intelligence' },
  { key: 'regulatory', label: 'Regulatory Environment' },
];

const endpointMap: Record<string, string> = {
  resource: '/api/analytics/resource-calendar',
  competitors: '/api/analytics/competitors',
  vendor: '/api/analytics/vendor-outcomes',
  audience: '/api/analytics/audience-profiles',
  trends: '/api/analytics/global-trends',
  events: '/api/analytics/event-schedule',
  market: '/api/analytics/market-intelligence',
  regulatory: '/api/analytics/regulatory-environment',
};

const COLORS = ['#6366f1', '#3b82f6', '#06b6d4', '#f59e42', '#f43f5e', '#a21caf', '#16a34a'];

const AnalyticsDashboard: React.FC = () => {
  const [tab, setTab] = useState('resource');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [aiInsights, setAiInsights] = useState<string[]>([]);

  useEffect(() => {
    setLoading(true);
    fetch(getApiUrl(endpointMap[tab]))
      .then(res => res.json())
      .then((d) => {
        setData(d);
        setAiInsights(d.ai_insights || []);
        setLoading(false);
      });
  }, [tab]);

  // --- Chart renderers for each tab ---
  function renderChart() {
    if (!data || !Array.isArray(data.data) || data.data.length === 0) {
      return <div className="text-center text-gray-500 py-12">No data available.</div>;
    }
    switch (tab) {
      case 'resource': {
        // Radar: city-wise avg crews, Area: total crews over time
        const cityGroups = Array.from(new Set(data.data.map((d: any) => d.city)));
        const radarData = cityGroups.map(city => ({
          city,
          avg_crews: data.data.filter((d: any) => d.city === city).reduce((sum: number, d: any) => sum + d.available_crews, 0) / data.data.filter((d: any) => d.city === city).length
        }));
        const dateGroups = Array.from(new Set(data.data.map((d: any) => d.date)));
        const areaData = dateGroups.map(date => {
          const totalCrews = data.data.filter((d: any) => d.date === date).reduce((sum: number, d: any) => sum + d.available_crews, 0);
          return { date, totalCrews };
        });
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Crew Availability by City (Radar)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart cx="50%" cy="50%" outerRadius={110} data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="city" tick={{ fill: '#6366f1', fontWeight: 600 }} />
                  <PolarRadiusAxis angle={30} domain={[0, Math.max(...radarData.map((d: any) => d.avg_crews)) * 1.2]} tick={false} />
                  <Radar name="Avg Crews" dataKey="avg_crews" stroke="#3b82f6" fill="#6366f1" fillOpacity={0.5} />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Total Crew Availability Over Time (Area)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={areaData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorCrews" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.2}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" minTickGap={10} tick={{ fontSize: 12 }}/>
                  <YAxis allowDecimals={false} />
                  <CartesianGrid strokeDasharray="3 3" />
                  <Tooltip />
                  <Area type="monotone" dataKey="totalCrews" stroke="#6366f1" fillOpacity={1} fill="url(#colorCrews)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      case 'competitors': {
        // Bar: avg footfall by company, Pie: tier distribution
        const barData = data.data.slice(0, 15).map((d: any) => ({ company: d.company, avg_footfall: d.avg_footfall }));
        const tierCounts: Record<string, number> = {};
        data.data.forEach((d: any) => { tierCounts[d.tier] = (tierCounts[d.tier] || 0) + 1; });
        const pieData = Object.entries(tierCounts).map(([tier, count]) => ({ name: tier, value: count }));
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Avg Footfall (Top 15 Companies)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData}>
                  <XAxis dataKey="company" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="avg_footfall" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Tier Distribution</h3>
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                    {pieData.map((entry: any, idx: number) => (
                      <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      case 'vendor': {
        // Bar: ROI by company (top 15), Pie: satisfaction distribution
        const barData = data.data.slice(0, 15).map((d: any) => ({ company: d.company, roi: d.roi_score }));
        const satisfactionCounts: Record<string, number> = {};
        data.data.forEach((d: any) => { satisfactionCounts[d.satisfaction] = (satisfactionCounts[d.satisfaction] || 0) + 1; });
        const pieData = Object.entries(satisfactionCounts).map(([s, count]) => ({ name: `Satisfaction ${s}`, value: count }));
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">ROI by Company (Top 15)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData}>
                  <XAxis dataKey="company" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="roi" fill="#06b6d4" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Satisfaction Distribution</h3>
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                    {pieData.map((entry: any, idx: number) => (
                      <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      case 'audience': {
        // Bar: match score by event, Pie: audience type frequency
        const barData = data.data.map((d: any) => ({ event: d.event_name, match: d.match_score }));
        const audienceTypeCounts: Record<string, number> = {};
        data.data.forEach((d: any) => {
          (d.audience_types || []).forEach((type: string) => {
            audienceTypeCounts[type] = (audienceTypeCounts[type] || 0) + 1;
          });
        });
        const pieData = Object.entries(audienceTypeCounts).map(([type, count]) => ({ name: type, value: count }));
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Audience Match Score by Event</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData}>
                  <XAxis dataKey="event" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="match" fill="#a21caf" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Audience Type Frequency</h3>
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                    {pieData.map((entry: any, idx: number) => (
                      <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      case 'trends': {
        // Bar: avg budget by industry, Pie: top country frequency
        const barData = data.data.map((d: any) => ({ industry: d.industry, avg_budget: d.avg_budget }));
        const countryCounts: Record<string, number> = {};
        data.data.forEach((d: any) => {
          (d.top_countries || []).forEach((country: string) => {
            countryCounts[country] = (countryCounts[country] || 0) + 1;
          });
        });
        const pieData = Object.entries(countryCounts).map(([country, count]) => ({ name: country, value: count }));
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Avg Budget by Industry</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData}>
                  <XAxis dataKey="industry" tick={{ fontSize: 12 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="avg_budget" fill="#f59e42" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Top Country Frequency</h3>
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                    {pieData.map((entry: any, idx: number) => (
                      <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      case 'events': {
        // Bar: expected footfall by event, Pie: tier distribution
        const barData = data.data.map((d: any) => ({ event: d.event_name, footfall: d.expected_footfall, cost: d.cost_per_sqm }));
        const tierCounts: Record<string, number> = {};
        data.data.forEach((d: any) => { tierCounts[d.tier] = (tierCounts[d.tier] || 0) + 1; });
        const pieData = Object.entries(tierCounts).map(([tier, count]) => ({ name: tier, value: count }));
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Saudi Events: Footfall vs Cost</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData}>
                  <XAxis dataKey="event" tick={{ fontSize: 8 }} interval={0} angle={-45} textAnchor="end" />
                  <YAxis yAxisId="left" allowDecimals={false} />
                  <YAxis yAxisId="right" orientation="right" allowDecimals={false} />
                  <Tooltip />
                  <Bar yAxisId="left" dataKey="footfall" fill="#16a34a" name="Expected Footfall" />
                  <Bar yAxisId="right" dataKey="cost" fill="#f59e42" name="Cost per SqM" />
                  <Legend />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100 flex flex-col items-center">
              <h3 className="font-semibold text-blue-700 mb-2">Event Tier Distribution</h3>
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                    {pieData.map((entry: any, idx: number) => (
                      <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      case 'market': {
        // Market intelligence visualization
        if (!data.data || !data.data.industry_growth_forecasts) {
          return <div className="text-center text-gray-500 py-12">Loading market intelligence...</div>;
        }
        const industryData = Object.entries(data.data.industry_growth_forecasts).map(([industry, details]: [string, any]) => ({
          industry,
          growth_2025: details.growth_rate_2025,
          market_size: details.market_size_usd / 1e9,
          govt_spending: details.government_spending_usd / 1e9
        }));
        const regionalData = Object.entries(data.data.regional_insights || {}).map(([region, details]: [string, any]) => ({
          region,
          gdp_contribution: details.gdp_contribution * 100,
          business_density: details.business_density * 100,
          innovation_hubs: details.innovation_hubs
        }));
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100">
              <h3 className="font-semibold text-blue-700 mb-2">Industry Growth Forecasts 2025</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={industryData}>
                  <XAxis dataKey="industry" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="growth_2025" fill="#6366f1" name="Growth Rate %" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100">
              <h3 className="font-semibold text-blue-700 mb-2">Regional Business Distribution</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={regionalData}>
                  <XAxis dataKey="region" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="gdp_contribution" fill="#3b82f6" name="GDP Contribution %" />
                  <Bar dataKey="business_density" fill="#06b6d4" name="Business Density %" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      case 'regulatory': {
        // Regulatory environment visualization
        if (!data.data || !data.data.investment_requirements) {
          return <div className="text-center text-gray-500 py-12">Loading regulatory data...</div>;
        }
        const partnershipData = Object.entries(data.data.investment_requirements?.local_partner_required || {}).map(([industry, required]: [string, any]) => ({
          industry,
          partnership_required: required ? 1 : 0,
          govt_approval: data.data.investment_requirements?.government_approval_required?.[industry] ? 1 : 0
        }));
        const saudizationData = Object.entries(data.data.compliance_requirements?.saudization_quotas || {}).map(([industry, quota]: [string, any]) => ({
          industry,
          quota: quota * 100
        }));
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100">
              <h3 className="font-semibold text-blue-700 mb-2">Regulatory Requirements by Industry</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={partnershipData}>
                  <XAxis dataKey="industry" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="partnership_required" fill="#f43f5e" name="Partnership Required" />
                  <Bar dataKey="govt_approval" fill="#a21caf" name="Govt Approval Required" />
                  <Legend />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white rounded-xl shadow-lg p-6 border border-blue-100">
              <h3 className="font-semibold text-blue-700 mb-2">Saudization Quotas by Industry</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={saudizationData}>
                  <XAxis dataKey="industry" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="quota" fill="#16a34a" name="Saudization %" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      }
      default:
        return null;
    }
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
      <Sidebar />
      <div className="flex-1 ml-64 overflow-y-auto">
        <div className="p-8 space-y-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center">
            <Bot className="w-7 h-7 text-blue-600 mr-2" /> 🇸🇦 Saudi Arabia Market Intelligence Dashboard
          </h2>
          <p className="text-gray-600 mb-4">Comprehensive analytics powered by machine learning and Vision 2030 insights</p>
      <Tabs value={tab} onValueChange={setTab} className="w-full">
        <TabsList className="mb-6">
          {tabConfigs.map(t => (
            <TabsTrigger key={t.key} value={t.key}>{t.label}</TabsTrigger>
          ))}
        </TabsList>
        {tabConfigs.map(t => (
          <TabsContent key={t.key} value={t.key} className="w-full">
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
              </div>
            ) : (
              <>
                {renderChart()}
                {/* AI Insights Section */}
                <div className="mt-8 bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl shadow p-6 border border-blue-200">
                  <h3 className="font-semibold text-blue-800 mb-2 flex items-center"><AlertTriangle className="w-5 h-5 text-blue-500 mr-2" /> AI Insights</h3>
                  <ul className="list-disc pl-6 text-blue-900 space-y-1">
                    {aiInsights.map((insight, idx) => (
                      <li key={idx}>{insight}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </TabsContent>
        ))}
      </Tabs>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard; 