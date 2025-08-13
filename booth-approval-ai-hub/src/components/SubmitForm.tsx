import { zodResolver } from "@hookform/resolvers/zod";
import { useFieldArray, useForm } from "react-hook-form";
import * as z from "zod";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { API_ENDPOINTS } from "@/lib/api";
import Sidebar from './Sidebar';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/use-toast";
import {
  Building,
  Calendar,
  DollarSign,
  MapPin,
  PlusCircle,
  Trash2,
  Send,
  CheckCircle,
  Layers,
  Users,
  Info,
} from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

const industries = [
  "Technology & IT",
  "Healthcare & Medical",
  "Energy & Utilities", 
  "Financial Services",
  "Manufacturing",
  "Construction & Real Estate",
  "Retail & E-commerce",
  "Automotive",
  "Food & Beverage",
  "Education",
  "Tourism & Hospitality",
  "Other",
];

const SAUDI_EVENTS = [
  "LEAP Technology Conference",
  "Saudi Green Initiative Summit", 
  "NEOM Tech & Digital Summit",
  "Future Investment Initiative",
  "Saudi International Healthcare Exhibition",
  "Big 5 Saudi Construction",
  "Saudi Manufacturing Summit",
  "Riyadh Energy Forum",
  "Global Blockchain Summit Saudi",
  "Saudi Fintech Summit",
  "Custom Event"
];

const SAUDI_LOCATIONS = [
  "Riyadh",
  "Jeddah", 
  "Dammam",
  "NEOM",
  "Al Khobar",
  "Mecca",
  "Medina",
  "Other"
];

const submitFormSchema = z.object({
  company_name: z.string().min(2, {
    message: "Company name must be at least 2 characters.",
  }),
  booth_size_sqm: z.coerce.number().min(9).max(200, {
    message: "Booth size must be between 9 and 200 square meters.",
  }),
  budget_sar: z.coerce.number().min(30000).max(500000, {
    message: "Budget must be between SAR 30,000 and SAR 500,000.",
  }),
  event_name: z.string().min(2, {
    message: "Event name must be at least 2 characters.",
  }),
  event_date: z.string().min(1, { message: "Event date is required" }),
  industry: z.string({
    required_error: "Please select an industry.",
  }),
  location: z.string().min(2, {
    message: "Location is required.",
  }),
  product_description: z.string().min(10, {
    message: "Product description must be at least 10 characters.",
  }),
  target_audience: z.string().min(10, {
    message: "Target audience must be at least 10 characters.",
  }),
  objectives: z.string().min(10, {
    message: "Exhibition objectives must be at least 10 characters.",
  }),
  special_requirements: z.string().optional(),
  preferred_style: z.enum(['modern', 'luxury', 'tech', 'eco', 'traditional'], {
    required_error: "Please select a preferred style.",
  }),
  past_events: z
    .array(
      z.object({
        value: z.string().min(2, { message: "Event name cannot be empty." }),
      })
    )
    .optional(),
});

type SubmitFormValues = z.infer<typeof submitFormSchema>;

const defaultValues: Partial<SubmitFormValues> = {
  company_name: "",
  event_name: "",
  location: "Riyadh",
  booth_size_sqm: 36,
  budget_sar: 75000,
  product_description: "",
  target_audience: "",
  objectives: "",
  special_requirements: "",
  preferred_style: 'modern' as const,
  past_events: [{ value: "" }],
};

// Simple SVG logo placeholder
const BoothLogo = () => (
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="4" y="12" width="40" height="28" rx="6" fill="#3b82f6" />
    <rect x="12" y="20" width="24" height="12" rx="3" fill="#fff" />
    <rect x="20" y="26" width="8" height="6" rx="2" fill="#3b82f6" />
  </svg>
);

export default function SubmitForm() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const navigate = useNavigate();

  const form = useForm<SubmitFormValues>({
    resolver: zodResolver(submitFormSchema),
    defaultValues,
    mode: "onChange",
  });

  const { fields, append, remove } = useFieldArray({
    name: "past_events",
    control: form.control,
  });

  // Progress bar logic: count completed sections
  const values = form.watch();
  const sectionChecks = [
    !!values.company_name && !!values.industry && !!values.product_description,
    !!values.event_name && !!values.event_date && !!values.location,
    !!values.booth_size_sqm && !!values.budget_sar && !!values.preferred_style,
    !!values.target_audience && !!values.objectives,
    !!values.past_events && values.past_events.length > 0 && values.past_events.every((e: any) => e.value && e.value.length > 1),
  ];
  const progress = (sectionChecks.filter(Boolean).length / sectionChecks.length) * 100;

  async function onSubmit(data: SubmitFormValues) {
    setIsSubmitting(true);
    
    // Transform data for backend compatibility
    const submissionData = {
      company_name: data.company_name,
      booth_size_sqm: data.booth_size_sqm,
      budget_inr: data.budget_sar, // Backend expects budget_inr but we'll send SAR
      event_name: data.event_name,
      event_date: data.event_date,
      industry: data.industry,
      location: data.location,
      past_events: data.past_events?.map((p) => p.value),
      // Enhanced fields for better AI design generation
      product_description: data.product_description,
      target_audience: data.target_audience,
      objectives: data.objectives,
      special_requirements: data.special_requirements,
      preferred_style: data.preferred_style,
    };

    try {
      const response = await fetch(
        API_ENDPOINTS.SUBMIT_BOOTH_REQUEST,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(submissionData),
        }
      );

      if (response.ok) {
        // Generate unique session ID for chat/design
        const sessionId = `session_${Date.now()}`;
        
        // Navigate to unified AI Assistant with enhanced data
        navigate(`/chat/${sessionId}`, { 
          state: { 
            boothRequest: submissionData,
            shouldGeneratePreset: true 
          } 
        });
      } else {
        throw new Error("Failed to submit form");
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      toast({
        variant: "destructive",
        title: "Submission Failed",
        description: "Failed to submit form. Please try again.",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  if (submitSuccess) {
    return (
      <div className="h-full flex items-center justify-center bg-white">
        <div className="text-center">
          <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Redirecting to AI Booth Designer...
          </h2>
          <p className="text-gray-600 mb-6">
            Your request has been submitted! The AI is now generating your custom booth design.
          </p>
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span className="text-blue-600 font-medium">Creating your booth...</span>
          </div>
          <Button onClick={() => setSubmitSuccess(false)} variant="outline" className="px-6 py-2">
            Submit Another Request
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
      <Sidebar />
      <div className="flex-1 ml-64 overflow-y-auto bg-gradient-to-br from-blue-100 via-blue-50 to-white min-h-screen">
        <div className="max-w-3xl mx-auto p-8">
        {/* Logo and Progress Bar */}
        <div className="flex flex-col items-center mb-6">
          <BoothLogo />
          <h2 className="text-2xl font-bold text-gray-900 mt-2 mb-1">
            Submit Booth Request
          </h2>
          <p className="text-gray-600 mb-4 text-center max-w-lg">
            Fill out the form below to submit your booth approval request. Our AI will analyze your requirements and provide recommendations.
          </p>
          {/* Progress Bar */}
          <div className="w-full max-w-lg h-3 bg-blue-200 rounded-full overflow-hidden shadow mb-2">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="w-full max-w-lg flex justify-between text-xs text-blue-700 font-medium mb-2">
            <span>Company</span>
            <span>Event</span>
            <span>Booth</span>
            <span>Strategy</span>
            <span>History</span>
          </div>
        </div>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            {/* Responsive grid for all sections */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Company Information */}
              <div className="bg-white/90 p-6 rounded-2xl shadow-lg border border-blue-100 flex flex-col gap-4 col-span-1 md:col-span-2">
                <div className="flex items-center mb-2">
                  <Building className="w-5 h-5 mr-2 text-blue-600" />
                  <h3 className="text-lg font-semibold text-blue-900">Company Information</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="company_name"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Company Name *</FormLabel>
                        <FormControl>
                          <Input placeholder="Enter company name" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="industry"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Industry *</FormLabel>
                        <Select onValueChange={field.onChange} value={field.value}>
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select an industry" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {industries.map((ind) => (
                              <SelectItem key={ind} value={ind}>
                                {ind}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
                <FormField
                  control={form.control}
                  name="product_description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Product/Service Description *</FormLabel>
                      <FormControl>
                        <Textarea 
                          placeholder="Describe your products or services that will be showcased at the booth"
                          className="min-h-[80px]"
                          {...field} 
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
              {/* Event Details */}
              <div className="bg-white/90 p-6 rounded-2xl shadow-lg border border-blue-100 flex flex-col gap-4 col-span-1 md:col-span-2">
                <div className="flex items-center mb-2">
                  <Calendar className="w-5 h-5 mr-2 text-blue-600" />
                  <h3 className="text-lg font-semibold text-blue-900">Saudi Exhibition Details</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <FormField
                    control={form.control}
                    name="event_name"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Saudi Event *</FormLabel>
                        <Select onValueChange={field.onChange} value={field.value}>
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select Saudi event" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {SAUDI_EVENTS.map((event) => (
                              <SelectItem key={event} value={event}>
                                {event}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="event_date"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Event Date *</FormLabel>
                        <FormControl>
                          <Input type="date" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="location"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Saudi Location *</FormLabel>
                        <Select onValueChange={field.onChange} value={field.value}>
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select location" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {SAUDI_LOCATIONS.map((location) => (
                              <SelectItem key={location} value={location}>
                                {location}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </div>
              {/* Booth Specifications */}
              <div className="bg-white/90 p-6 rounded-2xl shadow-lg border border-blue-100 flex flex-col gap-4">
                <div className="flex items-center mb-2">
                  <Layers className="w-5 h-5 mr-2 text-blue-600" />
                  <h3 className="text-lg font-semibold text-blue-900">Booth Specifications</h3>
                </div>
                <div className="space-y-6">
                  <FormField
                    control={form.control}
                    name="booth_size_sqm"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Booth Size (Square Meters) *</FormLabel>
                        <FormControl>
                          <div className="px-3">
                            <input
                              type="range"
                              min="9"
                              max="200"
                              value={field.value || 36}
                              onChange={(e) => field.onChange(parseInt(e.target.value))}
                              className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-sm text-gray-500 mt-1">
                              <span>9 sqm</span>
                              <span className="font-medium text-blue-600">{field.value || 36} sqm</span>
                              <span>200 sqm</span>
                            </div>
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="budget_sar"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Budget (Saudi Riyals) *</FormLabel>
                        <FormControl>
                          <div className="px-3">
                            <input
                              type="range"
                              min="30000"
                              max="500000"
                              step="5000"
                              value={field.value || 75000}
                              onChange={(e) => field.onChange(parseInt(e.target.value))}
                              className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-sm text-gray-500 mt-1">
                              <span>SAR 30K</span>
                              <span className="font-medium text-green-600">SAR {(field.value || 75000).toLocaleString()}</span>
                              <span>SAR 500K</span>
                            </div>
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="preferred_style"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Preferred Style *</FormLabel>
                        <FormControl>
                          <div className="grid grid-cols-2 gap-3">
                            {[
                              { value: 'modern', label: '🏢 Modern', desc: 'Clean & Contemporary' },
                              { value: 'luxury', label: '💎 Luxury', desc: 'Premium & Elegant' },
                              { value: 'tech', label: '🚀 Tech', desc: 'Futuristic & Digital' },
                              { value: 'eco', label: '🌱 Eco', desc: 'Sustainable & Natural' }
                            ].map(style => (
                              <button
                                key={style.value}
                                type="button"
                                onClick={() => field.onChange(style.value)}
                                className={`p-3 rounded-lg border text-left transition-all ${
                                  field.value === style.value
                                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                                    : 'border-gray-300 hover:border-gray-400'
                                }`}
                              >
                                <div className="font-medium">{style.label}</div>
                                <div className="text-xs text-gray-500">{style.desc}</div>
                              </button>
                            ))}
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </div>
              
              {/* Marketing Strategy */}
              <div className="bg-white/90 p-6 rounded-2xl shadow-lg border border-blue-100 flex flex-col gap-4">
                <div className="flex items-center mb-2">
                  <Users className="w-5 h-5 mr-2 text-blue-600" />
                  <h3 className="text-lg font-semibold text-blue-900">Marketing Strategy</h3>
                </div>
                <div className="space-y-4">
                  <FormField
                    control={form.control}
                    name="target_audience"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Target Audience *</FormLabel>
                        <FormControl>
                          <Textarea 
                            placeholder="Who are you trying to reach? (e.g., IT managers, healthcare professionals, Saudi government officials)"
                            className="min-h-[60px]"
                            {...field} 
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="objectives"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Exhibition Objectives *</FormLabel>
                        <FormControl>
                          <Textarea 
                            placeholder="What do you want to achieve? (e.g., generate leads, brand awareness, Vision 2030 partnerships)"
                            className="min-h-[60px]"
                            {...field} 
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="special_requirements"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Special Requirements (Optional)</FormLabel>
                        <FormControl>
                          <Textarea 
                            placeholder="Any special needs? (e.g., prayer area, halal catering, Arabic signage, VIP reception)"
                            className="min-h-[60px]"
                            {...field} 
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </div>
              {/* Past Event Participation */}
              <div className="bg-white/90 p-6 rounded-2xl shadow-lg border border-blue-100 flex flex-col gap-4 col-span-1 md:col-span-2">
                <div className="flex items-center mb-2">
                  <Users className="w-5 h-5 mr-2 text-blue-600" />
                  <h3 className="text-lg font-semibold text-blue-900">Past Event Participation</h3>
                </div>
                <div className="space-y-2">
                  {fields.map((fieldItem, idx) => (
                    <div key={fieldItem.id} className="flex gap-2 items-center">
                      <FormField
                        control={form.control}
                        name={`past_events.${idx}.value`}
                        render={({ field }) => (
                          <FormItem className="flex-1">
                            <FormLabel className="sr-only">Past Event Name</FormLabel>
                            <FormControl>
                              <Input placeholder="Enter a past event name" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => remove(idx)}
                        className="text-red-500 hover:bg-red-100"
                        title="Remove"
                        disabled={fields.length === 1}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="mt-2"
                    onClick={() => append({ value: "" })}
                  >
                    <PlusCircle className="w-4 h-4 mr-1" /> Add Past Event
                  </Button>
                  <FormDescription>
                    List some of the past events you have participated in.
                  </FormDescription>
                </div>
              </div>
            </div>
            {/* Submit Button */}
            <Button
              type="submit"
              className="w-full py-4 text-lg mt-4 flex items-center justify-center bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 shadow-xl transform hover:scale-105 transition-all duration-300"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Creating Your Saudi Booth Design...
                </>
              ) : (
                <>
                  🚀 Generate My Saudi Booth Design <Send className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </form>
        </Form>
        </div>
      </div>
    </div>
  );
}
