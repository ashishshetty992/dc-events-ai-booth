# DC Events - AI Booth Approval System

A comprehensive AI-powered booth approval system with React frontend and FastAPI backend.

## 🏗️ Project Structure

```
dc-events/
├── booth-approval-ai-hub/    # React Frontend (Vite + TypeScript)
├── ai_booth_agent/          # FastAPI Backend (Python)
├── api/                     # Vercel API functions
├── vercel.json             # Vercel deployment config
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Features

- **AI-Powered Booth Approval**: Intelligent analysis and recommendations
- **3D Booth Designer**: Interactive booth design with Three.js
- **Real-time Analytics**: Comprehensive dashboard with insights
- **Chat Interface**: AI assistant for booth planning
- **Session Management**: Track approval workflows
- **Multi-environment Support**: Development and production configurations

## 🛠️ Technologies

### Frontend
- React 18 + TypeScript
- Vite for build tooling
- Tailwind CSS + shadcn/ui
- Three.js for 3D visualization
- React Router for navigation
- TanStack Query for state management

### Backend
- FastAPI (Python)
- SQLAlchemy for database ORM
- WebSocket support for real-time features
- AI integration for booth analysis
- Comprehensive analytics endpoints

## 🧑‍💻 Development Setup

### Prerequisites
- Node.js 18+ 
- Python 3.9+
- npm or yarn

### Frontend Setup
```bash
cd booth-approval-ai-hub
npm install
npm run dev
```

### Backend Setup
```bash
cd ai_booth_agent
pip install -r requirements.txt
python start_server.py
```

## 🚀 Deployment

This project is configured for deployment on Vercel with both frontend and backend:

1. **Push to GitHub**
2. **Connect to Vercel**
3. **Deploy automatically**

The `vercel.json` configuration handles:
- Frontend build (Vite)
- Backend serverless functions (Python)
- API routing
- Static file serving

## 📊 API Endpoints

### Core Endpoints
- `POST /submit_booth_request` - Submit booth approval request
- `GET /chat_history` - Retrieve chat sessions
- `POST /api/approval/save` - Save approval decisions

### Analytics Endpoints
- `GET /api/analytics/competitors` - Competitor analysis
- `GET /api/analytics/vendor-outcomes` - Vendor performance
- `GET /api/analytics/market-intelligence` - Market insights
- `GET /api/analytics/resource-calendar` - Resource planning

### Design Endpoints
- `POST /api/booth-design/chat` - AI design assistant
- `POST /api/booth-design/save` - Save booth designs
- `POST /api/finalize-design` - Finalize booth design

## 🌍 Environment Configuration

### Development
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

### Production
- Frontend: Served via Vercel
- Backend: Serverless functions on Vercel
- API: `/api/*` routes

## 📁 Key Files

- `vercel.json` - Deployment configuration
- `api/index.py` - Serverless function entry point
- `booth-approval-ai-hub/src/lib/api.ts` - API configuration
- `ai_booth_agent/main.py` - FastAPI application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.
