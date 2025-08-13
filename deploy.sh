#!/bin/bash

echo "🚀 DC Events AI Booth - 5-Minute Railway Deployment"
echo "=================================================="

# Check if git repo is ready
if [[ -z $(git remote -v) ]]; then
    echo "❌ No git remote found. Please push your code to GitHub first."
    exit 1
fi

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

echo "🔐 Please login to Railway..."
railway login

echo "🎯 Initializing Railway project..."
railway init

echo "🏗️  Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "🎉 Next steps:"
echo "1. Go to railway.app dashboard"
echo "2. Find your project and both services (backend & frontend)"
echo "3. Get the backend URL and set it as VITE_API_BASE_URL in frontend service"
echo "4. Your app will be live in ~2-3 minutes!"
echo ""
echo "📱 Access your services:"
echo "   Frontend: https://your-frontend-service.railway.app"
echo "   Backend: https://your-backend-service.railway.app"
