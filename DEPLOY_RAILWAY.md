# 🚀 5-Minute Railway Deployment Guide

Deploy your AI Booth app to Railway.app in under 5 minutes!

## ⚡ Why Railway?
- **Free tier**: $5/month credit (enough for small apps)
- **Auto HTTPS**: Automatic SSL certificates
- **GitHub sync**: Auto-deploy on push
- **Fast deployment**: Usually under 2 minutes
- **Zero config**: Detects Dockerfiles automatically

## 🏃‍♂️ Quick Start (5 Minutes)

### Step 1: Push to GitHub (1 min)
```bash
# If not already done
git add .
git commit -m "Add Railway deployment config"
git push origin main
```

### Step 2: Deploy to Railway (3 mins)

**Option A: Web Interface (Easiest)**
1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Choose "Deploy from GitHub repo"
4. Select your repository
5. Railway will detect both Dockerfiles and create 2 services automatically
6. Click "Deploy" 

**Option B: CLI (Advanced)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Step 3: Configure Environment (1 min)
1. Go to your Railway dashboard
2. Click on **Frontend** service → Variables
3. Add: `VITE_API_BASE_URL` = `https://your-backend-service.railway.app`
4. Click on **Backend** service → Variables  
5. Add: `NODE_ENV` = `production`

### Step 4: Done! 🎉
- Frontend: `https://your-frontend-service.railway.app`
- Backend: `https://your-backend-service.railway.app`

## 🔧 Environment Variables

### Backend Service
```
NODE_ENV=production
PYTHONPATH=/app
DATABASE_URL=sqlite:///./booth_agent.db
```

### Frontend Service  
```
NODE_ENV=production
VITE_API_BASE_URL=https://your-backend-service.railway.app
```

## 📱 Alternative 5-Min Platforms

### 🥈 Render.com
1. Connect GitHub repo
2. Create **Web Service** (Backend): Docker
3. Create **Static Site** (Frontend): `npm run build`, publish `dist`
4. Set environment variables
⏱️ **Time**: ~5-7 minutes

### 🥉 Fly.io  
```bash
# Install CLI
curl -L https://fly.io/install.sh | sh

# Deploy backend
cd ai_booth_agent && fly launch --name dc-backend

# Deploy frontend  
cd ../booth-approval-ai-hub && fly launch --name dc-frontend
```
⏱️ **Time**: ~6-8 minutes

### 🥉 Vercel + Railway
- **Frontend**: Deploy to Vercel (2 mins)
- **Backend**: Deploy to Railway (3 mins)
⏱️ **Time**: ~5 minutes

## 🚨 Troubleshooting

**Port Issues?**
- Railway assigns `$PORT` automatically
- Our Dockerfiles handle this correctly

**Build Failing?**
- Check logs in Railway dashboard
- Ensure all dependencies in `requirements.txt` and `package.json`

**CORS Errors?**
- Update `main.py` CORS origins to include Railway domains
- Add your Railway URLs to allowed origins

**Environment Variables?**
- Make sure `VITE_API_BASE_URL` points to backend Railway URL
- Check that both services are running

## 🎯 Pro Tips

1. **Custom Domains**: Railway supports custom domains on paid plans
2. **Monitoring**: Built-in metrics and logs in dashboard  
3. **Auto-Deploy**: Push to main branch = automatic deployment
4. **Scaling**: Easy horizontal scaling with Railway Pro
5. **Database**: Consider Railway's PostgreSQL for production

## 🔄 Update Deployment
```bash
# Just push to GitHub
git add .
git commit -m "Update app"
git push origin main
# Railway auto-deploys! 🚀
```

---

**🎉 Your app will be live in under 5 minutes!**  
Railway is perfect for rapid prototyping and production deployments.
