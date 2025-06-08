# 🚀 Render Deployment Guide - Legal Intelligence Platform

This guide will walk you through deploying your Legal Intelligence Platform to Render.

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be pushed to https://github.com/erickyegon/legal-doc-analyzer
2. **Render Account**: Sign up at https://render.com
3. **Repository Access**: Ensure your GitHub repository is public or connected to Render

## 🎯 Deployment Options

### Option 1: One-Click Blueprint Deployment (Recommended)

1. **Click the Deploy Button**:
   [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/erickyegon/legal-doc-analyzer)

2. **Configure Services**:
   - The `render.yaml` file will automatically configure both backend and frontend
   - Review the configuration and click "Apply"

### Option 2: Manual Deployment

#### Step 1: Deploy Backend API

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub account if not already connected
   - Select the `legal-doc-analyzer` repository

3. **Configure Backend Service**:
   ```
   Name: legal-intelligence-api
   Region: Oregon (US West)
   Branch: main
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install --upgrade pip && pip install -r requirements-render.txt && python -m spacy download en_core_web_sm
   Start Command: python main.py
   Plan: Starter ($7/month)
   ```

4. **Environment Variables**:
   ```
   ENVIRONMENT=production
   PORT=10000
   EURI_API_KEY=test-key
   SECRET_KEY=[Auto-generated]
   CORS_ORIGINS=*
   PYTHONPATH=/opt/render/project/src
   ```

5. **Advanced Settings**:
   - Health Check Path: `/health`
   - Auto-Deploy: Yes

#### Step 2: Deploy Frontend (Optional)

1. **Create Static Site**:
   - Click "New +" → "Static Site"
   - Select the same repository

2. **Configure Frontend Service**:
   ```
   Name: legal-intelligence-frontend
   Branch: main
   Root Directory: frontend
   Build Command: npm ci && npm run build
   Publish Directory: dist
   ```

3. **Environment Variables**:
   ```
   VITE_API_URL=https://legal-intelligence-api.onrender.com
   VITE_APP_NAME=Legal Intelligence Platform
   VITE_APP_VERSION=1.0.0
   ```

## 🔧 Configuration Details

### Backend Configuration

The backend service includes:
- **FastAPI Application**: Production-ready API server
- **AI Tools**: All 6 advanced tools (PDF Parser, Clause Library, Summarizer, NER, Regex Search, External API)
- **spaCy Models**: Automatically downloads English language model
- **Health Checks**: Monitoring endpoint at `/health`
- **CORS**: Configured for cross-origin requests

### Environment Variables Explained

| Variable | Description | Required |
|----------|-------------|----------|
| `ENVIRONMENT` | Deployment environment (production) | Yes |
| `PORT` | Port for the application (10000) | Yes |
| `EURI_API_KEY` | EURI AI API key (use "test-key" for demo) | Optional |
| `SECRET_KEY` | JWT secret key (auto-generated) | Yes |
| `CORS_ORIGINS` | Allowed CORS origins | Yes |
| `PYTHONPATH` | Python path for imports | Yes |

## 🚀 Post-Deployment

### 1. Verify Backend Deployment

Once deployed, your backend will be available at:
```
https://legal-intelligence-api.onrender.com
```

**Test the API**:
- Health Check: `GET https://legal-intelligence-api.onrender.com/health`
- API Docs: `https://legal-intelligence-api.onrender.com/docs`

### 2. Test Document Analysis

```bash
curl -X POST "https://legal-intelligence-api.onrender.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This agreement may be terminated by either party upon 30 days written notice. Payment shall be due within 30 days.",
    "analysis_type": "comprehensive"
  }'
```

### 3. Monitor Performance

- **Render Dashboard**: Monitor logs, metrics, and performance
- **Health Checks**: Automatic monitoring of `/health` endpoint
- **Logs**: View real-time application logs

## 🔍 Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check that `requirements-render.txt` exists in the backend directory
   - Verify Python version compatibility
   - Check build logs for specific error messages

2. **spaCy Model Download Issues**:
   - The build command includes `python -m spacy download en_core_web_sm`
   - If it fails, the app will still work with reduced NLP capabilities

3. **Memory Issues**:
   - Starter plan has 512MB RAM
   - Consider upgrading to Standard plan for better performance

4. **CORS Issues**:
   - Update `CORS_ORIGINS` environment variable
   - Set to your frontend URL or `*` for development

### Performance Optimization

1. **Upgrade Plan**: Consider Standard plan for better performance
2. **Caching**: The application includes built-in caching
3. **Database**: Add PostgreSQL for persistence (optional)

## 🎯 Production Enhancements

### 1. Add Database (Optional)

1. **Create PostgreSQL Database**:
   - In Render Dashboard: "New +" → "PostgreSQL"
   - Name: `legal-intelligence-db`

2. **Update Environment Variables**:
   ```
   DATABASE_URL=[Auto-generated connection string]
   ```

### 2. Custom Domain

1. **Add Custom Domain**:
   - Go to your service settings
   - Add your custom domain
   - Configure DNS records

### 3. SSL Certificate

- Render automatically provides SSL certificates
- Your API will be available over HTTPS

## 📊 Expected Performance

- **Cold Start**: ~30-60 seconds (first request after inactivity)
- **Warm Requests**: ~2-3 seconds per document analysis
- **Concurrent Users**: 10-50 on Starter plan
- **Uptime**: 99.9% SLA

## 🆘 Support

If you encounter issues:

1. **Check Logs**: View deployment and runtime logs in Render Dashboard
2. **GitHub Issues**: Report issues at https://github.com/erickyegon/legal-doc-analyzer/issues
3. **Render Support**: Contact Render support for platform-specific issues

## 🎉 Success!

Once deployed, your Legal Intelligence Platform will be live and ready to analyze legal documents with advanced AI capabilities!

**Your API will be available at**: `https://legal-intelligence-api.onrender.com`

**Features Available**:
- ✅ Document analysis with entity extraction
- ✅ Legal clause detection and risk assessment
- ✅ Real-time processing with professional results
- ✅ Comprehensive API documentation
- ✅ Health monitoring and logging
