#!/bin/bash

# Legal Intelligence Platform - Deployment Script
# This script helps prepare and deploy the application to Render

echo "🚀 Legal Intelligence Platform - Deployment Preparation"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "render.yaml" ]; then
    echo "❌ Error: render.yaml not found. Please run this script from the project root."
    exit 1
fi

echo "📋 Checking prerequisites..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
    git remote add origin https://github.com/erickyegon/legal-doc-analyzer.git
fi

# Check if files are committed
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 Committing changes..."
    git add .
    git commit -m "Deploy: Updated Legal Intelligence Platform for Render deployment"
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Repository updated successfully!"
echo ""
echo "🎯 Next Steps for Render Deployment:"
echo "1. Go to https://dashboard.render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository: legal-doc-analyzer"
echo "4. Use these settings:"
echo ""
echo "   📊 Backend Service Configuration:"
echo "   - Name: legal-intelligence-api"
echo "   - Region: Oregon (US West)"
echo "   - Branch: main"
echo "   - Root Directory: backend"
echo "   - Runtime: Python 3"
echo "   - Build Command: pip install --upgrade pip && pip install -r requirements-render.txt && python -m spacy download en_core_web_sm"
echo "   - Start Command: python main.py"
echo "   - Plan: Starter"
echo ""
echo "   🔧 Environment Variables:"
echo "   - ENVIRONMENT=production"
echo "   - PORT=10000"
echo "   - EURI_API_KEY=test-key"
echo "   - SECRET_KEY=[Auto-generated]"
echo "   - CORS_ORIGINS=*"
echo "   - PYTHONPATH=/opt/render/project/src"
echo ""
echo "   ⚡ Advanced Settings:"
echo "   - Health Check Path: /health"
echo "   - Auto-Deploy: Yes"
echo ""
echo "🌐 Alternative: One-Click Deployment"
echo "Visit: https://render.com/deploy?repo=https://github.com/erickyegon/legal-doc-analyzer"
echo ""
echo "📖 For detailed instructions, see: RENDER_DEPLOYMENT.md"
echo ""
echo "🎉 Your Legal Intelligence Platform is ready for deployment!"
