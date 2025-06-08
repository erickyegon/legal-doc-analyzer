# Legal Intelligence Platform - Deployment Guide

This guide provides comprehensive instructions for deploying the Legal Intelligence Platform on Render with professional UI and advanced multimodal AI capabilities.

## 🏗️ Architecture Overview

The platform consists of:
- **Backend**: FastAPI with LangChain/LangGraph agents, multimodal extraction, and EURI AI integration
- **Frontend**: React with TypeScript, Material-UI, and professional design
- **Database**: PostgreSQL for production data storage
- **AI Agents**: Specialized agents for contract analysis, multimodal extraction, and document processing

## 🚀 Render Deployment

### Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Push your code to GitHub
3. **EURI API Key**: Obtain from your EURI AI provider
4. **Environment Variables**: Prepare all required environment variables

### Backend Deployment

1. **Create Web Service**:
   - Go to Render Dashboard → New → Web Service
   - Connect your GitHub repository
   - Select the `backend` directory as the root directory

2. **Configure Build Settings**:
   ```yaml
   Build Command: pip install --upgrade pip && pip install -r requirements.txt
   Start Command: chmod +x start.sh && ./start.sh
   ```

3. **Environment Variables**:
   ```bash
   ENVIRONMENT=production
   EURI_API_KEY=your_euri_api_key_here
   SECRET_KEY=your_super_secret_jwt_key_here
   DATABASE_URL=postgresql://user:password@host:port/dbname
   CORS_ORIGINS=https://your-frontend-url.onrender.com
   PYTHONPATH=/opt/render/project/src
   ```

4. **Database Setup**:
   - Create PostgreSQL database in Render
   - Use the connection string in `DATABASE_URL`

### Frontend Deployment

1. **Create Static Site**:
   - Go to Render Dashboard → New → Static Site
   - Connect your GitHub repository
   - Select the `frontend` directory as the root directory

2. **Configure Build Settings**:
   ```yaml
   Build Command: npm ci && npm run build
   Publish Directory: dist
   ```

3. **Environment Variables**:
   ```bash
   VITE_API_URL=https://your-backend-url.onrender.com
   VITE_APP_NAME=Legal Intelligence Platform
   VITE_APP_VERSION=1.0.0
   ```

## 🔧 Local Development Setup

### Backend Setup

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd legal-intelligence-platform/backend
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run Database Migrations**:
   ```bash
   # If using Alembic
   alembic upgrade head
   ```

6. **Start Development Server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### Frontend Setup

1. **Navigate to Frontend**:
   ```bash
   cd ../frontend
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start Development Server**:
   ```bash
   npm run dev
   ```

## 🤖 AI Agents Configuration

### Multimodal Extraction Agent

The platform includes advanced multimodal extraction capabilities:

- **Table Extraction**: Uses Camelot, Tabula, and PDFPlumber
- **Image Processing**: OpenCV and EasyOCR for image analysis
- **Signature Detection**: Computer vision-based signature identification
- **Layout Analysis**: LayoutParser for document structure analysis

### Contract Analysis Agent

Specialized LangChain agent for contract analysis:

- **Contract Type Identification**
- **Party Extraction**
- **Clause Analysis**
- **Risk Assessment**
- **Financial Terms Extraction**
- **Important Dates Identification**

### Agent Orchestrator

LangGraph-based orchestrator that coordinates multiple agents:

- **Intelligent Routing**: Determines optimal analysis strategy
- **Parallel Processing**: Runs multiple agents concurrently
- **Result Synthesis**: Combines results from all agents
- **Error Handling**: Robust error handling and fallback mechanisms

## 🔐 Security Configuration

### Backend Security

1. **JWT Configuration**:
   ```python
   SECRET_KEY=your-super-secret-key-change-in-production
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=1440
   ```

2. **CORS Configuration**:
   ```python
   CORS_ORIGINS=["https://your-frontend-domain.com"]
   ```

3. **Database Security**:
   - Use strong passwords
   - Enable SSL connections
   - Regular backups

### Frontend Security

1. **Environment Variables**:
   - Never expose sensitive data in frontend
   - Use VITE_ prefix for public variables

2. **Content Security Policy**:
   ```html
   <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline';">
   ```

## 📊 Monitoring and Analytics

### Health Checks

Both frontend and backend include health check endpoints:

- Backend: `GET /health`
- Frontend: Built-in Render health checks

### Logging

- **Backend**: Structured logging with different levels
- **Frontend**: Error boundary and console logging
- **Production**: Log aggregation and monitoring

### Performance Monitoring

- **Backend**: Request timing and database query monitoring
- **Frontend**: Core Web Vitals and user experience metrics

## 🔄 CI/CD Pipeline

### Automatic Deployments

Render automatically deploys when you push to your main branch:

1. **Backend**: Builds and deploys FastAPI application
2. **Frontend**: Builds React application and deploys static files
3. **Database**: Runs migrations automatically

### Testing

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   - Check DATABASE_URL format
   - Verify database is running
   - Check network connectivity

2. **CORS Errors**:
   - Verify CORS_ORIGINS configuration
   - Check frontend URL in backend settings

3. **Build Failures**:
   - Check dependency versions
   - Verify environment variables
   - Review build logs

4. **AI Agent Errors**:
   - Verify EURI_API_KEY is set
   - Check model availability
   - Review agent configuration

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Backend
DEBUG=true

# Frontend
VITE_ENABLE_DEBUG=true
```

## 📈 Scaling Considerations

### Backend Scaling

- **Horizontal Scaling**: Multiple Render instances
- **Database**: Connection pooling and read replicas
- **Caching**: Redis for session and data caching
- **File Storage**: External storage for uploaded documents

### Frontend Scaling

- **CDN**: Render provides global CDN
- **Caching**: Browser caching and service workers
- **Code Splitting**: Lazy loading for better performance

## 🔮 Advanced Features

### LangServe Integration

The platform includes LangServe endpoints for AI agents:

- **Multimodal Extraction**: `/multimodal-extraction`
- **Contract Analysis**: `/contract-analysis`
- **Orchestrated Analysis**: `/orchestrated-analysis`

### API Documentation

- **FastAPI Docs**: Available at `/docs` (development only)
- **ReDoc**: Available at `/redoc` (development only)
- **OpenAPI**: Full OpenAPI 3.0 specification

### Professional UI Features

- **Material-UI**: Professional component library
- **Responsive Design**: Mobile-first responsive layout
- **Dark/Light Theme**: Theme switching capability
- **Accessibility**: WCAG 2.1 AA compliance
- **Internationalization**: Multi-language support ready

## 📞 Support

For deployment issues or questions:

1. Check the troubleshooting section
2. Review Render documentation
3. Check application logs
4. Contact the development team

## 🔄 Updates and Maintenance

### Regular Updates

1. **Dependencies**: Keep dependencies updated
2. **Security Patches**: Apply security updates promptly
3. **Database Maintenance**: Regular backups and optimization
4. **Performance Monitoring**: Monitor and optimize performance

### Backup Strategy

1. **Database**: Automated daily backups
2. **Files**: Regular backup of uploaded documents
3. **Configuration**: Version control for all configuration
4. **Disaster Recovery**: Documented recovery procedures
