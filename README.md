# Legal Intelligence Platform

A comprehensive legal document analysis platform powered by AI, built with FastAPI backend and React frontend.

## 🚀 Features

- **Document Upload & Analysis**: Upload legal documents for AI-powered analysis
- **Intelligent Agents**: Specialized AI agents for different legal document types
- **User Authentication**: Secure JWT-based authentication system
- **Professional Dashboard**: Modern, responsive UI for document management
- **Real-time Analysis**: Get instant insights from legal documents
- **Multi-format Support**: Support for PDF, DOCX, and text files

## 🏗️ Architecture

### Backend (FastAPI)
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM for database operations
- **JWT Authentication**: Secure token-based authentication
- **EURI AI Integration**: Advanced AI capabilities using EURI client
- **PostgreSQL**: Production-ready database

### Frontend (React + TypeScript)
- **React 18**: Modern React with hooks and functional components
- **TypeScript**: Type-safe development
- **Material-UI**: Professional component library
- **React Router**: Client-side routing
- **Axios**: HTTP client for API communication
- **React Query**: Data fetching and caching

## 🛠️ Technology Stack

### Backend
- Python 3.11+
- FastAPI
- SQLAlchemy
- PostgreSQL
- JWT Authentication
- EURI AI Client
- Uvicorn (ASGI server)

### Frontend
- React 18
- TypeScript
- Material-UI (MUI)
- React Router v6
- Axios
- React Query
- Vite (Build tool)

## 📦 Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL (for production)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Configure your environment variables
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Deployment on Render

This application is configured for deployment on Render with separate services for frontend and backend.

### Backend Deployment
- Automatically deploys from `backend/` directory
- Uses PostgreSQL database
- Environment variables configured in Render dashboard

### Frontend Deployment
- Automatically deploys from `frontend/` directory
- Static site deployment with React build
- Configured with proper routing for SPA

## 🔧 Environment Variables

### Backend (.env)
```
EURI_API_KEY=your_euri_api_key
DATABASE_URL=postgresql://user:password@host:port/dbname
SECRET_KEY=your_jwt_secret_key
ENVIRONMENT=production
```

### Frontend (.env)
```
VITE_API_URL=https://your-backend-url.onrender.com
```

## 📚 API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔐 Authentication

The platform uses JWT-based authentication:
1. Login with credentials to receive access token
2. Include token in Authorization header for protected routes
3. Tokens expire after 24 hours (configurable)

## 📄 Document Analysis Workflow

1. **Upload**: Users upload legal documents through the web interface
2. **Processing**: Documents are processed and stored securely
3. **Analysis**: AI agents analyze documents based on type and content
4. **Results**: Users receive detailed analysis reports
5. **Management**: Documents and analyses are stored for future reference

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Enhanced UI and additional document types
- **v1.2.0**: Improved AI analysis and performance optimizations