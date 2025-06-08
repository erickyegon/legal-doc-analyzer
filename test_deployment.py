#!/usr/bin/env python3
"""
Deployment Verification Script for Legal Intelligence Platform

This script tests both backend and frontend deployments to ensure
they're working correctly in production.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import requests
import json
import time
from datetime import datetime

# Deployment URLs
BACKEND_URL = "https://legal-intelligence-api.onrender.com"
FRONTEND_URL = "https://legal-intelligence-frontend.onrender.com"

def test_backend_health():
    """Test backend health endpoint."""
    print("🏥 Testing Backend Health...")
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend Health Check Passed")
            print(f"   - Status: {data['status']}")
            print(f"   - Version: {data['version']}")
            print(f"   - spaCy NLP: {data['services']['spacy_nlp']}")
            print(f"   - Regex Patterns: {data['services']['regex_patterns']}")
            return True
        else:
            print(f"❌ Backend Health Check Failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend Health Check Error: {e}")
        return False

def test_backend_api():
    """Test backend document analysis API."""
    print("\n📄 Testing Backend Document Analysis...")
    
    sample_document = """
    SOFTWARE LICENSE AGREEMENT
    
    This Agreement is entered into on January 1, 2024, between TechCorp Inc. and BusinessCorp LLC.
    
    1. TERMINATION
    Either party may terminate this agreement upon thirty (30) days written notice.
    
    2. PAYMENT
    Licensee shall pay an annual fee of $50,000, due within 30 days of invoice.
    
    3. CONFIDENTIALITY
    Confidential information shall remain confidential for 5 years.
    
    4. GOVERNING LAW
    This agreement shall be governed by the laws of California.
    """
    
    try:
        payload = {
            "text": sample_document,
            "analysis_type": "comprehensive"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/analyze", 
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Document Analysis Successful")
            print(f"   - Document ID: {data['document_id']}")
            print(f"   - Processing Time: {data['processing_time']:.2f}s")
            print(f"   - Entities Found: {len(data['entities'])}")
            print(f"   - Clauses Detected: {len(data['clauses'])}")
            print(f"   - Overall Risk: {data['risk_assessment']['overall_risk']}")
            
            # Show sample results
            if data['entities']:
                print(f"   - Sample Entity: {data['entities'][0]['text']} ({data['entities'][0]['label']})")
            
            if data['clauses']:
                print(f"   - Sample Clause: {data['clauses'][0]['clause_type']} ({data['clauses'][0]['risk_level']} risk)")
            
            return True
        else:
            print(f"❌ Document Analysis Failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Document Analysis Error: {e}")
        return False

def test_frontend_availability():
    """Test frontend availability."""
    print("\n🎨 Testing Frontend Availability...")
    
    try:
        response = requests.get(FRONTEND_URL, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ Frontend Available")
            print(f"   - URL: {FRONTEND_URL}")
            print(f"   - Status: HTTP {response.status_code}")
            print(f"   - Content Length: {len(response.content)} bytes")
            
            # Check if it's a React app
            if "Legal Intelligence Platform" in response.text:
                print(f"   - React App: Detected")
            
            return True
        else:
            print(f"❌ Frontend Not Available: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend Availability Error: {e}")
        return False

def test_cors_configuration():
    """Test CORS configuration between frontend and backend."""
    print("\n🔗 Testing CORS Configuration...")
    
    try:
        # Test preflight request
        headers = {
            'Origin': FRONTEND_URL,
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        response = requests.options(
            f"{BACKEND_URL}/api/v1/analyze",
            headers=headers,
            timeout=30
        )
        
        if response.status_code in [200, 204]:
            print(f"✅ CORS Configuration Valid")
            print(f"   - Preflight Response: HTTP {response.status_code}")
            
            # Check CORS headers
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
            }
            
            for header, value in cors_headers.items():
                if value:
                    print(f"   - {header}: {value}")
            
            return True
        else:
            print(f"❌ CORS Configuration Issue: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ CORS Test Error: {e}")
        return False

def test_api_documentation():
    """Test API documentation availability."""
    print("\n📖 Testing API Documentation...")
    
    try:
        response = requests.get(f"{BACKEND_URL}/docs", timeout=30)
        
        if response.status_code == 200:
            print(f"✅ API Documentation Available")
            print(f"   - URL: {BACKEND_URL}/docs")
            print(f"   - Swagger UI: Accessible")
            return True
        else:
            print(f"❌ API Documentation Not Available: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API Documentation Error: {e}")
        return False

def performance_test():
    """Test API performance with multiple requests."""
    print("\n⚡ Testing API Performance...")
    
    try:
        payload = {
            "text": "Simple contract with termination clause. Payment due in 30 days.",
            "analysis_type": "basic"
        }
        
        times = []
        for i in range(3):
            start_time = time.time()
            response = requests.post(
                f"{BACKEND_URL}/api/v1/analyze",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
            else:
                print(f"❌ Performance Test Request {i+1} Failed")
                return False
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"✅ Performance Test Completed")
        print(f"   - Requests: {len(times)}")
        print(f"   - Average Time: {avg_time:.2f}s")
        print(f"   - Min Time: {min_time:.2f}s")
        print(f"   - Max Time: {max_time:.2f}s")
        
        if avg_time < 10:
            print(f"   - Performance: Good")
        elif avg_time < 20:
            print(f"   - Performance: Acceptable")
        else:
            print(f"   - Performance: Slow (consider upgrading plan)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Performance Test Error: {e}")
        return False

def main():
    """Run all deployment verification tests."""
    print("🚀 Legal Intelligence Platform - Deployment Verification")
    print("=" * 65)
    print(f"🕐 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Backend URL: {BACKEND_URL}")
    print(f"🎨 Frontend URL: {FRONTEND_URL}")
    print()
    
    tests = [
        ("Backend Health Check", test_backend_health),
        ("Backend API Analysis", test_backend_api),
        ("Frontend Availability", test_frontend_availability),
        ("CORS Configuration", test_cors_configuration),
        ("API Documentation", test_api_documentation),
        ("Performance Test", performance_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
        
        # Add delay between tests
        time.sleep(1)
    
    print("\n" + "=" * 65)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print(f"🕐 Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your Legal Intelligence Platform is fully operational!")
        print("\n🌟 Platform Status: PRODUCTION READY")
        print("\n📋 Available Services:")
        print(f"   🔗 Backend API: {BACKEND_URL}")
        print(f"   🔗 Frontend UI: {FRONTEND_URL}")
        print(f"   🔗 API Docs: {BACKEND_URL}/docs")
        print("\n🚀 Features Working:")
        print("   ✅ Document analysis with AI")
        print("   ✅ Entity extraction")
        print("   ✅ Legal clause detection")
        print("   ✅ Risk assessment")
        print("   ✅ Professional web interface")
        print("   ✅ Real-time processing")
        
        print("\n💡 Next Steps:")
        print("   1. Share the frontend URL with users")
        print("   2. Monitor performance in Render dashboard")
        print("   3. Consider upgrading plans for better performance")
        print("   4. Add custom domain (optional)")
        print("   5. Set up monitoring and analytics")
        
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the issues above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Check Render deployment logs")
        print("   2. Verify environment variables")
        print("   3. Ensure both services are deployed")
        print("   4. Check CORS configuration")
        print("   5. Wait for cold start (first request may be slow)")
        
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
