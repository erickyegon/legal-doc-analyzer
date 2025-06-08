import React, { useState } from 'react';
import { 
  ThemeProvider, 
  createTheme, 
  Container, 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  Card, 
  CardContent, 
  Button, 
  TextField, 
  Grid, 
  Chip, 
  Alert,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import { Toaster, toast } from 'react-hot-toast';
import axios from 'axios';

// Theme configuration
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface AnalysisResult {
  document_id: string;
  entities: Array<{
    text: string;
    label: string;
    confidence: number;
  }>;
  clauses: Array<{
    clause_type: string;
    description: string;
    risk_level: string;
    confidence: number;
  }>;
  summary: string;
  risk_assessment: {
    overall_risk: string;
    risk_breakdown: Record<string, number>;
    recommendations: string[];
  };
  processing_time: number;
}

function DemoApp() {
  const [documentText, setDocumentText] = useState('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<'unknown' | 'healthy' | 'error'>('unknown');

  // Check API health on component mount
  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      if (response.status === 200) {
        setApiStatus('healthy');
        toast.success('Connected to Legal Intelligence API');
      }
    } catch (error) {
      setApiStatus('error');
      toast.error('Cannot connect to API. Please ensure the backend is running.');
    }
  };

  const analyzeDocument = async () => {
    if (!documentText.trim()) {
      toast.error('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/analyze`, {
        text: documentText,
        analysis_type: 'comprehensive'
      });

      setAnalysisResult(response.data);
      toast.success('Document analysis completed!');
    } catch (error) {
      console.error('Analysis failed:', error);
      toast.error('Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const loadSampleDocument = () => {
    const sampleText = `SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of January 1, 2024,
by and between TechCorp Inc., a Delaware corporation ("Licensor"), and BusinessCorp LLC,
a California limited liability company ("Licensee").

1. GRANT OF LICENSE
Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee
a non-exclusive, non-transferable license to use the Software solely for Licensee's
internal business purposes.

2. TERM AND TERMINATION
This Agreement shall commence on the Effective Date and shall continue for a period of
three (3) years. Either party may terminate this agreement upon thirty (30) days written notice.

3. FEES AND PAYMENT
In consideration for the license granted hereunder, Licensee shall pay Licensor an
annual license fee of $50,000, payable within 30 days of invoice date.

4. CONFIDENTIALITY
Each party acknowledges that it may have access to confidential information of the other party.
Such confidential information shall remain confidential for a period of 5 years.

5. LIMITATION OF LIABILITY
In no event shall either party's liability exceed $100,000 in aggregate.

6. GOVERNING LAW
This Agreement shall be governed by the laws of California.`;

    setDocumentText(sampleText);
    toast.success('Sample document loaded');
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* Header */}
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🏛️ Legal Intelligence Platform - Demo
          </Typography>
          <Chip 
            label={apiStatus === 'healthy' ? 'API Connected' : apiStatus === 'error' ? 'API Disconnected' : 'Checking...'}
            color={apiStatus === 'healthy' ? 'success' : apiStatus === 'error' ? 'error' : 'default'}
            size="small"
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Input Section */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  📄 Document Analysis
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Button 
                    variant="outlined" 
                    onClick={loadSampleDocument}
                    sx={{ mr: 1 }}
                  >
                    Load Sample Document
                  </Button>
                  <Button 
                    variant="outlined" 
                    onClick={() => setDocumentText('')}
                  >
                    Clear
                  </Button>
                </Box>

                <TextField
                  fullWidth
                  multiline
                  rows={15}
                  variant="outlined"
                  label="Enter legal document text"
                  value={documentText}
                  onChange={(e) => setDocumentText(e.target.value)}
                  placeholder="Paste your legal document here for analysis..."
                  sx={{ mb: 2 }}
                />

                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  onClick={analyzeDocument}
                  disabled={loading || !documentText.trim()}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                >
                  {loading ? 'Analyzing...' : 'Analyze Document'}
                </Button>
              </CardContent>
            </Card>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12} md={6}>
            {analysisResult ? (
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    📊 Analysis Results
                  </Typography>

                  {/* Processing Info */}
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Analysis completed in {analysisResult.processing_time.toFixed(2)} seconds
                  </Alert>

                  {/* Risk Assessment */}
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      ⚠️ Risk Assessment
                    </Typography>
                    <Chip 
                      label={`Overall Risk: ${analysisResult.risk_assessment.overall_risk.toUpperCase()}`}
                      color={getRiskColor(analysisResult.risk_assessment.overall_risk)}
                      sx={{ mb: 1 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      High: {analysisResult.risk_assessment.risk_breakdown.high} | 
                      Medium: {analysisResult.risk_assessment.risk_breakdown.medium} | 
                      Low: {analysisResult.risk_assessment.risk_breakdown.low}
                    </Typography>
                  </Paper>

                  {/* Entities */}
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      🏷️ Entities Found ({analysisResult.entities.length})
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {analysisResult.entities.slice(0, 10).map((entity, index) => (
                        <Chip
                          key={index}
                          label={`${entity.text} (${entity.label})`}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                      {analysisResult.entities.length > 10 && (
                        <Chip label={`+${analysisResult.entities.length - 10} more`} size="small" />
                      )}
                    </Box>
                  </Paper>

                  {/* Clauses */}
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      📋 Legal Clauses ({analysisResult.clauses.length})
                    </Typography>
                    <List dense>
                      {analysisResult.clauses.slice(0, 5).map((clause, index) => (
                        <React.Fragment key={index}>
                          <ListItem>
                            <ListItemText
                              primary={clause.description}
                              secondary={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  <Chip 
                                    label={clause.risk_level} 
                                    size="small" 
                                    color={getRiskColor(clause.risk_level)}
                                  />
                                  <Typography variant="caption">
                                    {(clause.confidence * 100).toFixed(0)}% confidence
                                  </Typography>
                                </Box>
                              }
                            />
                          </ListItem>
                          {index < analysisResult.clauses.length - 1 && <Divider />}
                        </React.Fragment>
                      ))}
                    </List>
                  </Paper>

                  {/* Summary */}
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      📝 Summary
                    </Typography>
                    <Typography variant="body2">
                      {analysisResult.summary}
                    </Typography>
                  </Paper>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    🚀 Welcome to Legal Intelligence Platform
                  </Typography>
                  <Typography variant="body1" paragraph>
                    This platform provides advanced AI-powered legal document analysis with:
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText primary="🔍 Named Entity Recognition" secondary="Extract people, organizations, dates, and monetary amounts" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="📋 Legal Clause Detection" secondary="Identify termination, payment, liability, and other key clauses" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="⚠️ Risk Assessment" secondary="Automatic risk scoring and recommendations" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="📝 Document Summarization" secondary="Generate concise summaries of complex documents" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="⚡ Real-time Processing" secondary="Fast analysis with detailed results" />
                    </ListItem>
                  </List>
                  <Typography variant="body2" color="text.secondary">
                    Enter a legal document in the text area to get started, or click "Load Sample Document" to try with example content.
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      </Container>

      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
    </ThemeProvider>
  );
}

export default DemoApp;
