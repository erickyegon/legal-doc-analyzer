"""
External API Tool for the Legal Intelligence Platform.

This tool provides integration with external legal search APIs including
Westlaw, LexisNexis, and other legal research databases for comparison
and validation of legal analysis.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import hashlib
import time

# LangChain imports
from langchain.tools import BaseTool

# Custom imports
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported external API providers."""
    WESTLAW = "westlaw"
    LEXISNEXIS = "lexisnexis"
    JUSTIA = "justia"
    GOOGLE_SCHOLAR = "google_scholar"
    COURTLISTENER = "courtlistener"
    CASELAW_ACCESS = "caselaw_access"
    LEGAL_API = "legal_api"


class SearchType(Enum):
    """Types of legal searches."""
    CASE_LAW = "case_law"
    STATUTES = "statutes"
    REGULATIONS = "regulations"
    SECONDARY_SOURCES = "secondary_sources"
    NEWS = "news"
    FORMS = "forms"
    BRIEFS = "briefs"


@dataclass
class SearchQuery:
    """Data class for search queries."""
    query: str
    search_type: SearchType
    jurisdiction: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    court: Optional[str] = None
    practice_area: Optional[str] = None
    max_results: int = 10


@dataclass
class SearchResult:
    """Data class for search results."""
    title: str
    citation: str
    url: str
    snippet: str
    date: Optional[str]
    court: Optional[str]
    jurisdiction: Optional[str]
    relevance_score: float
    source: str
    metadata: Dict[str, Any]


@dataclass
class APIResponse:
    """Data class for API responses."""
    provider: str
    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    search_time: float
    api_cost: Optional[float]
    rate_limit_remaining: Optional[int]
    success: bool
    error_message: Optional[str]


class ExternalAPITool(BaseTool):
    """
    External API Tool for legal research integration.
    
    This tool provides:
    - Integration with major legal research APIs
    - Unified search interface across multiple providers
    - Rate limiting and cost management
    - Result caching and deduplication
    - Citation validation and formatting
    - Comparative analysis across sources
    """
    
    name = "external_api"
    description = "Search external legal databases and APIs for case law, statutes, and legal research"
    
    def __init__(self):
        """Initialize the External API Tool."""
        super().__init__()
        
        # API configurations
        self.api_configs = self._initialize_api_configs()
        
        # Rate limiting
        self.rate_limits = {}
        self.last_request_times = {}
        
        # Caching
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Session for HTTP requests
        self.session = None
    
    def _initialize_api_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize API configurations for different providers."""
        configs = {}
        
        # Westlaw API Configuration
        configs[APIProvider.WESTLAW.value] = {
            "base_url": "https://api.westlaw.com/v1",
            "auth_type": "oauth",
            "rate_limit": 100,  # requests per minute
            "cost_per_request": 0.10,  # USD
            "supported_searches": [SearchType.CASE_LAW, SearchType.STATUTES, SearchType.SECONDARY_SOURCES],
            "headers": {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        }
        
        # LexisNexis API Configuration
        configs[APIProvider.LEXISNEXIS.value] = {
            "base_url": "https://api.lexisnexis.com/v1",
            "auth_type": "api_key",
            "rate_limit": 60,  # requests per minute
            "cost_per_request": 0.15,  # USD
            "supported_searches": [SearchType.CASE_LAW, SearchType.STATUTES, SearchType.NEWS],
            "headers": {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        }
        
        # Justia API Configuration (Free)
        configs[APIProvider.JUSTIA.value] = {
            "base_url": "https://api.justia.com/v1",
            "auth_type": "none",
            "rate_limit": 1000,  # requests per day
            "cost_per_request": 0.0,  # Free
            "supported_searches": [SearchType.CASE_LAW, SearchType.STATUTES],
            "headers": {
                "Accept": "application/json"
            }
        }
        
        # CourtListener API Configuration (Free)
        configs[APIProvider.COURTLISTENER.value] = {
            "base_url": "https://www.courtlistener.com/api/rest/v3",
            "auth_type": "token",
            "rate_limit": 5000,  # requests per hour
            "cost_per_request": 0.0,  # Free
            "supported_searches": [SearchType.CASE_LAW, SearchType.BRIEFS],
            "headers": {
                "Accept": "application/json"
            }
        }
        
        # Caselaw Access Project (Free)
        configs[APIProvider.CASELAW_ACCESS.value] = {
            "base_url": "https://api.case.law/v1",
            "auth_type": "api_key",
            "rate_limit": 10000,  # requests per day
            "cost_per_request": 0.0,  # Free
            "supported_searches": [SearchType.CASE_LAW],
            "headers": {
                "Accept": "application/json"
            }
        }
        
        return configs
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def _run(self, query: str, **kwargs) -> str:
        """
        Run the external API tool.
        
        Args:
            query (str): Search query
            **kwargs: Additional arguments
            
        Returns:
            str: Search results summary
        """
        try:
            # Run async search
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_search(query, **kwargs))
                    results = future.result()
            else:
                results = asyncio.run(self._async_search(query, **kwargs))
            
            # Format summary
            if results:
                summary = f"Found {len(results)} results across {len(set(r.provider for r in results))} providers:\n"
                
                for result in results[:5]:  # Show top 5 results
                    summary += f"- {result.title} ({result.source})\n"
                    summary += f"  Citation: {result.citation}\n"
                    summary += f"  Relevance: {result.relevance_score:.2f}\n\n"
            else:
                summary = "No results found from external APIs."
            
            return summary
            
        except Exception as e:
            logger.error(f"External API search failed: {e}")
            return f"External API search failed: {str(e)}"
    
    async def _async_search(self, query: str, **kwargs) -> List[APIResponse]:
        """Async wrapper for search functionality."""
        search_query = SearchQuery(
            query=query,
            search_type=SearchType(kwargs.get('search_type', 'case_law')),
            jurisdiction=kwargs.get('jurisdiction'),
            date_range=kwargs.get('date_range'),
            court=kwargs.get('court'),
            practice_area=kwargs.get('practice_area'),
            max_results=kwargs.get('max_results', 10)
        )
        
        providers = kwargs.get('providers', [APIProvider.JUSTIA, APIProvider.COURTLISTENER])
        
        return await self.search_multiple_apis(search_query, providers)
    
    async def search_multiple_apis(self,
                                  query: SearchQuery,
                                  providers: List[APIProvider],
                                  parallel: bool = True) -> List[APIResponse]:
        """
        Search multiple APIs in parallel or sequentially.
        
        Args:
            query (SearchQuery): Search query object
            providers (List[APIProvider]): List of API providers to search
            parallel (bool): Whether to search APIs in parallel
            
        Returns:
            List[APIResponse]: Results from all APIs
        """
        if parallel:
            # Search all APIs in parallel
            tasks = []
            for provider in providers:
                task = self.search_single_api(query, provider)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_responses = []
            for response in responses:
                if isinstance(response, APIResponse):
                    valid_responses.append(response)
                else:
                    logger.error(f"API search failed: {response}")
            
            return valid_responses
        else:
            # Search APIs sequentially
            responses = []
            for provider in providers:
                try:
                    response = await self.search_single_api(query, provider)
                    responses.append(response)
                except Exception as e:
                    logger.error(f"API search failed for {provider}: {e}")
            
            return responses
    
    async def search_single_api(self, query: SearchQuery, provider: APIProvider) -> APIResponse:
        """
        Search a single API provider.
        
        Args:
            query (SearchQuery): Search query object
            provider (APIProvider): API provider to search
            
        Returns:
            APIResponse: API response with results
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, provider)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                logger.info(f"Using cached result for {provider.value}")
                return cached_result
            
            # Check rate limits
            if not self._check_rate_limit(provider):
                return APIResponse(
                    provider=provider.value,
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    api_cost=0.0,
                    rate_limit_remaining=0,
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            # Get API configuration
            config = self.api_configs.get(provider.value)
            if not config:
                raise ValueError(f"No configuration found for provider: {provider.value}")
            
            # Check if search type is supported
            if query.search_type not in config["supported_searches"]:
                return APIResponse(
                    provider=provider.value,
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    api_cost=0.0,
                    rate_limit_remaining=None,
                    success=False,
                    error_message=f"Search type {query.search_type.value} not supported"
                )
            
            # Perform the actual API search
            if provider == APIProvider.JUSTIA:
                response = await self._search_justia(query, config)
            elif provider == APIProvider.COURTLISTENER:
                response = await self._search_courtlistener(query, config)
            elif provider == APIProvider.CASELAW_ACCESS:
                response = await self._search_caselaw_access(query, config)
            elif provider == APIProvider.WESTLAW:
                response = await self._search_westlaw(query, config)
            elif provider == APIProvider.LEXISNEXIS:
                response = await self._search_lexisnexis(query, config)
            else:
                raise ValueError(f"Provider {provider.value} not implemented")
            
            # Update rate limiting
            self._update_rate_limit(provider)
            
            # Cache the result
            self._cache_result(cache_key, response)
            
            search_time = time.time() - start_time
            response.search_time = search_time
            
            return response
            
        except Exception as e:
            logger.error(f"API search failed for {provider.value}: {e}")
            
            return APIResponse(
                provider=provider.value,
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                api_cost=0.0,
                rate_limit_remaining=None,
                success=False,
                error_message=str(e)
            )
    
    async def _search_justia(self, query: SearchQuery, config: Dict[str, Any]) -> APIResponse:
        """Search Justia API."""
        session = await self._get_session()
        
        # Build search URL
        base_url = config["base_url"]
        
        if query.search_type == SearchType.CASE_LAW:
            url = f"{base_url}/cases"
        elif query.search_type == SearchType.STATUTES:
            url = f"{base_url}/codes"
        else:
            raise ValueError(f"Unsupported search type for Justia: {query.search_type}")
        
        # Build parameters
        params = {
            "q": query.query,
            "limit": query.max_results
        }
        
        if query.jurisdiction:
            params["jurisdiction"] = query.jurisdiction
        
        # Make request
        async with session.get(url, params=params, headers=config["headers"]) as response:
            if response.status == 200:
                data = await response.json()
                results = self._parse_justia_results(data)
                
                return APIResponse(
                    provider=APIProvider.JUSTIA.value,
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=0.0,  # Will be set by caller
                    api_cost=0.0,
                    rate_limit_remaining=None,
                    success=True,
                    error_message=None
                )
            else:
                raise Exception(f"Justia API error: {response.status}")
    
    async def _search_courtlistener(self, query: SearchQuery, config: Dict[str, Any]) -> APIResponse:
        """Search CourtListener API."""
        session = await self._get_session()
        
        # Build search URL
        base_url = config["base_url"]
        
        if query.search_type == SearchType.CASE_LAW:
            url = f"{base_url}/search/"
        else:
            raise ValueError(f"Unsupported search type for CourtListener: {query.search_type}")
        
        # Build parameters
        params = {
            "q": query.query,
            "type": "o",  # Opinions
            "order_by": "score desc",
            "format": "json"
        }
        
        # Add authentication if available
        headers = config["headers"].copy()
        courtlistener_token = getattr(settings, 'COURTLISTENER_TOKEN', None)
        if courtlistener_token:
            headers["Authorization"] = f"Token {courtlistener_token}"
        
        # Make request
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                results = self._parse_courtlistener_results(data)
                
                return APIResponse(
                    provider=APIProvider.COURTLISTENER.value,
                    query=query,
                    results=results,
                    total_results=data.get("count", len(results)),
                    search_time=0.0,
                    api_cost=0.0,
                    rate_limit_remaining=None,
                    success=True,
                    error_message=None
                )
            else:
                raise Exception(f"CourtListener API error: {response.status}")
    
    async def _search_caselaw_access(self, query: SearchQuery, config: Dict[str, Any]) -> APIResponse:
        """Search Caselaw Access Project API."""
        session = await self._get_session()
        
        # Build search URL
        base_url = config["base_url"]
        url = f"{base_url}/cases/"
        
        # Build parameters
        params = {
            "search": query.query,
            "full_case": "true",
            "format": "json"
        }
        
        if query.jurisdiction:
            params["jurisdiction"] = query.jurisdiction
        
        # Add API key if available
        headers = config["headers"].copy()
        caselaw_api_key = getattr(settings, 'CASELAW_API_KEY', None)
        if caselaw_api_key:
            headers["Authorization"] = f"Token {caselaw_api_key}"
        
        # Make request
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                results = self._parse_caselaw_access_results(data)
                
                return APIResponse(
                    provider=APIProvider.CASELAW_ACCESS.value,
                    query=query,
                    results=results,
                    total_results=data.get("count", len(results)),
                    search_time=0.0,
                    api_cost=0.0,
                    rate_limit_remaining=None,
                    success=True,
                    error_message=None
                )
            else:
                raise Exception(f"Caselaw Access API error: {response.status}")
    
    async def _search_westlaw(self, query: SearchQuery, config: Dict[str, Any]) -> APIResponse:
        """Search Westlaw API (placeholder implementation)."""
        # This would require actual Westlaw API credentials and implementation
        # For now, return empty results
        return APIResponse(
            provider=APIProvider.WESTLAW.value,
            query=query,
            results=[],
            total_results=0,
            search_time=0.0,
            api_cost=config["cost_per_request"],
            rate_limit_remaining=None,
            success=False,
            error_message="Westlaw API not implemented - requires subscription"
        )
    
    async def _search_lexisnexis(self, query: SearchQuery, config: Dict[str, Any]) -> APIResponse:
        """Search LexisNexis API (placeholder implementation)."""
        # This would require actual LexisNexis API credentials and implementation
        # For now, return empty results
        return APIResponse(
            provider=APIProvider.LEXISNEXIS.value,
            query=query,
            results=[],
            total_results=0,
            search_time=0.0,
            api_cost=config["cost_per_request"],
            rate_limit_remaining=None,
            success=False,
            error_message="LexisNexis API not implemented - requires subscription"
        )

    def _parse_justia_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Justia API results."""
        results = []

        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("title", ""),
                citation=item.get("citation", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                date=item.get("date"),
                court=item.get("court"),
                jurisdiction=item.get("jurisdiction"),
                relevance_score=item.get("score", 0.5),
                source="Justia",
                metadata=item
            )
            results.append(result)

        return results

    def _parse_courtlistener_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse CourtListener API results."""
        results = []

        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("caseName", ""),
                citation=item.get("citation", {}).get("neutral", ""),
                url=item.get("absolute_url", ""),
                snippet=item.get("snippet", ""),
                date=item.get("dateFiled"),
                court=item.get("court", ""),
                jurisdiction=item.get("jurisdiction"),
                relevance_score=item.get("score", 0.5),
                source="CourtListener",
                metadata=item
            )
            results.append(result)

        return results

    def _parse_caselaw_access_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Caselaw Access Project API results."""
        results = []

        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("name", ""),
                citation=item.get("citations", [{}])[0].get("cite", ""),
                url=item.get("url", ""),
                snippet=item.get("preview", ""),
                date=item.get("decision_date"),
                court=item.get("court", {}).get("name", ""),
                jurisdiction=item.get("jurisdiction", {}).get("name", ""),
                relevance_score=0.7,  # Default score
                source="Caselaw Access Project",
                metadata=item
            )
            results.append(result)

        return results

    def _generate_cache_key(self, query: SearchQuery, provider: APIProvider) -> str:
        """Generate cache key for query and provider."""
        query_str = f"{query.query}_{query.search_type.value}_{query.jurisdiction}_{provider.value}"
        return hashlib.md5(query_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[APIResponse]:
        """Get cached result if still valid."""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]

            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                return cached_data
            else:
                # Remove expired cache entry
                del self.cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, response: APIResponse):
        """Cache API response."""
        self.cache[cache_key] = (response, datetime.now().timestamp())

    def _check_rate_limit(self, provider: APIProvider) -> bool:
        """Check if API call is within rate limits."""
        config = self.api_configs.get(provider.value, {})
        rate_limit = config.get("rate_limit", 1000)

        current_time = datetime.now()

        # Initialize rate limit tracking
        if provider.value not in self.rate_limits:
            self.rate_limits[provider.value] = []

        # Clean old requests (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.rate_limits[provider.value] = [
            req_time for req_time in self.rate_limits[provider.value]
            if req_time > cutoff_time
        ]

        # Check if under rate limit
        return len(self.rate_limits[provider.value]) < rate_limit

    def _update_rate_limit(self, provider: APIProvider):
        """Update rate limit tracking after API call."""
        if provider.value not in self.rate_limits:
            self.rate_limits[provider.value] = []

        self.rate_limits[provider.value].append(datetime.now())

    def compare_results(self, responses: List[APIResponse]) -> Dict[str, Any]:
        """
        Compare results across different API providers.

        Args:
            responses (List[APIResponse]): API responses to compare

        Returns:
            Dict: Comparison analysis
        """
        if not responses:
            return {"error": "No responses to compare"}

        # Aggregate all results
        all_results = []
        provider_stats = {}

        for response in responses:
            if response.success:
                all_results.extend(response.results)
                provider_stats[response.provider] = {
                    "total_results": response.total_results,
                    "search_time": response.search_time,
                    "api_cost": response.api_cost,
                    "success": True
                }
            else:
                provider_stats[response.provider] = {
                    "total_results": 0,
                    "search_time": response.search_time,
                    "api_cost": 0.0,
                    "success": False,
                    "error": response.error_message
                }

        # Find overlapping results (same citation or similar title)
        overlaps = self._find_overlapping_results(all_results)

        # Calculate diversity metrics
        unique_courts = len(set(r.court for r in all_results if r.court))
        unique_jurisdictions = len(set(r.jurisdiction for r in all_results if r.jurisdiction))

        # Calculate average relevance scores by provider
        provider_relevance = {}
        for response in responses:
            if response.success and response.results:
                avg_relevance = sum(r.relevance_score for r in response.results) / len(response.results)
                provider_relevance[response.provider] = avg_relevance

        return {
            "total_results": len(all_results),
            "unique_results": len(all_results) - len(overlaps),
            "overlapping_results": len(overlaps),
            "provider_stats": provider_stats,
            "diversity_metrics": {
                "unique_courts": unique_courts,
                "unique_jurisdictions": unique_jurisdictions
            },
            "provider_relevance": provider_relevance,
            "overlaps": overlaps[:10]  # Show first 10 overlaps
        }

    def _find_overlapping_results(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Find overlapping results across providers."""
        overlaps = []

        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                # Check for same citation
                if (result1.citation and result2.citation and
                    result1.citation.strip() == result2.citation.strip()):

                    overlaps.append({
                        "type": "same_citation",
                        "citation": result1.citation,
                        "sources": [result1.source, result2.source],
                        "titles": [result1.title, result2.title]
                    })

                # Check for similar titles (simple similarity)
                elif self._titles_similar(result1.title, result2.title):
                    overlaps.append({
                        "type": "similar_title",
                        "title1": result1.title,
                        "title2": result2.title,
                        "sources": [result1.source, result2.source]
                    })

        return overlaps

    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar."""
        if not title1 or not title2:
            return False

        # Simple word overlap similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    def get_api_status(self) -> Dict[str, Any]:
        """
        Get status of all configured APIs.

        Returns:
            Dict: API status information
        """
        status = {
            "providers": {},
            "rate_limits": {},
            "cache_stats": {
                "total_cached": len(self.cache),
                "cache_ttl": self.cache_ttl
            }
        }

        for provider_name, config in self.api_configs.items():
            # Rate limit status
            current_requests = len(self.rate_limits.get(provider_name, []))
            rate_limit = config.get("rate_limit", 0)

            status["providers"][provider_name] = {
                "base_url": config["base_url"],
                "auth_type": config["auth_type"],
                "cost_per_request": config["cost_per_request"],
                "supported_searches": [st.value for st in config["supported_searches"]],
                "available": True  # Could add actual health checks
            }

            status["rate_limits"][provider_name] = {
                "current_requests": current_requests,
                "limit": rate_limit,
                "remaining": max(0, rate_limit - current_requests),
                "percentage_used": (current_requests / rate_limit * 100) if rate_limit > 0 else 0
            }

        return status

    def clear_cache(self):
        """Clear the API response cache."""
        self.cache.clear()
        logger.info("API cache cleared")

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    asyncio.run(self.session.close())
            except Exception:
                pass  # Ignore cleanup errors
