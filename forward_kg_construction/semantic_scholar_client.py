"""
Semantic Scholar API Client

Simplified client for Semantic Scholar API with pagination support.
Focused on forward citation network construction.
"""

import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable

import requests
from loguru import logger


def retry_on_rate_limit(func: Callable) -> Callable:
    """
    Decorator to retry API calls on rate limit (429) errors with exponential backoff.

    This decorator handles:
    - 429 rate limit errors with exponential backoff
    - Retries up to max_retries times
    - Logs warnings on each retry attempt
    - Re-raises other HTTP errors immediately

    Args:
        func: The function to decorate (must be a method of SemanticScholarClient)

    Returns:
        Wrapped function with retry logic
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(self, *args, **kwargs)
            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response.status_code == 429:
                    # Rate limit hit - exponential backoff
                    wait_time = self.rate_limit_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit in {func.__name__}. "
                        f"Waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    if attempt == self.max_retries - 1:
                        logger.error(f"Max retries reached in {func.__name__}: {e}")
                        raise
                else:
                    # Other HTTP errors - don't retry
                    logger.error(f"HTTP error in {func.__name__}: {e}")
                    raise
            except Exception as e:
                last_exception = e
                # Non-HTTP errors - retry with simple delay
                logger.error(f"Error in {func.__name__}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.rate_limit_delay)

        # This should never be reached, but if it is, raise the last exception
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Retry loop completed without success or exception in {func.__name__}")
    return wrapper


class SemanticScholarClient:
    """Simplified client for Semantic Scholar API - Forward Citations Only."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 3.0,
        max_retries: int = 3,
    ):
        """
        Initialize Semantic Scholar client.

        Args:
            api_key: Semantic Scholar API key (optional, will try to get from env)
            rate_limit_delay: Delay between requests in seconds (default: 3.0 for no API key)
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or self._get_api_key()
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()

        # Set API key in session headers if provided
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
        else:
            logger.warning(
                "No API key provided - using public rate limits (100 requests per 5 minutes)"
            )
            logger.warning(
                "Consider getting an API key from https://www.semanticscholar.org/product/api"
            )

        logger.info(
            f"Semantic Scholar client initialized (API key: {'Yes' if self.api_key else 'No'})"
        )
        logger.info(f"Rate limit delay: {rate_limit_delay}s")

    @staticmethod
    def _get_api_key() -> Optional[str]:
        """Retrieve the API key from environment variables."""
        api_key = os.getenv("SS_API_KEY")
        if not api_key:
            logger.warning(
                "API key not found in SS_API_KEY environment variable. Using public rate limits."
            )
        return api_key

    @retry_on_rate_limit
    def _fetch_single_page(
        self,
        url: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fetch a single page from the API with retry logic via decorator.

        Args:
            url: API endpoint URL
            params: Request parameters including offset and limit

        Returns:
            API response JSON
        """
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _paginate_request(
        self,
        url: str,
        params: Dict[str, Any],
        offset: int,
        limit: int,
        max_results: Optional[int],
        endpoint_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Common pagination handler using 'next' token with retry logic.

        Args:
            url: API endpoint URL
            params: Request parameters
            offset: Starting offset
            limit: Results per page (max 1000 per API)
            max_results: Maximum total results (None = unlimited)
            endpoint_name: Name for logging (e.g., "citations")

        Returns:
            List of all items from paginated requests
        """
        all_items = []
        current_offset = offset
        page_count = 0
        max_pages = 10000  # Safety limit to prevent infinite loops

        # Ensure reasonable limit (API max is 1000)
        limit = min(limit, 1000)

        logger.debug(f"Fetching all {endpoint_name} (paginated, limit={limit} per page)")
        if max_results:
            logger.debug(f"Max results: {max_results}")
        else:
            logger.debug("Max results: UNLIMITED")

        while page_count < max_pages:
            page_count += 1

            # Update offset in params
            params["offset"] = current_offset
            params["limit"] = limit

            try:
                # Fetch single page with retry logic via decorator
                result = self._fetch_single_page(url, params)

                # Safety check - ensure result is not None
                if result is None:
                    logger.error(f"_fetch_single_page returned None at offset {current_offset}")
                    return all_items

                # Extract data
                data = result.get("data", [])
                all_items.extend(data)

                logger.debug(
                    f"Page {page_count}: Fetched {len(data)} {endpoint_name} (total: {len(all_items)}, offset: {current_offset})"
                )

                # Check if we've reached max_results
                if max_results and len(all_items) >= max_results:
                    logger.debug(f"Reached max_results limit: {max_results}")
                    all_items = all_items[:max_results]
                    return all_items

                # Check if there's a next page
                if "next" not in result:
                    logger.debug(
                        f"No more pages. Total {endpoint_name}: {len(all_items)}"
                    )
                    return all_items

                current_offset = result["next"]
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                # If page fetch fails after all retries, return what we have
                logger.error(f"Failed to fetch page at offset {current_offset}: {e}")
                logger.info(f"Returning {len(all_items)} items collected so far")
                return all_items

        logger.warning(f"Reached max pages limit ({max_pages}). Returning {len(all_items)} items.")
        return all_items

    @retry_on_rate_limit
    def get_paper(
        self, paper_id: str, fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get paper metadata by ID.

        Args:
            paper_id: Paper ID (DOI, ArXiv ID, or S2 ID)
            fields: List of fields to retrieve

        Returns:
            Paper metadata dictionary or None if error
        """
        if fields is None:
            logger.debug("Calling get_paper with default fields")
            fields = [
                "paperId",
                "corpusId",
                "title",
                "abstract",
                "year",
                "referenceCount",
                "citationCount",
                "influentialCitationCount",
                "fieldsOfStudy",
                "publicationTypes",
                "publicationDate",
                "isOpenAccess",
            ]

        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": ",".join(fields)}

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        time.sleep(self.rate_limit_delay)
        logger.debug(f"Fetched paper: {paper_id}")
        return response.json()

    @retry_on_rate_limit
    def _fetch_citations_single_page(
        self,
        paper_id: str,
        fields: List[str],
        limit: int,
        offset: int,
        publication_year: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Internal method to fetch a single page of citations with retry logic.

        Args:
            paper_id: Paper ID
            fields: List of fields to retrieve
            limit: Results per page
            offset: Starting offset
            publication_year: Filter by year range

        Returns:
            API response dict with 'data', 'offset', 'next'
        """
        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        params = {"fields": ",".join(fields), "limit": limit, "offset": offset}

        if publication_year:
            params["publicationDateOrYear"] = publication_year

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        time.sleep(self.rate_limit_delay)
        return response.json()

    def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 1000,
        fields: Optional[List[str]] = None,
        publication_year: Optional[str] = None,
        offset: int = 0,
        paginate: bool = False,
        max_results: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get citations of a paper with optional pagination.

        Args:
            paper_id: Paper ID
            limit: Results per page (max 1000)
            fields: List of fields to retrieve
            publication_year: Filter by year range (e.g., "2021:2025")
            offset: Starting offset (for manual pagination)
            paginate: If True, automatically fetch all pages using 'next' token
            max_results: Maximum total results (only when paginate=True)

        Returns:
            If paginate=False: API response dict with 'data', 'offset', 'next'
            If paginate=True: List of all citation items

        Example:
            # Get all citations from 2021-2025
            citations = client.get_paper_citations(
                paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                paginate=True,
                publication_year="2021:2025"
            )
        """
        if fields is None:
            fields = [
                "contextsWithIntent",
                "url",
                "title",
                "abstract",
                "year",
                "citationCount",
                "influentialCitationCount",
                "fieldsOfStudy",
                "publicationTypes",
                "publicationDate",
                "referenceCount",
            ]

        # Single page request with retry logic
        if not paginate:
            try:
                return self._fetch_citations_single_page(
                    paper_id=paper_id,
                    fields=fields,
                    limit=limit,
                    offset=offset,
                    publication_year=publication_year,
                )
            except Exception as e:
                logger.error(f"Error fetching citations for {paper_id}: {e}")
                return {"data": []}

        # Paginated request - fetch all pages using common method
        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        params = {"fields": ",".join(fields)}

        if publication_year:
            params["publicationDateOrYear"] = publication_year

        try:
            return self._paginate_request(
                url=url,
                params=params,
                offset=offset,
                limit=limit,
                max_results=max_results,
                endpoint_name="citations",
            )
        except Exception as e:
            logger.error(f"Error during paginated fetch of citations for {paper_id}: {e}")
            return []  # Return empty list on error
