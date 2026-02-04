import os
from typing import Any, Dict, Optional

import requests
from loguru import logger


class SemanticScholarClient:
    """Client for interacting with Semantic Scholar API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": self.api_key}

    @staticmethod
    def _get_api_key() -> str:
        """Retrieve the API key from environment variables."""
        api_key = os.getenv("SS_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Please set the SS_API_KEY environment variable."
            )
        return api_key

    def get_paper_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch paper JSON from Semantic Scholar API based on the paper title."""
        try:
            url = f"{self.base_url}/paper/search/match"
            query_params = {
                "query": title,
                "fields": "paperId,corpusId,url,title,abstract,venue,publicationVenue,year,"
                "referenceCount,citationCount,influentialCitationCount,isOpenAccess,"
                "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,"
                "publicationDate,journal,authors",
            }
            response = requests.get(
                url, params=query_params, headers=self.headers
            ).json()

            if "error" in response:
                print(f"API Error for query '{title}': {response['error']}")
                return None

            if not response.get("data"):
                return None

            return response["data"][0]
        except Exception as e:
            print(f"Error fetching paper JSON for query '{title}': {e}")
            return None

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetch paper JSON from Semantic Scholar API based on the paper ID."""
        try:
            url = f"{self.base_url}/paper/{paper_id}"
            query_params = {
                "fields": "paperId,corpusId,url,title,abstract,venue,publicationVenue,year,"
                "referenceCount,citationCount,influentialCitationCount,isOpenAccess,"
                "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,"
                "publicationDate,journal,authors",
            }
            response = requests.get(
                url, params=query_params, headers=self.headers
            ).json()

            if "error" in response:
                print(f"API Error for paper ID '{paper_id}': {response['error']}")
                return None

            return response
        except Exception as e:
            print(f"Error fetching paper JSON for paper ID '{paper_id}': {e}")
            return None

    def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
        publication_year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch papers that cite the given paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to fetch (max 1000)
            offset: Pagination offset
            publication_year: Filter references by publication year, e.g., 2022:2023

        Returns:
            Dictionary containing citation data with structure:
            {
                "data": [
                    {
                        "citingPaper": {
                            "paperId": "...",
                            "title": "...",
                            "year": 2023,
                            ...
                        }
                    }
                ],
                "offset": 0,
                "next": 100
            }
            Returns None if error occurs.
        """
        try:
            url = f"{self.base_url}/paper/{paper_id}/citations"
            query_params = {
                "fields": "paperId,corpusId,url,title,abstract,venue,publicationVenue,year,"
                "referenceCount,citationCount,isInfluential,influentialCitationCount,isOpenAccess,"
                "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,"
                "publicationDate,journal,authors",
                "publicationDateOrYear": publication_year,
                "limit": min(limit, 1000),  # API max is 1000
                "offset": offset,
            }
            response = requests.get(
                url, params=query_params, headers=self.headers
            ).json()

            if "error" in response:
                print(
                    f"API Error fetching citations for paper '{paper_id}': {response['error']}"
                )
                return None

            return response
        except Exception as e:
            print(f"Error fetching citations for paper '{paper_id}': {e}")
            return None

    def get_paper_references(
        self,
        paper_id: str,
        limit: int = 500,
        offset: int = 0,
        publication_year: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch papers that cite the given paper.
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to fetch (max 1000)
            offset: Pagination offset
            publication_year: Filter references by publication year, e.g., 2022:2023


        """
        try:
            url = f"{self.base_url}/paper/{paper_id}/references"
            query_params = {
                "fields": "paperId,corpusId,url,title,abstract,venue,publicationVenue,year,"
                "referenceCount,citationCount,influentialCitationCount,isOpenAccess,"
                "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,"
                "publicationDate,journal,authors",
                "publicationDateOrYear": publication_year,
                "limit": min(limit, 1000),  # API max is 1000
                "offset": offset,
            }
            response = requests.get(
                url, params=query_params, headers=self.headers
            ).json()

            if "error" in response:
                logger.error(
                    f"API Error fetching citations for paper '{paper_id}': {response['error']}"
                )
                return None

            return response

        except Exception as e:
            print(f"Error fetching citations for paper '{paper_id}': {e}")
            return None
