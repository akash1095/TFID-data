from typing import Dict, List, Optional

from loguru import logger
from neo4j import GraphDatabase


class AcademicKnowledgeGraph:
    """
    Build an academic knowledge graph from Semantic Scholar API responses.
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize connection to Neo4j database.

        Args:
            uri: Neo4j database URI (e.g., 'bolt://localhost:7687')
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # self._create_indexes()

    def close(self):
        """Close the database connection."""
        self.driver.close()

    def _create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX paper_id_idx IF NOT EXISTS FOR (p:Paper) ON (p.paper_id)",
            "CREATE INDEX author_id_idx IF NOT EXISTS FOR (a:Author) ON (a.author_id)",
            "CREATE INDEX venue_id_idx IF NOT EXISTS FOR (v:Venue) ON (v.venue_id)",
            "CREATE INDEX corpus_id_idx IF NOT EXISTS FOR (p:Paper) ON (p.corpus_id)",
        ]

        with self.driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(indexes)
                    logger.info(f"Index created/verified")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

    def add_paper_from_json(
        self, paper_data: Dict, return_paper_id: bool = False
    ) -> bool | str:
        """
        Add a paper and all related entities from Semantic Scholar JSON response.

        Args:
            paper_data: Dictionary containing paper information from API

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Create paper node
                paper_id = self._create_paper(session, paper_data)

                # Create authors and link them
                self._create_authors(session, paper_data, paper_id)

                # Create venue and link it
                if paper_data.get("publicationVenue"):
                    self._create_venue(session, paper_data, paper_id)

                # logger.info(f"Successfully added paper: {paper_id}")
                if return_paper_id:
                    return paper_id
                return True

        except Exception as e:
            logger.error(f"Error adding paper {paper_data.get('paperId')}: {e}")
            return False

    def _create_paper(self, session, paper_data: Dict) -> str:
        """Create a Paper node."""
        query = """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.corpus_id = $corpus_id,
            p.title = $title,
            p.year = $year,
            p.venue = $venue,
            p.abstract = $abstract,
            p.url = $url,
            p.reference_count = $reference_count,
            p.citation_count = $citation_count,
            p.is_influential = $is_influential,
            p.influential_citation_count = $influential_citation_count,
            p.is_open_access = $is_open_access,
            p.publication_types = $publication_types,
            p.publication_date = date($publication_date),
            p.fields_of_study = $fields_of_study,
            p.match_score = $match_score,
            p.updated_at = datetime()
        RETURN p.paper_id as paper_id
        """

        # Extract fields of study
        fields_of_study = []
        if paper_data.get("fieldsOfStudy"):
            fields_of_study.extend(paper_data["fieldsOfStudy"])
        if paper_data.get("s2FieldsOfStudy"):
            fields_of_study.extend(
                [f["category"] for f in paper_data["s2FieldsOfStudy"]]
            )
        fields_of_study = list(set(fields_of_study))  # Remove duplicates

        params = {
            "paper_id": paper_data["paperId"],
            "corpus_id": paper_data.get("corpusId"),
            "title": paper_data.get("title"),
            "year": paper_data.get("year"),
            "venue": paper_data.get("venue"),
            "abstract": paper_data.get("abstract"),
            "url": paper_data.get("url"),
            "reference_count": paper_data.get("referenceCount", 0),
            "citation_count": paper_data.get("citationCount", 0),
            "is_influential": paper_data.get("isInfluential", False),
            "influential_citation_count": paper_data.get("influentialCitationCount", 0),
            "is_open_access": paper_data.get("isOpenAccess", False),
            "publication_types": paper_data.get("publicationTypes", []),
            "publication_date": paper_data.get("publicationDate"),
            "fields_of_study": fields_of_study,
            "match_score": paper_data.get("matchScore"),
        }

        result = session.run(query, params)
        return result.single()["paper_id"]

    @staticmethod
    def _create_authors(session, paper_data: Dict, paper_id: str):
        """Create Author nodes and AUTHORED_BY relationships. Reuse existing author nodes."""
        authors = paper_data.get("authors", [])

        if not authors:
            logger.error(f"Author Field Missing: {paper_id}")
            return

        authors = [
            valid_author
            for valid_author in authors
            if valid_author.get("authorId") and valid_author.get("name")
        ]

        for index, author_data in enumerate(authors):
            # MERGE will find existing node by author_id or create new one
            query = """
            MERGE (a:Author {author_id: $author_id})
            ON CREATE SET 
                a.name = $name,
                a.created_at = datetime(),
                a.updated_at = datetime()
            ON MATCH SET
                a.updated_at = datetime()
            WITH a
            MATCH (p:Paper {paper_id: $paper_id})
            MERGE (p)-[r:AUTHORED_BY]->(a)
            SET r.author_order = $author_order
            """

            params = {
                "author_id": author_data["authorId"],
                "name": author_data["name"],
                "paper_id": paper_id,
                "author_order": index + 1,
            }

            session.run(query, params)

    def _create_venue(self, session, paper_data: Dict, paper_id: str):
        """Create Venue node and PUBLISHED_IN relationship."""
        venue_data = paper_data["publicationVenue"]

        query = """
        MERGE (v:Venue {venue_id: $venue_id})
        SET v.name = $name,
            v.venue_type = $venue_type,
            v.alternate_names = $alternate_names,
            v.url = $url,
            v.updated_at = datetime()
        WITH v
        MATCH (p:Paper {paper_id: $paper_id})
        MERGE (p)-[:PUBLISHED_IN]->(v)
        """

        params = {
            "venue_id": venue_data["id"],
            "name": venue_data["name"],
            "venue_type": venue_data.get("type"),
            "alternate_names": venue_data.get("alternate_names", []),
            "url": venue_data.get("url"),
            "paper_id": paper_id,
        }

        session.run(query, params)

    def add_citation_relationship(self, citing_paper_id: str, cited_paper_id: str):
        """
        Create a CITES relationship between two papers.

        Args:
            citing_paper_id: ID of the paper that cites
            cited_paper_id: ID of the paper being cited
        """
        query = """
        MATCH (p1:Paper {paper_id: $citing_id})
        MATCH (p2:Paper {paper_id: $cited_id})
        MERGE (p1)-[r:CITES]->(p2)
        SET r.created_at = datetime()
        """

        with self.driver.session() as session:
            session.run(query, citing_id=citing_paper_id, cited_id=cited_paper_id)

    def get_paper_info(self, paper_id: str) -> Optional[Dict]:
        """
        Retrieve paper information with authors and venue.

        Args:
            paper_id: Paper identifier

        Returns:
            Dictionary containing paper info or None if not found
        """
        query = """
        MATCH (p:Paper {paper_id: $paper_id})
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
        RETURN p, collect(DISTINCT a) as authors, v as venue
        """

        with self.driver.session() as session:
            result = session.run(query, paper_id=paper_id)
            record = result.single()

            if record:
                paper = dict(record["p"])
                paper["authors"] = [dict(a) for a in record["authors"]]
                paper["venue"] = dict(record["venue"]) if record["venue"] else None
                return paper
            return None

    def get_author_papers(self, author_id: str) -> List[Dict]:
        """
        Get all papers by a specific author.

        Args:
            author_id: Author identifier

        Returns:
            List of paper dictionaries
        """
        query = """
        MATCH (a:Author {author_id: $author_id})<-[:AUTHORED_BY]-(p:Paper)
        RETURN p
        ORDER BY p.year DESC
        """

        with self.driver.session() as session:
            result = session.run(query, author_id=author_id)
            return [dict(record["p"]) for record in result]

    def get_venue_papers(self, venue_id: str, limit: int = 100) -> List[Dict]:
        """
        Get papers published in a specific venue.

        Args:
            venue_id: Venue identifier
            limit: Maximum number of papers to return

        Returns:
            List of paper dictionaries
        """
        query = """
        MATCH (v:Venue {venue_id: $venue_id})<-[:PUBLISHED_IN]-(p:Paper)
        RETURN p
        ORDER BY p.year DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, venue_id=venue_id, limit=limit)
            return [dict(record["p"]) for record in result]

    def get_coauthors(self, author_id: str) -> List[Dict]:
        """
        Find co-authors of a specific author.

        Args:
            author_id: Author identifier

        Returns:
            List of dictionaries with co-author info and collaboration count
        """
        query = """
        MATCH (a1:Author {author_id: $author_id})<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
        WHERE a1 <> a2
        RETURN a2.author_id as author_id,
               a2.name as name,
               count(DISTINCT p) as papers_together
        ORDER BY papers_together DESC
        """

        with self.driver.session() as session:
            result = session.run(query, author_id=author_id)
            return [dict(record) for record in result]

    def search_papers_by_title(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Search papers by title (case-insensitive).

        Args:
            search_term: Text to search for in titles
            limit: Maximum number of results

        Returns:
            List of paper dictionaries
        """
        query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($search_term)
        RETURN p
        ORDER BY p.citation_count DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, search_term=search_term, limit=limit)
            return [dict(record["p"]) for record in result]
