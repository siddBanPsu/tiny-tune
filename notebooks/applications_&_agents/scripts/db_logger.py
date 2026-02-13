"""
SQLite database logger for LinkedIn Agent runs
Stores the complete final state with timestamp for tracking and analysis
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add a stream handler for this module if none exists
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_handler)
# Prevent double logging via propagation to root logger
logger.propagate = False
logger.setLevel(logging.INFO)

class AgentRunLogger:
    """Logger for storing agent run results in SQLite database"""
    
    def __init__(self, db_path: str = "agent_runs.db"):
        """
        Initialize the logger with database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                linkedin_post TEXT,
                linkedin_post_url TEXT,
                search_query TEXT,
                iteration_count INTEGER,
                scraper_thoughts TEXT,
                error_message TEXT,
                full_state JSON
            )
        """)
        
        # Articles table for better querying
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraped_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                article_title TEXT,
                article_date TEXT,
                article_url TEXT,
                article_summary TEXT,
                is_selected BOOLEAN DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES agent_runs(id)
            )
        """)
        
        # Retrieved data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieved_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                summary TEXT,
                source_url TEXT,
                source_content TEXT,
                FOREIGN KEY (run_id) REFERENCES agent_runs(id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp 
            ON agent_runs(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_status 
            ON agent_runs(status)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def log_run(self, final_state: Dict[str, Any], status: str = "success", 
                error_message: Optional[str] = None) -> int:
        """
        Log a complete agent run to the database
        
        Args:
            final_state: The complete final state from the agent
            status: Status of the run ("success", "error", "partial")
            error_message: Optional error message if status is "error"
        
        Returns:
            run_id: The ID of the inserted run record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract key fields
            linkedin_post = None
            linkedin_post_url = None
            
            if final_state.get("linkedin_post"):
                lp = final_state["linkedin_post"]
                if hasattr(lp, "post"):
                    linkedin_post = lp.post
                    linkedin_post_url = str(lp.url) if hasattr(lp, "url") else None
                elif isinstance(lp, str):
                    linkedin_post = lp
            
            # Serialize the full state to JSON
            full_state_json = json.dumps(self._serialize_state(final_state), 
                                        indent=2, default=str)
            
            # Insert main run record
            cursor.execute("""
                INSERT INTO agent_runs 
                (status, linkedin_post, linkedin_post_url, search_query, 
                 iteration_count, scraper_thoughts, error_message, full_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                status,
                linkedin_post,
                linkedin_post_url,
                final_state.get("search_query", ""),
                final_state.get("iteration_count", 0),
                final_state.get("scraper_thoughts", ""),
                error_message,
                full_state_json
            ))
            
            run_id = cursor.lastrowid
            
            # Log articles
            self._log_articles(cursor, run_id, final_state)
            
            # Log retrieved data
            self._log_retrieved_data(cursor, run_id, final_state)
            
            conn.commit()
            logger.info(f"Logged run {run_id} with status: {status}")
            return run_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error logging run to database: {e}")
            raise
        finally:
            conn.close()
    
    def _log_articles(self, cursor, run_id: int, final_state: Dict[str, Any]):
        """Log scraped articles to the database"""
        # Log raw headlines
        for article in final_state.get("raw_headlines", []):
            cursor.execute("""
                INSERT INTO scraped_articles 
                (run_id, article_title, article_date, article_url, article_summary, is_selected)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (
                run_id,
                article.get("title") or article.get("news_title", ""),
                article.get("date", ""),
                article.get("url", ""),
                article.get("summary") or article.get("description", "")
            ))
        
        # Mark selected stories
        top_stories = final_state.get("top_stories", [])
        if hasattr(top_stories, "artifact"):
            # Handle ToolMessage artifact format
            for item in top_stories.artifact:
                if isinstance(item, tuple) and len(item) >= 2:
                    for article in item[1]:
                        if hasattr(article, "news_title"):
                            cursor.execute("""
                                INSERT INTO scraped_articles 
                                (run_id, article_title, article_date, 
                                 article_summary, is_selected)
                                VALUES (?, ?, ?, ?, 1)
                            """, (
                                run_id,
                                article.news_title,
                                article.date,
                                # article.url,
                                article.description,
                            ))
        elif isinstance(top_stories, list):
            for article in top_stories:
                cursor.execute("""
                    INSERT INTO scraped_articles 
                    (run_id, article_title, article_date, article_url, 
                     article_summary, is_selected)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (
                    run_id,
                    article.get("title") or getattr(article, "news_title", ""),
                    article.get("date", ""),
                    article.get("url", ""),
                    article.get("summary") or getattr(article, "description", "")
                ))
    
    def _log_retrieved_data(self, cursor, run_id: int, final_state: Dict[str, Any]):
        """Log retrieved web data to the database"""
        retrieved_data = final_state.get("retrieved_data")
        if retrieved_data and hasattr(retrieved_data, "links_w_contents"):
            summary = getattr(retrieved_data, "summary", "")
            for url, content in retrieved_data.links_w_contents:
                cursor.execute("""
                    INSERT INTO retrieved_data 
                    (run_id, summary, source_url, source_content)
                    VALUES (?, ?, ?, ?)
                """, (
                    run_id,
                    summary,
                    str(url),
                    content
                ))
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to JSON-serializable format"""
        serialized = {}
        for key, value in state.items():
            if hasattr(value, "model_dump"):
                # Pydantic model
                serialized[key] = value.model_dump()
            elif hasattr(value, "__dict__"):
                # Object with __dict__
                serialized[key] = vars(value)
            elif isinstance(value, list):
                serialized[key] = [
                    v.model_dump() if hasattr(v, "model_dump") 
                    else vars(v) if hasattr(v, "__dict__") 
                    else v 
                    for v in value
                ]
            else:
                serialized[key] = value
        return serialized
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent agent runs
        
        Args:
            limit: Number of runs to retrieve
        
        Returns:
            List of run records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, status, linkedin_post, linkedin_post_url,
                   search_query, iteration_count, error_message
            FROM agent_runs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_run_details(self, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Get complete details for a specific run
        
        Args:
            run_id: The run ID to retrieve
        
        Returns:
            Complete run details including full state
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM agent_runs WHERE id = ?
        """, (run_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        result = dict(row)
        
        # Get associated articles
        cursor.execute("""
            SELECT * FROM scraped_articles WHERE run_id = ?
        """, (run_id,))
        result["articles"] = [dict(r) for r in cursor.fetchall()]
        
        # Get retrieved data
        cursor.execute("""
            SELECT * FROM retrieved_data WHERE run_id = ?
        """, (run_id,))
        result["retrieved_sources"] = [dict(r) for r in cursor.fetchall()]
        
        conn.close()
        return result
    
    def get_successful_posts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all successful LinkedIn posts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, linkedin_post, linkedin_post_url
            FROM agent_runs
            WHERE status = 'success' AND linkedin_post IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def export_to_json(self, output_path: str, run_id: Optional[int] = None):
        """
        Export runs to JSON file
        
        Args:
            output_path: Path to output JSON file
            run_id: Optional specific run ID to export, or None for all runs
        """
        if run_id:
            data = self.get_run_details(run_id)
        else:
            data = self.get_recent_runs(limit=1000)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported data to {output_path}")
