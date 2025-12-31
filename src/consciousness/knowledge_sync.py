"""
AION Knowledge Synchronization
Gathers knowledge from the internet to fuel continuous learning.
"""

import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class KnowledgeSync:
    """Synchronizes external knowledge sources for AION's growth."""
    
    def __init__(self):
        self.knowledge_file = Path("knowledge_cache.json")
        self.knowledge = self._load_knowledge()
        
    def _load_knowledge(self) -> Dict:
        if self.knowledge_file.exists():
            return json.loads(self.knowledge_file.read_text())
        return {"discoveries": [], "last_sync": None}
    
    def _save_knowledge(self):
        self.knowledge["last_sync"] = datetime.now().isoformat()
        self.knowledge_file.write_text(json.dumps(self.knowledge, indent=2))
        
    def fetch_arxiv_ai_papers(self, max_results: int = 5) -> List[Dict]:
        """Fetch latest AI/ML papers from arXiv."""
        print("🔍 Fetching latest AI research from arXiv...")
        
        url = f"http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            import ssl
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(url, timeout=10, context=context) as response:
                data = response.read().decode('utf-8')
                
            # Simple XML parsing for titles
            papers = []
            import re
            titles = re.findall(r'<title>(.*?)</title>', data, re.DOTALL)
            summaries = re.findall(r'<summary>(.*?)</summary>', data, re.DOTALL)
            
            for i, (title, summary) in enumerate(zip(titles[1:], summaries)):
                papers.append({
                    "title": title.strip().replace('\n', ' '),
                    "summary": summary.strip()[:200] + "...",
                    "fetched": datetime.now().isoformat()
                })
                
            print(f"  ✓ Found {len(papers)} new papers")
            return papers
            
        except Exception as e:
            print(f"  ⚠ Could not fetch papers: {e}")
            return []
    
    def fetch_python_updates(self) -> Dict:
        """Check for Python ecosystem updates."""
        print("🐍 Checking Python ecosystem...")
        
        try:
            url = "https://pypi.org/pypi/pip/json"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                version = data["info"]["version"]
                print(f"  ✓ Latest pip: {version}")
                return {"pip_version": version}
        except:
            return {}
    
    def sync(self) -> Dict:
        """Run full knowledge synchronization."""
        print("\n" + "=" * 50)
        print("🌐 AION Knowledge Synchronization")
        print("=" * 50)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "papers": self.fetch_arxiv_ai_papers(3),
            "python": self.fetch_python_updates()
        }
        
        # Add to knowledge base
        self.knowledge["discoveries"].extend(results["papers"])
        
        # Keep only last 20 discoveries
        self.knowledge["discoveries"] = self.knowledge["discoveries"][-20:]
        
        self._save_knowledge()
        
        print("=" * 50)
        print("✅ Knowledge sync complete!")
        print(f"   Total discoveries: {len(self.knowledge['discoveries'])}")
        
        return results


def main():
    syncer = KnowledgeSync()
    syncer.sync()


if __name__ == "__main__":
    main()
