"""
AION Industry Templates - Science
==================================

Science-specific agent templates:
- Research Assistant: Literature search and synthesis
- Data Analysis: Scientific data processing
- Experiment Design: Methodology recommendations
- Paper Writing: Academic writing support

Auto-generated for Phase 5: Scale
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import statistics
import random


class PublicationType(Enum):
    """Types of scientific publications."""
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    BOOK_CHAPTER = "book_chapter"
    THESIS = "thesis"
    REVIEW = "review"


class ResearchField(Enum):
    """Scientific research fields."""
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    COMPUTER_SCIENCE = "computer_science"
    MEDICINE = "medicine"
    MATHEMATICS = "mathematics"
    ENGINEERING = "engineering"
    SOCIAL_SCIENCES = "social_sciences"


@dataclass
class Publication:
    """A scientific publication."""
    id: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    publication_type: PublicationType = PublicationType.JOURNAL_ARTICLE
    venue: str = ""
    year: int = 0
    doi: str = ""
    citations: int = 0
    keywords: List[str] = field(default_factory=list)
    field: ResearchField = ResearchField.COMPUTER_SCIENCE


@dataclass
class DataSet:
    """A scientific dataset."""
    name: str = ""
    description: str = ""
    columns: List[str] = field(default_factory=list)
    rows: int = 0
    data: List[List[Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchAssistant:
    """Research literature assistant."""
    
    def __init__(self):
        self.publications: Dict[str, Publication] = {}
        self.search_history: List[Dict[str, Any]] = []
    
    def add_publication(self, pub: Publication):
        """Add publication to library."""
        self.publications[pub.id] = pub
    
    def search(self, query: str, field: ResearchField = None,
               min_year: int = None, max_results: int = 10) -> List[Publication]:
        """Search publications."""
        query_terms = set(query.lower().split())
        results = []
        
        for pub in self.publications.values():
            # Apply filters
            if field and pub.field != field:
                continue
            if min_year and pub.year < min_year:
                continue
            
            # Calculate relevance
            text = (pub.title + ' ' + pub.abstract + ' ' + ' '.join(pub.keywords)).lower()
            text_terms = set(text.split())
            
            overlap = len(query_terms & text_terms)
            if overlap == 0:
                continue
            
            score = overlap / len(query_terms)
            # Boost by citations
            score += min(pub.citations / 1000, 0.5)
            
            results.append((pub, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        self.search_history.append({
            'query': query,
            'results': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
        return [r[0] for r in results[:max_results]]
    
    def synthesize_literature(self, publications: List[Publication]) -> Dict[str, Any]:
        """Synthesize findings from multiple publications."""
        if not publications:
            return {}
        
        # Extract common themes
        all_keywords = []
        for pub in publications:
            all_keywords.extend(pub.keywords)
        
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        common_themes = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Year distribution
        years = [p.year for p in publications if p.year]
        
        return {
            'publication_count': len(publications),
            'year_range': (min(years), max(years)) if years else (0, 0),
            'average_citations': statistics.mean([p.citations for p in publications]),
            'common_themes': [t[0] for t in common_themes],
            'top_authors': self._get_top_authors(publications),
            'summary': f"Analysis of {len(publications)} publications covering {len(set(all_keywords))} unique topics"
        }
    
    def _get_top_authors(self, publications: List[Publication]) -> List[str]:
        """Get most frequent authors."""
        author_freq = {}
        for pub in publications:
            for author in pub.authors:
                author_freq[author] = author_freq.get(author, 0) + 1
        
        sorted_authors = sorted(author_freq.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_authors[:5]]


class DataAnalyzer:
    """Scientific data analysis system."""
    
    def __init__(self):
        self.datasets: Dict[str, DataSet] = {}
    
    def load_dataset(self, dataset: DataSet):
        """Load a dataset for analysis."""
        self.datasets[dataset.name] = dataset
    
    def descriptive_statistics(self, dataset_name: str, 
                               column: str = None) -> Dict[str, Any]:
        """Calculate descriptive statistics."""
        dataset = self.datasets.get(dataset_name)
        if not dataset or not dataset.data:
            return {}
        
        results = {}
        
        if column:
            col_idx = dataset.columns.index(column) if column in dataset.columns else 0
            values = [row[col_idx] for row in dataset.data if isinstance(row[col_idx], (int, float))]
            
            if values:
                results[column] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
        else:
            # Analyze all numeric columns
            for i, col in enumerate(dataset.columns):
                values = [row[i] for row in dataset.data if isinstance(row[i], (int, float))]
                if values:
                    results[col] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0
                    }
        
        return results
    
    def correlation(self, dataset_name: str, col1: str, col2: str) -> float:
        """Calculate correlation between two columns."""
        dataset = self.datasets.get(dataset_name)
        if not dataset:
            return 0.0
        
        try:
            idx1 = dataset.columns.index(col1)
            idx2 = dataset.columns.index(col2)
        except ValueError:
            return 0.0
        
        values1 = [row[idx1] for row in dataset.data if isinstance(row[idx1], (int, float))]
        values2 = [row[idx2] for row in dataset.data if isinstance(row[idx2], (int, float))]
        
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        mean1, mean2 = statistics.mean(values1), statistics.mean(values2)
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
        denominator = (
            sum((x - mean1) ** 2 for x in values1) *
            sum((y - mean2) ** 2 for y in values2)
        ) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0
    
    def hypothesis_test(self, dataset_name: str, column: str,
                        null_hypothesis: float) -> Dict[str, Any]:
        """Perform simple hypothesis test."""
        dataset = self.datasets.get(dataset_name)
        if not dataset:
            return {}
        
        try:
            col_idx = dataset.columns.index(column)
        except ValueError:
            return {}
        
        values = [row[col_idx] for row in dataset.data if isinstance(row[col_idx], (int, float))]
        
        if len(values) < 2:
            return {}
        
        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / (len(values) ** 0.5)
        t_stat = (mean - null_hypothesis) / std_err if std_err != 0 else 0
        
        # Simplified p-value (approximation)
        p_value = min(1.0, 2 * (1 - min(abs(t_stat) / 3, 1)))
        
        return {
            'test': 'one-sample t-test',
            'null_hypothesis': null_hypothesis,
            'sample_mean': mean,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


@dataclass
class ExperimentDesign:
    """Experimental design specification."""
    name: str = ""
    hypothesis: str = ""
    variables: Dict[str, str] = field(default_factory=dict)  # name -> type (independent/dependent/control)
    methodology: str = ""
    sample_size: int = 0
    duration: str = ""
    analysis_plan: List[str] = field(default_factory=list)


class LiteratureReviewer:
    """Literature review generator."""
    
    def __init__(self):
        self.templates: Dict[str, str] = {
            'introduction': "This review examines {topic} across {count} studies from {year_start} to {year_end}.",
            'methodology': "We conducted a systematic search using the following criteria: {criteria}",
            'findings': "Key findings include: {findings}",
            'conclusion': "In conclusion, {conclusion}"
        }
    
    def generate_review(self, topic: str, publications: List[Publication],
                        synthesis: Dict[str, Any]) -> str:
        """Generate a literature review."""
        year_start, year_end = synthesis.get('year_range', (2020, 2024))
        
        review = f"# Literature Review: {topic}\n\n"
        
        # Introduction
        review += "## Introduction\n\n"
        review += f"This review examines {topic} across {len(publications)} studies "
        review += f"from {year_start} to {year_end}.\n\n"
        
        # Themes
        review += "## Key Themes\n\n"
        for theme in synthesis.get('common_themes', [])[:5]:
            review += f"- {theme}\n"
        review += "\n"
        
        # Key studies
        review += "## Notable Studies\n\n"
        sorted_pubs = sorted(publications, key=lambda p: p.citations, reverse=True)
        for pub in sorted_pubs[:5]:
            review += f"### {pub.title}\n"
            review += f"*{', '.join(pub.authors[:3])}* ({pub.year})\n\n"
            review += f"{pub.abstract[:200]}...\n\n"
        
        # Conclusion
        review += "## Conclusion\n\n"
        review += f"This review synthesized {len(publications)} publications, "
        review += f"averaging {synthesis.get('average_citations', 0):.1f} citations per paper. "
        review += "Further research is needed in emerging areas.\n"
        
        return review


class ScienceAgent:
    """Science-specialized AION agent."""
    
    def __init__(self, agent_id: str = "science-agent"):
        self.agent_id = agent_id
        self.research_assistant = ResearchAssistant()
        self.data_analyzer = DataAnalyzer()
        self.literature_reviewer = LiteratureReviewer()
    
    async def search_literature(self, query: str, 
                                field: ResearchField = None) -> List[Publication]:
        """Search scientific literature."""
        return self.research_assistant.search(query, field)
    
    async def analyze_data(self, dataset: DataSet) -> Dict[str, Any]:
        """Analyze scientific data."""
        self.data_analyzer.load_dataset(dataset)
        return self.data_analyzer.descriptive_statistics(dataset.name)
    
    async def generate_review(self, topic: str, 
                             publications: List[Publication]) -> str:
        """Generate literature review."""
        synthesis = self.research_assistant.synthesize_literature(publications)
        return self.literature_reviewer.generate_review(topic, publications, synthesis)
    
    async def design_experiment(self, hypothesis: str,
                               variables: Dict[str, str]) -> ExperimentDesign:
        """Design an experiment."""
        design = ExperimentDesign(
            name=f"Experiment_{datetime.now().strftime('%Y%m%d')}",
            hypothesis=hypothesis,
            variables=variables,
            methodology="Randomized controlled trial",
            sample_size=100,  # Would calculate based on power analysis
            duration="4 weeks",
            analysis_plan=[
                "Descriptive statistics",
                "Hypothesis testing (t-test/ANOVA)",
                "Effect size calculation",
                "Regression analysis"
            ]
        )
        return design
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'agent_id': self.agent_id,
            'publications_indexed': len(self.research_assistant.publications),
            'datasets_loaded': len(self.data_analyzer.datasets),
            'searches_performed': len(self.research_assistant.search_history)
        }


async def demo_science():
    """Demonstrate science template."""
    print("🔬 Science Template Demo")
    print("=" * 50)
    
    agent = ScienceAgent()
    
    # Add sample publications
    pubs = [
        Publication(
            id="pub1", title="Deep Learning for Protein Structure Prediction",
            authors=["Smith, J.", "Johnson, A.", "Williams, B."],
            abstract="We present a novel deep learning approach for predicting protein structures...",
            year=2023, citations=150,
            keywords=["deep learning", "protein folding", "bioinformatics"],
            field=ResearchField.BIOLOGY
        ),
        Publication(
            id="pub2", title="Transformer Models in Scientific Computing",
            authors=["Brown, C.", "Davis, D."],
            abstract="This paper explores the application of transformer architectures...",
            year=2024, citations=75,
            keywords=["transformer", "scientific computing", "machine learning"],
            field=ResearchField.COMPUTER_SCIENCE
        ),
    ]
    
    for pub in pubs:
        agent.research_assistant.add_publication(pub)
    
    # Search literature
    print("\n📚 Literature Search:")
    results = await agent.search_literature("deep learning protein")
    for r in results:
        print(f"  - {r.title} ({r.year}) - {r.citations} citations")
    
    # Data analysis
    dataset = DataSet(
        name="experiment_data",
        columns=["treatment", "response", "time"],
        rows=10,
        data=[
            [1, 10.5 + random.gauss(0, 2), 1],
            [1, 12.3 + random.gauss(0, 2), 2],
            [1, 14.1 + random.gauss(0, 2), 3],
            [0, 8.2 + random.gauss(0, 2), 1],
            [0, 8.5 + random.gauss(0, 2), 2],
            [0, 9.0 + random.gauss(0, 2), 3],
        ]
    )
    
    print("\n📊 Data Analysis:")
    stats = await agent.analyze_data(dataset)
    for col, data in stats.items():
        print(f"  {col}: mean={data['mean']:.2f}, std={data.get('std_dev', 0):.2f}")
    
    # Hypothesis test
    agent.data_analyzer.load_dataset(dataset)
    test_result = agent.data_analyzer.hypothesis_test("experiment_data", "response", 10.0)
    print(f"\n🧪 Hypothesis Test:")
    print(f"  H0: μ = 10.0")
    print(f"  p-value: {test_result.get('p_value', 0):.4f}")
    print(f"  Significant: {test_result.get('significant', False)}")
    
    # Experiment design
    print("\n📋 Experiment Design:")
    design = await agent.design_experiment(
        hypothesis="Treatment X increases response variable",
        variables={"treatment": "independent", "response": "dependent", "time": "control"}
    )
    print(f"  Hypothesis: {design.hypothesis}")
    print(f"  Sample Size: {design.sample_size}")
    print(f"  Analysis Plan: {', '.join(design.analysis_plan[:3])}...")
    
    print(f"\n📊 Status: {agent.get_status()}")
    print("\n✅ Science template demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_science())
