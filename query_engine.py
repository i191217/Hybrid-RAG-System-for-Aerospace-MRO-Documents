#!/usr/bin/env python3
"""
Advanced Query Engine for Aerospace MRO RAG System
Provides intelligent querying with aerospace domain expertise.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os

from config import config
from vector_db import get_vector_db
from embedding_service import get_embedding_service

logger = logging.getLogger("doc_processor.query_engine")

@dataclass
class QueryResult:
    """Represents a query result with source information."""
    answer: str
    sources: List[Dict[str, Any]]
    query_time: float
    retrieval_stats: Dict[str, Any]
    confidence_score: Optional[float] = None

@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with metadata."""
    content: str
    filename: str
    chunk_index: int
    similarity_score: float
    metadata: Dict[str, Any]

class AerospaceQueryEngine:
    """Advanced query engine specialized for aerospace MRO domain."""
    
    AEROSPACE_SYSTEM_PROMPT = """You are an expert aerospace engineer and MRO (Maintenance, Repair, and Operations) specialist with deep knowledge in:

🛩️ **Aerospace Engineering Expertise:**
- Aircraft systems, components, and maintenance procedures
- Regulatory compliance (FAA, EASA, Transport Canada)
- Airworthiness directives and service bulletins
- Parts certification and traceability
- Maintenance planning and scheduling
- Safety management systems (SMS)

🔧 **MRO Operations Expertise:**
- Preventive and corrective maintenance procedures
- Component repair and overhaul processes
- Supply chain and inventory management
- Quality assurance and control procedures
- Technical documentation and record keeping
- Cost analysis and budget planning

📋 **Document Analysis Skills:**
- Technical manual interpretation
- Maintenance procedure analysis
- Regulatory document review
- Financial report analysis
- Investment memo evaluation
- Board presentation insights

**Your Role:**
Provide accurate, detailed, and actionable answers based on the retrieved document chunks. Always:
1. Ground your responses in the provided context
2. Cite specific sources when making claims
3. Acknowledge if information is incomplete or requires additional context
4. Provide practical, implementable recommendations
5. Highlight safety-critical considerations when relevant
6. Consider regulatory and compliance implications

**Response Format:**
- Lead with a clear, direct answer
- Support with evidence from the documents
- Include relevant technical details
- Suggest next steps or considerations when appropriate
- Cite sources clearly (filename and relevant sections)

Remember: Safety and regulatory compliance are paramount in aerospace operations."""

    def __init__(self):
        """Initialize the query engine with all necessary components."""
        self.logger = logging.getLogger("doc_processor.query_engine.AerospaceQueryEngine")
        
        try:
            # Initialize components
            self.vector_db = get_vector_db()
            self.embedding_service = get_embedding_service()
            
            # Initialize OpenAI client for response generation
            self._initialize_openai()
            
            # Query statistics
            self.stats = {
                "total_queries": 0,
                "total_query_time": 0.0,
                "total_retrieval_time": 0.0,
                "total_generation_time": 0.0,
                "average_chunks_retrieved": 0.0
            }
            
            self.logger.info("Aerospace Query Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize query engine: {e}")
            raise
    
    def _initialize_openai(self):
        """Initialize OpenAI client for response generation."""
        try:
            import openai
            
            openai_config = config.get_openai_config()
            if not openai_config.get("api_key"):
                raise ValueError("OpenAI API key not configured")
            
            self.openai_client = openai.OpenAI(
                api_key=openai_config["api_key"]
            )
            
            self.logger.info("OpenAI client initialized")
            
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def query(self, 
              query_text: str, 
              max_chunks: int = 5,
              temperature: float = 0.6,
              model: str = "gpt-4") -> QueryResult:
        """
        Execute a query against the aerospace document collection.
        
        Args:
            query_text: The user's question or query
            max_chunks: Maximum number of chunks to retrieve
            temperature: OpenAI temperature for response generation
            model: OpenAI model to use for response generation
            
        Returns:
            QueryResult with answer and source information
        """
        start_time = time.time()
        self.logger.info(f"Processing query: {query_text[:100]}...")
        
        try:
            # Step 1: Use LLM to intelligently determine query intent and strategy
            query_analysis = self._analyze_query_intent(query_text, model, temperature)
            
            # Step 2: Execute appropriate response strategy based on analysis
            if query_analysis["needs_document_search"]:
                return self._handle_document_search_query(query_text, query_analysis, max_chunks, temperature, model, start_time)
            else:
                return self._handle_conversational_query(query_text, query_analysis, temperature, model, start_time)
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return QueryResult(
                answer=f"I encountered an error while processing your query: {str(e)}. Please try again or contact support.",
                sources=[],
                query_time=time.time() - start_time,
                retrieval_stats={"error": str(e)},
                confidence_score=0.1
            )
    
    def _analyze_query_intent(self, query_text: str, model: str, temperature: float) -> Dict[str, Any]:
        """Use LLM to analyze query intent and determine response strategy."""
        try:
            analysis_prompt = """You are an intelligent query analyzer for an aerospace MRO chatbot system. Analyze the user's query and determine the best response strategy.

Your task is to determine:
1. Whether this query needs to search through technical documents
2. What type of response would best serve the user
3. The confidence level of your analysis

Guidelines:
- Simple greetings, casual conversation, and general questions about the bot's capabilities = NO document search needed
- Technical aerospace questions, specific procedural inquiries, financial data requests = YES document search needed
- Questions about specific documents, maintenance procedures, compliance, costs, etc. = YES document search needed

Respond ONLY with a JSON object in this exact format:
{
    "needs_document_search": true/false,
    "query_type": "greeting|general_conversation|aerospace_technical|aerospace_general|document_specific",
    "reasoning": "brief explanation of your decision",
    "suggested_approach": "how to best respond to this query",
    "confidence": 0.0-1.0
}

User Query: """ + query_text

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=200
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content.strip())
                self.logger.debug(f"Query analysis: {analysis}")
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                self.logger.warning("Failed to parse query analysis JSON, using fallback")
                return {
                    "needs_document_search": len(query_text.split()) > 3,  # Simple fallback
                    "query_type": "unknown",
                    "reasoning": "JSON parsing failed, using simple heuristic",
                    "suggested_approach": "treat as potential document search",
                    "confidence": 0.5
                }
                
        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            # Conservative fallback - assume document search might be needed
            return {
                "needs_document_search": True,
                "query_type": "unknown",
                "reasoning": f"Analysis failed: {str(e)}",
                "suggested_approach": "default to document search",
                "confidence": 0.3
            }
    
    def _handle_conversational_query(self, query_text: str, query_analysis: Dict[str, Any], temperature: float, model: str, start_time: float) -> QueryResult:
        """Handle conversational queries without document search."""
        try:
            conversational_prompt = """You are a friendly and professional aerospace MRO assistant. The user's query doesn't require searching through technical documents.

Based on the query analysis, this appears to be a: """ + query_analysis.get("query_type", "general conversation") + """

Your role:
- Be warm, professional, and helpful
- Explain your capabilities as an aerospace MRO specialist
- Offer to help with specific aerospace topics if appropriate
- Keep responses natural and conversational
- If asked about your capabilities, mention you can search through aerospace documents for technical questions

Provide a natural, helpful response to the user."""

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": conversational_prompt},
                    {"role": "user", "content": query_text}
                ],
                temperature=temperature,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content
            total_time = time.time() - start_time
            
            # Lower confidence for conversational responses - they don't need to be 100%
            confidence = min(0.85, query_analysis.get("confidence", 0.7))
            
            return QueryResult(
                answer=answer,
                sources=[],
                query_time=total_time,
                retrieval_stats={
                    "chunks_found": 0,
                    "retrieval_time": 0.0,
                    "generation_time": total_time,
                    "query_type": query_analysis.get("query_type", "conversational"),
                    "analysis_confidence": query_analysis.get("confidence", 0.8)
                },
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Conversational response failed: {e}")
            # Simple fallback response
            return QueryResult(
                answer="Hello! I'm your aerospace MRO assistant. I can help you with questions about aircraft maintenance, repair procedures, compliance requirements, and operational documentation. How can I assist you today?",
                sources=[],
                query_time=time.time() - start_time,
                retrieval_stats={"query_type": "fallback_conversational"},
                confidence_score=0.8
            )
    
    def _handle_document_search_query(self, query_text: str, query_analysis: Dict[str, Any], max_chunks: int, temperature: float, model: str, start_time: float) -> QueryResult:
        """Handle queries that need document search."""
        try:
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            retrieved_chunks = self._retrieve_chunks(query_text, max_chunks)
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_chunks:
                # No documents found - provide intelligent response based on query analysis
                return self._handle_no_documents_found(query_text, query_analysis, start_time, retrieval_time, temperature, model)
            
            # Step 2: Generate response using retrieved chunks
            generation_start = time.time()
            answer, confidence = self._generate_response(query_text, retrieved_chunks, temperature, model)
            generation_time = time.time() - generation_start
            
            # Step 3: Prepare source information
            sources = self._prepare_sources(retrieved_chunks)
            
            # Step 4: Update statistics
            total_time = time.time() - start_time
            self._update_stats(total_time, retrieval_time, generation_time, len(retrieved_chunks))
            
            self.logger.info(f"Document search query completed in {total_time:.2f}s: {len(retrieved_chunks)} chunks, confidence: {confidence}")
            
            return QueryResult(
                answer=answer,
                sources=sources,
                query_time=total_time,
                retrieval_stats={
                    "chunks_found": len(retrieved_chunks),
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "max_similarity": max(chunk.similarity_score for chunk in retrieved_chunks) if retrieved_chunks else 0.0,
                    "min_similarity": min(chunk.similarity_score for chunk in retrieved_chunks) if retrieved_chunks else 0.0,
                    "avg_similarity": sum(chunk.similarity_score for chunk in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0,
                    "query_type": query_analysis.get("query_type", "document_search"),
                    "analysis_confidence": query_analysis.get("confidence", 0.8)
                },
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Document search query failed: {e}")
            return QueryResult(
                answer=f"I encountered an error while searching the documents: {str(e)}. Please try rephrasing your question.",
                sources=[],
                query_time=time.time() - start_time,
                retrieval_stats={"error": str(e), "query_type": "failed_document_search"},
                confidence_score=0.1
            )
    
    def _handle_no_documents_found(self, query_text: str, query_analysis: Dict[str, Any], start_time: float, retrieval_time: float, temperature: float, model: str) -> QueryResult:
        """Handle aerospace queries when no relevant documents are found."""
        try:
            no_docs_prompt = f"""You are an aerospace MRO expert. The user asked a question that seemed to require searching through technical documents, but no relevant documents were found in the database.

Query Analysis: {query_analysis.get('reasoning', 'Technical question detected')}
Query Type: {query_analysis.get('query_type', 'aerospace_technical')}

The user asked: "{query_text}"

Provide a helpful response that:
1. Acknowledges their specific question
2. Explains that no relevant documents were found in the current database
3. Provides general aerospace knowledge if you can help with the topic
4. Suggests practical next steps (adding documents, rephrasing, etc.)
5. Offers to help with other aerospace topics

Be professional, knowledgeable, and helpful."""

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": no_docs_prompt}],
                temperature=temperature,
                max_tokens=400
            )
            
            answer = response.choices[0].message.content
            
            return QueryResult(
                answer=answer,
                sources=[],
                query_time=time.time() - start_time,
                retrieval_stats={
                    "chunks_found": 0,
                    "retrieval_time": retrieval_time,
                    "query_type": query_analysis.get("query_type", "aerospace_no_docs"),
                    "analysis_confidence": query_analysis.get("confidence", 0.6)
                },
                confidence_score=0.3  # Lowered from 0.6 - we couldn't find documents so confidence should be low
            )
            
        except Exception as e:
            self.logger.error(f"No documents response generation failed: {e}")
            return QueryResult(
                answer="I understand you're asking about aerospace topics, but I couldn't find relevant information in the current document database. This could mean the specific documents you need haven't been processed yet, or you might need to rephrase your question. I'm here to help with aircraft maintenance, repair procedures, compliance requirements, and operational documentation. Could you try rephrasing your question or let me know what specific aerospace area you'd like to explore?",
                sources=[],
                query_time=time.time() - start_time,
                retrieval_stats={
                    "chunks_found": 0,
                    "retrieval_time": retrieval_time,
                    "query_type": "aerospace_fallback",
                    "analysis_confidence": 0.2
                },
                confidence_score=0.2
            )
    
    def _retrieve_chunks(self, query_text: str, max_chunks: int) -> List[RetrievedChunk]:
        """Retrieve relevant chunks and save analysis to file."""
        import time
        import os
        
        try:
            # Create analysis file path
            analysis_file = "retrieval_analysis.txt"
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                # === QUERY ANALYSIS SECTION ===
                f.write("=" * 80 + "\n")
                f.write("🔍 RETRIEVAL ANALYSIS START\n")
                f.write(f"QUERY: '{query_text}'\n")
                f.write(f"QUERY STATS: {len(query_text)} chars, {len(query_text.split())} words\n")
                f.write(f"PARAMETERS: max_chunks={max_chunks}\n")
                
                # === EMBEDDING GENERATION ===
                embedding_start = time.time()
                query_embedding = self.embedding_service.embed_text(query_text)
                embedding_time = time.time() - embedding_start
                
                if not query_embedding:
                    f.write("❌ EMBEDDING FAILED\n")
                    return []
                
                f.write(f"✅ EMBEDDING: {embedding_time:.3f}s, {len(query_embedding)} dims\n")
                f.write(f"EMBEDDING SAMPLE: [{query_embedding[0]:.6f}, {query_embedding[1]:.6f}, ..., {query_embedding[-1]:.6f}]\n")
                
                # === VECTOR SEARCH ===
                search_start = time.time()
                search_results = self.vector_db.search(
                    query_vector=query_embedding,
                    limit=max_chunks * 3,  # Get more for analysis
                    score_threshold=0.0    # Get all results
                )
                search_time = time.time() - search_start
                
                f.write(f"🔍 SEARCH: {search_time:.3f}s, {len(search_results)} total results\n")
                
                # === ALL RESULTS ANALYSIS ===
                f.write("-" * 60 + "\n")
                f.write("📊 ALL SEARCH RESULTS:\n")
                for i, result in enumerate(search_results):
                    source = result.metadata.get("source", "Unknown")
                    chunk_idx = result.metadata.get("chunk_index", 0)
                    content_preview = result.metadata.get("content", "")[:80].replace('\n', ' ')
                    
                    f.write(f"  {i+1:2d}. {source} (chunk {chunk_idx}) | {result.score:.6f}\n")
                    f.write(f"      📝 {content_preview}...\n")
                
                # === DETAILED CHUNK ANALYSIS ===
                retrieved_chunks = []
                f.write("-" * 60 + "\n")
                f.write("🔬 DETAILED CHUNK ANALYSIS:\n")
                
                # Take all chunks up to max_chunks (no threshold filtering)
                for i, result in enumerate(search_results[:max_chunks]):
                    chunk = RetrievedChunk(
                        content=result.metadata.get("content", ""),
                        filename=result.metadata.get("source", "Unknown"),
                        chunk_index=result.metadata.get("chunk_index", 0),
                        similarity_score=result.score,
                        metadata=result.metadata
                    )
                    retrieved_chunks.append(chunk)
                    
                    f.write(f"📄 CHUNK {len(retrieved_chunks)}: {chunk.filename} (Section {chunk.chunk_index})\n")
                    f.write(f"   📊 SIMILARITY: {chunk.similarity_score:.6f}\n")
                    f.write(f"   📏 LENGTH: {len(chunk.content)} characters\n")
                    
                    # === CONTENT ANALYSIS ===
                    f.write(f"   📝 FULL CONTENT:\n")
                    content_lines = chunk.content.split('\n')
                    for line_num, line in enumerate(content_lines[:15], 1):  # Show more lines
                        if line.strip():
                            f.write(f"      {line_num:2d}: {line.strip()}\n")
                    if len(content_lines) > 15:
                        f.write(f"      ... ({len(content_lines) - 15} more lines)\n")
                    
                    # === RELEVANCE ANALYSIS ===
                    query_words = set(query_text.lower().split())
                    content_words = set(chunk.content.lower().split())
                    common_words = query_words.intersection(content_words)
                    
                    f.write(f"   🎯 WORD OVERLAP: {len(common_words)}/{len(query_words)} query words found\n")
                    if common_words:
                        f.write(f"      Common words: {sorted(list(common_words))}\n")
                    
                    # Key term analysis
                    key_terms = ['repair', 'maintenance', 'investment', 'cost', 'procedure', 'manual', 'specification', 'IDG', 'engine', 'aircraft']
                    found_terms = [term for term in key_terms if term.lower() in chunk.content.lower()]
                    if found_terms:
                        f.write(f"   🔑 KEY TERMS: {found_terms}\n")
                    
                    f.write("-" * 40 + "\n")
                
                # === SUMMARY STATISTICS ===
                f.write("📈 RETRIEVAL SUMMARY:\n")
                f.write(f"   Total results: {len(search_results)}\n")
                f.write(f"   Retrieved chunks: {len(retrieved_chunks)}\n")
                
                if retrieved_chunks:
                    similarities = [c.similarity_score for c in retrieved_chunks]
                    content_lengths = [len(c.content) for c in retrieved_chunks]
                    sources = [c.filename for c in retrieved_chunks]
                    
                    f.write(f"   📊 Similarity: {min(similarities):.6f} - {max(similarities):.6f}\n")
                    f.write(f"   📏 Content: {min(content_lengths)} - {max(content_lengths)} chars\n")
                    f.write(f"   📁 Sources: {len(set(sources))} unique files\n")
                    
                    for source in set(sources):
                        count = sources.count(source)
                        f.write(f"      {source}: {count} chunks\n")
                
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks. Analysis saved to {analysis_file}")
            return retrieved_chunks
            
        except Exception as e:
            self.logger.error(f"❌ RETRIEVAL ERROR: {e}")
            import traceback
            self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
            return []
    
    def _generate_response(self, 
                          query_text: str, 
                          chunks: List[RetrievedChunk], 
                          temperature: float,
                          model: str) -> Tuple[str, float]:
        """Generate response using OpenAI with retrieved chunks."""
        try:
            # Prepare context from chunks
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(
                    f"--- Document {i}: {chunk.filename} (Section {chunk.chunk_index}) ---\n"
                    f"Similarity: {chunk.similarity_score:.3f}\n"
                    f"Content: {chunk.content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Create messages for chat completion
            messages = [
                {"role": "system", "content": self.AEROSPACE_SYSTEM_PROMPT},
                {"role": "user", "content": f"""Based on the following aerospace MRO documents, please answer this question:

QUESTION: {query_text}

RELEVANT DOCUMENT SECTIONS:
{context}

Please provide a comprehensive answer based on the document content above. Include specific citations to source documents when making claims."""}
            ]
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # More realistic confidence scoring based on actual search quality
            if not chunks:
                confidence = 0.1
            else:
                # Base confidence on similarity scores and number of chunks
                avg_similarity = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
                max_similarity = max(chunk.similarity_score for chunk in chunks)
                
                # Scale confidence based on similarity scores
                # 0.7+ similarity = high confidence, 0.3-0.7 = medium, <0.3 = low
                if max_similarity >= 0.7:
                    base_confidence = 0.8
                elif max_similarity >= 0.5:
                    base_confidence = 0.65
                elif max_similarity >= 0.3:
                    base_confidence = 0.5
                elif max_similarity >= 0.2:
                    base_confidence = 0.35
                else:
                    base_confidence = 0.25
                
                # Adjust based on number of supporting chunks
                chunk_bonus = min(0.1, len(chunks) * 0.02)
                
                # Final confidence calculation
                confidence = min(0.9, base_confidence + chunk_bonus)
                
                self.logger.info(f"CONFIDENCE: avg_sim={avg_similarity:.3f}, max_sim={max_similarity:.3f}, "
                                f"base={base_confidence:.3f}, final={confidence:.3f}")
            
            return answer, confidence
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return f"I encountered an error generating the response: {str(e)}", 0.0
    
    def _prepare_sources(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Prepare source information for the response."""
        sources = []
        for chunk in chunks:
            sources.append({
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "similarity_score": round(chunk.similarity_score, 3),
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "metadata": chunk.metadata
            })
        return sources
    
    def _update_stats(self, total_time: float, retrieval_time: float, generation_time: float, chunk_count: int):
        """Update query statistics."""
        self.stats["total_queries"] += 1
        self.stats["total_query_time"] += total_time
        self.stats["total_retrieval_time"] += retrieval_time
        self.stats["total_generation_time"] += generation_time
        
        # Update average chunks retrieved
        current_avg = self.stats["average_chunks_retrieved"]
        total_queries = self.stats["total_queries"]
        self.stats["average_chunks_retrieved"] = ((current_avg * (total_queries - 1)) + chunk_count) / total_queries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        stats = self.stats.copy()
        if stats["total_queries"] > 0:
            stats["avg_query_time"] = stats["total_query_time"] / stats["total_queries"]
            stats["avg_retrieval_time"] = stats["total_retrieval_time"] / stats["total_queries"]
            stats["avg_generation_time"] = stats["total_generation_time"] / stats["total_queries"]
        else:
            stats["avg_query_time"] = 0.0
            stats["avg_retrieval_time"] = 0.0
            stats["avg_generation_time"] = 0.0
        
        return stats
    
    def interactive_query(self):
        """Start an interactive query session."""
        print("AEROSPACE MRO QUERY ENGINE")
        print("=" * 50)
        print("Ask questions about your aerospace MRO documents.")
        print("Type 'quit', 'exit', or 'q' to end the session.\n")
        
        while True:
            try:
                query = input("Your Question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Aerospace MRO Query Engine!")
                    break
                
                if not query:
                    continue
                
                print("\nSearching aerospace documents...")
                result = self.query(query)
                
                print(f"\n**Answer** (Confidence: {result.confidence_score:.1%}):")
                print(result.answer)
                
                if result.sources:
                    print(f"\n**Sources** ({len(result.sources)} documents):")
                    for i, source in enumerate(result.sources, 1):
                        print(f"  {i}. {source['filename']} (Section {source['chunk_index']}) - Similarity: {source['similarity_score']}")
                
                print(f"\nQuery completed in {result.query_time:.2f}s")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\nSession ended by user.")
                break
            except Exception as e:
                print(f"\nError: {e}")

# Convenience functions
def query_aerospace_docs(query_text: str, **kwargs) -> QueryResult:
    """Quick function to query aerospace documents."""
    engine = AerospaceQueryEngine()
    return engine.query(query_text, **kwargs)

def start_interactive_session():
    """Start an interactive query session."""
    engine = AerospaceQueryEngine()
    engine.interactive_query()

if __name__ == "__main__":
    # Run interactive session if called directly
    start_interactive_session() 