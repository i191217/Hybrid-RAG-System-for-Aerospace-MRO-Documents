#!/usr/bin/env python3
"""
Simple web interface for the Aerospace Query Engine
Provides an easy-to-use web UI for querying aerospace documents.
"""

try:
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Flask is required for the web interface. Install with: pip install flask flask-cors")
    exit(1)

import logging
from pathlib import Path
import json
import time

from core.query_engine import AerospaceQueryEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Initialize query engine
query_engine = None

def init_query_engine():
    """Initialize the query engine with error handling."""
    global query_engine
    try:
        query_engine = AerospaceQueryEngine()
        logger.info("Query engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}")
        return False

@app.route('/')
def index():
    """Main page with query interface."""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aerospace MRO Query Engine</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f7fa;
                color: #333;
            }
            .header {
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .container {
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
            .query-box {
                width: 100%;
                padding: 15px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 16px;
                resize: vertical;
                min-height: 100px;
                font-family: inherit;
            }
            .query-box:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
            }
            .submit-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 15px;
                transition: transform 0.2s;
            }
            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .submit-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #667eea;
            }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .result {
                display: none;
                margin-top: 2rem;
            }
            .answer {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin: 15px 0;
                border-radius: 0 8px 8px 0;
                line-height: 1.6;
            }
            .sources {
                margin-top: 20px;
            }
            .source-item {
                background: #fff;
                border: 1px solid #e1e5e9;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .source-header {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 8px;
            }
            .source-preview {
                color: #666;
                font-size: 14px;
                line-height: 1.4;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                font-size: 14px;
            }
            .stat-item {
                text-align: center;
            }
            .stat-value {
                font-weight: bold;
                color: #667eea;
                font-size: 18px;
            }
            .examples {
                margin-top: 2rem;
            }
            .example-btn {
                background: #f8f9fa;
                border: 1px solid #e1e5e9;
                padding: 10px 15px;
                margin: 5px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s;
            }
            .example-btn:hover {
                background: #e9ecef;
                border-color: #667eea;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Aerospace MRO Query Engine</h1>
            <p>Ask questions about your aerospace maintenance, repair, and operations documents</p>
        </div>

        <div class="container">
            <div>
                <label for="query" style="font-weight: bold; display: block; margin-bottom: 10px;">
                    Your Question:
                </label>
                <textarea 
                    id="query" 
                    class="query-box" 
                    placeholder="Ask about maintenance procedures, financial reports, compliance requirements, or any aerospace-related topic..."
                ></textarea>
                <button onclick="submitQuery()" class="submit-btn" id="submitBtn">
                    Search Documents
                </button>
            </div>

            <div class="examples">
                <h3>Example Questions:</h3>
                <button class="example-btn" onclick="setQuery('What maintenance procedures are mentioned in the documents?')">
                    Maintenance Procedures
                </button>
                <button class="example-btn" onclick="setQuery('What are the key financial metrics or budgets discussed?')">
                    Financial Metrics
                </button>
                <button class="example-btn" onclick="setQuery('Are there any compliance or regulatory requirements mentioned?')">
                    Compliance Requirements
                </button>
                <button class="example-btn" onclick="setQuery('What aerospace components or systems are referenced?')">
                    Components & Systems
                </button>
                <button class="example-btn" onclick="setQuery('What operational procedures are documented?')">
                    Operational Procedures
                </button>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Searching aerospace documents...</div>
        </div>

        <div class="result" id="result">
            <div class="container">
                <div id="answer" class="answer"></div>
                <div id="sources" class="sources"></div>
                <div id="stats" class="stats"></div>
            </div>
        </div>

        <script>
            function setQuery(text) {
                document.getElementById('query').value = text;
            }

            async function submitQuery() {
                const query = document.getElementById('query').value.trim();
                if (!query) {
                    alert('Please enter a question');
                    return;
                }

                // Show loading, hide result
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('submitBtn').disabled = true;

                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });

                    const data = await response.json();

                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }

                    // Display results
                    displayResults(data);

                } catch (error) {
                    alert('Network error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('submitBtn').disabled = false;
                }
            }

            function displayResults(data) {
                // Display answer
                const answerDiv = document.getElementById('answer');
                const confidence = data.confidence_score ? Math.round(data.confidence_score * 100) : 0;
                answerDiv.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="margin: 0;"> Answer</h3>
                        <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px;">
                            Confidence: ${confidence}%
                        </span>
                    </div>
                    <div>${data.answer.replace(/\\n/g, '<br>')}</div>
                `;

                // Display sources
                const sourcesDiv = document.getElementById('sources');
                if (data.sources && data.sources.length > 0) {
                    let sourcesHtml = `<h3> Sources (${data.sources.length} documents)</h3>`;
                    data.sources.forEach((source, index) => {
                        const similarity = Math.round(source.similarity_score * 100);
                        sourcesHtml += `
                            <div class="source-item">
                                <div class="source-header">
                                    ${index + 1}. ${source.filename} (Section ${source.chunk_index})
                                    <span style="float: right; color: #28a745; font-size: 12px;">
                                        ${similarity}% match
                                    </span>
                                </div>
                                <div class="source-preview">${source.content_preview}</div>
                            </div>
                        `;
                    });
                    sourcesDiv.innerHTML = sourcesHtml;
                } else {
                    sourcesDiv.innerHTML = '<p>No sources found</p>';
                }

                // Display stats
                const statsDiv = document.getElementById('stats');
                const stats = data.retrieval_stats;
                statsDiv.innerHTML = `
                    <div class="stat-item">
                        <div class="stat-value">${stats.chunks_found || 0}</div>
                        <div>Chunks Found</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${(data.query_time || 0).toFixed(2)}s</div>
                        <div>Total Time</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${((stats.avg_similarity || 0) * 100).toFixed(0)}%</div>
                        <div>Avg Similarity</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${(stats.generation_time || 0).toFixed(2)}s</div>
                        <div>AI Generation</div>
                    </div>
                `;

                document.getElementById('result').style.display = 'block';
            }

            // Allow Enter key to submit
            document.getElementById('query').addEventListener('keydown', function(event) {
                if (event.ctrlKey && event.key === 'Enter') {
                    submitQuery();
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for processing queries."""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        if not query_engine:
            return jsonify({'error': 'Query engine not initialized'}), 500
        
        # Process the query (removed similarity_threshold parameter)
        result = query_engine.query(
            query_text,
            max_chunks=5,
            temperature=0.6
        )
        
        # Convert result to JSON-serializable format
        response = {
            'answer': result.answer,
            'sources': result.sources,
            'query_time': result.query_time,
            'retrieval_stats': result.retrieval_stats,
            'confidence_score': result.confidence_score
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for query engine statistics."""
    try:
        if not query_engine:
            return jsonify({'error': 'Query engine not initialized'}), 500
        
        stats = query_engine.get_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'query_engine_ready': query_engine is not None
    })

def test_chat():
    """
    Interactive test chat function to test the query engine directly.
    Run this function to start a command-line chat interface.
    """
    print("AEROSPACE MRO QUERY ENGINE - TEST CHAT")
    print("=" * 60)
    print("This is a test interface to try out the query engine.")
    print("Type your questions about aerospace documents.")
    print("Type 'quit', 'exit', or 'q' to end the session.\n")
    
    # Initialize query engine if not already done
    global query_engine
    if not query_engine:
        print("Initializing query engine...")
        if not init_query_engine():
            print("ERROR: Failed to initialize query engine. Check your configuration.")
            return
        print("Query engine ready!\n")
    
    # Chat loop
    while True:
        try:
            # Get user input
            query = input(" Your Question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q', '']:
                print("\n Thank you for testing the Aerospace MRO Query Engine!")
                break
            
            # Process the query
            print("\n Searching aerospace documents...")
            start_time = time.time()
            
            result = query_engine.query(
                query,
                max_chunks=5,
                temperature=0.6
            )
            
            # Display results
            print(f"\n **Answer** (Confidence: {result.confidence_score:.1%}):")
            print("-" * 50)
            print(result.answer)
            
            if result.sources:
                print(f"\n **Sources** ({len(result.sources)} documents):")
                print("-" * 50)
                for i, source in enumerate(result.sources, 1):
                    similarity_pct = source['similarity_score'] * 100
                    print(f"  {i}. {source['filename']} (Section {source['chunk_index']}) - {similarity_pct:.1f}% match")
                    if len(source['content_preview']) > 100:
                        preview = source['content_preview'][:100] + "..."
                    else:
                        preview = source['content_preview']
                    print(f"      {preview}")
                    print()
            else:
                print("\n **Sources:** No relevant documents found")
            
            # Display stats
            stats = result.retrieval_stats
            print(f" **Query Stats:**")
            print(f"     Total time: {result.query_time:.2f}s")
            print(f"    Chunks found: {stats.get('chunks_found', 0)}")
            if stats.get('avg_similarity'):
                print(f"    Avg similarity: {stats['avg_similarity']:.1%}")
            print(f"    Generation time: {stats.get('generation_time', 0):.2f}s")
            print(f"    Query type: {stats.get('query_type', 'unknown')}")
            
            print("\n" + "=" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n Session ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again or type 'quit' to exit.\n")

def main():
    """Main function to run the web server."""
    print(" Starting Aerospace Query Engine Web Interface")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path(".env").exists():
        print(" Error: .env file not found. Please run from the improved-doc-processor directory.")
        return
    
    # Initialize query engine
    print(" Initializing query engine...")
    if not init_query_engine():
        print(" Failed to initialize query engine. Check your configuration.")
        return
    
    print(" Query engine initialized successfully!")
    print("\n Starting web server...")
    print(" Open your browser and go to: http://localhost:5000")
    print(" Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n Web server stopped. Goodbye!")

if __name__ == "__main__":
    import sys
    import time
    
    # Check if user wants to run test chat instead of web server
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['test', 'chat', 'cli']:
        test_chat()
    else:
        main() 