#!/usr/bin/env python3
"""
Simple script to run the enhanced PDF extraction comparison.
This script creates organized output folders and comprehensive performance metrics.
"""

import sys
from pathlib import Path
from enhanced_pdf_comparison import EnhancedPDFComparison

def run_enhanced_test(pdf_file_path: str, output_dir: str = "pdf_comparison_results"):
    """
    Run enhanced PDF extraction comparison with organized output.
    
    Args:
        pdf_file_path: Path to the PDF file to test
        output_dir: Base directory for organized results
    """
    pdf_path = Path(pdf_file_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    print(f"Starting Enhanced PDF Extraction Comparison")
    print(f"Testing file: {pdf_path.name}")
    print(f"File size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 70)
    
    # Create enhanced comparison instance
    comparator = EnhancedPDFComparison(output_dir)
    
    try:
        # Run comprehensive comparison
        results = comparator.run_comprehensive_comparison(pdf_path)
        
        # Display summary results
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        successful_results = [(name, result) for name, result in results.items() if result.success]
        failed_results = [(name, result) for name, result in results.items() if not result.success]
        
        if successful_results:
            # Sort by speed for quick overview
            speed_sorted = sorted(successful_results, key=lambda x: x[1].metrics.chars_per_second, reverse=True)
            
            print(f"✅ Successfully tested: {len(successful_results)}/{len(results)} libraries")
            print()
            
            for i, (name, result) in enumerate(speed_sorted, 1):
                m = result.metrics
                print(f"{i}. {name}:")
                print(f"   ⚡ Speed: {m.chars_per_second:,.0f} chars/sec ({m.extraction_time:.2f}s)")
                print(f"   📝 Text: {m.character_count:,} chars, {m.word_count:,} words")
                print(f"   💾 Memory: {m.memory_usage_mb:.1f} MB (efficiency: {m.memory_efficiency:,.0f} chars/MB)")
                print(f"   📊 Quality: {m.readability_score:.1f}/100")
                print()
        
        if failed_results:
            print(f"❌ Failed libraries:")
            for name, result in failed_results:
                print(f"   {name}: {result.error_message}")
            print()
        
        # Show output organization
        latest_dir = max(comparator.output_base_dir.glob("test_*"), key=lambda p: p.stat().st_mtime)
        print(f"📂 Results saved to: {latest_dir}")
        print(f"   📄 Extracted texts: {latest_dir / 'extracted_texts'}")
        print(f"   📊 Reports: {latest_dir / 'reports'}")
        print(f"   📈 Metrics: {latest_dir / 'metrics'}")
        
        # Quick recommendations
        if successful_results:
            print(f"\n🎯 QUICK RECOMMENDATIONS:")
            print("-" * 50)
            
            fastest = max(successful_results, key=lambda x: x[1].metrics.chars_per_second)
            most_text = max(successful_results, key=lambda x: x[1].metrics.character_count)
            most_efficient = max(successful_results, key=lambda x: x[1].metrics.memory_efficiency)
            best_quality = max(successful_results, key=lambda x: x[1].metrics.readability_score)
            
            print(f"🚀 Fastest: {fastest[0]} ({fastest[1].metrics.chars_per_second:,.0f} chars/sec)")
            print(f"📄 Most text: {most_text[0]} ({most_text[1].metrics.character_count:,} chars)")
            print(f"💾 Most efficient: {most_efficient[0]} ({most_efficient[1].metrics.memory_efficiency:,.0f} chars/MB)")
            print(f"✨ Best quality: {best_quality[0]} ({best_quality[1].metrics.readability_score:.1f}/100)")
            
            print(f"\n💡 For aerospace MRO documents:")
            print(f"   • Tables & forms → pdfplumber")
            print(f"   • Complex layouts → PDFMiner.six")
            print(f"   • Speed critical → PyMuPDF")
            print(f"   • Simple docs → PyPDF2")
        
        print(f"\n✅ Comparison complete! Check the reports for detailed analysis.")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

def compare_multiple_pdfs(pdf_directory: str, output_dir: str = "batch_comparison_results"):
    """
    Compare extraction libraries across multiple PDF files.
    
    Args:
        pdf_directory: Directory containing PDF files to test
        output_dir: Base directory for results
    """
    pdf_dir = Path(pdf_directory)
    
    if not pdf_dir.exists():
        print(f"❌ Error: Directory not found: {pdf_dir}")
        return
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {pdf_dir}")
        return
    
    print(f"🔍 Batch PDF Extraction Comparison")
    print(f"📁 Testing {len(pdf_files)} PDF files from: {pdf_dir}")
    print("=" * 70)
    
    all_results = {}
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        print("-" * 50)
        
        comparator = EnhancedPDFComparison(f"{output_dir}/{pdf_file.stem}")
        
        try:
            results = comparator.run_comprehensive_comparison(pdf_file)
            all_results[pdf_file.name] = results
            
            # Quick summary for this file
            successful = sum(1 for r in results.values() if r.success)
            print(f"   ✅ {successful}/{len(results)} libraries successful")
            
            if successful > 0:
                fastest = max([r for r in results.values() if r.success], 
                            key=lambda x: x.metrics.chars_per_second)
                print(f"   🚀 Fastest: {fastest.library_name} ({fastest.metrics.chars_per_second:,.0f} chars/sec)")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_results[pdf_file.name] = None
    
    # Overall summary
    print(f"\n📊 BATCH COMPARISON SUMMARY:")
    print("=" * 70)
    
    successful_files = [name for name, results in all_results.items() if results is not None]
    print(f"✅ Successfully processed: {len(successful_files)}/{len(pdf_files)} files")
    
    if successful_files:
        # Library success rates across all files
        library_success = {}
        library_speeds = {}
        
        for file_name, results in all_results.items():
            if results:
                for lib_name, result in results.items():
                    if lib_name not in library_success:
                        library_success[lib_name] = {"success": 0, "total": 0}
                        library_speeds[lib_name] = []
                    
                    library_success[lib_name]["total"] += 1
                    if result.success:
                        library_success[lib_name]["success"] += 1
                        library_speeds[lib_name].append(result.metrics.chars_per_second)
        
        print(f"\n📈 Library Performance Across All Files:")
        print("-" * 50)
        
        for lib_name, stats in library_success.items():
            success_rate = stats["success"] / stats["total"] * 100
            avg_speed = sum(library_speeds[lib_name]) / len(library_speeds[lib_name]) if library_speeds[lib_name] else 0
            
            print(f"{lib_name}:")
            print(f"   Success Rate: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
            if avg_speed > 0:
                print(f"   Avg Speed: {avg_speed:,.0f} chars/sec")
            print()
    
    print(f"📂 Individual results saved in: {output_dir}/")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Enhanced PDF Extraction Comparison Tool")
        print("=" * 50)
        print()
        print("Usage:")
        print("  python run_enhanced_comparison.py <pdf_file> [output_dir]")
        print("  python run_enhanced_comparison.py --batch <pdf_directory> [output_dir]")
        print()
        print("Examples:")
        print("  python run_enhanced_comparison.py document.pdf")
        print("  python run_enhanced_comparison.py document.pdf my_results")
        print("  python run_enhanced_comparison.py --batch RAG_input/ batch_results")
        print()
        print("Features:")
        print("  ✅ Comprehensive performance metrics (speed, memory, CPU)")
        print("  📄 Organized text output in separate folders")
        print("  📊 Multiple report formats (TXT, JSON, CSV)")
        print("  🎯 Specific recommendations for aerospace documents")
        print("  💾 Memory usage and efficiency analysis")
        print("  📈 Text quality and readability scoring")
        return
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("❌ Error: --batch requires a directory path")
            return
        
        pdf_directory = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else "batch_comparison_results"
        compare_multiple_pdfs(pdf_directory, output_dir)
    else:
        pdf_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "pdf_comparison_results"
        run_enhanced_test(pdf_file, output_dir)

if __name__ == "__main__":
    main() 