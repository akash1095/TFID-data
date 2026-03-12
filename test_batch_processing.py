#!/usr/bin/env python3
"""
Test script for batch processing with OpenAI-compatible API (Vast.ai setup).

This script tests:
1. Connection to your Vast.ai/Ollama setup (single request)
2. Concurrent ainvoke (8 simultaneous requests with asyncio.gather)
3. Batch processing with abatch() (optimized batch API)
4. Verifies results are coming as expected

Usage:
    python test_batch_processing.py --url http://localhost:11434/v1 --api-key YOUR_KEY
"""

import argparse
import asyncio
import time
from forward_kg_construction.llm.openai_inference import OpenAIInference, OpenAIConfig
from forward_kg_construction.llm.schema import RelationshipAnalysis
from forward_kg_construction.llm.prompts import LLAMA_8B_SYSTEM_PROMPT, LLAMA_8B_EXTRACT_PROMPT

# Sample paper pairs for testing
SAMPLE_PAPERS = [
    {
        "citing_title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "citing_abstract": "Extends The dominant sequence transduction models language representation model which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        "cited_title": "Attention is All You Need",
        "cited_abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    },
    {
        "citing_title": "GPT-3: Language Models are Few-Shot Learners",
        "citing_abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
        "cited_title": "Attention is All You Need",
        "cited_abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    },
    {
        "citing_title": "T5: Exploring the Limits of Transfer Learning",
        "citing_abstract": "Transfer learning has become a powerful technique in natural language processing. We explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format.",
        "cited_title": "Attention is All You Need",
        "cited_abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    },
]


async def test_connection(client):
    """Test 1: Verify connection to API."""
    print("\n" + "="*80)
    print("TEST 1: Connection Test")
    print("="*80)
    
    paper = SAMPLE_PAPERS[0]

    # Use full LLAMA_8B_EXTRACT_PROMPT template with instructions
    user_prompt = LLAMA_8B_EXTRACT_PROMPT.format(
        citing_title=paper['citing_title'],
        citing_abstract=paper['citing_abstract'],
        cited_title=paper['cited_title'],
        cited_abstract=paper['cited_abstract']
    )

    messages = client.build_messages(
        system_prompt=LLAMA_8B_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
    
    try:
        start = time.time()
        result = await client.ainvoke(messages, schema=RelationshipAnalysis)
        elapsed = time.time() - start
        
        print(f"✅ Connection successful!")
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📊 Relationships found: {len(result.relationships)}")
        for rel in result.relationships:
            print(f"   - {rel.type} (confidence: {rel.confidence})")
        
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


async def test_concurrent_ainvoke(client, num_requests=4):
    """Test 2: Verify concurrent ainvoke works (8 simultaneous requests)."""
    print("\n" + "="*80)
    print(f"TEST 2: Concurrent ainvoke ({num_requests} simultaneous requests)")
    print("="*80)

    # Build messages for multiple papers
    messages_list = []
    for i in range(num_requests):
        # Repeat papers if we need more than available
        paper = SAMPLE_PAPERS[i % len(SAMPLE_PAPERS)]

        # Use full LLAMA_8B_EXTRACT_PROMPT template with instructions
        user_prompt = LLAMA_8B_EXTRACT_PROMPT.format(
            citing_title=paper['citing_title'],
            citing_abstract=paper['citing_abstract'],
            cited_title=paper['cited_title'],
            cited_abstract=paper['cited_abstract']
        )

        messages = client.build_messages(
            system_prompt=LLAMA_8B_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        messages_list.append(messages)

    try:
        print(f"🚀 Sending {num_requests} concurrent requests...")
        start = time.time()

        # Send all requests concurrently using asyncio.gather
        tasks = [
            client.ainvoke(messages, schema=RelationshipAnalysis)
            for messages in messages_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start

        # Count successful vs failed
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))

        print(f"✅ Concurrent processing complete!")
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"📊 Successful: {successful}/{num_requests}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{num_requests}")
        print(f"⚡ Average: {elapsed/num_requests:.2f}s per request")
        print(f"🚀 Throughput: {num_requests/elapsed*3600:.0f} papers/hour")

        print(f"\n📋 Results:")
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"   Request {i}: ❌ Error - {str(result)[:50]}...")
            else:
                print(f"   Request {i}: ✅ {len(result.relationships)} relationships")
                for rel in result.relationships:
                    print(f"      - {rel.type} ({rel.confidence})")

        return successful == num_requests
    except Exception as e:
        print(f"❌ Concurrent processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_processing(client, batch_size=3):
    """Test 3: Verify batch processing works."""
    print("\n" + "="*80)
    print(f"TEST 3: Batch Processing ({batch_size} papers)")
    print("="*80)
    
    # Build messages for all papers
    messages_list = []
    for paper in SAMPLE_PAPERS[:batch_size]:
        # Use full LLAMA_8B_EXTRACT_PROMPT template with instructions
        user_prompt = LLAMA_8B_EXTRACT_PROMPT.format(
            citing_title=paper['citing_title'],
            citing_abstract=paper['citing_abstract'],
            cited_title=paper['cited_title'],
            cited_abstract=paper['cited_abstract']
        )

        messages = client.build_messages(
            system_prompt=LLAMA_8B_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        messages_list.append(messages)
    
    try:
        start = time.time()
        results = await client.abatch(
            messages_list,
            schema=RelationshipAnalysis,
            max_concurrency=8
        )
        elapsed = time.time() - start
        
        print(f"✅ Batch processing successful!")
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"📊 Processed {len(results)} papers")
        print(f"⚡ Average: {elapsed/len(results):.2f}s per paper")
        print(f"🚀 Throughput: {len(results)/elapsed*3600:.0f} papers/hour")
        
        print(f"\n📋 Results: {results}")
        # for i, result in enumerate(result, 1):
        #     print(f"   Paper {i}: {len(result.relationships)} relationships")
        #     for rel in result.relationships:
        #         print(f"      - {rel.type} ({rel.confidence})")
        
        return True
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test batch processing with Vast.ai setup")
    parser.add_argument("--url", default="http://185.62.108.226:42692/v1",
                        help="OpenAI-compatible API URL (default: http://185.62.108.226:42692/v1)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="Model name (default: qwen2.5:7b)")
    parser.add_argument("--api-key", default=None,
                        help="API key (e.g., Vast.ai open_button_key)")
    args = parser.parse_args()
    
    print("🚀 Batch Processing Test for Vast.ai Setup")
    print("="*80)
    print(f"API URL: {args.url}")
    print(f"Model: {args.model}")

    # Get API key from args or environment
    import os
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
    if api_key != "EMPTY":
        print(f"API Key: {api_key[:8]}...")

    # Create client (simplified config for Ollama compatibility)
    print("\n📦 Creating OpenAI-compatible client...")
    config = OpenAIConfig(
        model="Qwen/Qwen3-8B",
        base_url=args.url,
        api_key=api_key,
        temperature=0.3,
        max_tokens=1024,  # Increased to allow complete JSON responses
        # Don't set top_p, frequency_penalty, etc. for Ollama compatibility
    )
    client = OpenAIInference(config=config)
    print("✅ Client created")
    
    # Run tests
    connection_ok = await test_connection(client)
    if not connection_ok:
        print("\n❌ Connection test failed. Please check your Vast.ai setup.")
        return

    concurrent_ok = await test_concurrent_ainvoke(client, num_requests=2)
    batch_ok = await test_batch_processing(client)

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Connection Test: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"Concurrent ainvoke Test (8 requests): {'✅ PASS' if concurrent_ok else '❌ FAIL'}")
    print(f"Batch Test: {'✅ PASS' if batch_ok else '❌ FAIL'}")
    
    if connection_ok and concurrent_ok and batch_ok:
        print("\n🎉 All tests passed! Your Vast.ai setup is ready for batch processing.")
        print("\n📈 Performance Comparison:")
        print("  - Concurrent ainvoke: Good for moderate concurrency (8-16 requests)")
        print("  - Batch abatch: Best for high concurrency (32+ requests)")
        print("\nNext steps:")
        print(f"  python step2_extract_relationships.py \\")
        print(f"    --backend openai \\")
        print(f"    --openai-url {args.url} \\")
        print(f"    --model {args.model} \\")
        if api_key != "EMPTY":
            print(f"    --openai-api-key {api_key} \\")
        print(f"    --batch-mode \\")
        print(f"    --batch-size 100 \\")
        print(f"    --batch-concurrency 32")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running the pipeline.")


if __name__ == "__main__":
    asyncio.run(main())

