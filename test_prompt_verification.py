#!/usr/bin/env python3
"""
Test script to verify the prompt is properly constructed and working.

This script:
1. Shows the exact prompt being sent to the model
2. Sends a single request
3. Shows the raw response
4. Verifies relationships are being extracted

Usage:
    python test_prompt_verification.py --url http://YOUR_IP:PORT/v1 --api-key YOUR_KEY
"""

import argparse
import asyncio
from forward_kg_construction.llm.openai_inference import OpenAIInference, OpenAIConfig
from forward_kg_construction.llm.schema import RelationshipAnalysis
from forward_kg_construction.llm.prompts import LLAMA_8B_SYSTEM_PROMPT, LLAMA_8B_EXTRACT_PROMPT

# Test paper pair (BERT cites Transformer - should have EXTENDS relationship)
TEST_PAPER = {
    "citing_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "citing_abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.",
    "cited_title": "Attention is All You Need",
    "cited_abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
}


async def main():
    parser = argparse.ArgumentParser(description="Verify prompt construction")
    parser.add_argument("--url", default="http://185.62.108.226:42692/v1",
                        help="API URL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="Model name")
    parser.add_argument("--api-key", default=None,
                        help="API key")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")

    print("🔍 Prompt Verification Test")
    print("="*80)
    
    # Build the prompt
    user_prompt = LLAMA_8B_EXTRACT_PROMPT.format(
        citing_title=TEST_PAPER["citing_title"],
        citing_abstract=TEST_PAPER["citing_abstract"],
        cited_title=TEST_PAPER["cited_title"],
        cited_abstract=TEST_PAPER["cited_abstract"]
    )
    
    print("\n📝 SYSTEM PROMPT:")
    print("-"*80)
    print(LLAMA_8B_SYSTEM_PROMPT)
    print("-"*80)
    
    print("\n📝 USER PROMPT:")
    print("-"*80)
    print(user_prompt)
    print("-"*80)
    
    # Create client
    print("\n📦 Creating client...")
    config = OpenAIConfig(
        model=args.model,
        base_url=args.url,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
        top_p=0.95
    )
    client = OpenAIInference(config=config)
    
    # Build messages
    messages = client.build_messages(
        system_prompt=LLAMA_8B_SYSTEM_PROMPT,
        user_prompt= user_prompt
    )
    
    print(f"\n🚀 Sending request to {args.url}...")
    print(f"Model: {args.model}")
    
    try:
        # Send request
        invoke = client.invoke(messages, schema=RelationshipAnalysis)
        print("\n✅ Response received!")
        print("="*80)
        print(f"\n📊 Number of relationships: {len(invoke.relationships)}")
        if invoke.relationships:
            print("\n📋 Extracted Relationships:")
            for i, rel in enumerate(invoke.relationships, 1):
                print(f"\n  {i}. Type: {rel.type}")
                print(f"     Confidence: {rel.confidence}")
                print(f"     Evidence: {rel.evidence}")
                print(f"     Explanation: {rel.explanation}")

        result = await client.ainvoke(messages, schema=RelationshipAnalysis)
        
        print("\n✅ Response received!")
        print("="*80)
        print(f"\n📊 Number of relationships: {len(result.relationships)}")
        
        if result.relationships:
            print("\n📋 Extracted Relationships:")
            for i, rel in enumerate(result.relationships, 1):
                print(f"\n  {i}. Type: {rel.type}")
                print(f"     Confidence: {rel.confidence}")
                print(f"     Evidence: {rel.evidence}")
                print(f"     Explanation: {rel.explanation}")
        else:
            print("\n⚠️  No relationships extracted!")
        
        # Check if we got valid relationships
        valid_rels = [
            r for r in result.relationships
            if r.type.upper().replace("-", "_").replace(" ", "_") != "NO_RELATION"
        ]
        
        print("\n" + "="*80)
        print("📈 ANALYSIS:")
        print("="*80)
        print(f"Total relationships: {len(result.relationships)}")
        print(f"Valid relationships: {len(valid_rels)}")
        print(f"No-Relation: {len(result.relationships) - len(valid_rels)}")
        
        if valid_rels:
            print("\n✅ SUCCESS: Model is extracting relationships correctly!")
            print(f"Valid types: {[r.type for r in valid_rels]}")
        else:
            print("\n❌ ISSUE: All relationships are No-Relation")
            print("\nPossible causes:")
            print("1. Model doesn't understand the prompt")
            print("2. Prompt is too strict")
            print("3. Model needs different instructions")
            print("4. Papers actually have no relationship (unlikely for BERT->Transformer)")
        
        return len(valid_rels) > 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

