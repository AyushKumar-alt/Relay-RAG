import sys
from pathlib import Path

# Add src/ to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipeline import RelayPipeline

def main():
    print("=" * 70)
    print("       TESTING RELAY PIPELINE (4 STEPS, TOOL CALL, BRANCH/LOOP)")
    print("=" * 70)

    pipeline = RelayPipeline()

    test_queries = [
        ("req_001", "What are the dimensions dk and dv used for each head in the Transformer multi-head attention mechanism?"),
        ("req_002", "What is the BLEU score of the Moshi dialogue model on the WMT 2014 English-to-German translation task?"),
        ("req_003", "What is the training batch size and learning rate used to train the LLaMA-3 foundation model?")
    ]

    for req_id, q_text in test_queries:
        print(f"\n--- Running Request: {req_id} ---")
        print(f"Question: {q_text}")
        
        result = pipeline.process_request(req_id, q_text, engine="standin", scenario="test")
        
        print(f"Status:       {result['status']}")
        print(f"Grade:        {result['answer_grade']}")
        print(f"TTFT:         {result['first_token_ms']} ms")
        print(f"Total Time:   {result['total_ms']} ms")
        print(f"Retries/Loop: {result['retries']}")
        print(f"Answer:       {result['answer']}")
        print(f"Reason:       {result['reason']}")

    print("\n" + "=" * 70)
    print("Pipeline test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
