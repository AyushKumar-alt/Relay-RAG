import sys
import time
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipeline import RelayPipeline, GROQ_API_KEY

def print_result(res: dict):
    print("\n" + "=" * 75)
    print(f" QUESTION: {res.get('question', '')}")
    print("=" * 75)
    print(f" STATUS:          {res['status'].upper()} (Grade: {res['answer_grade']})")
    print(f" ENGINE:          {res['source'].upper()}")
    print(f" TIME TO 1ST TOK: {res['first_token_ms'] if res['first_token_ms'] is not None else 'N/A (Refused before model)'} ms")
    print(f" TOTAL TIME:      {res['total_ms']} ms")
    print(f" TOKENS:          In: {res['tokens_in']} | Out: {res['tokens_out']}")
    print(f" REASON:          {res.get('reason', '')}")
    print("-" * 75)
    print(" ANSWER:")
    print(res['answer'])
    print("=" * 75 + "\n")

def main():
    engine = "real" if GROQ_API_KEY else "standin"
    if "--standin" in sys.argv:
        engine = "standin"
        sys.argv.remove("--standin")
    elif "--real" in sys.argv:
        engine = "real"
        sys.argv.remove("--real")

    pipe = RelayPipeline()

    # If query passed as command-line arguments
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Processing query using [{engine.upper()}] engine...")
        res = pipe.process_request("query_cli", query, engine=engine)
        res["question"] = query
        print_result(res)
        return

    # Interactive mode
    print("=" * 75)
    print(f" THE RELAY: INTERACTIVE RESEARCH ASSISTANT (Engine: {engine.upper()})")
    print(" Grounded on 3 arXiv papers: Attention Is All You Need, Moshi, Dynamic Chunking")
    print(" Type your question and press Enter. (Type 'exit' or 'quit' to stop)")
    print(" Tip: Add '--standin' to run locally without using Groq API quota.")
    print("=" * 75)

    while True:
        try:
            q = input("\nEnter your question > ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break

            res = pipe.process_request("interactive", q, engine=engine)
            res["question"] = q
            print_result(res)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    main()

