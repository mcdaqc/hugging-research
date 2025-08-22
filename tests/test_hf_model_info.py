import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFModelInfoTool


def main():
    tool = HFModelInfoTool()
    # Example: a well-known repo id
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    result_json_str = tool.forward(repo_id=repo_id)
    try:
        data = json.loads(result_json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(result_json_str)


if __name__ == "__main__":
    main()


