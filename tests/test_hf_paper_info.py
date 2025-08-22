import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFPaperInfoTool


def main():
    tool = HFPaperInfoTool()
    arxiv_id = "1706.03762"  # Attention Is All You Need
    result_json_str = tool.forward(arxiv_id=arxiv_id)
    try:
        data = json.loads(result_json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(result_json_str)


if __name__ == "__main__":
    main()


