import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFSiteSearchTool


def main():
    tool = HFSiteSearchTool()
    result_json_str = tool.forward(query="fine-tuning tutorial", limit=5)
    try:
        data = json.loads(result_json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(result_json_str)


if __name__ == "__main__":
    main()


