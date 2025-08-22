import json
import os
import sys

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFModelsSearchTool


def main():
    tool = HFModelsSearchTool()
    result_json_str = tool.forward(
        query="stable diffusion",
        task="text-to-image",
        sort="downloads",
        direction="descending",
        limit=5,
    )
    try:
        data = json.loads(result_json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(result_json_str)


if __name__ == "__main__":
    main()


