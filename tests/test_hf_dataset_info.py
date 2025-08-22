import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFDatasetInfoTool


def main():
    tool = HFDatasetInfoTool()
    repo_id = "glue"
    result_json_str = tool.forward(repo_id=repo_id)
    try:
        data = json.loads(result_json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(result_json_str)


if __name__ == "__main__":
    main()


