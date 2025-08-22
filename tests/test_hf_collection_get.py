import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFCollectionGetTool


def main():
    tool = HFCollectionGetTool()
    # Example namespace/slug - replace with a public collection if needed
    namespace = "huggingface"
    slug_id = "trending-models"  # may vary; adjust to an existing collection
    result_json_str = tool.forward(namespace=namespace, slug_id=slug_id)
    try:
        data = json.loads(result_json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(result_json_str)


if __name__ == "__main__":
    main()


