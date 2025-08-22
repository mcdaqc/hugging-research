import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFReportGenerateTool


def main():
    tool = HFReportGenerateTool()
    # Minimal aggregated data example
    data = {
        "Models": [{
            "type": "model",
            "id": "bert-base-uncased",
            "owner": "google",
            "url": "https://huggingface.co/bert-base-uncased",
            "downloads": 100,
            "likes": 10,
            "updatedAt": "2024-01-01",
            "description": "BERT base cased model"
        }],
        "Datasets": [],
        "Spaces": []
    }
    import json
    html = tool.forward(data_json=json.dumps(data), title="Test Report")
    print(html[:500])  # print first 500 chars


if __name__ == "__main__":
    main()


