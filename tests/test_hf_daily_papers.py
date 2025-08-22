import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFDailyPapersTool


def main():
    tool = HFDailyPapersTool()
    result_json_str = tool.forward()
    try:
        data = json.loads(result_json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        try:
            print(result_json_str.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
        except Exception:
            print("<unprintable> due to encoding error")


if __name__ == "__main__":
    main()


