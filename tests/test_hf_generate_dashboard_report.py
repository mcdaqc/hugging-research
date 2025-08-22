import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.hf_tools import HFDashboardReportTool


def main():
    tool = HFDashboardReportTool()
    html = tool.forward(query="semantic search", limit=5)
    print(html[:500])  # print first 500 chars


if __name__ == "__main__":
    main()


