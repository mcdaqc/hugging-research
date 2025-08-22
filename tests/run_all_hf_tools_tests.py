import subprocess
import sys
import os
from datetime import datetime

TESTS = [
    "test_hf_models_search.py",
    "test_hf_model_info.py",
    "test_hf_datasets_search.py",
    "test_hf_dataset_info.py",
    "test_hf_spaces_search.py",
    "test_hf_space_info.py",
    "test_hf_user_info.py",
    "test_hf_collections_list.py",
    "test_hf_collection_get.py",
    "test_hf_paper_info.py",
    "test_hf_paper_repos.py",
    "test_hf_daily_papers.py",
    "test_hf_repo_info.py",
    "test_hf_site_search.py",
    "test_hf_report_generate.py",
    "test_hf_generate_dashboard_report.py",
]


def main():
    base = os.path.dirname(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(base, f"hf_tools_tests_output_{timestamp}.txt")

    ok = True
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"Hugging Face Tools Test Run — {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        for t in TESTS:
            path = os.path.join(base, t)
            header = f"=== Running {t} ===\n"
            print("\n" + header, end="")
            f.write(header)
            # Write simple INPUT info: command and forward() snippet if found
            f.write(f"Command: {sys.executable} {path}\n")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as tf:
                    src = tf.read()
                snippet = ""
                key = "tool.forward("
                idx = src.find(key)
                if idx != -1:
                    end = idx
                    # capture up to 500 chars from forward( to show parameters
                    snippet = src[idx: idx + 500]
                else:
                    # fallback: show main() body first 400 chars
                    m_idx = src.find("def main():")
                    snippet = src[m_idx: m_idx + 400] if m_idx != -1 else src[:400]
                f.write("--- INPUT (snippet) ---\n")
                f.write(snippet)
                if not snippet.endswith("\n"):
                    f.write("\n")
            except Exception as e:
                f.write(f"--- INPUT (unavailable): {e}\n")
            try:
                result = subprocess.run(
                    [sys.executable, path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                f.write("--- STDOUT ---\n")
                f.write(result.stdout or "")
                if not result.stdout.endswith("\n"):
                    f.write("\n")
                f.write("--- STDERR ---\n")
                f.write(result.stderr or "")
                if not result.stderr.endswith("\n"):
                    f.write("\n")
                status_line = f"[OK] {t}\n" if result.returncode == 0 else f"[FAIL] {t}\n"
                f.write(status_line)
                f.write("-" * 80 + "\n\n")
                if result.returncode != 0:
                    ok = False
                print(status_line.strip())
            except Exception as e:
                ok = False
                err_line = f"[ERROR] {t}: {e}\n"
                f.write(err_line)
                f.write("-" * 80 + "\n\n")
                print(err_line.strip())

    print(f"\nResults saved to: {out_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


