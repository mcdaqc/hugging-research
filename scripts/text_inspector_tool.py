from typing import Optional
import os

from smolagents import Tool
from smolagents.models import MessageRole, Model

from .mdconvert import MarkdownConverter


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx", ".mjs", ".js"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"
    md_converter = MarkdownConverter()

    def __init__(self, model: Model, text_limit: int):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def forward_initial_exam_mode(self, file_path, question):
        try:
            # Only allow reading files from uploads directory
            uploads_dir = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
            candidate_path = os.path.abspath(file_path)
            if not candidate_path.startswith(uploads_dir + os.sep):
                # Fallback to uploads/<basename>
                candidate_path = os.path.join(uploads_dir, os.path.basename(file_path))

            result = self.md_converter.convert(candidate_path)

            if file_path[-4:] in [".png", ".jpg"]:
                raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

            if ".zip" in file_path:
                return result.text_content

            if not question:
                return result.text_content

            if len(result.text_content) < 4000:
                return "Document content: " + result.text_content

            # For larger files, just return the content without model processing to avoid freezing
            return f"Document title: {result.title}\n\nDocument content:\n{result.text_content[:self.text_limit]}"
            
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}. Access is restricted to files uploaded via the interface."

    def forward(self, file_path, question: Optional[str] = None) -> str:
        try:
            # Only allow reading files from uploads directory
            uploads_dir = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
            candidate_path = os.path.abspath(file_path)
            if not candidate_path.startswith(uploads_dir + os.sep):
                # Fallback to uploads/<basename>
                candidate_path = os.path.join(uploads_dir, os.path.basename(file_path))

            result = self.md_converter.convert(candidate_path)

            if file_path[-4:] in [".png", ".jpg"]:
                raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

            if ".zip" in file_path:
                return result.text_content

            if not question:
                return result.text_content

            # For questions, return the content with a note about the question
            return f"Question: {question}\n\nDocument title: {result.title}\n\nDocument content:\n{result.text_content[:self.text_limit]}"
            
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}. Access is restricted to files uploaded via the interface."
