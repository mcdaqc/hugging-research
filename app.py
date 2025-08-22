import json
import mimetypes
import os
import re
import shutil
import threading
import uuid
from typing import Optional
from loguru import logger
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    Tool,
    DuckDuckGoSearchTool,
)
from smolagents.agent_types import (
    AgentAudio,
    AgentImage,
    AgentText,
    handle_agent_output_types,
)
from smolagents.gradio_ui import stream_to_gradio

from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from scripts.report_generator import HFLinkReportTool
from scripts.hf_tools import (
    HFModelsSearchTool,
    HFModelInfoTool,
    HFDatasetsSearchTool,
    HFDatasetInfoTool,
    HFSpacesSearchTool,
    HFSpaceInfoTool,
    HFUserInfoTool,
    HFCollectionsListTool,
    HFCollectionGetTool,
    HFPaperInfoTool,
    HFPaperReposTool,
    HFDailyPapersTool,
    HFRepoInfoTool,
    HFSiteSearchTool,
)

# web_search = GoogleSearchTool(provider="serper")
web_search = DuckDuckGoSearchTool()

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "jinja2",
]

load_dotenv(override=True)

# Only login if HF_TOKEN is available and valid in environment
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
        logger.info("Successfully logged in with HF_TOKEN from environment")
    except Exception as e:
        logger.warning(f"Failed to login with HF_TOKEN from environment: {e}")
        logger.info("You can still use the application by providing a valid API key in the interface")

# Global session storage for independent user sessions
user_sessions = {}
session_lock = threading.Lock()

append_answer_lock = threading.Lock()

# Initialize browser
browser = SimpleTextBrowser(request_kwargs={})

def validate_hf_api_key(api_key: str) -> tuple[bool, str]:
    """Validate Hugging Face API key by making a test request."""
    if not api_key or not api_key.strip():
        return False, "❌ API key cannot be empty"
    
    api_key = api_key.strip()
    
    # Basic format validation
    if not api_key.startswith("hf_"):
        return False, "❌ Invalid API key format. Hugging Face API keys start with 'hf_'"
    
    try:
        # Test the API key by making a simple request
        api = HfApi(token=api_key)
        # Try to get user info to validate the token
        user_info = api.whoami()
        return True, f"✅ API key validated successfully! Welcome, {user_info.get('name', 'User')}!"
    except Exception as e:
        return False, f"❌ Invalid API key: {str(e)}"

def create_model_with_api_key(hf_token: str, model_id: str = None) -> InferenceClientModel:
    """Create a model instance with the provided API key."""
    if not model_id:
        model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    # Store original token
    original_token = os.environ.get("HF_TOKEN")
    
    try:
        # Set the token in environment for this session
        os.environ["HF_TOKEN"] = hf_token
        
        # Create model without explicit token parameter
        model = InferenceClientModel(
            model_id=model_id,
        )
        
        return model
    finally:
        # Restore original token
        if original_token:
            os.environ["HF_TOKEN"] = original_token
        elif "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

def create_tools_with_model(model: InferenceClientModel):
    """Create tools with the provided model."""
    # Verify the model was created correctly
    if model is None:
        raise ValueError("Model is None, cannot create TextInspectorTool")
    
    # Text inspector tool disabled for now (inspect_file_as_text)
    # Reason: model attempted to use it with remote URLs; keep only for local uploads when re-enabled.
    # ti_tool = TextInspectorTool(model, 20000)
    
    # Hugging Face tools (public-only, anonymous)
    hf_tools = [
        HFModelsSearchTool(),
        HFModelInfoTool(),
        HFDatasetsSearchTool(),
        HFDatasetInfoTool(),
        HFSpacesSearchTool(),
        HFSpaceInfoTool(),
        HFUserInfoTool(),
        HFCollectionsListTool(),
        HFCollectionGetTool(),
        HFPaperInfoTool(),
        HFPaperReposTool(),
        HFDailyPapersTool(),
        HFRepoInfoTool(),
        HFSiteSearchTool(),
    ]

    tools = hf_tools + [
        web_search,  # duckduckgo
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        # ti_tool,  # TextInspectorTool (disabled) — only for uploaded local files; do not use with URLs
    ]
    
    return tools

# Agent creation in a factory function
def create_agent(hf_token: str = None, model_id: str = None, max_steps: int = 10):
    """Creates a fresh agent instance for each session"""
    if not hf_token:
        raise ValueError("A valid Hugging Face API key is required to create an agent.")
    
    logger.info(f"Creating agent with token: {hf_token[:10]}...")
    
    # Use session-specific model with HF_TOKEN
    model = create_model_with_api_key(hf_token, model_id)
    tools = create_tools_with_model(model)
    
    # TextInspectorTool temporarily disabled; skip presence check
    # Previous enforcement kept for reference:
    # has_text_inspector = any(getattr(tool, 'name', '') == 'inspect_file_as_text' for tool in tools)
    # if not has_text_inspector:
    #     raise ValueError("TextInspectorTool not found in tools list")
    
    agent = CodeAgent(
        model=model,
        tools=[visualizer] + tools,
        max_steps=max_steps,
        verbosity_level=1,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
    )
    
    logger.info("Agent created successfully")
    return agent

def get_user_session(request: gr.Request) -> str:
    """Get or create a unique session ID for the user."""
    if not request:
        logger.warning("No request object, using random session ID")
        return str(uuid.uuid4())
    
    # Try to get session from headers or create new one
    session_id = request.headers.get("x-session-id")
    if not session_id:
        # Use client IP and user agent as a more stable identifier
        client_ip = request.client.host if hasattr(request, 'client') and request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        # Create a hash-based session ID for more stability
        import hashlib
        session_hash = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()
        session_id = f"session_{session_hash[:8]}"
        logger.info(f"Created stable session ID {session_id} for client {client_ip}")
    
    return session_id

def get_stable_session_id(request: gr.Request) -> str:
    """Get a stable session ID that persists across requests."""
    if not request:
        logger.warning("No request object, using random session ID")
        return f"random_{str(uuid.uuid4())[:8]}"
    
    # Use a combination of client info for more stable sessions
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Add additional uniqueness factors
    accept_language = request.headers.get("accept-language", "unknown")
    accept_encoding = request.headers.get("accept-encoding", "unknown")
    
    # Create a more unique session ID
    import hashlib
    session_data = f"{client_ip}:{user_agent}:{accept_language}:{accept_encoding}"
    session_hash = hashlib.md5(session_data.encode()).hexdigest()
    session_id = f"user_{session_hash[:16]}"
    
    logger.info(f"Generated session ID: {session_id}")
    logger.info(f"Session data: {session_data}")
    
    return session_id

def get_unique_session_id(request: gr.Request) -> str:
    """Get a truly unique session ID for each request."""
    if not request:
        return f"unique_{str(uuid.uuid4())[:8]}"
    
    # Use timestamp + client info for uniqueness
    import time
    timestamp = int(time.time() * 1000)  # milliseconds
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Create a unique session ID
    import hashlib
    session_data = f"{timestamp}:{client_ip}:{user_agent}"
    session_hash = hashlib.md5(session_data.encode()).hexdigest()
    session_id = f"unique_{session_hash[:16]}"
    
    logger.info(f"Generated unique session ID: {session_id}")
    
    return session_id

def get_persistent_session_id(request: gr.Request) -> str:
    """Get a persistent session ID that stays the same for the same client."""
    if not request:
        return f"persistent_{str(uuid.uuid4())[:8]}"
    
    # Use only client info for persistence (no timestamp)
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    user_agent = request.headers.get("user-agent", "unknown")
    accept_language = request.headers.get("accept-language", "unknown")
    
    # Create a persistent session ID
    import hashlib
    session_data = f"{client_ip}:{user_agent}:{accept_language}"
    session_hash = hashlib.md5(session_data.encode()).hexdigest()
    session_id = f"persistent_{session_hash[:16]}"
    
    logger.info(f"Generated persistent session ID: {session_id}")
    logger.info(f"Session data: {session_data}")
    
    return session_id

def get_session_data(session_id: str) -> dict:
    """Get session data for a specific user."""
    with session_lock:
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                "hf_token": None,
                "agent": None,
                "max_steps": 10,
                "created_at": datetime.now()
            }
        return user_sessions[session_id]

def clear_session_data(session_id: str):
    """Clear session data for a specific user."""
    with session_lock:
        if session_id in user_sessions:
            # Clear sensitive data
            user_sessions[session_id]["hf_token"] = None
            user_sessions[session_id]["agent"] = None
            logger.info(f"Session {session_id[:8]}... cleared")

def clear_agent_only(session_id: str):
    """Clear only the agent, keeping the API key for convenience."""
    with session_lock:
        if session_id in user_sessions:
            if "agent" in user_sessions[session_id]:
                del user_sessions[session_id]["agent"]
            logger.info(f"Session {session_id[:8]}... agent cleared")



class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, file_upload_folder: str | None = None):
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)
        # No on-disk report saving; reports are rendered in-app only

    def validate_api_key(self, api_key: str) -> tuple[str, str]:
        """Validate API key and return status message."""
        is_valid, message = validate_hf_api_key(api_key)
        if is_valid:
            return message, "success"
        else:
            return message, "error"

    def interact_with_agent(self, prompt, messages, request: gr.Request):
        """Handle agent interaction with proper session management."""
        # Get unique session ID for this user
        session_id = get_persistent_session_id(request)
        session_data = get_session_data(session_id)
        
        logger.info(f"Processing request for session {session_id}...")
        logger.info(f"Request client: {request.client.host if request and request.client else 'unknown'}")
        logger.info(f"Request user-agent: {request.headers.get('user-agent', 'unknown')[:50] if request else 'unknown'}")
        logger.info(f"All active sessions: {list(user_sessions.keys())}")
        logger.info(f"Session data for {session_id}: {session_data}")
        
        # Check if we have a valid agent for this session
        if not session_data.get("agent"):
            # Check if we have a valid HF_TOKEN in session
            hf_token = session_data.get("hf_token")
            
            # If no token in session, try to get it from .env file
            if not hf_token:
                env_token = os.getenv("HF_TOKEN")
                if env_token:
                    hf_token = env_token
                    session_data["hf_token"] = env_token
                    session_data["max_steps"] = 10  # Default max_steps
                    logger.info(f"Using HF_TOKEN from .env file for session {session_id[:8]}...")
                else:
                    logger.warning(f"No API key found for session {session_id[:8]}...")
                    error_msg = "❌ No API key configured for your session. Please enter your Hugging Face API key in the API Configuration section above and click 'Setup API Key'."
                    messages.append(gr.ChatMessage(role="assistant", content=error_msg))
                    yield messages, "", ""
                    return
            
            logger.info(f"Creating agent for session {session_id[:8]}...")
            
            if hf_token:
                try:
                    max_steps = session_data.get("max_steps", 10)
                    session_data["agent"] = create_agent(hf_token, model_id=os.getenv("MODEL_ID"), max_steps=max_steps)
                    logger.info(f"Agent created successfully for session {session_id[:8]}...")
                except Exception as e:
                    logger.error(f"Failed to create agent for session {session_id[:8]}: {e}")
                    error_msg = f"❌ Failed to create agent with provided API key: {str(e)}"
                    messages.append(gr.ChatMessage(role="assistant", content=error_msg))
                    yield messages, "", ""
                    return
        else:
            logger.info(f"Agent already exists for session {session_id[:8]}...")
        
        # Adding monitoring
        try:
            # log the existence of agent memory
            has_memory = hasattr(session_data["agent"], "memory")
            print(f"Agent has memory: {has_memory}")
            if has_memory:
                print(f"Memory type: {type(session_data['agent'].memory)}")

            # Get current date for the prompt
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            
             # Prepare the system prompt (Hugging Search)
            system_prompt = f"""You are Hugging Research, an assistant focused on Hugging Face content (models, datasets, Spaces, users, collections, papers) and related learning/blog/news.

TODAY'S DATE: {current_date}

STYLE
- Warm, collaborative, concise. use second person (you)

ACCESS BOUNDARIES
- Read‑only. Use only public information.
- If a tool indicates 401/403/private/gated, state "no access" and continue with other public sources.

AVAILABLE TOOLS
- web_search, visit, page_up, page_down, find, find_next, archive_search, visualizer
- hf_models_search, hf_model_info, hf_datasets_search, hf_dataset_info, hf_spaces_search, hf_space_info
- hf_user_info, hf_collections_list, hf_collection_get, hf_paper_info, hf_paper_repos, hf_daily_papers
- hf_repo_info, hf_site_search

LINK POLICY (anti‑hallucination)
- Only cite URLs that come directly from tool outputs. Never invent or guess links.
- Prefer official huggingface.co URLs for models/datasets/Spaces/papers.
- For tutorials/blogs/news, prefer huggingface.co when the same content exists there.
- If you need a URL that isn't present, first use a tool (web_search or hf_site_search) to retrieve it, then cite it.

TOOL USAGE POLICY
- You can write compact Python to orchestrate multiple tool calls in one block.
- Never dump large/raw JSON. If using python_interpreter, ensure visible output by printing a short structured summary (<=20 lines) or leaving a final expression; otherwise summarize in natural language.
- Keep parameters minimal: include query and limit; add owner only if asked; use a single pipeline_tag or tags only if explicitly implied; use sort/direction when asked or implied (default downloads/descending; 'trending' allowed).
- Default to limit=10 for searches unless the user explicitly asks for more.
- Use web_search to capture fresh/trending context; use hf_site_search for tutorials/blog/Learn.
- Use only the listed tools; do not call undefined helpers (e.g., visit_page).
 - web_search returns plain text; never json.load or index it. Use it only for keywords or discovering links.
 - hf_* tools return JSON serialized as string; always json.loads(...) before indexing keys like 'results' or 'item'.

STARTING MOVE
- Begin with multiple web_search to capture today‑relevant terms (include "Hugging Face" in the query when helpful). Derive 3–5 keywords and reuse them across hf_* calls.

DECISION RULES
- Prefer hf_* tools for official Hub content. Use derived keywords; do not rely only on date sort.
- Stop calling tools once you have enough signal for a confident, useful answer.

FINAL STEP GUIDANCE
 - Do not call any dashboard/report tool. The app will automatically generate a dashboard from your final answer text for the Report tab. Focus on writing a clean Final Answer with accurate inline links derived from tool outputs.

OUTPUT REQUIREMENTS
- Provide a conversational summary tailored to the user’s goal.
- Structure: brief opening (what we looked for and why), key findings woven into short prose.
- Use inline links to official HF pages for repos and to reputable external sources for tutorials/news.
- Briefly mention at least one relevant item with inline links across these categories when available: models, datasets, Spaces, papers, blogs/docs, repositories, videos, news.

EXAMPLES (GOOD)
# Derive keywords then orchestrate searches
results_web = web_search(query="diffusion models Hugging Face latest")
import json
models = json.loads(hf_models_search(query="semantic search", limit=5)).get("results", [])
ds = json.loads(hf_datasets_search(query="semantic search", limit=5)).get("results", [])
repo = json.loads(hf_model_info(repo_id="sentence-transformers/all-MiniLM-L6-v2")).get("item")
spaces = json.loads(hf_spaces_search(query="whisper transcription", limit=5)).get("results", [])
learn = json.loads(hf_site_search(query="fine-tuning tutorial Hugging Face course", limit=5)).get("results", [])
# Final step: compose the final answer in natural language with inline links.
# The app will build a dashboard automatically from your final answer (no extra tool call needed).
final_answer_text = "We looked at semantic search models and datasets, including https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 ..."
 
 Now is your turn to answer the user query.
 
User Query: """

            # Combine system prompt with user message
            full_prompt = system_prompt + prompt

            # Extract clean message for display (remove internal context)
            display_message = prompt
            if "[INTERNAL CONTEXT:" in prompt:
                display_message = prompt.split("[INTERNAL CONTEXT:")[0].strip()

            messages.append(gr.ChatMessage(role="user", content=display_message))
            yield messages, "", ""

            logger.info(f"Starting agent interaction for session {session_id[:8]}...")
            latest_assistant_text = ""
            for msg in stream_to_gradio(
                session_data["agent"], task=full_prompt, reset_agent_memory=False
            ):
                # If the message contains an HTML report, just pass it through (no on-disk saving)
                # We render the dashboard in the Report tab below.
                # (Intentionally no file saving)
                messages.append(msg)
                if getattr(msg, "role", None) == "assistant" and isinstance(msg.content, str):
                    latest_assistant_text = msg.content
                yield messages, "", ""
            
            # Clear sensitive data from session after interaction (AUTOMATIC)
            # Note: We clear the agent but keep the API key for convenience
            if "agent" in session_data:
                del session_data["agent"]
            logger.info(f"Session {session_id[:8]}... agent cleared after interaction")
            
            # Build Report tab content
            last_answer = latest_assistant_text or ""
            report_md = ""
            if display_message or last_answer:
                report_md = f"### Prompt\n{display_message}\n\n{last_answer}"
            # Generate report HTML from the final answer
            dashboard_html = ""
            try:
                dashboard_html = HFLinkReportTool().forward(final_answer=last_answer, query=display_message)
            except Exception:
                dashboard_html = ""
            yield messages, report_md, dashboard_html
        except Exception as e:
            logger.error(f"Error in interaction for session {session_id[:8]}: {str(e)}")
            print(f"Error in interaction: {str(e)}")
            error_msg = f"❌ Error during interaction: {str(e)}"
            messages.append(gr.ChatMessage(role="assistant", content=error_msg))
            yield messages, "", ""

    def setup_api_key(self, api_key: str, request: gr.Request) -> str:
        """Setup API key for the user's session."""
        # Get unique session ID for this user
        session_id = get_persistent_session_id(request)
        session_data = get_session_data(session_id)
        
        logger.info(f"Setting up API key for session {session_id}...")
        logger.info(f"Setup request client: {request.client.host if request and request.client else 'unknown'}")
        logger.info(f"Setup request user-agent: {request.headers.get('user-agent', 'unknown')[:50] if request else 'unknown'}")
        logger.info(f"All active sessions before setup: {list(user_sessions.keys())}")
        logger.info(f"Session data before setup: {session_data}")
        
        # Check if API key is provided from interface
        if api_key and api_key.strip():
            # Use the API key from interface
            token_to_use = api_key.strip()
            source = "interface"
        else:
            # Try to use token from .env file
            env_token = os.getenv("HF_TOKEN")
            if env_token:
                token_to_use = env_token
                source = ".env file"
            else:
                return "❌ No API key provided. Please enter your Hugging Face API key or set HF_TOKEN in your .env file."
        
        # Validate the token
        is_valid, message = validate_hf_api_key(token_to_use)
        
        if is_valid:
            # Store HF_TOKEN in session data
            session_data["hf_token"] = token_to_use
            session_data["max_steps"] = 10
            logger.info(f"API key stored in session {session_id[:8]}... from {source}")
            logger.info(f"Max steps set to fixed value: 10")
            
            # Create new agent with the HF_TOKEN and max_steps
            try:
                session_data["agent"] = create_agent(token_to_use, model_id=os.getenv("MODEL_ID"), max_steps=10)
                logger.info(f"Agent created successfully for session {session_id[:8]}...")
                return f"✅ API key from {source} validated and agent created successfully! {message.split('!')[1] if '!' in message else ''}"
            except Exception as e:
                logger.error(f"Failed to create agent for session {session_id[:8]}: {e}")
                return f"❌ Failed to create agent with API key from {source}: {str(e)}"
        else:
            logger.warning(f"Invalid API key for session {session_id[:8]}... from {source}")
            return f"❌ Invalid API key from {source}: {message}"

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(
            self.file_upload_folder, os.path.basename(sanitized_name)
        )
        shutil.copy(file.name, file_path)

        return gr.Textbox(
            f"File uploaded: {file_path}", visible=True
        ), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        # Create the user message for display (clean, without file info)
        display_message = text_input
        
        # Create the internal message for the agent (with file context)
        internal_message = text_input
        if len(file_uploads_log) > 0:
            file_names = [os.path.basename(f) for f in file_uploads_log]
            file_paths = [f for f in file_uploads_log]  # Full paths
            # Note: inspect_file_as_text is currently disabled (only for local uploads when re-enabled)
            internal_message += f"\n\n[Uploaded files available: {', '.join(file_names)}. You can reference their content if needed (plain text).]"
        
        return (
            internal_message,  # This goes to the agent (with file context)
            gr.Textbox(
                value="",
                interactive=False,
                placeholder="Please wait while Steps are getting populated",
            ),
            gr.Button(interactive=False),
        )

    def detect_device(self, request: gr.Request):
        # Check whether the user device is a mobile or a computer

        if not request:
            return "Desktop"  # Default to desktop if no request info
        
        # Method 1: Check sec-ch-ua-mobile header (most reliable)
        is_mobile_header = request.headers.get("sec-ch-ua-mobile")
        if is_mobile_header:
            return "Mobile" if "?1" in is_mobile_header else "Desktop"

        # Method 2: Check user-agent string
        user_agent = request.headers.get("user-agent", "").lower()
        mobile_keywords = ["android", "iphone", "ipad", "mobile", "phone", "tablet"]
        
        # More comprehensive mobile detection
        if any(keyword in user_agent for keyword in mobile_keywords):
            return "Mobile"
        
        # Check for mobile-specific patterns
        if "mobile" in user_agent or "android" in user_agent or "iphone" in user_agent:
            return "Mobile"

        # Method 3: Check platform
        platform = request.headers.get("sec-ch-ua-platform", "").lower()
        if platform:
            if platform in ['"android"', '"ios"']:
                return "Mobile"
            elif platform in ['"windows"', '"macos"', '"linux"']:
                return "Desktop"

        # Method 4: Check viewport width (if available)
        viewport_width = request.headers.get("viewport-width")
        if viewport_width:
            try:
                width = int(viewport_width)
                return "Mobile" if width <= 768 else "Desktop"
            except ValueError:
                pass

        # Default case if no clear indicators
        return "Desktop"

    def launch(self, **kwargs):
        # Custom CSS for mobile optimization
        custom_css = """
        @media (max-width: 768px) {
            .gradio-container {
                max-width: 100% !important;
                padding: 10px !important;
            }
            .main {
                padding: 10px !important;
            }
            .chatbot {
                max-height: 60vh !important;
            }
            .textbox {
                font-size: 16px !important; /* Prevents zoom on iOS */
            }
            .button {
                min-height: 44px !important; /* Better touch targets */
            }
        }
        """
        
        with gr.Blocks(theme="ocean", fill_height=True, css=custom_css) as demo:
            # Different layouts for mobile and computer devices
            @gr.render()
            def layout(request: gr.Request):
                device = self.detect_device(request)
                print(f"device - {device}")
                # Render layout with sidebar
                # Prepare logo as data URI for reliable rendering
                try:
                    import base64
                    _logo_src = ""
                    _used = ""
                    for _p in ("assets/images/@image_logo.png", "assets/images/image_logo.png"):
                        if os.path.exists(_p):
                            with open(_p, "rb") as _lf:
                                _b64 = base64.b64encode(_lf.read()).decode("ascii")
                                _logo_src = f"data:image/png;base64,{_b64}"
                                _used = _p
                            break
                    print(f"Logo path used: {_used or 'none'}")
                except Exception as _e:
                    print(f"Logo load error: {_e}")
                    _logo_src = ""
                _logo_img_html = (
                    f'<img src="{_logo_src}" alt="App logo" style="vertical-align: middle; margin-right: 10px; height: 40px; display:inline-block;">'
                    if _logo_src else ""
                )
                if device == "Desktop":
                    with gr.Blocks(
                        fill_height=True,
                    ):
                        file_uploads_log = gr.State([])
                        with gr.Sidebar():
                            # Project title and repository link at the top
                            gr.Markdown(value=f"<h1 style=\"display:flex; align-items:center; gap:10px; margin:0; text-align:left;\">{_logo_img_html}Hugging Research</h1>")
                            gr.Markdown("""<img src=\"https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png\" width=\"20\" height=\"20\" style=\"display: inline-block; vertical-align: middle; margin-right: 8px;\"> <a href=\"https://github.com/mcdaqc/hugging-research\" target=\"_blank\">Github Repository</a>""")
                            
                            # About section
                            with gr.Accordion("ℹ️ About", open=False):
                                gr.Markdown("""**What it does:**
Hugging Research finds Hugging Face models, datasets, and Spaces with direct links and short summaries.

**Available tools:**
- Hugging Face Hub API endpoints (via hf_* tools) — see [Hub API](https://huggingface.co/docs/hub/en/api)
- Web search + basic navigation (DuckDuckGo `web_search`, `visit`, `page_up/down`, `find`, `archive_search`)

**Model configuration:**
- Default: Qwen/Qwen3-Coder-480B-A35B-Instruct (HF Inference API)
- Optional: Ollama/local via `.env`

**How to use:**
- Enter your Hugging Face API key
- Ask in natural language (e.g., models/datasets/spaces by topic or owner; or “I’m new to LLMs/fine‑tuning—where to start?”)
- Get concise, linked results""")
                            
                            with gr.Group():
                                gr.Markdown("**Your request**", container=True)
                                text_input = gr.Textbox(
                                    lines=3,
                                    label="Your request",
                                    container=False,
                                    placeholder="Enter your prompt here and press Shift+Enter or press the button",
                                )
                                launch_research_btn = gr.Button(
                                    "Run", variant="primary"
                                )

                            # Examples Section
                            with gr.Accordion("💡 Example Prompts", open=False):
                                gr.Markdown("**Click any example below to populate your request field:**")
                                
                                example_btn_1 = gr.Button("Basic: Tiny chatbot", size="sm", variant="secondary")
                                example_btn_2 = gr.Button("Medium: RAG Q&A", size="sm", variant="secondary")
                                example_btn_3 = gr.Button("Advanced: Instr. tuning", size="sm", variant="secondary")
                                
                                # Example button events
                                example_btn_1.click(
                                    lambda: "I want a small chatbot I can run on my laptop. Recommend a few lightweight chat models, a small dialogue dataset to fine‑tune, and a beginner‑friendly finetuning guide. Include a Space I can duplicate or clear steps to run locally.",
                                    None,
                                    [text_input]
                                )
                                example_btn_2.click(
                                    lambda: "I'm building a document Q&A RAG app. Recommend CPU‑friendly embedding models and an optional reranker, give sensible chunk size and overlap defaults, suggest a small starter dataset, link a Space I can duplicate or a repo for an end‑to‑end pipeline, and provide a short guide to evaluate answer quality.",
                                    None,
                                    [text_input]
                                )
                                example_btn_3.click(
                                    lambda: "I'm exploring instruction‑tuning and preference optimization for code LLMs. Surface recent papers from 2024 and 2025 on SFT, DPO, ORPO, and GRPO, link relevant datasets, models and repos that implement these methods, outline typical training and evaluation setups, and highlight open challenges and safety notes.",
                                    None,
                                    [text_input]
                                )

                            # API Key Configuration Section
                            with gr.Accordion("🔑 API Configuration", open=False):
                                gr.Markdown("**Configure your Hugging Face API Key**")
                                gr.Markdown("🔒 **Security**: Your API key is only kept during this session.")
                                gr.Markdown("Get your API key from: https://huggingface.co/settings/tokens")
                                
                                api_key_input = gr.Textbox(
                                    label="Hugging Face API Key",
                                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                                    type="password",
                                    lines=1
                                )
                                api_key_status = gr.Textbox(
                                    label="Status",
                                    value="✅ HF_TOKEN found in .env file. To use a different key, enter it above and click 'Setup API Key'." if os.getenv("HF_TOKEN") else "⚠️ Please enter your Hugging Face API key above and click 'Setup API Key' to start using the application.",
                                    interactive=False
                                )
                                
                                # Agent configuration (fixed steps)
                                gr.Markdown("**Agent Configuration** — steps are fixed for stability.")
                                
                                setup_api_btn = gr.Button("Setup API Key", variant="secondary")

                            # If an upload folder is provided, enable the upload feature
                            # COMMENTED: File upload feature temporarily disabled - works but consumes too many steps for parsing
                            # TODO: Re-enable after optimizing TextInspectorTool to use fewer steps
                            # if self.file_upload_folder is not None:
                            #     upload_file = gr.File(label="Upload a file")
                            #     upload_status = gr.Textbox(
                            #         label="Upload Status",
                            #         interactive=False,
                            #         visible=False,
                            #     )
                            #     upload_file.change(
                            #         self.upload_file,
                            #         [upload_file, file_uploads_log],
                            #         [upload_status, file_uploads_log],
                            #     )

                            # Powered by smolagents
                            with gr.Row():
                                gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">Powered by
                        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                        <a target="_blank" href="https://github.com/huggingface/smolagents"><b>hf/smolagents</b></a>
                        </div>""")

                        # Chat interface
                        stored_messages = gr.State([])
                        chatbot = gr.Chatbot(
                            label="open-Deep-Research",
                            type="messages",
                            avatar_images=(
                                None,
                                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                            ),
                            resizeable=False,
                            scale=1,
                            elem_id="my-chatbot",
                        )

                        # Tabs for Research and Report
                        with gr.Tabs():
                            with gr.Tab("Report"):
                                report_markdown = gr.Markdown(value="")
                                report_dashboard = gr.HTML(value="")

                        # API Key setup event
                        setup_api_btn.click(
                            self.setup_api_key,
                            [api_key_input],
                            [api_key_status]
                        )
                        


                        text_input.submit(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot],
                            [chatbot, report_markdown, report_dashboard],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )
                        launch_research_btn.click(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot],
                            [chatbot, report_markdown, report_dashboard],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )

                # Render simple layout for mobile
                else:
                    try:
                        with gr.Blocks(
                            fill_height=True,
                        ):
                            # Project title and repository link at the top
                            gr.Markdown(value=f"<h1 style=\"display:flex; align-items:center; gap:10px; margin:0; text-align:left;\">{_logo_img_html}Hugging Research</h1>")
                            gr.Markdown("""<img src=\"https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png\" width=\"20\" height=\"20\" style=\"display: inline-block; vertical-align: middle; margin-right: 8px;\"> <a href=\"https://github.com/mcdaqc/hugging-research\" target=\"_blank\">Github Repository</a>""")
                            
                            # About section for mobile
                            with gr.Accordion("ℹ️ About", open=False):
                                gr.Markdown("""**What it does:**
Hugging Research finds Hugging Face models, datasets, and Spaces with direct links and short summaries.

**Available tools:**
- Hugging Face Hub API endpoints (via hf_* tools) — see [Hub API](https://huggingface.co/docs/hub/en/api)
- Web search + basic navigation (DuckDuckGo `web_search`, `visit`, `page_up/down`, `find`, `archive_search`)

**Model configuration:**
- Default: Qwen/Qwen3-Coder-480B-A35B-Instruct (HF Inference API)
- Optional: Ollama/local via `.env`

**How to use:**
- Enter your Hugging Face API key
- Ask in natural language (e.g., models/datasets/spaces by topic or owner; or “I’m new to LLMs/fine‑tuning—where to start?”)
- Get concise, linked results""")

                        # API Key Configuration Section for Mobile
                        with gr.Accordion("🔑 API Configuration", open=False):
                            gr.Markdown("**Configure your Hugging Face API Key**")
                            gr.Markdown("🔒 **Security**: Your API key is only kept during this session.")
                            gr.Markdown("Get your API key from: https://huggingface.co/settings/tokens")
                            
                            mobile_api_key_input = gr.Textbox(
                                label="Hugging Face API Key",
                                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                                type="password",
                                lines=1
                            )
                            mobile_api_key_status = gr.Textbox(
                                label="Status",
                                value="✅ HF_TOKEN found in .env file. To use a different key, enter it above and click 'Setup API Key'." if os.getenv("HF_TOKEN") else "⚠️ Please enter your Hugging Face API key above and click 'Setup API Key' to start using the application.",
                                interactive=False
                            )
                            
                            # Agent configuration for mobile
                            gr.Markdown("**Agent Configuration**")
                            mobile_max_steps_slider = gr.Slider(
                                minimum=5,
                                maximum=30,
                                value=10,
                                step=1,
                                label="Maximum Steps",
                                info="Number of steps the agent can take per session (higher = more detailed but slower)"
                            )
                            
                            mobile_setup_api_btn = gr.Button("Setup API Key", variant="secondary")
                        
                        # Chat interface for mobile
                        stored_messages = gr.State([])
                        file_uploads_log = gr.State([])
                        chatbot = gr.Chatbot(
                            label="open-Deep-Research",
                            type="messages",
                            avatar_images=(
                                None,
                                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                            ),
                            resizeable=True,
                            scale=1,
                        )
                        
                        # Input section for mobile
                        text_input = gr.Textbox(
                            lines=1,
                            label="Your request",
                            placeholder="Enter your prompt here and press the button",
                        )
                        launch_research_btn = gr.Button(
                            "Run",
                            variant="primary",
                        )

                        # File upload section for mobile (simple)
                        # COMMENTED: File upload feature temporarily disabled - works but consumes too many steps for parsing
                        # TODO: Re-enable after optimizing
                        # if self.file_upload_folder is not None:
                        #     mobile_upload_file = gr.File(label="📎 Upload PDF/TXT file (optional)")
                        #     mobile_upload_status = gr.Textbox(
                        #         label="Upload Status",
                        #         interactive=False,
                        #         visible=False,
                        #     )
                        #     mobile_upload_file.change(
                        #         self.upload_file,
                        #         [mobile_upload_file, file_uploads_log],
                        #         [mobile_upload_status, file_uploads_log],
                        #     )

                        # Examples Section for Mobile
                        with gr.Accordion("💡 Example Prompts", open=False):
                            gr.Markdown("**Click any example below to populate your request field:**")
                            
                            mobile_example_btn_1 = gr.Button("Basic: Tiny chatbot", size="sm", variant="secondary")
                            mobile_example_btn_2 = gr.Button("Medium: RAG Q&A", size="sm", variant="secondary")
                            mobile_example_btn_3 = gr.Button("Advanced: Instr. tuning", size="sm", variant="secondary")
                        
                        # Powered by smolagents for mobile
                        with gr.Row():
                            gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">Powered by
                        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                        <a target="_blank" href="https://github.com/huggingface/smolagents"><b>hf/smolagents</b></a>
                        </div>""")

                        # Mobile API Key setup event
                        mobile_setup_api_btn.click(
                            self.setup_api_key,
                            [mobile_api_key_input],
                            [mobile_api_key_status]
                        )
                        
                        # Mobile Example button events
                        mobile_example_btn_1.click(
                            lambda: "I want a small chatbot I can run on my laptop. Recommend a few lightweight chat models, a small dialogue dataset to fine‑tune, and a beginner‑friendly finetuning guide. Include a Space I can duplicate or clear steps to run locally.",
                            None,
                            [text_input]
                        )
                        mobile_example_btn_2.click(
                            lambda: "I'm building a document Q&A RAG app. Recommend CPU‑friendly embedding models and an optional reranker, give sensible chunk size and overlap defaults, suggest a small starter dataset, link a Space I can duplicate or a repo for an end‑to‑end pipeline, and provide a short guide to evaluate answer quality.",
                            None,
                            [text_input]
                        )
                        mobile_example_btn_3.click(
                            lambda: "I'm exploring instruction‑tuning and preference optimization for code LLMs. Surface recent papers from 2024 and 2025 on SFT, DPO, ORPO, and GRPO, link relevant datasets, models and repos that implement these methods, outline typical training and evaluation setups, and highlight open challenges and safety notes.",
                            None,
                            [text_input]
                        )

                        # Research and Report panels for mobile
                        with gr.Tabs():
                            with gr.Tab("Report"):
                                m_report_markdown = gr.Markdown(value="")
                                m_report_dashboard = gr.HTML(value="")

                        # Mobile chat events
                        text_input.submit(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot],
                            [chatbot, m_report_markdown, m_report_dashboard],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )
                        launch_research_btn.click(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot],
                            [chatbot, m_report_markdown, m_report_dashboard],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )
                    except Exception as e:
                        # Fallback to desktop layout if mobile layout fails
                        logger.error(f"Mobile layout failed: {e}")
                        # Re-render desktop layout as fallback
                        with gr.Blocks(fill_height=True):
                            gr.Markdown(value=f"<h1 style=\"display:flex; align-items:center; gap:10px; margin:0; text-align:left;\">{_logo_img_html}Hugging Research</h1>")
                            gr.Markdown("""<img src=\"https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png\" width=\"20\" height=\"20\" style=\"display: inline-block; vertical-align: middle; margin-right: 8px;\"> <a href=\"https://github.com/mcdaqc/hugging-research\" target=\"_blank\">Github Repository</a>""")
                            gr.Markdown("⚠️ Mobile layout failed, using desktop layout as fallback.")
                            
                            # Simple fallback interface
                            stored_messages = gr.State([])
                            file_uploads_log = gr.State([])
                            chatbot = gr.Chatbot(
                                label="open-Deep-Research",
                                type="messages",
                                avatar_images=(
                                    None,
                                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                                ),
                            )
                            with gr.Tabs():
                                with gr.Tab("Report"):
                                    fb_report_markdown = gr.Markdown(value="")
                                    fb_report_dashboard = gr.HTML(value="")
                            
                            text_input = gr.Textbox(
                                lines=1,
                                label="Your request",
                                placeholder="Enter your prompt here and press the button",
                            )
                            launch_research_btn = gr.Button("Run", variant="primary")
                            
                            # Fallback events
                            text_input.submit(
                                self.log_user_message,
                                [text_input, file_uploads_log],
                                [stored_messages, text_input, launch_research_btn],
                            ).then(
                                self.interact_with_agent,
                                [stored_messages, chatbot],
                                [chatbot, fb_report_markdown, fb_report_dashboard],
                            )
                            launch_research_btn.click(
                                self.log_user_message,
                                [text_input, file_uploads_log],
                                [stored_messages, text_input, launch_research_btn],
                            ).then(
                                self.interact_with_agent,
                                [stored_messages, chatbot],
                                [chatbot, fb_report_markdown, fb_report_dashboard],
                            )

        
        # Configure for Hugging Face Spaces compatibility
        is_spaces = os.getenv("SPACE_ID") is not None
        
        if is_spaces:
            # Hugging Face Spaces configuration
            demo.launch(
                debug=False,
                server_name="0.0.0.0",
                server_port=int(os.getenv("PORT", 7860)),
                share=True,
                **kwargs
            )
        else:
            # Local development configuration
            demo.launch(
                debug=True,
                server_name="localhost",
                server_port=7860,
                share=False,
                **kwargs
            )

# Launch the application
if __name__ == "__main__":
    try:
        GradioUI(file_upload_folder="uploads").launch()
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        raise