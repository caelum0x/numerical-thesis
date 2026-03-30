"""
Integration layer — connects autoresearch and MiroFish into the main pipeline.

Modules:
    autoresearch_bridge : Reads best model config from autoresearch experiments
    mirofish_bridge     : Reads MiroFish swarm signals for risk overlay + features
    orchestrator        : End-to-end closed-loop runner
"""

from src.integration.autoresearch_bridge import AutoResearchBridge
from src.integration.mirofish_bridge import MiroFishBridge
from src.integration.feedback_loop import run_feedback_loop
