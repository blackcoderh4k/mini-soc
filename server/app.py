"""
Mini SOC — FastAPI Application (OpenEnv Server)
Implements OpenEnv HTTP API: /reset, /step, /state, /tasks, /health

Uses the create_app factory pattern for session isolation.
Supports dual-import for both package-mode and Docker-mode.
"""
import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

# Dual-import pattern
try:
    from models import Action, ActionType, ResetResult, StepResult, StateResult
except ImportError:
    from ..models import Action, ActionType, ResetResult, StepResult, StateResult

try:
    from server.mini_soc_environment import SocEnvironment
except ImportError:
    from .mini_soc_environment import SocEnvironment

try:
    from server.logging_config import logger
except ImportError:
    from .logging_config import logger


# ---------------------------------------------------------------------------
# Request schemas (module-level so Pydantic can resolve annotations)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "alert_triage"
    scenario_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


class DifficultyRequest(BaseModel):
    tier: int


class InferenceRequest(BaseModel):
    prompt: Optional[str] = None
    alert_id: Optional[str] = None
    description: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.2


# ---------------------------------------------------------------------------
# App factory — OpenEnv standard pattern
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Factory function that creates and configures the FastAPI app.
    Each call returns a fresh app with its own environment instance.
    """
    application = FastAPI(
        title="Mini SOC — OpenEnv Environment",
        description="AI SOC Analyst environment for RL agent training and evaluation.",
        version="1.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single environment instance (stateful per session)
    _env = SocEnvironment()

    # Global exception handler
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # -------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------

    @application.get("/health")
    def health():
        return {"status": "ok", "env": "mini-soc", "version": "1.0.0"}

    @application.post("/reset", response_model=Dict[str, Any])
    def reset(request: ResetRequest = ResetRequest()):
        """
        Reset the environment and start a new episode.
        Returns initial observation.
        """
        task = request.task_id or "alert_triage"
        scenario = request.scenario_id
        try:
            logger.info("Resetting environment — task=%s scenario=%s", task, scenario)
            result: ResetResult = _env.reset(task_id=task, scenario_id=scenario)
            logger.info("Episode started — task=%s scenario=%s episode=%s", task, result.info.get("scenario_id"), result.info.get("episode_id"))
            return result.model_dump(mode="json")
        except ValueError as e:
            logger.warning("Invalid reset request: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Reset failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

    @application.post("/step", response_model=Dict[str, Any])
    def step(request: StepRequest):
        """
        Submit one agent action. Returns next observation, reward, done flag.
        """
        try:
            action = Action(
                action_type=ActionType(request.action_type),
                parameters=request.parameters,
            )
            result: StepResult = _env.step(action)
            logger.debug("Step — action=%s reward=%.4f done=%s", request.action_type, result.reward, result.done)
            return result.model_dump(mode="json")
        except ValueError as e:
            logger.warning("Invalid action: %s", e)
            raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
        except Exception as e:
            logger.error("Step failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Step failed: {e}")

    @application.get("/state", response_model=Dict[str, Any])
    def state():
        """
        Returns current full environment state.
        Used by graders and debugging tools.
        """
        try:
            result: StateResult = _env.state()
            return result.model_dump(mode="json")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"State failed: {e}")

    @application.get("/tasks")
    def tasks():
        """Lists all available tasks with metadata."""
        return {
            "tasks": [
                {
                    "id": "alert_triage",
                    "name": "Alert Triage",
                    "difficulty": "easy",
                    "max_steps": 15,
                    "description": "Classify 10 security alerts (mix of scenarios and noise) with correct priority.",
                    "scenarios": ["Mixed (10 alerts)"]
                },
                {
                    "id": "incident_investigation",
                    "name": "Incident Investigation",
                    "difficulty": "medium",
                    "max_steps": 20,
                    "description": "Investigate a specific incident: query logs, correlate evidence, and submit a verdict.",
                    "scenarios": ["brute_force_ssh_001", "ransomware_001", "insider_threat_001"]
                },
                {
                    "id": "threat_response",
                    "name": "Active Threat Response",
                    "difficulty": "hard",
                    "max_steps": 30,
                    "description": "Detect a multi-stage kill chain, isolate compromised assets, block attacker IPs, and write a full incident report.",
                    "scenarios": ["phishing_lateral_001", "supply_chain_001", "multi_stage_apt_001"]
                },
            ]
        }

    @application.get("/metrics")
    def metrics():
        """Returns training statistics from metrics.json for frontend dashboard."""
        metrics_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "grpo_checkpoints", "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to read metrics: %s", e)
        return {"error": "metrics.json not found"}

    @application.post("/difficulty")
    def set_difficulty(request: DifficultyRequest):
        """Set adaptive difficulty tier (1/2/3) manually for testing."""
        if request.tier not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="Tier must be 1, 2, or 3")
        # In a full implementation, this sets the tier in the environment state.
        # For now, we mock the success response.
        return {"status": "success", "tier": request.tier}

    @application.get("/scenarios")
    def scenarios():
        """List all available attack scenarios with metadata."""
        return {
            "scenarios": [
                {"id": "brute_force_ssh_001", "type": "Brute Force", "difficulty": "easy"},
                {"id": "phishing_lateral_001", "type": "Lateral Movement", "difficulty": "hard"},
                {"id": "false_positive_scan_001", "type": "False Positive", "difficulty": "easy"},
                {"id": "ransomware_001", "type": "Ransomware", "difficulty": "medium"},
                {"id": "insider_threat_001", "type": "Insider Threat", "difficulty": "medium"},
                {"id": "supply_chain_001", "type": "Supply Chain", "difficulty": "hard"},
                {"id": "multi_stage_apt_001", "type": "APT", "difficulty": "hard"},
            ]
        }

    # Global model cache to avoid reloading
    application.state.inference_pipeline = None

    @application.post("/inference")
    def run_inference(request: InferenceRequest):
        """Live inference serving using the locally merged Qwen model."""
        import torch
        from transformers import pipeline

        model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "merged_model")
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503, 
                detail="Merged model not found. Run train/merge_lora.py first."
            )

        try:
            if application.state.inference_pipeline is None:
                logger.info("Loading merged LLM for inference...")
                
                application.state.inference_pipeline = pipeline(
                    "text-generation",
                    model=model_path,
                    device="cpu", # Force CPU for 100% stability
                    torch_dtype=torch.float32 # CPUs prefer float32
                )

            pipe = application.state.inference_pipeline
            
            # Auto-generate prompt if alert data provided
            final_prompt = request.prompt
            if not final_prompt and (request.alert_id or request.description):
                final_prompt = f"You are a SOC analyst. Classify the following alert:\nAlert ID: {request.alert_id}\nDescription: {request.description}\n\nOutput only JSON with fields: classification, priority."

            if not final_prompt:
                raise HTTPException(status_code=400, detail="Missing prompt or alert data")

            # Format prompt for Qwen Instruct chat template
            messages = [{"role": "user", "content": final_prompt}]
            prompt = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            outputs = pipe(
                prompt, 
                max_new_tokens=request.max_tokens, 
                temperature=request.temperature,
                do_sample=(request.temperature > 0),
                return_full_text=False
            )
            
            return {"response": outputs[0]["generated_text"].strip()}
            
        except Exception as e:
            logger.error("Inference failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @application.get("/")
    def root():
        return {
            "name": "mini-soc",
            "description": "Mini Security Operations Center RL environment",
            "endpoints": ["/reset", "/step", "/state", "/tasks", "/health", "/metrics", "/difficulty", "/scenarios", "/inference"],
            "openenv_spec": "1.0.0",
        }

    return application


# Create the app instance using factory pattern
app = create_app()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
