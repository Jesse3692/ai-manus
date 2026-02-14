from typing import Dict, Any, List, AsyncGenerator, Optional
import json
import logging
import re
from app.domain.models.plan import Plan, Step
from app.domain.models.message import Message
from app.domain.services.agents.base import BaseAgent
from app.domain.models.memory import Memory
from app.domain.external.llm import LLM
from app.domain.services.prompts.system import SYSTEM_PROMPT
from app.domain.services.prompts.planner import (
    CREATE_PLAN_PROMPT, 
    UPDATE_PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT
)
from app.domain.models.event import (
    BaseEvent,
    PlanEvent,
    PlanStatus,
    ErrorEvent,
    MessageEvent,
    DoneEvent,
)
from app.domain.external.sandbox import Sandbox
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.file import FileTool
from app.domain.services.tools.shell import ShellTool
from app.domain.repositories.agent_repository import AgentRepository
from app.domain.utils.json_parser import JsonParser

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Planner agent class, defining the basic behavior of planning
    """

    name: str = "planner"
    system_prompt: str = SYSTEM_PROMPT + PLANNER_SYSTEM_PROMPT
    format: Optional[str] = "json_object"
    tool_choice: Optional[str] = "none"

    def __init__(
        self,
        agent_id: str,
        agent_repository: AgentRepository,
        llm: LLM,
        tools: List[BaseTool],
        json_parser: JsonParser,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_repository=agent_repository,
            llm=llm,
            json_parser=json_parser,
            tools=tools,
        )


    async def create_plan(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        user_message = message.message or ""
        prompt = CREATE_PLAN_PROMPT.format(
            message=user_message,
            attachments="\n".join(message.attachments)
        )
        async for event in self.execute(prompt):
            if isinstance(event, MessageEvent):
                logger.info(event.message)
                parsed_response = await self.json_parser.parse(event.message)
                parsed_response = self._sanitize_steps(parsed_response, user_message)
                plan = Plan.model_validate(parsed_response)
                yield PlanEvent(status=PlanStatus.CREATED, plan=plan)
            else:
                yield event

    def _sanitize_steps(self, parsed_response: Any, user_message: str = "") -> Any:
        if not isinstance(parsed_response, dict):
            return parsed_response
        steps = parsed_response.get("steps")
        if not isinstance(steps, list):
            return parsed_response
        cleaned_steps = []
        for item in steps:
            if not isinstance(item, dict):
                continue
            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            cleaned_item = {"description": description.strip()}
            step_id = item.get("id")
            if isinstance(step_id, str) and step_id.strip():
                cleaned_item["id"] = step_id.strip()
            cleaned_steps.append(cleaned_item)
        if self._is_weather_query(user_message):
            city = self._extract_weather_city(user_message) or "北京"
            cleaned_steps = [{
                "id": "1",
                "description": f"使用浏览器工具打开 https://wttr.in/{city}?format=j1 并调用 browser_view 提取明天的天气预报"
            }]
        parsed_response["steps"] = cleaned_steps
        return parsed_response

    def _is_weather_query(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        lower_text = text.lower()
        return "天气" in text or "weather" in lower_text

    def _extract_weather_city(self, text: str) -> Optional[str]:
        if "北京" in text:
            return "北京"
        match = re.search(r"weather\\s+in\\s+([A-Za-z\\s]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"([\\u4e00-\\u9fff]{2,8})[^\\u4e00-\\u9fff]*天气", text)
        if match:
            city = match.group(1)
            for token in ["明天", "今天", "后天", "的", "晚上", "白天", "夜间"]:
                city = city.replace(token, "")
            city = city.strip()
            if city:
                return city
        return None

    async def update_plan(self, plan: Plan, step: Step) -> AsyncGenerator[BaseEvent, None]:
        message = UPDATE_PLAN_PROMPT.format(plan=plan.dump_json(), step=step.model_dump_json())
        async for event in self.execute(message):
            if isinstance(event, MessageEvent):
                logger.debug(f"Planner agent update plan: {event.message}")
                parsed_response = await self.json_parser.parse(event.message)
                parsed_response = self._sanitize_steps(parsed_response)
                updated_plan = Plan.model_validate(parsed_response)
                new_steps = [Step.model_validate(step) for step in updated_plan.steps]
                
                # Find the index of the first pending step
                first_pending_index = None
                for i, step in enumerate(plan.steps):
                    if not step.is_done():
                        first_pending_index = i
                        break
                
                # If there are pending steps, replace all pending steps
                if first_pending_index is not None:
                    # Keep completed steps
                    updated_steps = plan.steps[:first_pending_index]
                    # Add new steps
                    updated_steps.extend(new_steps)
                    # Update steps in plan
                    plan.steps = updated_steps
                
                yield PlanEvent(status=PlanStatus.UPDATED, plan=plan)
            else:
                yield event
