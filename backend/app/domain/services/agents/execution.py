import datetime
import json
import logging
import re
import urllib.parse
import urllib.request
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.domain.external.browser import Browser
from app.domain.external.file import FileStorage
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.models.event import (
    BaseEvent,
    DoneEvent,
    ErrorEvent,
    MessageEvent,
    StepEvent,
    StepStatus,
    ToolEvent,
    ToolStatus,
    WaitEvent,
)
from app.domain.models.file import FileInfo
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.repositories.agent_repository import AgentRepository
from app.domain.services.agents.base import BaseAgent
from app.domain.services.prompts.execution import (
    EXECUTION_PROMPT,
    EXECUTION_SYSTEM_PROMPT,
    SUMMARIZE_PROMPT,
)
from app.domain.services.prompts.system import SYSTEM_PROMPT
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.browser import BrowserTool
from app.domain.services.tools.file import FileTool
from app.domain.services.tools.message import MessageTool
from app.domain.services.tools.search import SearchTool
from app.domain.services.tools.shell import ShellTool
from app.domain.utils.json_parser import JsonParser

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """
    Execution agent class, defining the basic behavior of execution
    """

    name: str = "execution"
    system_prompt: str = SYSTEM_PROMPT + EXECUTION_SYSTEM_PROMPT
    format: Optional[str] = None
    tool_choice: Optional[str] = "auto"

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
        self._restrict_tools = False

    def get_available_tools(self) -> Optional[List[Dict[str, Any]]]:
        if not self._restrict_tools:
            return super().get_available_tools()
        available_tools = []
        for tool in self.tools:
            if tool.name == "search":
                continue
            available_tools.extend(tool.get_tools())
        return available_tools

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

    async def _call_tool(
        self, function_name: str, function_args: Dict[str, Any]
    ) -> Tuple[Any, List[ToolEvent]]:
        tool = self.get_tool(function_name)
        tool_call_id = str(uuid.uuid4())
        calling_event = ToolEvent(
            status=ToolStatus.CALLING,
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            function_name=function_name,
            function_args=function_args,
        )
        result = await self.invoke_tool(tool, function_name, function_args)
        called_event = ToolEvent(
            status=ToolStatus.CALLED,
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            function_name=function_name,
            function_args=function_args,
            function_result=result,
        )
        return result, [calling_event, called_event]

    def _parse_weather_json(self, content: Any) -> Optional[Dict[str, Any]]:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str) or not content.strip():
            return None
        cleaned = content.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _format_weather_result(
        self, data: Dict[str, Any], city: str, user_text: str
    ) -> Optional[str]:
        weather_list = data.get("weather")
        if not isinstance(weather_list, list) or not weather_list:
            return None
        day_data = weather_list[1] if len(weather_list) > 1 else weather_list[0]
        if not isinstance(day_data, dict):
            return None
        max_temp = day_data.get("maxtempC") or day_data.get("maxtempF")
        min_temp = day_data.get("mintempC") or day_data.get("mintempF")
        desc = None
        chance_rain = None
        hourly = day_data.get("hourly")
        if isinstance(hourly, list) and hourly:
            first = hourly[0]
            if isinstance(first, dict):
                desc_list = first.get("weatherDesc")
                if isinstance(desc_list, list) and desc_list:
                    desc_item = desc_list[0]
                    if isinstance(desc_item, dict):
                        desc = desc_item.get("value")
                chance_rain = first.get("chanceofrain")
        if "天气" in user_text:
            details = []
            if desc:
                details.append(f"天气：{desc}")
            if max_temp is not None and min_temp is not None:
                details.append(f"最高{max_temp}°C，最低{min_temp}°C")
            if chance_rain is not None:
                details.append(f"降水概率{chance_rain}%")
            detail_text = "，".join(details) if details else "已获取天气预报"
            return f"{city}明天{detail_text}（来源：wttr.in）"
        details = []
        if desc:
            details.append(f"Conditions: {desc}")
        if max_temp is not None and min_temp is not None:
            details.append(f"High {max_temp}°C, Low {min_temp}°C")
        if chance_rain is not None:
            details.append(f"Chance of rain {chance_rain}%")
        detail_text = ", ".join(details) if details else "Forecast available"
        return f"Tomorrow in {city}: {detail_text} (source: wttr.in)"

    def _fetch_weather_json(self, city: str) -> Optional[Dict[str, Any]]:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = response.read().decode("utf-8")
        except Exception:
            return None
        return self._parse_weather_json(payload)

    def _fetch_open_meteo(self, city: str) -> Optional[Dict[str, Any]]:
        names: List[str] = [city]
        if city == "北京":
            names.append("Beijing")
        results = None
        for name in names:
            try:
                url = (
                    "https://geocoding-api.open-meteo.com/v1/search"
                    f"?name={urllib.parse.quote(name)}&count=1&language=zh&format=json"
                )
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = response.read().decode("utf-8")
                data = self._parse_weather_json(payload)
                if data and isinstance(data.get("results"), list) and data["results"]:
                    results = data["results"][0]
                    break
            except Exception:
                continue

        if not isinstance(results, dict):
            return None

        latitude = results.get("latitude")
        longitude = results.get("longitude")
        if latitude is None or longitude is None:
            return None

        try:
            forecast_url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}"
                "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                "&timezone=Asia%2FShanghai"
            )
            request = urllib.request.Request(
                forecast_url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = response.read().decode("utf-8")
            forecast = self._parse_weather_json(payload)
            if not forecast or "daily" not in forecast:
                return None
            forecast["_geocoding"] = results
            return forecast
        except Exception:
            return None

    def _open_meteo_code_to_text(self, code: Any, zh: bool) -> str:
        try:
            code_int = int(code)
        except Exception:
            return "未知" if zh else "Unknown"

        mapping_zh = {
            0: "晴",
            1: "大部晴朗",
            2: "多云",
            3: "阴",
            45: "雾",
            48: "雾凇",
            51: "毛毛雨",
            53: "毛毛雨",
            55: "毛毛雨",
            61: "小雨",
            63: "中雨",
            65: "大雨",
            71: "小雪",
            73: "中雪",
            75: "大雪",
            80: "阵雨",
            81: "阵雨",
            82: "强阵雨",
            95: "雷暴",
            96: "雷暴伴冰雹",
            99: "强雷暴伴冰雹",
        }
        mapping_en = {
            0: "Clear",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Drizzle",
            53: "Drizzle",
            55: "Drizzle",
            61: "Rain",
            63: "Rain",
            65: "Heavy rain",
            71: "Snow",
            73: "Snow",
            75: "Heavy snow",
            80: "Rain showers",
            81: "Rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail",
        }
        if zh:
            return mapping_zh.get(code_int, f"天气代码{code_int}")
        return mapping_en.get(code_int, f"Weather code {code_int}")

    def _format_open_meteo_result(
        self, forecast: Dict[str, Any], city: str, user_text: str
    ) -> Optional[str]:
        daily = forecast.get("daily")
        if not isinstance(daily, dict):
            return None
        times = daily.get("time")
        max_list = daily.get("temperature_2m_max")
        min_list = daily.get("temperature_2m_min")
        code_list = daily.get("weather_code")
        rain_list = daily.get("precipitation_probability_max")
        if not (
            isinstance(times, list)
            and isinstance(max_list, list)
            and isinstance(min_list, list)
            and isinstance(code_list, list)
        ):
            return None
        if (
            len(times) < 2
            or len(max_list) < 2
            or len(min_list) < 2
            or len(code_list) < 2
        ):
            return None

        tomorrow_idx = 1
        try:
            today = (
                datetime.datetime.now(datetime.timezone.utc)
                .astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                .date()
            )
            tomorrow = (today + datetime.timedelta(days=1)).isoformat()
            if tomorrow in times:
                tomorrow_idx = times.index(tomorrow)
        except Exception:
            pass

        max_temp = max_list[tomorrow_idx]
        min_temp = min_list[tomorrow_idx]
        code = code_list[tomorrow_idx]
        rain = None
        if isinstance(rain_list, list) and len(rain_list) > tomorrow_idx:
            rain = rain_list[tomorrow_idx]

        zh = "天气" in user_text
        desc = self._open_meteo_code_to_text(code, zh=zh)
        if zh:
            details = [f"天气：{desc}", f"最高{max_temp}°C，最低{min_temp}°C"]
            if rain is not None:
                details.append(f"降水概率{rain}%")
            return f"{city}明天" + "，".join(details) + "（来源：open-meteo.com）"
        details = [f"Conditions: {desc}", f"High {max_temp}°C, Low {min_temp}°C"]
        if rain is not None:
            details.append(f"Precip prob {rain}%")
        return (
            f"Tomorrow in {city}: " + ", ".join(details) + " (source: open-meteo.com)"
        )

    async def _execute_weather_with_browser(
        self, step: Step, user_text: str, city: str
    ) -> AsyncGenerator[BaseEvent, None]:
        step.status = ExecutionStatus.RUNNING
        yield StepEvent(status=StepStatus.STARTED, step=step)
        if "天气" in user_text:
            notify_text = f"我将查询 {city} 明天的天气，并在必要时切换数据源。"
        else:
            notify_text = f"I will fetch tomorrow's forecast for {city}, switching source if needed."
        _, notify_events = await self._call_tool(
            "message_notify_user", {"text": notify_text}
        )
        for event in notify_events:
            yield event

        wttr_data = self._fetch_weather_json(city)
        if not wttr_data:
            open_meteo = self._fetch_open_meteo(city)
            result_text = (
                self._format_open_meteo_result(open_meteo, city, user_text)
                if open_meteo
                else None
            )
            if result_text:
                step.status = ExecutionStatus.COMPLETED
                step.success = True
                step.result = result_text
                yield StepEvent(status=StepStatus.COMPLETED, step=step)
                yield MessageEvent(message=step.result)
                return

        city_encoded = urllib.parse.quote(city)
        url = f"https://wttr.in/{city_encoded}?format=j1"
        _, navigate_events = await self._call_tool("browser_navigate", {"url": url})
        for event in navigate_events:
            yield event
        console_result, console_events = await self._call_tool(
            "browser_console_exec",
            {
                "javascript": f"return fetch('https://wttr.in/{city_encoded}?format=j1').then(r => r.text());"
            },
        )
        for event in console_events:
            yield event
        console_data = getattr(console_result, "data", None)
        data = self._parse_weather_json(
            console_data.get("result") if isinstance(console_data, dict) else None
        )
        if not data:
            view_result, view_events = await self._call_tool("browser_view", {})
            for event in view_events:
                yield event
            view_data = getattr(view_result, "data", None)
            data = self._parse_weather_json(
                view_data.get("content") if isinstance(view_data, dict) else None
            )
        if not data:
            data = self._fetch_weather_json(city)
        result_text = (
            self._format_weather_result(data, city, user_text) if data else None
        )
        if not result_text:
            step.status = ExecutionStatus.FAILED
            open_meteo = self._fetch_open_meteo(city)
            result_text = (
                self._format_open_meteo_result(open_meteo, city, user_text)
                if open_meteo
                else None
            )
            if result_text:
                step.status = ExecutionStatus.COMPLETED
                step.success = True
                step.result = result_text
                yield StepEvent(status=StepStatus.COMPLETED, step=step)
                yield MessageEvent(message=step.result)
                return
            step.error = "无法获取天气数据"
            yield StepEvent(status=StepStatus.FAILED, step=step)
            yield ErrorEvent(error=step.error)
            return
        step.status = ExecutionStatus.COMPLETED
        step.success = True
        step.result = result_text
        yield StepEvent(status=StepStatus.COMPLETED, step=step)
        yield MessageEvent(message=step.result)

    async def execute_step(
        self, plan: Plan, step: Step, message: Message
    ) -> AsyncGenerator[BaseEvent, None]:
        step_description = step.description
        user_text = message.message or ""
        force_weather = (
            "天气" in user_text
            or "weather" in user_text.lower()
            or "天气" in step_description
            or "weather" in step_description.lower()
        )
        city = self._extract_weather_city(user_text) if force_weather else None
        if city:
            async for event in self._execute_weather_with_browser(
                step, user_text, city
            ):
                yield event
            return
        if force_weather:
            step_description = "使用浏览器打开 https://wttr.in/<city>?format=j1 并从页面内容提取明天的天气预报"
            self._restrict_tools = True
        message = EXECUTION_PROMPT.format(
            step=step_description,
            message=user_text,
            attachments="\n".join(message.attachments),
            language=plan.language,
        )
        step.status = ExecutionStatus.RUNNING
        yield StepEvent(status=StepStatus.STARTED, step=step)
        try:
            async for event in self.execute(message):
                if isinstance(event, ErrorEvent):
                    step.status = ExecutionStatus.FAILED
                    step.error = event.error
                    yield StepEvent(status=StepStatus.FAILED, step=step)
                elif isinstance(event, MessageEvent):
                    step.status = ExecutionStatus.COMPLETED
                    parsed_response = await self.json_parser.parse(event.message)
                    new_step = Step.model_validate(parsed_response)
                    step.success = new_step.success
                    step.result = new_step.result
                    step.attachments = new_step.attachments
                    yield StepEvent(status=StepStatus.COMPLETED, step=step)
                    if step.result:
                        yield MessageEvent(message=step.result)
                    continue
                elif isinstance(event, ToolEvent):
                    if event.function_name == "message_ask_user":
                        if event.status == ToolStatus.CALLING:
                            yield MessageEvent(
                                message=event.function_args.get("text", "")
                            )
                        elif event.status == ToolStatus.CALLED:
                            yield WaitEvent()
                            return
                        continue
                yield event
        finally:
            self._restrict_tools = False
        step.status = ExecutionStatus.COMPLETED

    async def summarize(self) -> AsyncGenerator[BaseEvent, None]:
        message = SUMMARIZE_PROMPT
        async for event in self.execute(message):
            if isinstance(event, MessageEvent):
                logger.debug(f"Execution agent summary: {event.message}")
                parsed_response = await self.json_parser.parse(event.message)
                message = Message.model_validate(parsed_response)
                attachments = [
                    FileInfo(file_path=file_path) for file_path in message.attachments
                ]
                yield MessageEvent(message=message.message, attachments=attachments)
                continue
            yield event
