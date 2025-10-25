"""
Task Executor

Executes UI automation actions (click, type, scroll, etc.) via Playwright or PyAutoGUI.
"""

import logging
import json
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    Executes actions on the UI using Playwright (recommended) or PyAutoGUI fallback.
    """
    
    def __init__(self, use_playwright: bool = True):
        """
        Initialize task executor.
        
        Args:
            use_playwright: If True, use Playwright; else PyAutoGUI
        """
        self.use_playwright = use_playwright
        self.browser = None
        self.page = None
        
        if use_playwright:
            self._init_playwright()
    
    def _init_playwright(self):
        """Initialize Playwright browser."""
        try:
            from playwright.async_api import async_playwright
            logger.info("Playwright initialized (async mode)")
        except ImportError:
            logger.warning("Playwright not installed. Install with: pip install playwright")
            self.use_playwright = False
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a single action.
        
        Args:
            action: Dict with {type, target, position, args, ...}
                - type: "click", "type", "scroll", "wait", "navigate"
                - target: UI element identifier (e.g., button label)
                - position: (x, y) coordinates (for click)
                - args: Additional arguments (e.g., text for type)
        
        Returns:
            True if successful, False otherwise
        """
        action_type = action.get("type", "").lower()
        
        try:
            if action_type == "click":
                return self._handle_click(action)
            elif action_type == "type":
                return self._handle_type(action)
            elif action_type == "scroll":
                return self._handle_scroll(action)
            elif action_type == "wait":
                return self._handle_wait(action)
            elif action_type == "navigate":
                return self._handle_navigate(action)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return False
    
    def _handle_click(self, action: Dict) -> bool:
        """Handle click action."""
        position = action.get("position", None)
        target = action.get("target", None)
        
        if not position:
            logger.warning("No position for click action")
            return False
        
        x, y = position
        
        if self.use_playwright and self.page:
            try:
                self.page.click(f"button:has-text('{target}')")
                logger.info(f"Clicked {target}")
                return True
            except:
                pass
        
        # Fallback: PyAutoGUI
        try:
            import pyautogui
            pyautogui.click(x, y)
            logger.info(f"Clicked at ({x}, {y})")
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False
    
    def _handle_type(self, action: Dict) -> bool:
        """Handle type action."""
        text = action.get("args", {}).get("text", "")
        
        if not text:
            logger.warning("No text for type action")
            return False
        
        try:
            import pyautogui
            pyautogui.typewrite(text)
            logger.info(f"Typed: {text}")
            return True
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return False
    
    def _handle_scroll(self, action: Dict) -> bool:
        """Handle scroll action."""
        direction = action.get("args", {}).get("direction", "down")
        amount = action.get("args", {}).get("amount", 3)
        
        try:
            import pyautogui
            if direction.lower() == "down":
                pyautogui.scroll(-amount)
            else:
                pyautogui.scroll(amount)
            logger.info(f"Scrolled {direction} by {amount}")
            return True
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return False
    
    def _handle_wait(self, action: Dict) -> bool:
        """Handle wait action."""
        seconds = action.get("args", {}).get("seconds", 1)
        time.sleep(seconds)
        logger.info(f"Waited {seconds} seconds")
        return True
    
    def _handle_navigate(self, action: Dict) -> bool:
        """Handle navigate action."""
        url = action.get("args", {}).get("url", "")
        
        if self.use_playwright and self.page:
            try:
                self.page.goto(url)
                logger.info(f"Navigated to {url}")
                return True
            except Exception as e:
                logger.error(f"Navigate failed: {e}")
                return False
        
        logger.warning("Playwright not available for navigate")
        return False
    
    def close(self):
        """Close browser/session."""
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()


class ActionParser:
    """Parses LLM-generated action strings into executable action dicts."""
    
    @staticmethod
    def parse_json_action(action_str: str) -> Dict[str, Any]:
        """
        Parse JSON action string.
        
        Expected format:
            {
                "action": "click",
                "target": "Save Button",
                "position": [125, 65]
            }
        
        Args:
            action_str: JSON string
        
        Returns:
            Parsed action dict
        """
        try:
            action = json.loads(action_str)
            # Rename "action" to "type" for TaskExecutor
            if "action" in action:
                action["type"] = action.pop("action")
            return action
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON action: {e}")
            return {}
    
    @staticmethod
    def parse_natural_action(text: str) -> Dict[str, Any]:
        """
        Parse natural language action description.
        
        Examples:
            "Click on the Save button"
            "Type hello world"
            "Scroll down 3 times"
        """
        text_lower = text.lower()
        
        if "click" in text_lower:
            # Extract target between 'on' and other keywords
            target = text.split("click on")[-1].split("at")[0].strip()
            return {"type": "click", "target": target}
        
        elif "type" in text_lower:
            # Extract text after 'type'
            parts = text.split("type")
            if len(parts) > 1:
                text_to_type = parts[1].strip().strip("'\"")
                return {"type": "type", "args": {"text": text_to_type}}
        
        elif "scroll" in text_lower:
            direction = "down" if "down" in text_lower else "up"
            return {"type": "scroll", "args": {"direction": direction}}
        
        elif "wait" in text_lower:
            return {"type": "wait", "args": {"seconds": 1}}
        
        return {}
