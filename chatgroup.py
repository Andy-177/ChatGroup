import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog, colorchooser
import json
import os
import threading
import queue
import time
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
import requests
from enum import Enum
import re
from pathlib import Path

# -------------------------- 常量定义 --------------------------
PROMPT_DIR = Path("prompt")
ROBOT_DIR = Path("robot")
PROMPT_DIR.mkdir(exist_ok=True)
ROBOT_DIR.mkdir(exist_ok=True)
DEFAULT_ROBOT_COLORS = [
    "#228b22",  # 绿色
    "#9933ff",  # 紫色
    "#ff6600",  # 橙色
    "#cc0000"   # 红色
]

# -------------------------- 数据模型定义 --------------------------
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    FUNCTION = "function"
    TOOL = "tool"

class Message(BaseModel):
    role: Role
    content: str
    name: str  # 发送者名字
    target: Optional[str] = None  # 接收者名字，用于@功能

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    n: Optional[int] = Field(1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    presence_penalty: Optional[float] = Field(0.0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class AIClient(BaseModel):
    api_key: str
    base_url: HttpUrl = Field(default="https://api.openai.com/v1")
    timeout: int = Field(30, ge=1)
    proxy: Optional[str] = None
    model: str = "gpt-3.5-turbo"

    class Config:
        arbitrary_types_allowed = True

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _validate_base_url(self) -> None:
        url_str = str(self.base_url)
        if not re.match(r'^https?://.+/v\d+$', url_str):
            raise ValueError(
                f"无效的API基础URL格式: {url_str}\n"
                "正确格式示例: https://api.openai.com/v1 或 http://xxx.xxx.xxx/v1"
            )

    def chat_completion(self, request: ChatCompletionRequest, robot_name: str = None) -> ChatCompletionResponse:
        self._validate_base_url()

        url_str = str(self.base_url).rstrip('/')
        url = f"{url_str}/chat/completions"

        request_kwargs = {
            "url": url,
            "headers": self._get_headers(),
            "data": request.model_dump_json(),
            "timeout": self.timeout
        }

        if self.proxy:
            request_kwargs["proxies"] = {
                "http": self.proxy,
                "https": self.proxy
            }

        try:
            response = requests.post(**request_kwargs)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                raise ValueError(
                    f"API返回了非JSON响应 (类型: {content_type})\n"
                    f"响应内容: {response.text[:300]}..."
                )

            try:
                response_data = response.json()
            except json.JSONDecodeError:
                raise ValueError(
                    f"无法解析API响应为JSON\n"
                    f"响应内容: {response.text[:300]}..."
                )

            if robot_name:
                for choice in response_data.get('choices', []):
                    if choice.get('message', {}).get('role') not in [r.value for r in Role]:
                        choice['message']['role'] = 'assistant'
                    if 'name' not in choice.get('message', {}):
                        choice['message']['name'] = robot_name

            return ChatCompletionResponse(**response_data)

        except requests.exceptions.SSLError:
            raise ValueError("SSL 错误，请检查 API 地址是否为 HTTPS 或配置代理")
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求失败: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\n状态码: {e.response.status_code}"
                error_msg += f"\n响应内容: {e.response.text[:300]}..."
            raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"处理响应失败: {str(e)}")

# -------------------------- 机器人配置模型 --------------------------
class RobotConfig(BaseModel):
    name: str  # 机器人唯一名称
    prompt: str = ""  # 机器人专属提示词
    prompt_file: Optional[str] = None  # 关联的prompt文件
    enabled: bool = True  # 是否默认启用
    auto_respond_to_ai: bool = True  # 是否自动回应其他AI的消息
    auto_respond_to_all: bool = False  # 是否自动回应所有公共消息
    color: Optional[str] = None  # 机器人在聊天中的显示颜色

# -------------------------- 配置管理 --------------------------
class ConfigManager:
    def __init__(self):
        self.main_config_file = "main_config.json"
        self.default_main_config = {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "proxy": "",
            "default_robots": [],  # 默认加载的机器人名称列表
            "user_name": "用户",  # 默认用户名
            "allow_ai_conversations": True,  # 是否允许AI间自动对话
            "max_ai_conversation_turns": 10,  # 最大自动对话轮次
        }
        self.load_main_config()

    def load_main_config(self):
        try:
            if os.path.exists(self.main_config_file):
                with open(self.main_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.main_config = {**self.default_main_config, **config}
            else:
                self.main_config = self.default_main_config.copy()
                self.save_main_config()
            return self.main_config
        except Exception as e:
            messagebox.showerror("配置错误", f"加载主配置失败: {str(e)}\n使用默认配置")
            self.main_config = self.default_main_config.copy()
            return self.main_config

    def save_main_config(self):
        try:
            with open(self.main_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.main_config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            messagebox.showerror("配置错误", f"保存主配置失败: {str(e)}")
            return False

    def list_robots(self) -> List[str]:
        return [f.stem for f in ROBOT_DIR.glob("*.json")]

    def load_robot(self, robot_name: str) -> Optional[RobotConfig]:
        robot_file = ROBOT_DIR / f"{robot_name}.json"
        if not robot_file.exists():
            return None
        try:
            with open(robot_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "auto_respond_to_ai" not in data:
                    data["auto_respond_to_ai"] = True
                if "auto_respond_to_all" not in data:
                    data["auto_respond_to_all"] = False
                if "color" not in data:
                    all_robots = sorted(self.list_robots())
                    index = all_robots.index(robot_name)
                    data["color"] = DEFAULT_ROBOT_COLORS[index % len(DEFAULT_ROBOT_COLORS)]
                return RobotConfig(**data)
        except Exception as e:
            messagebox.showerror("机器人配置错误", f"加载{robot_name}失败: {str(e)}")
            return None

    def load_all_robots(self) -> Dict[str, RobotConfig]:
        all_robots = {}
        for robot_name in self.list_robots():
            robot_config = self.load_robot(robot_name)
            if robot_config:
                all_robots[robot_name] = robot_config
        return all_robots

    def save_robot(self, robot_config: RobotConfig) -> bool:
        if not robot_config.name.strip():
            messagebox.showerror("错误", "机器人名称不能为空")
            return False

        if robot_config.name == self.get("user_name"):
            messagebox.showerror("错误", f"不能使用用户名'{robot_config.name}'作为机器人名")
            return False

        existing_robots = self.list_robots()
        if robot_config.name in existing_robots:
            existing_config = self.load_robot(robot_config.name)
            if not existing_config or existing_config.name != robot_config.name:
                messagebox.showerror("错误", f"机器人'{robot_config.name}'已存在")
                return False

        robot_file = ROBOT_DIR / f"{robot_config.name}.json"
        try:
            with open(robot_file, 'w', encoding='utf-8') as f:
                json.dump(robot_config.model_dump(), f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            messagebox.showerror("错误", f"保存{robot_config.name}失败: {str(e)}")
            return False

    def delete_robot(self, robot_name: str) -> bool:
        robot_file = ROBOT_DIR / f"{robot_name}.json"
        if not robot_file.exists():
            return False
        try:
            os.remove(robot_file)
            default_robots = self.get("default_robots")
            if robot_name in default_robots:
                default_robots.remove(robot_name)
                self.set("default_robots", default_robots)
            return True
        except Exception as e:
            messagebox.showerror("错误", f"删除{robot_name}失败: {str(e)}")
            return False

    def list_prompts(self) -> List[str]:
        return [f.name for f in PROMPT_DIR.glob("*.txt")]

    def load_prompt(self, prompt_filename: str) -> Optional[str]:
        prompt_file = PROMPT_DIR / prompt_filename
        if not prompt_file.exists():
            return None
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            messagebox.showerror("Prompt错误", f"加载{prompt_filename}失败: {str(e)}")
            return None

    def save_prompt(self, prompt_filename: str, content: str) -> bool:
        if not prompt_filename.endswith(".txt"):
            prompt_filename += ".txt"
        prompt_file = PROMPT_DIR / prompt_filename
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            messagebox.showerror("Prompt错误", f"保存{prompt_filename}失败: {str(e)}")
            return False

    def get(self, key: str, default=None):
        return self.main_config.get(key, default)

    def set(self, key: str, value):
        self.main_config[key] = value
        return self.save_main_config()

# -------------------------- GUI应用 --------------------------
class AIGroupChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI群组聊天助手 - 支持@功能")
        self.root.geometry("1000x700")
        self.root.minsize(800, 500)

        self.config_manager = ConfigManager()
        self.active_robots: Dict[str, RobotConfig] = {}
        self.chat_history: List[Message] = []
        self.ai_client: Optional[AIClient] = None
        self.ai_conversation_turns = 0  # 跟踪AI自动对话轮次
        self.infinite_loop = False  # 无限轮回选项，不保存到配置
        self.dev_mode = False  # 开发者模式，临时性的，不保存到配置
        self.dev_windows_lock = threading.Lock()  # 用于线程安全创建窗口

        self.message_queue = queue.Queue()
        self.ai_processing = False

        self.create_widgets()
        self.load_all_robots_on_start()
        self.init_ai_client()
        self.start_message_processor()

    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 左侧：机器人管理面板
        robot_frame = ttk.LabelFrame(main_pane, text="活跃机器人", padding="10")
        main_pane.add(robot_frame, weight=1)
        self.robot_listbox = tk.Listbox(robot_frame, font=("SimHei", 10), selectbackground="#4a86e8")
        self.robot_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.update_robot_list()
        robot_btn_frame = ttk.Frame(robot_frame)
        robot_btn_frame.pack(fill=tk.X)
        ttk.Button(robot_btn_frame, text="新建机器人", command=self.open_robot_editor).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(robot_btn_frame, text="编辑选中", command=self.edit_selected_robot).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(robot_btn_frame, text="删除选中", command=self.delete_selected_robot).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(robot_btn_frame, text="刷新列表", command=self.update_robot_list).pack(side=tk.RIGHT, padx=2, pady=2)
        # 添加无限轮回选项
        loop_frame = ttk.Frame(robot_frame)
        loop_frame.pack(fill=tk.X, pady=10)
        self.infinite_loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            loop_frame,
            text="无限轮回对话",
            variable=self.infinite_loop_var,
            command=self.toggle_infinite_loop
        ).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Label(loop_frame, text="(忽略轮次限制，持续对话)").pack(side=tk.LEFT, padx=5)
        # 添加开发者模式选项
        dev_frame = ttk.Frame(robot_frame)
        dev_frame.pack(fill=tk.X, pady=5)
        self.dev_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            dev_frame,
            text="开发者模式",
            variable=self.dev_mode_var,
            command=self.toggle_dev_mode
        ).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Label(dev_frame, text="(显示AI提示词分发详情)").pack(side=tk.LEFT, padx=5)
        # 右侧：聊天主区域
        chat_frame = ttk.Frame(main_pane)
        main_pane.add(chat_frame, weight=9)
        history_frame = ttk.LabelFrame(chat_frame, text="聊天记录", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.chat_text = scrolledtext.ScrolledText(
            history_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg="#f5f5f5", font=("SimHei", 10)
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        self.chat_text.tag_config("user", foreground="#0066cc", font=("SimHei", 10, "bold"))
        self.chat_text.tag_config("system", foreground="#8b4513", font=("SimHei", 9, "italic"))
        self.chat_text.tag_config("mention", foreground="#ff4500", font=("SimHei", 10, "bold"))
        input_frame = ttk.LabelFrame(chat_frame, text="发送消息（使用@机器人名 来指定接收者）", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        user_name_frame = ttk.Frame(input_frame)
        user_name_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(user_name_frame, text="你的名字:").pack(side=tk.LEFT)
        self.user_name_var = tk.StringVar(value=self.config_manager.get("user_name"))
        self.user_name_entry = ttk.Entry(user_name_frame, textvariable=self.user_name_var, width=20)
        self.user_name_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(user_name_frame, text="保存", command=self.save_user_name).pack(side=tk.LEFT, padx=5)
        self.message_entry = scrolledtext.ScrolledText(
            input_frame, wrap=tk.WORD, height=4, font=("SimHei", 10)
        )
        self.message_entry.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 10))
        self.message_entry.focus_set()
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(side=tk.RIGHT)
        self.send_button = ttk.Button(btn_frame, text="发送", command=self.send_message)
        self.send_button.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="清空记录", command=self.clear_chat).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="配置", command=self.open_config_window).pack(fill=tk.X, pady=2)
        self.status_var = tk.StringVar(value=f"就绪 | 已加载机器人: {len(self.active_robots)} 个 | AI对话轮次: 0")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.message_entry.bind("<Return>", self.on_enter_press)

    def toggle_infinite_loop(self):
        """切换无限轮回状态"""
        self.infinite_loop = self.infinite_loop_var.get()
        max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
        loop_status = "启用" if self.infinite_loop else "禁用"
        dev_status = "启用" if self.dev_mode else "禁用"
        self.status_var.set(
            f"就绪 | 已加载机器人: {len(self.active_robots)} 个 | "
            f"当前轮次: {self.ai_conversation_turns}/{max_turns} | "
            f"无限轮回: {loop_status} | 开发者模式: {dev_status}"
        )

    def toggle_dev_mode(self):
        """切换开发者模式状态"""
        self.dev_mode = self.dev_mode_var.get()
        dev_status = "启用" if self.dev_mode else "禁用"
        max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
        loop_status = "启用" if self.infinite_loop else "禁用"
        self.status_var.set(
            f"就绪 | 已加载机器人: {len(self.active_robots)} 个 | "
            f"当前轮次: {self.ai_conversation_turns}/{max_turns} | "
            f"无限轮回: {loop_status} | 开发者模式: {dev_status}"
        )

    def start_message_processor(self):
        def process_messages():
            while True:
                try:
                    msg = self.message_queue.get(block=True)
                    self.root.after(0, lambda m=msg: self.add_message_to_history(m))

                    max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
                    if (msg.role == Role.ASSISTANT and
                        self.config_manager.get("allow_ai_conversations") and
                        (self.infinite_loop or self.ai_conversation_turns < max_turns)):
                        if not self.ai_processing:
                            self.root.after(0, lambda m=msg: self.trigger_ai_response_to_ai_message(m))

                    self.message_queue.task_done()
                except Exception as e:
                    print(f"消息处理错误: {str(e)}")

        threading.Thread(target=process_messages, daemon=True).start()

    def open_config_window(self):
        config_window = tk.Toplevel(self.root)
        config_window.title("系统配置")
        config_window.geometry("800x600")
        config_window.transient(self.root)
        config_window.grab_set()
        notebook = ttk.Notebook(config_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # API配置标签页
        api_frame = ttk.Frame(notebook, padding=10)
        notebook.add(api_frame, text="API设置")

        ttk.Label(api_frame, text="API密钥:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.api_key_var = tk.StringVar(value=self.config_manager.get("api_key", ""))
        ttk.Entry(api_frame, textvariable=self.api_key_var, width=50).grid(row=0, column=1, pady=5)
        ttk.Label(api_frame, text="API地址:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.base_url_var = tk.StringVar(value=self.config_manager.get("base_url", "https://api.openai.com/v1"))
        ttk.Entry(api_frame, textvariable=self.base_url_var, width=50).grid(row=1, column=1, pady=5)
        ttk.Label(api_frame, text="模型:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value=self.config_manager.get("model", "gpt-3.5-turbo"))
        ttk.Entry(api_frame, textvariable=self.model_var, width=50).grid(row=2, column=1, pady=5)
        ttk.Label(api_frame, text="超时时间(秒):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.timeout_var = tk.StringVar(value=str(self.config_manager.get("timeout", 30)))
        ttk.Entry(api_frame, textvariable=self.timeout_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Label(api_frame, text="代理:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.proxy_var = tk.StringVar(value=self.config_manager.get("proxy", ""))
        ttk.Entry(api_frame, textvariable=self.proxy_var, width=50).grid(row=4, column=1, pady=5)
        # AI对话设置标签页
        conv_frame = ttk.Frame(notebook, padding=10)
        notebook.add(conv_frame, text="AI对话设置")

        self.allow_ai_conv_var = tk.BooleanVar(value=self.config_manager.get("allow_ai_conversations", True))
        ttk.Checkbutton(
            conv_frame,
            text="允许AI之间自动对话",
            variable=self.allow_ai_conv_var
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=10)
        ttk.Label(conv_frame, text="最大自动对话轮次:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.max_turns_var = tk.StringVar(value=str(self.config_manager.get("max_ai_conversation_turns", 10)))
        ttk.Entry(conv_frame, textvariable=self.max_turns_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(conv_frame, text="(每轮指所有活跃机器人完成一次回应)").grid(row=1, column=2, sticky=tk.W, pady=5)
        # 默认机器人设置标签页
        robot_frame = ttk.Frame(notebook, padding=10)
        notebook.add(robot_frame, text="默认机器人")

        ttk.Label(robot_frame, text="默认加载机器人:").grid(row=0, column=0, sticky=tk.NW, pady=5)
        all_robots = self.config_manager.list_robots()
        default_robots = self.config_manager.get("default_robots", [])

        self.robot_vars = {}
        robot_check_frame = ttk.Frame(robot_frame)
        robot_check_frame.grid(row=0, column=1, sticky=tk.W, pady=5)

        for i, robot_name in enumerate(all_robots):
            var = tk.BooleanVar(value=robot_name in default_robots)
            self.robot_vars[robot_name] = var
            ttk.Checkbutton(robot_check_frame, text=robot_name, variable=var).grid(row=i, column=0, sticky=tk.W)
        btn_frame = ttk.Frame(config_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="保存所有配置", command=lambda: self.save_all_config(config_window)).pack(side=tk.RIGHT, padx=10)
        ttk.Button(btn_frame, text="取消", command=config_window.destroy).pack(side=tk.RIGHT, padx=10)

    def save_all_config(self, window):
        try:
            self.config_manager.set("api_key", self.api_key_var.get())
            self.config_manager.set("base_url", self.base_url_var.get())
            self.config_manager.set("model", self.model_var.get())

            try:
                timeout = int(self.timeout_var.get())
                if timeout <= 0:
                    raise ValueError("超时时间必须大于0")
                self.config_manager.set("timeout", timeout)
            except ValueError as e:
                messagebox.showerror("错误", f"无效的超时时间: {str(e)}")
                return

            self.config_manager.set("proxy", self.proxy_var.get())
            self.config_manager.set("allow_ai_conversations", self.allow_ai_conv_var.get())

            try:
                max_turns = int(self.max_turns_var.get())
                if max_turns < 1:
                    raise ValueError("最大轮次必须至少为1")
                self.config_manager.set("max_ai_conversation_turns", max_turns)
            except ValueError as e:
                messagebox.showerror("错误", f"无效的最大对话轮次: {str(e)}")
                return

            new_defaults = [name for name, var in self.robot_vars.items() if var.get()]
            self.config_manager.set("default_robots", new_defaults)
            self.active_robots.clear()
            self.load_all_robots_on_start()
            self.init_ai_client()
            self.update_robot_list()
            messagebox.showinfo("成功", "所有配置已保存，轮次限制已生效")
            window.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {str(e)}")

    def init_ai_client(self) -> bool:
        try:
            self.ai_client = AIClient(
                api_key=self.config_manager.get("api_key"),
                base_url=self.config_manager.get("base_url"),
                timeout=self.config_manager.get("timeout"),
                proxy=self.config_manager.get("proxy"),
                model=self.config_manager.get("model")
            )
            max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
            loop_status = "启用" if self.infinite_loop else "禁用"
            dev_status = "启用" if self.dev_mode else "禁用"
            self.status_var.set(f"API连接成功 | 已加载机器人: {len(self.active_robots)} 个 | 最大轮次: {max_turns} | 无限轮回: {loop_status} | 开发者模式: {dev_status}")
            return True
        except Exception as e:
            max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
            loop_status = "启用" if self.infinite_loop else "禁用"
            dev_status = "启用" if self.dev_mode else "禁用"
            self.status_var.set(f"API初始化失败 | 已加载机器人: {len(self.active_robots)} 个 | 最大轮次: {max_turns} | 无限轮回: {loop_status} | 开发者模式: {dev_status}")
            messagebox.showerror("API错误", f"创建AI客户端失败: {str(e)}")
            return False

    def load_all_robots_on_start(self) -> None:
        all_robots = self.config_manager.load_all_robots()
        user_name = self.config_manager.get("user_name", "用户")

        # 发送群聊名单给每个机器人
        robot_list_info = f"群聊成员名单：\n"
        robot_list_info += f"- 用户（human）：{user_name}\n"
        for robot_name, robot_config in all_robots.items():
            if robot_config.enabled:
                self.active_robots[robot_name] = robot_config
                robot_list_info += f"- 机器人（robot）：{robot_name}\n"

        for robot_name, robot_config in self.active_robots.items():
            if robot_config.prompt:
                self.chat_history.append(Message(
                    role=Role.SYSTEM,
                    content=f"机器人 {robot_name} 的提示词: {robot_config.prompt}",
                    name="系统"
                ))
            # 发送群聊名单
            self.chat_history.append(Message(
                role=Role.SYSTEM,
                content=f"""{robot_list_info}
你是机器人（robot），名字是 {robot_name}。
你的属性：
- 可以被 @ 并回应 @ 消息
- 可以回应其他 AI 的消息（根据系统设置）
- 可以回应公开消息（根据系统设置）

在这个群组聊天中：
- 你只能看到 @你 的消息或公开消息（无 @ 标记）。
- 如果消息 @ 了其他机器人，你看不到也不能回应。
- 如果消息 @ 了你，你必须回应。
- 如果消息是公开的（无 @），你可以选择回应（根据系统设置）。
回复时直接发表你的观点，不要在开头添加你的名字或任何前缀，你的名字会自动标记。""",
                name="系统"
            ))

        self.update_robot_list()
        max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
        loop_status = "启用" if self.infinite_loop else "禁用"
        dev_status = "启用" if self.dev_mode else "禁用"
        self.status_var.set(f"就绪 | 已加载机器人: {len(self.active_robots)} 个 | 最大轮次: {max_turns} | 无限轮回: {loop_status} | 开发者模式: {dev_status}")

    def update_robot_list(self) -> None:
        self.robot_listbox.delete(0, tk.END)
        for robot_name in self.active_robots.keys():
            self.robot_listbox.insert(tk.END, robot_name)

    def parse_mention(self, content: str) -> tuple[str, Optional[str]]:
        mention_pattern = r'^@(\w+)\s'
        match = re.match(mention_pattern, content)

        if match:
            target = match.group(1)
            if target in self.active_robots:
                cleaned_content = re.sub(mention_pattern, '', content)
                return cleaned_content, target

        return content, None

    def get_ai_color_tag(self, ai_name: str) -> str:
        tag_name = f"ai_{ai_name}"

        if tag_name not in self.chat_text.tag_names():
            robot_config = self.active_robots.get(ai_name)
            color = robot_config.color if robot_config and robot_config.color else DEFAULT_ROBOT_COLORS[0]
            self.chat_text.tag_config(tag_name, foreground=color, font=("SimHei", 10, "bold"))

        return tag_name

    def add_message_to_history(self, msg: Message) -> None:
        self.chat_text.config(state=tk.NORMAL)

        if msg.role == Role.SYSTEM:
            self.chat_text.insert(tk.END, f"系统消息: {msg.content}\n\n", "system")
        else:
            if msg.role == Role.USER:
                sender_tag = "user"
            else:
                sender_tag = self.get_ai_color_tag(msg.name)

            self.chat_text.insert(tk.END, f"{msg.name}: ", sender_tag)

            if msg.target:
                self.chat_text.insert(tk.END, f"@{msg.target} ", "mention")

            self.chat_text.insert(tk.END, msg.content + "\n\n")

        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)
        self.chat_history.append(msg)

    def send_message(self) -> None:
        user_input = self.message_entry.get("1.0", tk.END).strip()
        if not user_input:
            return

        user_name = self.user_name_var.get().strip() or "用户"
        self.message_entry.delete("1.0", tk.END)

        self.ai_conversation_turns = 0  # 用户发送消息后重置轮次计数

        content, target = self.parse_mention(user_input)

        user_message = Message(
            role=Role.USER,
            content=content,
            name=user_name,
            target=target
        )

        self.message_queue.put(user_message)

        if self.active_robots and self.ai_client:
            threading.Thread(target=self.trigger_ai_responses, args=(user_message,), daemon=True).start()
        else:
            if not self.active_robots:
                messagebox.showinfo("提示", "请先添加并启用机器人")
            elif not self.ai_client:
                messagebox.showinfo("提示", "请先配置API并确保连接成功")

    def trigger_ai_response_to_ai_message(self, ai_message: Message) -> None:
        if not self.active_robots or not self.ai_client:
            return

        max_turns = self.config_manager.get("max_ai_conversation_turns", 10)

        if not self.infinite_loop and self.ai_conversation_turns >= max_turns:
            self.message_queue.put(
                Message(
                    role=Role.SYSTEM,
                    content=f"已达到最大AI对话轮次({max_turns})。请用户发送消息以继续。",
                    name="系统"
                )
            )
            return

        threading.Thread(
            target=self.trigger_ai_responses,
            args=(ai_message, True),
            daemon=True
        ).start()

    def show_robot_dev_window(self, robot_name: str, messages: List[Message]):
        """显示单个AI的专属开发者窗口"""
        if not self.dev_mode:
            return

        with self.dev_windows_lock:
            dev_window = tk.Toplevel(self.root)
            dev_window.title(f"开发者模式 - {robot_name} 的提示词")
            dev_window.geometry("600x500")
            dev_window.transient(self.root)

            text_frame = ttk.Frame(dev_window, padding="10")
            text_frame.pack(fill=tk.BOTH, expand=True)

            dev_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("SimHei", 10))
            dev_text.pack(fill=tk.BOTH, expand=True)

            dev_text.insert(tk.END, f"=== 发送给 {robot_name} 的提示词 ===\n\n")
            for msg in messages:
                dev_text.insert(tk.END, f"角色: {msg.role}\n")
                dev_text.insert(tk.END, f"发送者: {msg.name}\n")
                if msg.target:
                    dev_text.insert(tk.END, f"目标: @{msg.target}\n")
                dev_text.insert(tk.END, f"内容: {msg.content}\n")
                dev_text.insert(tk.END, "-"*50 + "\n")

            dev_text.config(state=tk.DISABLED)

    def trigger_ai_responses(self, last_message: Message, is_ai_message: bool = False) -> None:
        self.ai_processing = True

        try:
            max_turns = self.config_manager.get("max_ai_conversation_turns", 10)

            if is_ai_message:
                self.ai_conversation_turns += 1
                loop_status = "启用" if self.infinite_loop else "禁用"
                self.root.after(0, lambda: self.status_var.set(
                    f"AI对话中... | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status}"
                ))

            responding_robots = []
            for robot_name, robot_config in self.active_robots.items():
                if robot_name == last_message.name:  # 不回应自己
                    continue
                if last_message.target and last_message.target != robot_name:  # 不是 @ 当前机器人，跳过
                    continue
                if is_ai_message and not robot_config.auto_respond_to_ai:  # 不自动回应 AI 消息
                    continue
                if not last_message.target and not robot_config.auto_respond_to_all:  # 不自动回应公开消息
                    continue
                responding_robots.append((robot_name, robot_config))

            for i, (robot_name, robot_config) in enumerate(responding_robots):
                if not self.infinite_loop and self.ai_conversation_turns > max_turns:
                    self.message_queue.put(
                        Message(
                            role=Role.SYSTEM,
                            content=f"已达到最大AI对话轮次({max_turns})。对话已停止。",
                            name="系统"
                        )
                    )
                    break

                loop_status = "启用" if self.infinite_loop else "禁用"
                self.root.after(0, lambda rn=robot_name:
                    self.status_var.set(f"等待{rn}响应... | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status}"))

                robot_specific_history = []

                if robot_config.prompt:
                    system_prompt = f"你的名字是{robot_name}。{robot_config.prompt}\n"
                else:
                    system_prompt = f"你的名字是{robot_name}。\n"

                # 发送群聊名单
                user_name = self.config_manager.get("user_name", "用户")
                robot_list_info = f"群聊成员名单：\n"
                robot_list_info += f"- 用户（human）：{user_name}\n"
                for rn, rc in self.active_robots.items():
                    robot_list_info += f"- 机器人（robot）：{rn}\n"

                system_prompt += f"""{robot_list_info}
你是机器人（robot），名字是 {robot_name}。
你的属性：
- 可以被 @ 并回应 @ 消息
- 可以回应其他 AI 的消息（根据系统设置）
- 可以回应公开消息（根据系统设置）

在这个群组聊天中：
- 你只能看到 @你 的消息或公开消息（无 @ 标记）。
- 如果消息 @ 了其他机器人，你看不到也不能回应。
- 如果消息 @ 了你，你必须回应。
- 如果消息是公开的（无 @），你可以选择回应（根据系统设置）。
回复时直接发表你的观点，不要在开头添加你的名字或任何前缀，你的名字会自动标记。"""

                robot_specific_history.append(Message(
                    role=Role.SYSTEM,
                    content=system_prompt,
                    name="系统"
                ))

                for msg in self.chat_history:
                    if msg.role == Role.SYSTEM:
                        if msg.content.startswith("机器人 "):
                            continue
                        robot_specific_history.append(msg)
                    elif msg.target is None:  # 公开消息，所有机器人都能看到
                        robot_specific_history.append(msg)
                    elif msg.target == robot_name:  # 只有被 @ 的机器人能看到
                        robot_specific_history.append(msg)
                    elif msg.name == robot_name:  # 机器人自己发送的消息（回应）
                        robot_specific_history.append(msg)

                # 如果开启开发者模式，显示该AI的专属窗口
                if self.dev_mode:
                    self.root.after(0, lambda rn=robot_name, rh=robot_specific_history: self.show_robot_dev_window(rn, rh))

                try:
                    request = ChatCompletionRequest(
                        model=self.ai_client.model,
                        messages=robot_specific_history,
                        temperature=0.7
                    )

                    response = self.ai_client.chat_completion(request, robot_name)

                    if response.choices and len(response.choices) > 0:
                        ai_response = response.choices[0].message
                        content, target = self.parse_mention(ai_response.content)

                        name_prefix = f"{robot_name}:"
                        if content.startswith(name_prefix):
                            content = content[len(name_prefix):].lstrip()

                        ai_message = Message(
                            role=Role.ASSISTANT,
                            content=content,
                            name=robot_name,
                            target=target
                        )
                        self.message_queue.put(ai_message)

                        loop_status = "启用" if self.infinite_loop else "禁用"
                        self.root.after(0, lambda rn=robot_name:
                            self.status_var.set(f"{rn} 已响应 | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status}"))

                except Exception as e:
                    error_msg = f"{robot_name} 响应失败: {str(e)}"
                    self.message_queue.put(
                        Message(role=Role.SYSTEM, content=error_msg, name="系统")
                    )
                    loop_status = "启用" if self.infinite_loop else "禁用"
                    self.root.after(0, lambda: self.status_var.set(
                        f"错误: 部分机器人响应失败 | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status}")
                )

                if i < len(responding_robots) - 1:
                    time.sleep(1)

        finally:
            self.ai_processing = False
            max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
            loop_status = "启用" if self.infinite_loop else "禁用"
            dev_status = "启用" if self.dev_mode else "禁用"
            self.root.after(0, lambda: self.status_var.set(
                f"就绪 | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status} | 开发者模式: {dev_status}")
            )

    def on_enter_press(self, event) -> str:
        if event.state & 0x1:
            return None
        else:
            self.send_message()
            return "break"

    def clear_chat(self) -> None:
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state=tk.DISABLED)

        self.ai_conversation_turns = 0
        max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
        loop_status = "启用" if self.infinite_loop else "禁用"
        dev_status = "启用" if self.dev_mode else "禁用"

        system_messages = [msg for msg in self.chat_history
                          if msg.role == Role.SYSTEM and msg.content.startswith("机器人 ")]
        self.chat_history = system_messages
        self.status_var.set(f"聊天记录已清空 | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status} | 开发者模式: {dev_status}")

    def save_user_name(self) -> None:
        new_name = self.user_name_var.get().strip()
        if not new_name:
            messagebox.showwarning("提示", "用户名不能为空")
            return

        if new_name in self.active_robots:
            messagebox.showerror("错误", f"用户名'{new_name}'与机器人重名")
            return

        self.config_manager.set("user_name", new_name)
        messagebox.showinfo("成功", "用户名已更新")

    def open_robot_editor(self, robot_config: Optional[RobotConfig] = None) -> None:
        is_new = robot_config is None
        if is_new:
            all_robots = sorted(self.config_manager.list_robots())
            default_color = DEFAULT_ROBOT_COLORS[len(all_robots) % len(DEFAULT_ROBOT_COLORS)]
            robot_config = RobotConfig(name="", color=default_color)

        editor_window = tk.Toplevel(self.root)
        editor_window.title(f"{'新建' if is_new else '编辑'}机器人")
        editor_window.geometry("600x550")
        editor_window.transient(self.root)
        editor_window.grab_set()
        frame = ttk.Frame(editor_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="机器人名称:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_var = tk.StringVar(value=robot_config.name)
        ttk.Entry(frame, textvariable=name_var, width=40).grid(row=0, column=1, pady=5)
        ttk.Label(frame, text="显示颜色:").grid(row=1, column=0, sticky=tk.W, pady=5)

        color_frame = ttk.Frame(frame, width=30, height=20, relief=tk.SUNKEN, borderwidth=1)
        color_frame.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        color_var = tk.StringVar(value=robot_config.color or DEFAULT_ROBOT_COLORS[0])

        def update_color_preview():
            color_frame.configure(style=f"color.TFrame")
            ttk.Style().configure(f"color.TFrame", background=color_var.get())

        update_color_preview()

        def choose_color():
            color = colorchooser.askcolor(title="选择机器人颜色", initialcolor=color_var.get())
            if color[1]:
                color_var.set(color[1])
                update_color_preview()

        ttk.Button(frame, text="选择颜色", command=choose_color).grid(row=1, column=2, padx=5)
        ttk.Label(frame, text="提示词文件:").grid(row=2, column=0, sticky=tk.W, pady=5)
        prompt_files = self.config_manager.list_prompts()
        prompt_file_var = tk.StringVar(value=robot_config.prompt_file or "")

        def load_selected_prompt(event):
            selected = prompt_file_var.get()
            if selected:
                content = self.config_manager.load_prompt(selected)
                if content:
                    prompt_text.delete("1.0", tk.END)
                    prompt_text.insert(tk.END, content)

        prompt_file_combo = ttk.Combobox(frame, textvariable=prompt_file_var, values=prompt_files, width=37)
        prompt_file_combo.grid(row=2, column=1, pady=5)
        prompt_file_combo.bind("<<ComboboxSelected>>", load_selected_prompt)
        ttk.Button(frame, text="刷新", command=lambda: prompt_file_combo.config(values=self.config_manager.list_prompts())).grid(row=2, column=2, padx=5)
        ttk.Label(frame, text="提示词内容:").grid(row=3, column=0, sticky=tk.NW, pady=5)
        prompt_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15, width=50)
        prompt_text.grid(row=3, column=1, pady=5)
        prompt_text.insert(tk.END, robot_config.prompt)
        def save_current_prompt():
            content = prompt_text.get("1.0", tk.END).strip()
            if not content:
                messagebox.showwarning("提示", "提示词内容为空")
                return

            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
                initialdir=str(PROMPT_DIR)
            )
            if filename:
                self.config_manager.save_prompt(os.path.basename(filename), content)
                prompt_file_combo.config(values=self.config_manager.list_prompts())
        ttk.Button(frame, text="保存提示词到文件", command=save_current_prompt).grid(row=4, column=1, pady=5, sticky=tk.W)
        behavior_frame = ttk.LabelFrame(frame, text="机器人行为")
        behavior_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=10, padx=5)

        enabled_var = tk.BooleanVar(value=robot_config.enabled)
        ttk.Checkbutton(behavior_frame, text="默认启用", variable=enabled_var).grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)

        auto_respond_var = tk.BooleanVar(value=robot_config.auto_respond_to_ai)
        ttk.Checkbutton(
            behavior_frame,
            text="自动回应其他AI的消息",
            variable=auto_respond_var
        ).grid(row=0, column=1, sticky=tk.W, pady=5, padx=10)

        auto_respond_all_var = tk.BooleanVar(value=robot_config.auto_respond_to_all)
        ttk.Checkbutton(
            behavior_frame,
            text="自动回应所有公共消息",
            variable=auto_respond_all_var
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5, padx=10)
        def save_robot():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("错误", "机器人名称不能为空")
                return

            if name == self.config_manager.get("user_name"):
                messagebox.showerror("错误", f"不能使用用户名'{name}'作为机器人名")
                return

            if is_new and name in self.config_manager.list_robots():
                messagebox.showerror("错误", f"机器人'{name}'已存在")
                return
            new_config = RobotConfig(
                name=name,
                prompt=prompt_text.get("1.0", tk.END).strip(),
                prompt_file=prompt_file_var.get() or None,
                enabled=enabled_var.get(),
                auto_respond_to_ai=auto_respond_var.get(),
                auto_respond_to_all=auto_respond_all_var.get(),
                color=color_var.get()
            )
            if self.config_manager.save_robot(new_config):
                if new_config.enabled:
                    self.active_robots[name] = new_config
                elif name in self.active_robots:
                    del self.active_robots[name]

                self.update_robot_list()
                max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
                loop_status = "启用" if self.infinite_loop else "禁用"
                dev_status = "启用" if self.dev_mode else "禁用"
                self.status_var.set(f"就绪 | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status} | 开发者模式: {dev_status}")
                messagebox.showinfo("成功", f"机器人{'创建' if is_new else '更新'}成功")
                editor_window.destroy()
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        ttk.Button(button_frame, text="保存", command=save_robot).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="取消", command=editor_window.destroy).pack(side=tk.LEFT, padx=10)

    def edit_selected_robot(self) -> None:
        selected = self.robot_listbox.curselection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个机器人")
            return

        robot_name = self.robot_listbox.get(selected[0])
        robot_config = self.config_manager.load_robot(robot_name)
        if robot_config:
            self.open_robot_editor(robot_config)

    def delete_selected_robot(self) -> None:
        selected = self.robot_listbox.curselection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个机器人")
            return

        robot_name = self.robot_listbox.get(selected[0])
        if messagebox.askyesno("确认", f"确定要删除机器人'{robot_name}'吗？"):
            if self.config_manager.delete_robot(robot_name):
                if robot_name in self.active_robots:
                    del self.active_robots[robot_name]
                self.update_robot_list()
                max_turns = self.config_manager.get("max_ai_conversation_turns", 10)
                loop_status = "启用" if self.infinite_loop else "禁用"
                dev_status = "启用" if self.dev_mode else "禁用"
                self.status_var.set(f"就绪 | 已加载机器人: {len(self.active_robots)} 个 | 当前轮次: {self.ai_conversation_turns}/{max_turns} | 无限轮回: {loop_status} | 开发者模式: {dev_status}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIGroupChatGUI(root)
    root.mainloop()
