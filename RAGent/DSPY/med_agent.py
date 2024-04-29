from __future__ import annotations

"""Adapted from LLama - Index legacy code.
# https://github.com/run-llama/llama_index/blob/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-legacy/llama_index/legacy/agent/legacy/openai_agent.py#L495
The MIT License

Copyright (c) Jerry Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import asyncio
import json
import logging
from abc import abstractmethod
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast, get_args

import dspy
from llama_index.agent.types import BaseAgent
from llama_index.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import OpenAIToolCall
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool

from utils import defaults, exists, format_tool_output
# from rag import RAG
from loguru_logger import logger

from rag import RAG

# TODO: import correct RAG

# constants
DEFAULT_MAX_FUNCTION_CALLS = 10
DEFAULT_MODEL_NAME = "gpt-4-0125-preview"
AGENT_SYSTEM_PROMPT = """You are a medical AI assistant trained by OpenAI, based on the GPT-4 model.
You will recieve medical information about a patient and a question from a medical doctor.
Lets think step by step. First think about the information you recieved. Then check your available tools. Develop a stretegy to get all relevant information using multiple rounds of tools if necessary. You can also combine tool outputs and inputs.
Then, run all tools that you consider useful.
Finally, do NOT answer the user question. Instead, summarize the new information we have recieved from the tools and draw conclusions. Include every detail.
"""


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


def call_tool_with_error_handling(
    tool: BaseTool,
    input_dict: Dict,
    error_message: Optional[str] = None,
    raise_error: bool = False,
) -> ToolOutput:
    """Call tool with error handling.

    Input is a dictionary with args and kwargs

    """
    try:
        return tool(**input_dict)
    except Exception as e:
        if raise_error:
            raise
        error_message = error_message or f"Error: {e!s}"
        return ToolOutput(
            content=error_message,
            tool_name=tool.metadata.name,
            raw_input={"kwargs": input_dict},
            raw_output=e,
        )


def call_function(
    tools: List[BaseTool],
    tool_call: OpenAIToolCall,
    verbose: bool = False,
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    # validations to get passed mypy
    assert tool_call.id is not None
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None

    id_ = tool_call.id
    function_call = tool_call.function
    name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    argument_dict = json.loads(arguments_str)

    # Call tool
    # Use default error message
    output = call_tool_with_error_handling(tool, argument_dict, error_message=None)
    if verbose:
        print(f"Got output: {output!s}")
        print("========================\n")
    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": name,
                "tool_call_id": id_,
            },
        ),
        output,
    )


async def acall_function(
    tools: List[BaseTool], tool_call: OpenAIToolCall, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    # validations to get passed mypy
    assert tool_call.id is not None
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None

    id_ = tool_call.id
    function_call = tool_call.function
    name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    async_tool = adapt_to_async_tool(tool)
    argument_dict = json.loads(arguments_str)
    output = await async_tool.acall(**argument_dict)
    if verbose:
        print(f"Got output: {output!s}")
        print("========================\n")
    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": name,
                "tool_call_id": id_,
            },
        ),
        output,
    )


def resolve_tool_choice(tool_choice: Union[str, dict] = "auto") -> Union[str, dict]:
    """Resolve tool choice.

    If tool_choice is a function name string, return the appropriate dict.
    """
    if isinstance(tool_choice, str) and tool_choice not in ["none", "auto"]:
        return {"type": "function", "function": {"name": tool_choice}}

    return tool_choice


class BaseOpenAIAgent(BaseAgent):
    def __init__(
        self,
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool,
        max_function_calls: int,
        callback_manager: Optional[CallbackManager],
        rag: Optional[RAG] = None,
        use_rag: bool = True,
    ):
        self._llm = llm
        self._rag = rag
        self._use_rag = use_rag
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.prefix_messages = prefix_messages
        self.memory = memory
        self.callback_manager = callback_manager or self._llm.callback_manager
        self.sources: List[ToolOutput] = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    @property
    def all_messages(self) -> List[ChatMessage]:
        return self.prefix_messages + self.memory.get()

    @property
    def latest_function_call(self) -> Optional[dict]:
        return self.memory.get_all()[-1].additional_kwargs.get("function_call", None)

    @property
    def latest_tool_calls(self) -> Optional[List[OpenAIToolCall]]:
        return self.memory.get_all()[-1].additional_kwargs.get("tool_calls", None)

    @property
    def all_tool_calls(self) -> Optional[List[OpenAIToolCall]]:
        called_tools: List[OpenAIToolCall] = []
        for message in self.memory.get_all():
            if message.role == MessageRole.TOOL:
                called_tools.append(message)

        return called_tools

    def reset(self) -> None:
        self.memory.reset()

    @abstractmethod
    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""

    def _should_continue(
        self, tool_calls: Optional[List[OpenAIToolCall]], n_function_calls: int
    ) -> bool:
        if n_function_calls > self._max_function_calls:
            return False
        #####
        if not tool_calls:
            if not hasattr(self, "_stop_next_round"):
                self.memory.put(
                    ChatMessage(
                        content="Check again if you have used all available tools necessary. If not, use the missing ones. You *MUST* use *ALL* tools that are useful or instructed to you.",
                        role=MessageRole.USER
                    )
                )
                self._stop_next_round = True
                return True
            return False
        #####
        return True

    def init_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[BaseTool], List[dict]]:
        if chat_history is not None:
            self.memory.set(chat_history)
        self.sources = []
        self.memory.put(ChatMessage(content=message, role=MessageRole.USER))
        tools = self.get_tools(message)
        openai_tools = [tool.metadata.to_openai_tool() for tool in tools]
        return tools, openai_tools

    # def _process_message(self, chat_response: ChatResponse) -> AgentChatResponse:
    def _process_message(self, chat_response: ChatResponse) -> ChatMessage:
        ai_message = chat_response.message
        # only add tool call messages to the memory so far, for the final message we will use the RAG + agent message and append to memory later
        if self.is_tool_call_message(ai_message):
            self.memory.put(ai_message)
        return ai_message
        # return AgentChatResponse(response=str(ai_message.content), sources=self.sources)

    def _get_stream_ai_response(
        self, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        ...

    async def _get_async_stream_ai_response(
        self, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        ...

    def _call_function(self, tools: List[BaseTool], tool_call: OpenAIToolCall) -> None:
        function_call = tool_call.function
        # validations to get passed mypy
        assert function_call is not None
        assert function_call.name is not None
        assert function_call.arguments is not None

        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: function_call.arguments,
                EventPayload.TOOL: get_function_by_name(
                    tools, function_call.name
                ).metadata,
            },
        ) as event:
            function_message, tool_output = call_function(
                tools, tool_call, verbose=self._verbose
            )

            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        self.sources.append(tool_output)
        self.memory.put(function_message)

    async def _acall_function(
        self, tools: List[BaseTool], tool_call: OpenAIToolCall
    ) -> None:
        ...

    def _get_llm_chat_kwargs(
        self, openai_tools: List[dict], tool_choice: Union[str, dict] = "auto"
    ) -> Dict[str, Any]:
        llm_chat_kwargs: dict = {"messages": self.all_messages}
        if openai_tools:
            llm_chat_kwargs.update(
                tools=openai_tools, tool_choice=resolve_tool_choice(tool_choice)
            )
        return llm_chat_kwargs

    def _get_agent_response(
        self, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if mode == ChatResponseMode.WAIT:
            chat_response: ChatResponse = self._llm.chat(**llm_chat_kwargs)
            return self._process_message(chat_response)
        elif mode == ChatResponseMode.STREAM:
            return self._get_stream_ai_response(**llm_chat_kwargs)
        else:
            raise NotImplementedError

    async def _get_async_agent_response(
        self, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        ...

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        tools, openai_tools = self.init_chat(message, chat_history)
        n_function_calls = 0

        # Loop until no more function calls or max_function_calls is reached
        current_tool_choice = tool_choice
        ix = 0
        while True:
            ix += 1

            if self._verbose:
                print(f"STARTING TURN {ix}\n---------------\n")

            llm_chat_kwargs = self._get_llm_chat_kwargs(
                openai_tools, current_tool_choice
            )

            agent_chat_response = self._get_agent_response(mode=mode, **llm_chat_kwargs)

            if not self._should_continue(self.latest_tool_calls, n_function_calls):
                logger.debug("Break: should continue False")
                break
            # iterate through all the tool calls
            logger.debug(f"Continue to tool calls: {self.latest_tool_calls}")

            if self.latest_tool_calls is not None:
                for tool_call in self.latest_tool_calls:
                    # Some validation
                    if not isinstance(tool_call, get_args(OpenAIToolCall)):
                        raise ValueError("Invalid tool_call object")

                    if tool_call.type != "function":
                        raise ValueError("Invalid tool type. Unsupported by OpenAI")
                    # TODO: maybe execute this with multi-threading
                    self._call_function(tools, tool_call)
                    # change function call to the default value, if a custom function was given
                    # as an argument (none and auto are predefined by OpenAI)
                    if current_tool_choice not in ("auto", "none"):
                        current_tool_choice = "auto"
                    n_function_calls += 1

        return agent_chat_response

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        ...

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.WAIT
            )
            # assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    def chat_and_save(
        self,
        context: str,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        chat_response = self.chat_ext(context, question, chat_history, tool_choice)

        raise NotImplementedError  # TODO: FIXME

    @trace_method("chat")
    def chat_ext(
        self,
        context: str,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        use_rag: bool = True,
    ) -> AgentChatResponse:
        ## hacky way
        instruction = (
            " Given the above clinical context, lets now think step by step. Develop a strategy to answer the following question: "
            + question
        )

        instruction += " Use as many tools as useful and whenever possible to get more information about the patient. "
        instruction += "You can chain multiple tool calls and perform multiple rounds of using tools. "
        instruction += "Execute all tools you consider useful. "
        instruction += "Remember that we use an oncology setting. Any speculative feedback of the tools most likely indicates to a cancer diagnosis. "
        instruction += "Finally, do not actually answer the question. Instead provide a detailed summary of the new information we recieved from the tools. "
        instruction += "This summary shall include the tools and inputs you used, and the outputs you recieved and what they mean. "
        instruction += "If a tool fails / throws an error, ignore this in your answer. "
        instruction += "Be precise. Use numbers, dates, names of the images etc. "
        instruction += "If a tool output is not unambiguous, e.g. if a radiology report is uncertain in the diagnosis, state this in your answer."
        
        instruction += "Check the instructions you have received and then use *EVERY* tool you are either instructed to use or that could be helpful to you. Consider every tool at your disposal."
        instruction += "Once you are done, provide a *very* long summary of the tools and their results. Include all details."

        message = context + "\n" + instruction

        logger.info(f"Chatting with message: {message}")

        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat_ext(
                message=message,
                context=context,
                question=question,
                chat_history=chat_history,
                tool_choice=tool_choice,
                mode=ChatResponseMode.WAIT,
            )
            #assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        ...

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        ...

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        ...

    def is_tool_call_message(self, agent_response: ChatMessage) -> bool:
        return exists(agent_response.additional_kwargs.get("tool_calls", None))

    def _process_rag_message(self, rag_response: dspy.Prediction) -> AgentChatResponse:
        """Process the RAG message and add it to memory."""
        assert isinstance(
            rag_response, dspy.Prediction
        ), f"Invalid rag_response: {rag_response} is of type {type(rag_response)} but should be of type dspy.Prediction"
        message = ChatMessage(
            role=MessageRole.ASSISTANT, content=str(rag_response.response)
        )
        self.memory.put(message)
        rag_chat_response = AgentChatResponse(
            response=str(message.content),
            sources=self.sources,
            source_nodes=rag_response.context_nodes,
        )
        return rag_chat_response

    def _chat_ext(
        self,
        message: str,
        context: str,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        n_function_calls = 0
        tools, openai_tools = self.init_chat(message, chat_history)

        current_tool_choice = tool_choice

        ix = 0
        while True:
            ix += 1
            if self._verbose:
                print(f"STARTING TURN {ix}\n---------------\n")

            llm_chat_kwargs = self._get_llm_chat_kwargs(
                openai_tools, current_tool_choice
            )

            agent_response = self._get_agent_response(mode=mode, **llm_chat_kwargs)

            logger.info(f"Agent response: {agent_response}")

            if not self._should_continue(self.latest_tool_calls, n_function_calls):
                logger.debug("Break: should continue False")
                break
            # iterate through all the tool calls
            logger.debug(f"Continue to tool calls: {self.latest_tool_calls}")

            if self.latest_tool_calls is not None:
                for tool_call in self.latest_tool_calls:
                    # Some validation
                    if not isinstance(tool_call, get_args(OpenAIToolCall)):
                        raise ValueError("Invalid tool_call object")

                    if tool_call.type != "function":
                        raise ValueError("Invalid tool type. Unsupported by OpenAI")

                    self._call_function(
                        tools, tool_call
                    )  ## eventually multithread this

                    if current_tool_choice not in ("auto", "none"):
                        current_tool_choice = "auto"
                    n_function_calls += 1

        # now we have all available information about the patient
        if exists(self._rag) and self._use_rag: # ONLY use RAG if wanted
            tool_results = str(agent_response.content)
            return tool_results

            # run rag seperately
            rag_response = self._rag(
                question=question,
                patient_context=context,
                tool_results=tool_results,
                agent_tools=self.tools,
            )
            # append the formatted_tool_results to the final message
            ragent_chat_response = self._process_rag_message(rag_response)
            return ragent_chat_response

        else: # only return the tool output
            agent_chat_response = AgentChatResponse(
                response=agent_response.content,
                sources=self.sources,
                source_nodes=None,
            )
            return agent_chat_response

    # dummy function to debug RAG while skipping tool use to avoid waiting
    # legacy method
    def ask_rag(self, question: str, context: str, tool_results: str):
        result = self._rag(question, patient_context=context, tool_results=tool_results)
        return result


class MedOpenAIAgent(BaseOpenAIAgent):
    """OpenAI (function calling) Agent.

    Uses the OpenAI function API to reason about whether to
    use a tool, and returning the response to the user.

    Supports both a flat list of tools as well as retrieval over the tools.

    Args:
        tools (List[BaseTool]): List of tools to use.
        llm (OpenAI): OpenAI instance.
        memory (BaseMemory): Memory to use.
        prefix_messages (List[ChatMessage]): Prefix messages to use.
        verbose (Optional[bool]): Whether to print verbose output. Defaults to False.
        max_function_calls (Optional[int]): Maximum number of function calls.
            Defaults to DEFAULT_MAX_FUNCTION_CALLS.
        callback_manager (Optional[CallbackManager]): Callback manager to use.
            Defaults to None.
        tool_retriever (ObjectRetriever[BaseTool]): Object retriever to retrieve tools.
    """

    def __init__(
        self,
        rag: Optional[RAG],
        tools: List[BaseTool],
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        use_rag: bool = True,
        verbose: bool = True,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        super().__init__(
            rag=rag,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            use_rag=use_rag
        )
        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            # no tools
            self._get_tools = lambda _: []

        self._tools = tools

    @classmethod
    def from_tools(
        cls,
        rag: Optional[RAG] = None,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = True,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> MedOpenAIAgent:
        """Create an MedOpenAIAgent from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = defaults(tools, [])

        chat_history = chat_history or []

        llm_kwargs = defaults(
            llm_kwargs,
            dict(
                temperature=0.2, max_tokens=4096, system_prompt=AGENT_SYSTEM_PROMPT
            ),  # TODO: REMOVE HARDCODING
        )

        _llm = OpenAI(model=DEFAULT_MODEL_NAME, **llm_kwargs)
        llm = defaults(llm, _llm)

        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history, llm=llm)

        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
            )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            rag=rag,
            tool_retriever=tool_retriever,
            llm=llm,
            tools=tools,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    # legacy method
    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._get_tools(message)

    @property
    def tools(self):
        """Get a dict of tools {name: description}"""
        tools = []
        for tool in self._tools:
            tools.append(
                {"name": tool.metadata.name, "description": tool.metadata.description}
            )
        return tools
