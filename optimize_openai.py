import re
import time

import httpx
from loguru import logger

import api_key_db
from modules import util


class ChatPaperAPI:
    """
    Official ChatGPT API
    """

    def __init__(
            self,
            proxy=None,
            max_tokens: int = 4000,
            temperature: float = 0.5,
            top_p: float = 1.0,
            model_name: str = "gpt-3.5-turbo",
            reply_count: int = 1,
            system_prompt="You are ChatPaper, A paper reading bot",
            apiTimeInterval: float = 20.0,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.apiTimeInterval = apiTimeInterval
        self.proxy = proxy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reply_count = reply_count
        self.decrease_step = 250
        self.conversation = {}

    async def _calculate_delay(self, api_key):
        elapsed_time = time.time() - api_key[0]
        if elapsed_time < self.apiTimeInterval:
            return self.apiTimeInterval - elapsed_time
        else:
            return 0

    def add_to_conversation(self,
                            message: str,
                            role: str,
                            convo_id: str = "default"):
        if convo_id not in self.conversation:
            self.reset(convo_id)
        self.conversation[convo_id].append({"role": role, "content": message})

    def __truncate_conversation(self, convo_id: str = "default"):
        """
        Truncate the conversation
        """
        last_dialog = self.conversation[convo_id][-1]
        query = str(last_dialog['content'])
        if util.token_str(query) > self.max_tokens:
            query = query[:int(1.5 * self.max_tokens)]
        while util.token_str(query) > self.max_tokens:
            query = query[:self.decrease_step]
        self.conversation[convo_id] = self.conversation[convo_id][:-1]
        full_conversation = ""
        for x in self.conversation[convo_id]:
            full_conversation = str(x["content"]) + "\n" + full_conversation
        while True:
            if (util.token_str(full_conversation + query) >
                    self.max_tokens):
                query = query[:self.decrease_step]
            else:
                break
        last_dialog['content'] = str(query)
        self.conversation[convo_id].append(last_dialog)

    async def ask_stream(self,
                         prompt: str,
                         role: str = "user",
                         convo_id: str = "default",
                         **kwargs) -> str:
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, "user", convo_id=convo_id)
        self.__truncate_conversation(convo_id=convo_id)
        api_key = api_key_db.get_single_alive_key()
        async with httpx.AsyncClient() as client:
            async with client.stream(
                    method="POST",
                    url="https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization":
                            f"Bearer {kwargs.get('api_key', api_key)}"
                    },
                    json={
                        "model": self.model_name,
                        "messages": self.conversation[convo_id],
                        # kwargs
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "n": self.reply_count,
                        "user": role,
                        "stream": True,
                    },
            ) as response:
                if response.status_code != 200:
                    if response.status_code == 403:
                        await api_key_db.update_api_key_is_alive(api_key, False)
                    raise Exception(
                        f"Error: {response.status_code}",
                    )
                content = []
                async for chunk in response.aiter_bytes():
                    content.append(chunk.decode())
                full_response = "".join(content)
                return full_response

    async def ask(self,
                  prompt: str,
                  role: str = "user",
                  convo_id: str = "default",
                  **kwargs):
        response = await self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            **kwargs,
        )
        pattern = r'"content":"([^"]*)"'
        matches = re.findall(pattern, response)
        full_response: str = "".join(matches)
        logger.info(full_response)
        self.add_to_conversation(full_response, role, convo_id=convo_id)
        usage_token = util.token_str(prompt)
        com_token = util.token_str(full_response)
        total_token = self.token_cost(convo_id=convo_id)
        return full_response, usage_token, com_token, total_token

    def reset(self, convo_id: str = "default", system_prompt=None):
        self.conversation[convo_id] = [
            {
                "role": "system",
                "content": str(system_prompt or self.system_prompt)
            },
        ]

    async def conversation_summary(self, convo_id: str = "default"):
        input = ""
        for conv in self.conversation[convo_id]:
            if conv["role"] == 'user':
                role = 'User'
            else:
                role = 'ChatGpt'
            input += role + ' : ' + conv['content'] + '\n'
        prompt = "Your goal is to summarize the provided conversation in English. Your summary should be concise and " \
                 "focus on the key information to facilitate better dialogue for the large language model.Ensure that " \
                 "you include all necessary details and relevant information while still reducing the length of the " \
                 "conversation as much as possible. Your summary should be clear and easily understandable for the " \
                 "ChatGpt model providing a comprehensive and concise summary of the conversation."
        if util.token_str(str(input) + prompt) > self.max_tokens:
            input = input[util.token_str(str(input)) - self.max_tokens:]
        while util.token_str(str(input) + prompt) > self.max_tokens:
            input = input[self.decrease_step:]
        prompt = prompt.replace("{conversation}", input)
        self.reset(convo_id='conversationSummary')
        response = await self.ask(prompt, convo_id='conversationSummary')
        while util.token_str(str(response)) > self.max_tokens:
            response = response[:-self.decrease_step]
        self.reset(convo_id='conversationSummary',
                   system_prompt='Summariaze our diaglog')
        self.conversation[convo_id] = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": "Summariaze our diaglog"
            },
            {
                "role": 'assistant',
                "content": response
            },
        ]
        return self.conversation[convo_id]

    def token_cost(self, convo_id: str = "default"):
        return util.token_str("\n".join(
            [x["content"] for x in self.conversation[convo_id]]))
