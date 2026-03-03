import openai
from ..memory.drive_memory import NCCLDriveMemory

class NullConeChat:
    def __init__(self, openai_api_key=None):
        if openai_api_key:
            openai.api_key = openai_api_key
        self.memory = NCCLDriveMemory()

    def chat(self, user_id, prompt, model="gpt-3.5-turbo"):
        # 1. 获取历史记忆
        context = self.memory.get_context(user_id)
        # 2. 构造prompt
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Previous conversation:\n{context}"})
        messages.append({"role": "user", "content": prompt})
        # 3. 调用OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        reply = response.choices[0].message.content
        # 4. 保存到记忆
        self.memory.add_message(user_id, "user", prompt)
        self.memory.add_message(user_id, "assistant", reply)
        return reply
