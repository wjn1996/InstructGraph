# -*- encoding: utf-8 -*-
'''
@File    :   instruction.py
@Time    :   2023/12/03 09:18:40
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

import os
from getpass import getpass
from langchain.llms import Replicate

REPLICATE_API_TOKEN = getpass()
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN



llama2_13b = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
llm = Replicate(
    model=llama2_13b,
    model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
)