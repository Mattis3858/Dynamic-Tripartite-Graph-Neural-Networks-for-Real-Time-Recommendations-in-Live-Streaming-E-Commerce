import torch
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from item_parsing_prompt.V1 import ITEM_PARSING_PROMPT


llama_api_key = "03cf8ca7-2cb3-44cd-804b-66721a0c7d13"


class ProductItem(BaseModel):
    clean_description: str
    predicted_category: str

class Products(BaseModel):
    items: List[ProductItem]

parser = JsonOutputParser(pydantic_object=Products)

def build_chain():
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
    )
    chain = llm
    return chain

def get_llm_answer(CHANNEL_NAME: str, item_name: str):
    chain = build_chain()
    prompt = ITEM_PARSING_PROMPT.format(CHANNEL_NAME=CHANNEL_NAME, item_name=item_name)
    response = chain.invoke(prompt)
    result = parser.parse(response.content)
    return result

if __name__ == "__main__":
    result = get_llm_answer("The Shining.周及及", "1702 高領麻花針織上衣")
    print(result)


# print(chain.invoke({
#     "CHANNEL_NAME": "The Shining.周及及",
#     "item_name": "1702 高領麻花針織上衣"
# }))