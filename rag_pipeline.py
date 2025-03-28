import os

from datasets import Dataset
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from query_pipeline import load_local_dataset, load_collection, do_query, format_query_result
from utils import load_json

PROMPT_DATA_PATH = "./data/prompt.json"
load_dotenv()


class ResponseGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.prompt_data = load_json(PROMPT_DATA_PATH)
        self.select_model()

    def select_model(self):
        """
        Select a model with image support
        """
        if self.model_name == "gpt-4o":
            api_key = os.getenv("OPENAI_API_KEY")
            self.model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
        elif self.model_name == "llava":
            self.model = OllamaLLM(model="llava")
        else:
            raise NotImplementedError

    def generate_response(self, query_result):
        if self.model_name == "gpt-4o":
            response = self._generate_response_gpt(query_result)
        elif self.model_name == "llava":
            response = self._generate_response_llava(query_result)
        else:
            raise NotImplementedError

        return response

    def _generate_response_gpt(self, query_result):
        output_parser = StrOutputParser()

        user_messages = [
            {
                "type": "text",
                "text": self.prompt_data[0]["system"] + query_result["user_query"],
            }
        ]
        for i in range(len(query_result["images"])):
            user_messages.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{{image_{i}}}",
            })

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_data[0]["system"]),
                ("user", user_messages)
            ]
        )

        prompt_input = {f"image_{i}": query_result["images"][i] for i in range(len(query_result["images"]))}

        chain = prompt_template | self.model | output_parser
        response = chain.invoke(prompt_input)

        return response

    def _generate_response_llava(self, query_result):
        print("Generating response with LLaVA...")

        output_parser = StrOutputParser()

        messages = [
            ("system", self.prompt_data[0]["system"]),
            ("user", self.prompt_data[0]["system"] + query_result["user_query"])
        ]

        model_with_image = self.model.bind(images=query_result["images"])
        response = model_with_image.invoke(messages)
        response = output_parser.invoke(response)

        return response


if __name__ == "__main__":
    ds = load_local_dataset()
    print(ds)
    collection = load_collection()
    print("collection count:", collection.count())

    query_text = "pink-themed flower"

    result = do_query(collection, query_text, 2)
    print(result)

    result_formatted = format_query_result(ds, query_text, result)

    generator = ResponseGenerator("llava")

    response = generator.generate_response(result_formatted)
    print("-- Response --")
    print(response)
