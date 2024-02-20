from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm import get_llm
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from third_parties.linkedin import scrape_linkedin_profile


if __name__ == '__main__':
    from agents.linked_lookup_agent import lookup

    res = lookup('Ivan Hahanov')

    print(res)