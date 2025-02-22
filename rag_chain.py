from langchain_core.prompts import PromptTemplate
import ollama
from config import LLM_MODEL

class RAGChain:
    def __init__(self):
        self.prompt_template = '''
###
CONTEXT:
"""
{context}
"""
###
RULES:
You are a helpful agent that answers questions.
Learn from the CONTEXT and answer the following question.
###
{question}
'''
        self.prompt_template_obj = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def format_prompt(self, context, question):
        """
        Retrieval sonuçları (context) ve soru (question) bilgilerini şablona yerleştirir.
        Eğer context, (doküman, skor) çiftleri şeklinde geliyorsa, yalnızca doküman içeriklerini birleştirir.
        """
        if isinstance(context, list) and len(context) > 0:
            if isinstance(context[0], tuple) and hasattr(context[0][0], "page_content"):
                context_str = "\n\n".join([doc.page_content for doc, _ in context])
            else:
                context_str = "\n\n".join(context)
        else:
            context_str = str(context)
        return self.prompt_template_obj.format(context=context_str, question=question)

    def query_llm(self, prompt):
        """Oluşturulan prompt'u yerel LLM modeline gönderir ve cevabı döner."""
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        return response['response']