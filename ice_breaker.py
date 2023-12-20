from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm import get_llm
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

info = """"
Lisa Valerie Kudrow (/ˈkuːdroʊ/ KOO-droh; born July 30, 1963) is an American actress. She rose to international fame for her role as Phoebe Buffay in the American television sitcom Friends, which aired from 1994 to 2004. The series earned her Primetime Emmy, Screen Actors Guild, Satellite, American Comedy and TV Guide awards. Phoebe has since been named one of the greatest television characters of all time and is considered to be Kudrow's breakout role, spawning her successful film career.

Kudrow starred in the cult comedy film Romy and Michele's High School Reunion (1997) and followed it with an acclaimed performance in the comedy/drama The Opposite of Sex (1998), which won her the New York Film Critics Circle Award for Best Supporting Actress and a nomination for the Independent Spirit Award for Best Supporting Female. She created, produced, wrote, and starred in the HBO mockumentary series The Comeback, which initially lasted for one season in 2005 but was revived for a critically acclaimed second and final season in 2014. She was nominated for the Primetime Emmy Award for Outstanding Lead Actress in a Comedy Series for both seasons.

In 2007, Kudrow received praise for her starring role in the film Kabluey and appeared in the critically panned box office hit film P.S. I Love You. She produced and starred in the Showtime program Web Therapy (2011–2015), which was nominated for a Primetime Emmy Award. She is a producer on the TLC/NBC reality program Who Do You Think You Are, which has been nominated for an Primetime Emmy Award five times. She currently is a voice actress on the animated series HouseBroken.

Kudrow has also had roles in Analyze This (1999) and its sequel Analyze That (2002), Dr. Dolittle 2 (2001), Bandslam (2008), Hotel for Dogs (2009), Easy A (2010), Neighbors (2014) and its sequel Neighbors 2: Sorority Rising (2016), The Girl on the Train (2016), The Boss Baby (2017), Long Shot (2019) and Booksmart (2019).
"""

if __name__ == '__main__':
    summary_template = """Given the info {info} about a person from I want you to create:
        1. short summary
        2. 2 interesting facts about them
    """

    summary_template_prompt = PromptTemplate(input_variables=['info'], template=summary_template)

    pipeline = get_llm()
    llm = HuggingFacePipeline(pipeline=pipeline)

    chain = LLMChain(llm=llm, prompt=summary_template_prompt)

    print(chain.run(info=info)) 