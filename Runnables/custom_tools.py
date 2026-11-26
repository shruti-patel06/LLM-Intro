from langchain.tools import tool

@tool #decorator for creating tool
def get_greet(name:str)->str: #type hints
    """
     Generate a greeting message for a user    
    """ #Docstring
    return f"Hello {name}, Welcome to the GENAI World"

result = get_greet.invoke({"name":"Pluto"})
print(result)

print(get_greet.name)
print(get_greet.description)
print(get_greet.args)