from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .llama_model import Llama3
import json

bot = Llama3("meta-llama/Meta-Llama-3.1-70B-Instruct")

@csrf_exempt
def chatbot_view(request):
    if request.method == "POST":
        # user_input = request.POST.get("input", "")
        data = json.loads(request.body)
        user_input = data['input']
        max_tokens = data.get('max_tokens', 2048)
        temperature = data.get('temperature', 0.1)
        top_p = data.get('top_p', 0.9)
        stream = data.get('stream', True)
        print("user input: ",user_input, "max_tokens:", max_tokens, "temperature: ", temperature, "top_p: ", top_p)

        if stream:
            response = StreamingHttpResponse(
                bot._chatbot(user_input, max_tokens, temperature, top_p),
                content_type='text/plain',
            )   
        else:
            pass

        return response
