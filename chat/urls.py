from django.urls import path
from .views import chatbot_view

urlpatterns = [
    path("chatbot/", chatbot_view, name="chatbot"),
]
