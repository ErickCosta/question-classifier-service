from django.http import HttpResponse
from .classifier import getClassifier

def index(request):
    question = (request.body).decode('utf-8')
    return HttpResponse(getClassifier(question))