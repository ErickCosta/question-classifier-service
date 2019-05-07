from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .classifier import getClassifier

def index(request):
    question = str(request.body)
    print(question.encode('utf-8'))
    return HttpResponse(getClassifier('Teste'))