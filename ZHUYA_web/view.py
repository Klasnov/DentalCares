from django.shortcuts import render
from django.http import HttpResponse
def base(request):
    return render(request,'base.html')

def home(request):
    return render(request,'home.html')

def single_upload(request):
    return render(request, 'single_upload.html')
def is_ya_upload(request):
    return render(request, 'is_ya_upload.html')

def multi_upload(request):
    return render(request, 'multi_upload.html')

def multi_list_upload(request):
    return render(request,'multi_list_upload.html')

def tar_upload(request):
    return render(request,'tar_upload.html')


def login(request):
    return render(request,'login.html')
def christmas(request,name ='Echo'):
    context = {}
    if name.lower() == 'rene':
        name = 'Rene giegie'
        # console.log('Rene傻逼')
    if name.lower() == 'jam':
        name = 'Rene的爹'
    if (name.lower() == 'miemie') or (name == '咩咩'):
        name = '富婆咩咩'
    if name == '小鱼' or name.lower() == 'xiaoyu':
        name = '美女小鱼'
    if name.lower() == 'alex':
        name = '铁汁Alex'
    if name.lower() == 'mathis':
        name = '好大哥哥Mathis'
    if name.lower() == 'chloroplast' or name == '梓元':
        name = ' 老公元宝'

    context['name'] = name

    return render(request,'christmas.html',context)

def test(request):
    return render(request,'base.html')
