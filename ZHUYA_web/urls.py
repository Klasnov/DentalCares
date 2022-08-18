"""ZHUYA_web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import controller
from . import  view
from . import  save_excel
urlpatterns = [
    path('admin/', admin.site.urls),
    path('base/', view.base),
    path('home/', view.home),
    path('single_upload/',view.single_upload),
    path('is_ya_upload/',view.is_ya_upload),
    path('multi_upload/',view.multi_upload),
    path('multi_list_upload/',view.multi_list_upload),
    path('tar_upload/',view.tar_upload),

    path('deal_single/',controller.deal_single),
    path('miniapp/',controller.get_image),
    path('is_ya_single/',controller.is_ya_single),
    path('deal_multi_list/',controller.deal_multi_list),
    path('deal_tar/',controller.deal_tar),
    path('download/',save_excel.download),
    path('login/', view.login),
    path('test/',view.test),
    path('christmas/',view.christmas),
    path('christmas/<str:name>/', view.christmas),
    path('down_image/',controller.down_image)
]
