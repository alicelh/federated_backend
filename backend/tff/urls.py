from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("serverinfo", views.findServer, name="ServerInfo"),
    path("serverparabyiter/<int:iter>", views.findServerParaByIter, name="ServerPara"),
    path(
        "clientinfobyindex/<str:index>",
        views.findClientByIndex,
        name="ClientInfoByIndex",
    ),
    path(
        "clientinfobyiter/<int:iter>", views.findClientByIter, name="ClientInfoByIter"
    ),
    path(
        "clientstastics/<int:miniter>/<int:maxiter>",
        views.findClientStas,
        name="ClientStas",
    ),
    path(
        "clientparabyiter/<int:iter>",
        views.findClientParaByIter,
        name="ClientParaByIter",
    ),
    path(
        "clientparabyiterindex/<int:iter>/<str:index>",
        views.findClientParaByIterIndex,
        name="ClientParaByInterIndex",
    ),
    path(
        "clientparabyiterindexarr/<int:iter>/<str:indexarr>",
        views.findClientParaByIterIndexarr,
        name="ClientParaByInterIndexarr",
    ),
    path(
        "confusionmatrixbyiterclientindex/<int:iter>/<str:index>",
        views.findConfusionMatrixByiIerClientIndex,
        name="confusionmatrixbyiterclientindex"
    )
]
