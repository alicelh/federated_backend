from .models import Server, Client, Serverpara, Clientpara


def find_server():
    return Server.objects.fields(id=0)


def find_client_stas(miniter, maxiter):
    return Client.objects(iter__gte=miniter, iter__lte=maxiter).fields(id=0)


def find_server_para(itervalue):
    return Serverpara.objects(iter=itervalue).fields(id=0)


def find_client_by_index(index):
    return Client.objects(index=index).fields(id=0, index=0)


def find_client_by_iter(iter):
    return Client.objects(iter=iter).fields(id=0)


def find_client_para_by_iter(iter):
    return Clientpara.objects(iter=iter).fields(id=0, iter=0)


def find_client_para_by_iter_indexarr(iter, indexarr):
    return Clientpara.objects(iter=iter, index__in=indexarr).fields(id=0, iter=0)


def find_client_para_by_iter_index(iter, index):
    return Clientpara.objects(iter=iter, index=index).fields(id=0)

