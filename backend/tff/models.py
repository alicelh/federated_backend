# from django.db import models
# Create your models here.
import mongoengine


class Server(mongoengine.Document):
    iter = mongoengine.IntField()
    loss = mongoengine.FloatField()
    acc = mongoengine.FloatField()
    num = mongoengine.IntField()


class Client(mongoengine.Document):
    index = mongoengine.StringField()
    iter = mongoengine.IntField()
    loss = mongoengine.FloatField()
    acc = mongoengine.FloatField()
    num = mongoengine.IntField()
    count = mongoengine.IntField()
    matrix = mongoengine.ListField()


class Serverpara(mongoengine.Document):
    iter = mongoengine.IntField()
    w1 = mongoengine.ListField()
    b1 = mongoengine.ListField()


class Clientpara(mongoengine.Document):
    index = mongoengine.StringField()
    iter = mongoengine.IntField()
    w1 = mongoengine.ListField()
    b1 = mongoengine.ListField()
