# 联邦学习可视化系统-后端

## 系统界面
![](https://i.imgur.com/T7RDwep.jpg)

## 后端框架+数据库
django + mongodb

## run server

pipenv shell

cd backend

python manage.py runserver 

## database
10.76.0.201 :27017

## 环境
使用pipenv作为虚拟环境

tutorial:

 Create a new project using Python 3.6, specifically:

   $ pipenv --python 3.6

   Install all dependencies for a project (including dev):

   $ pipenv install --dev

   Create a lockfile containing pre-releases:

   $ pipenv lock --pre

   Show a graph of your installed dependencies:

   $ pipenv graph

   Check your installed dependencies for security vulnerabilities:

   $ pipenv check

   Install a local setup.py into your virtual environment/Pipfile:

   $ pipenv install -e .

   Use a lower-level pip command:

   $ pipenv run pip freeze

