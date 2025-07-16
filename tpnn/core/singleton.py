# Singleton pattern realised as metaclass.
# Using this will prevent __init__ from launching when getting instance of singleton
#   through SomeSingleton(). Also you can use __init__ to get instance of your singletons
#
# Usecase:
# =========================================================================================
# Default realisation of singleton:
#   ```
#   class RandomSingleton:
#       def __new__(cls):
#           if not hasattr(cls, "instance"):
#               cls.instance = super(PipelineBuilder, cls).__new__(cls)
#           return cls.instance
#       def __init__(self):
#           from random import random
#           print(random())
#
#   s = RandomSingleton()  # will print something like 0.814625685213
#   s1 = RandomSingleton() # will also print 0.371772649199
#   ```
# That means re-initialization of all fields, that was declared in __init__
# =========================================================================================
# But with metaclass realization the problem solves itself
#   ```
#   class RandomSingleton(metaclass=Singleton):
#       def __init__(self):
#           from random import random
#           print(random())
#
#   s = RandomSingleton()  # will print random number 0.77183495977
#   s1 = RandomSingleton() # won't print anything, because instances of
#                         # a singletons will be stored in metaclass
#   ```
# =========================================================================================


class Singleton(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            cls.instance = cls.__instances[cls]
        return cls.__instances[cls]
