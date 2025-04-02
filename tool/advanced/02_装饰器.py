"""
演示装饰器的写法
"""

# 装饰器的一般写法（闭包）
# def outer(func):
#     def inner():
#         print("我睡觉了")
#         func()
#         print("我起床了")
#
#     return inner
#
#
# def sleep():
#     import random
#     import time
#     print("睡眠中......")
#     time.sleep(random.randint(1, 5))
#
#
# fn = outer(sleep)
# fn()

# 装饰器的快捷写法（语法糖）
def outer(func):
    def inner():
        print("我睡觉了")
        func()
        print("我起床了")

    return inner


@outer # 将 `sleep` 函数传递给 `outer` 函数
@outer
def sleep():
    import random
    import time
    print("睡眠中......")
    time.sleep(random.randint(1, 5))


sleep()

