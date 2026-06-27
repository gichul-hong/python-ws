print("Asdf")

a=[1,2,3,4]
b=[5,6]
c='asdf'

print(a+b)
print(a[1:1])
# print(a.append(b))
a.extend(b)
print(a)
a.append(b)
print(a)
# print(a.append(c))
# print(a.extend(c))

print("a", "b", "c", sep="/")

for i in range(2, 4):         # 2단부터 3단까지
    for j in range(1, 4):     # 1부터 3까지 곱하기
        print(f"{i} * {j} = {i*j}")

result = sum([x for x in range(101)])
print(result)

def factorial(n):
    if n == 1:
        return 1
    
    return n * factorial(n - 1)

print(factorial(6))


def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(20))