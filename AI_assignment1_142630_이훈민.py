#142630 이훈민
import math

def product(a,b):               #함수선언, a,b를 입력받습니다
    print(a,"*",b,"=",a*b)      #a*b를 출력하고 계산합니다.

def is_odd(num):                #함수선언, num을 입력받습니다.
    if num%2 == 0:              #홀,짝인지 구분하기 위해서 2를 나눠준 나머지가 0인지 아닌지 판별합니다.
        return True             #맞으면 짝수입니다.
    else :
        return False            #아니면 홀수입니다.

def my_stars(height):           #함수선언, height를 입력받습니다.
    for i in range(1,height+1): #반복문을 사용합니다.
        print("*"*i)            #*을 찍습니다.

def my_mean(*args):             #함수선언, *args는 몇개입력받을지 모를때 사용합니다.
    ave = sum(args)/len(args)   #args의 합을 args의 길이로 나눠주면 평균이 나옵니다.
    print("평균 : ", ave)        #평균을 출력합니다.

print("\n실습 1번\n-------------------")
product(3,4)

print("\n\n실습 2번\n----------------------")

print(is_odd(4))
print(is_odd(5), "\n\n실습 3번\n--------------------")

my_stars(3)

print("\n\n실습 4번\n-----------------------")
my_mean(1,2,3,4,5)