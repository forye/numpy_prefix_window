'''
Created on Oct 7, 2014

@author: Idan
'''



if __name__ == '__main__':
    pass


def is_swap_can_equal_sums(A, B, m):
    '''
    O(n)
    gets two positive int arrays (A,B) m is the maximal value of the elements
    return True if
    the sum of the two arrays 
     can be equaled with a single swap 
    '''
    n = len(A)
    sum_a = sum(A)
    sum_b = sum(B)
    d = sum_b - sum_a
    if d % 2 == 1:
        return False
    d //= 2
    count = counting(A, m)
    for i in xrange(n):
        if 0 <= B[i] - d and B[i] - d <= m and count[B[i] - d] > 0:
                return True
    return False

def counting(A, m):
    '''
    gets a list (A) and max value (m)
    return a list (count) 
    
    counts = of number of occurences of int = 0:m in A 
    '''
    n = len(A)
    count = [0] * (m + 1)
    for k in xrange(n):
        count[A[k]] += 1
    return count

def mushrooms(A, k, m):
    
    '''
    caculate the sum of m consecutal steps starting with k

    o(n+m)
    positive int array
    k- starting point , m- number of steps (windw)
    
    '''
    n = len(A)
    result = 0
    pref = prefix_sums(A)
    for p in xrange(min(m, k) + 1):
        left_pos = k - p
        right_pos = min(n - 1, max(k, k + m - 2 * p))
        result = max(result, count_total(pref, left_pos, right_pos))
    for p in xrange(min(m + 1, n - k)):
        right_pos = k + p
        left_pos = max(0, min(k, k - (m - 2 * p)))
        result = max(result, count_total(pref, left_pos, right_pos))
    return result

def count_total(P, x, y):
    return P[y + 1] - P[x]


def prefix_sums(A):
    '''
    returns a commutive sum of A
    len(P) - len(A) ==1
    '''    
    n = len(A)
    P = [0] * (n + 1)
    for k in xrange(n):
        P[k+1] = P[k] + A[k]
    return P

def quickSort(arr):
    '''
    sorting...)( n log(n))
    for each value, seperates the array to higer vals and lowe vals till the end
    
    '''
    if len(arr) <= 1:
        return arr    
    less = []
    pivotList = []
    more = []
    pivot = arr[0]
    for i in arr:
        if i < pivot:
            less.append(i)
        elif i > pivot:
            more.append(i)
        else:
            pivotList.append(i)
    less = quickSort(less)
    more = quickSort(more)
    return less + pivotList + more
    
def selectionSort(A):
    '''
     selection sort is o(n^2) each place in the array from start, find minimal, move to first
     and continue for the rest of the array
     sort in place!
    '''
    n = len(A)
    for k in xrange(n):
        minimal = k
        for j in xrange(k + 1, n):
            if (A[minimal] > A[j]):
                minimal = j
        A[k], A[minimal] = A[minimal], A[k]#swap!
    return A

def countingSort(A, k):
    '''
    for limited valued arrays x[i] = 0:k-> length of counters is k+1
    o(n+k)- time . memory- additional O(k) 
    1. countes the elements of an array in a counters list
    2. itterate trew the array in increasing order
    
    '''
    n = len(A)
    count = [0] * (k + 1)
    for i in xrange(n):
        count[A[i]] += 1
    p = 0
    for i in xrange(k + 1):
        for _ in xrange(count[i]):
            A[p] = i;
            p += 1;
    return A

def create_stack(N):
    '''
    example of calling a GLOBAL var, a var that twas declared before the functions
    '''
    stack = [0] * N
    size = 0
    def push(x):
        global size
        stack[size] = x
        size += 1
    def pop():
        global size
        size -= 1
        return stack[size]

class stack():
    '''
    a class stack realization
    '''
    def __init__(self,N):
        self.stack = [0] * N
        self.size = 0
    def push(self,x):
        self.stack[self.size] = x
        self.size += 1
    def pop(self):
        global size
        self.size -= 1
        return stack[size]
    
    
def queue(N): 
    '''
    we can store N - 1
    
    a cyclic queue
    '''
    queue = [0] * N
    head, tail = 0, 0
    def push(x):
        global tail        
        tail = (tail + 1) % N
        queue[tail] = x
    def pop():
        global head
        head = (head + 1) % N
        return queue[head]
    def size():
        return (tail - head + N) % N
    def empty():
        return head == tail
    
def groceryStore(A): 
    '''
    using only the queue size
    
    Problem: You are given a zero-indexed array A consisting of n integers: a0, a1, . . . , an-1.
    Array A represents a scenario in a grocery store, and contains only 0s and/or 1s:
    0 represents the action of a new person joining the line in the grocery store,
    1 represents the action of the person at the front of the queue being served and leaving
    the line.
    The goal is to count the minimum number of people who should have been in the line before
    the above scenario, so that the scenario is possible
    
    We should find the smallest negative number (size) to determine the size of the queue during the
    whole simulation
    o(n)
    
    '''
    n = len(A)
    size, result = 0, 0
    for i in xrange(n):
        if (A[i] == 0):
            size += 1
        else:
            size -= 1
            result = max(result, -size);
    return result

def goldenLeader(A):
    '''
    gets a golden leader on o(n)
    using a clip!
    '''
    n = len(A)
    size = 0
    for k in xrange(n):
        if (size == 0):
            size += 1
            value = A[k]
        else:
            if (value != A[k]):
                size -= 1
            else:
                size += 1
    candidate = -1
    if (size > 0):
        candidate = value
    leader = -1
    count = 0
    for k in xrange(n):
        if (A[k] == candidate):
            count += 1
    if (count > n // 2):
        leader = candidate
    return leader

def golden_max_slice(A):
    '''
    finds the maximum slice of an array (pand w) on o(n)
    max_ending-last summation if it was possitive (else 0)- accumulates the corrent sum, represent an increasement
    in the sum. it accumulate as long as it is not decresing from the total sum!!!
    max_slice-accumulate the  biggest sum so far
    
    '''
    max_ending = max_slice = 0# starting with the noenelements
    
    for a in A:# for each element:
        max_ending = max(0, max_ending + a)# is the element+the last sum is contributing to the sum? if so add it, else dont 
        max_slice = max(max_slice, max_ending) # if max ending is bigger then the corrent max sum replace- get the maximal sum
    return max_slice



def divisors(n):
    '''
    counts the number of divisor of n
    0(sqrt(n))
    
    all divisors must be in te sqrt(n) region
    
    '''
    i = 1
    result = 0
    while (i * i < n):#case if i< sqrt(n) it must have a partner which is larger then sqrt(n)
        if (n % i == 0):
            result += 2
            i += 1
    if (i * i == n):# case of squred value
        result += 1
    return result

def is_prime(n):
    '''
    works for n>=2
    0(sqrt(n))

    '''
    i = 2
    while (i * i <= n):
        if (n % i == 0):
            return False
        i += 1
    return True

def coin_fliping_problem(n):
    '''
    count the number of coins showing tails (1), heads==0
    (n log(n) )
    
    sum(1/1:n)~log(n)
    
    we can also tell the coin by its number of divisors pairty- (even is flipped)
    so if a number has two devisors it will be flipped twice
    
    meaning, we need to find the number that have the k^2 elemnt in them (flipped only once)
    there are exdactly floor(sqrt(n)) numbers like that
    findig sqrt(n) ~ o(log(n))
     
    '''
    result = 0
    coin = [0] * (n+1)
    for i in xrange(1,n+1):# for each number
        k=i # initiate first heap
        while k<=n:
            coin[k] = (coin[k] +1)%2 #flip coin
            k+=i# initiate next heap
        result +=coin[i]# add to count
    return result
            
     
def sieve(n):
    '''
    Sieve of Eratosthenes
    
    retruns a binary array that flags true on each prime number
    O(n log log n).
    '''
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    i = 2
    while (i * i <= n):
        if (sieve[i]):
            k = i * i
            while (k <= n):
                sieve[k] = False
                k += i
        i += 1
    return sieve

def factorization(x, F):
    '''
    log(x)
    prime factors decomposition according to
    Sieve of Eratosthenes
    '''
    primeFactors = []
    while (F[x] > 0):
        primeFactors += [F[x]]
        x /= F[x]
    primeFactors += [x]
    return primeFactors

def arrayF(n):
    '''
    prepering the array F-
     what factor has marked what value(index in an array)
     
    this array list the prime number ,arked by seive algorithm
    ex:
    [0,0,2,0,2,0,2,3,2,0,2,0,2,3,2,0,2,0,2] 
    
    #of factor of X < logx
    for
    prime factors decomposition according to
    Sieve of Eratosthenes
    '''
    F = [0] * (n + 1)
    i = 2
    while (i * i <= n):
        if (F[i] == 0):
            k = i * i
            while (k <= n):
                if (F[k] == 0):
                    F[k] = i;
        k += i
    i += 1
    return F

def gcd(a, b):
    '''
    o(log(a+b))
    greatest common devidsor
    
    b[k+1] >= b[k] + b[k-1] ~ (1-sqrt(5))^n/sqrt(5)
    
    lcm(a1, a2, . . . , an) = lcm(a1, lcm(a2, a3, . . . , an))

    '''
    if a % b == 0:#O(log n · log log n)
        return b
    else:
        return gcd(b, a % b)

def fibonacci(n):
    '''
    o(n)
    

    faster: Fibonacci(n) = [ [1,1],[1,0]]**n=[[f[n+1],f[n]],[f[n],f[n-1]]]
    
    fasterer: fib(n) = ( (1+sqrt(5))**n/2**n - (1-sqrt(5))**n/2**n)/sqrt(5)
    '''
    if (n <= 1):
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def binarySearch(A, x):
    '''
    search x in a sorted array A
    o(log(n))
    '''
    n = len(A)
    beg = 0
    end = n - 1
    result = -1
    while (beg <= end):
        mid = (beg + end) / 2
        if (A[mid] <= x):# demand condition check!- is valid or optimal?
            beg = mid + 1
            result = mid
        else:
            end = mid - 1
    return result

def boards(A, k):
    '''
    Binary search in O(log n).
    cover holes with a k boards. return the minimal size of a board
    
    '''
    n = len(A)
    beg = 1 #minimal size of board
    end = n # maximal size of board
    result = -1 # returnig the optimal size
    while (beg <= end):
        mid = (beg + end) / 2# test size of board
        if (check(A, mid) <= k):# if the size is enough
            end = mid - 1# try make it smaller
            result = mid# update
        else:
            beg = mid + 1# else try to make it biggre
    return result

def check(A, k):
    '''
    check how many boards in size k is enough to cover all holes
    '''
    n = len(A)
    boards = 0
    last = -1
    for i in xrange(n):
        if A[i] == 1 and last < i:# if there is a hole in i, and if the hole is in the covering board 
            boards += 1
            last = i + k - 1 # increase the count of boards
    return boards

def caterpillarMethod(A, s):
    '''
    fin sub_suquence_sum of s in A o(n)
    moves like a catterpillar (for each (half) step streach or bend)
    '''
    n = len(A)
    front, total = 0, 0
    for back in xrange(n):
        while (front < n and total + A[front] <= s):
            total += A[front]
            front += 1
        if total == s:
            return True
        total -= A[back]
    return False

def triangles(A):

    '''
    o(n^2)
     count the number of triangles tha can be made with these n stick o(n^2)
     count the number of triplets at indices x < y < z AND A[x]+A[y] > A[z]
    
    A- sorted array of sticks length (no equal length sticks)
    
    '''
    n = len(A)
    result = 0
    for x in xrange(n):# for minimal value, run on all the indexes- the back o the catterpillar
        z = 0
        for y in xrange(x + 1, n):# run on the rest of indexes since x- the front of the caterpillar
            while (z < n and A[x] + A[y] > A[z]): #find the index,z , that creates possible trianglewith longes A[z] 
                z += 1
            result += z - y - 1# add all possible options with smaller A[z] that follow the triangle rule 
    return result

'''
    Greedy programming is a method by which
    a solution is determined based on making the locally optimal choice at any given moment

    In other words, we choose the best decision from the viewpoint of the current stage of the
    solution
    
'''
def greedyCoinChanging(M, k):
    '''
    find minimum amount of coins (in M- coin map ( ex in floats:0.05 0.1 0.5 1 2 5 10...)) that pays k
    select coins from large to small, not exceeding k
    '''
    n = len(M)
    result = []
    for i in xrange(n - 1, -1, -1):# go from n-1 to end( says -1 cause it is NOT INCLUDING. it stped at 0) with backward steps
        result += [(M[i], k // M[i])]# append a coinSize, number of coins of this type
        k %= M[i]# get resedue
    return result




def greedyCanoeistA(W, k):
    '''
    The goal is to seat them in the minimum number of double canoes whose displacement (the
    maximum load) equals k. You may assume that wi ¬ k.
    
    for the heaviest fatso, we should find the heaviest skinny who can be
    seated with him
    the thinner the heaviest fatso is, the fatter skinny can be.
    o(n)
    '''

    N = len(W)
    skinny = []#queue()# a que object??
    fatso = []#queue()
    for i in xrange(N - 1):
        if W[i] + W[-1] <= k:
            skinny.append(W[i])
        else:
            fatso.append(W[i])
    fatso.append(W[-1])
    canoes = 0
    while (skinny or fatso):
        if len(skinny) > 0:# can we realese a canou with a skinny?
            skinny.pop(-1)
        fatso.pop(-1)
        canoes += 1
        if (not fatso and skinny):
            fatso.append(skinny.pop(-1)) # update the depleated fatso list
        while (len(fatso) > 1 and fatso[-1] + fatso[0] <= k):
            skinny.append(fatso.pop(0)) # create a new skinny list
    return canoes

def greedyCanoeistB(W, k):
    '''
    same as A, but cleaner?
    '''
    canoes = 0
    j = 0
    i = len(W) - 1
    while (i >= j):
        if W[i] + W[j] <= k:
            j += 1;
        canoes += 1;
        i -= 1
    return canoes
