import time
import numpy as np
import matplotlib.pyplot as plt

def mergeSort(array):
    if len(array) > 1:
        r = len(array) // 2
        l = array[:r]
        m = array[r:]
        mergeSort(l)
        mergeSort(m)
        i = j = k = 0
        while i < len(l) and j < len(m):
            if l[i] < m[j]:
                array[k] = l[i]
                i += 1
            else:
                array[k] = m[j]
                j += 1
            k += 1
        while i < len(l):
            array[k] = l[i]
            i += 1
            k += 1
        while j < len(m):
            array[k] = m[j]
            j += 1
            k += 1

def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i = i + 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1

def quickSort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quickSort(array, low, pi - 1)
        quickSort(array, pi + 1, high)

def selectionSort(array, size):
    for step in range(size):
        min_idx = step
        for i in range(step + 1, size):
            if array[i] < array[min_idx]:
                min_idx = i
        array[step], array[min_idx] = array[min_idx], array[step]

def read_input():
    a = []
    n = int(input("Enter the number of tv channels: "))
    print("Enter the number of viewers:")
    for i in range(0, n):
        l = int(input())
        a.append(l)
    return a

elements = []
times = []
labeldata = ""

print("1. Mergesort\n2. Quicksort\n3. Selectionsort")
ch = int(input("Enter the choice: "))

if ch == 1:
    array = read_input()
    mergeSort(array)
    labeldata = "mergesort"
    print("Sorted array is:")
    print(array)
elif ch == 2:
    array = read_input()
    size = len(array)
    labeldata = "quicksort"
    quickSort(array, 0, size - 1)
    print("Sorted array is:")
    print(array)
elif ch == 3:
    array = read_input()
    size = len(array)
    labeldata = "selectionsort"
    selectionSort(array, size)
    print("Sorted array is:")
    print(array)

print("******** Running Time Analysis ********")
for i in range(1, 10):
    array = np.random.randint(0, 1000 * i, 1000 * i)
    print(i)
    start = time.time()
    if ch == 1:
        mergeSort(array)
    elif ch == 2:
        size = len(array)
        quickSort(array, 0, size - 1)
    else:
        size = len(array)
        selectionSort(array, size)
    end = time.time()
    print(len(array), "elements sorted by", labeldata, "Time taken:", end - start)
    elements.append(len(array))
    times.append(end - start)

plt.xlabel("List Length")
plt.ylabel("Time Complexity")
plt.plot(elements, times, label=labeldata)
plt.grid()
plt.legend()
plt.show()
