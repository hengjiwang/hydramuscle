from multiprocessing import Pool

a = list(range(1000))

def task(j):
    print(a[j])

pool = Pool(processes=4)

for j in range(1000):
    pool.apply_async(task, args=(j,))

pool.close()
pool.join()