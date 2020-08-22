import multiprocessing as mp
import time

def job(q):
    res = 0
    for i in range(1000):
        res += i**2
    q.put(res)

def job_core(x):
    return x**2

def multi_core():
    pool = mp.Pool()
    res = pool.map(job_core, range(10))
    print(res)

def lock_job(v, num, l):
    l.acquire()
    for _ in range(10):
        time.sleep(0.1)
        v.value += num
        print(v.value)
    l.release()


if __name__ == '__main__':
    # q = mp.Queue()
    l = mp.Lock()
    v = mp.Value('i', 0) #共享内存
    p1 = mp.Process(target=lock_job, args=(v, 1, l))
    p2 = mp.Process(target=lock_job, args=(v, 3, l))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    # res1 = q.get()
    # res2 = q.get()
    # multi_core()