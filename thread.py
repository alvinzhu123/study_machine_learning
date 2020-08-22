import threading
import time
def add_thread():
    print("This is an add thread, number is %s"%threading.current_thread())

def thread_job():
    print("T1 start\n")
    for _ in range(5):
        time.sleep(0.1)
    print("T1 stop\n")

def job1():
    global A, lock
    lock.acquire()
    for _ in range(10):
        A += 1
        print("job1", A)
    lock.release()

def job2():
    global A, lock
    lock.acquire()
    for _ in range(10):
        A += 10
        print("job2", A)
    lock.release()

#python中同类型的数据，多线程并不一定是高效的，因为GIL的缘故，会锁住其他线程，实际在跑的仍然是单线程。


if __name__ == '__main__':
    lock = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job1, name='t1')
    t2 = threading.Thread(target=job2, name='t2')
    t1.start()
    t2.start()
    t1.join()
    t2.join()