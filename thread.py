import threading
import time
def add_thread():
    print("This is an add thread, number is %s"%threading.current_thread())

def thread_job():
    print("T1 start\n")
    for _ in range(5):
        time.sleep(0.1)
    print("T1 stop\n")

#python中同类型的数据，多线程并不一定是高效的，因为GIL的缘故，会锁住其他线程，实际在跑的仍然是单线程。
def main():
    added_thread = threading.Thread(target=thread_job(), name="T1")
    added_thread.start()
    added_thread.join()
    print("all done\n")

if __name__ == '__main__':
    main()