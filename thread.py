import threading
import time
def add_thread():
    print("This is an add thread, number is %s"%threading.current_thread())

def thread_job():
    print("T1 start\n")
    for _ in range(5):
        time.sleep(0.1)
    print("T1 stop\n")

def main():
    added_thread = threading.Thread(target=thread_job(), name="T1")
    added_thread.start()
    added_thread.join()
    print("all done\n")

if __name__ == '__main__':
    main()