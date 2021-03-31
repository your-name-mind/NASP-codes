# coding utf-8
# time 2021/3/30 1:00
# author wzy
import logging
import sys
logging.basicConfig(stream=sys.stdout,format='%(asctime)s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def write():
    logging.info("111")

write()

if __name__ == '__main__':
    for i,(x,y) in enumerate(zip(range(10),range(10,20))):
        print(i,x,y)