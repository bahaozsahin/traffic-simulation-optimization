import os, sys, time
from utils.parse_args import parse_cl_args
from utils.process.dist import Dist

def main():
    start_time = time.time()
    print('started running...')
    args = parse_cl_args()
    procs = Dist(args, args.tsc, args.mode)
    procs.run()
    print(args)
    print('...finished running')
    print('run time '+str((time.time()-start_time)/60)) #time in mins

if __name__ == '__main__':
    main()
