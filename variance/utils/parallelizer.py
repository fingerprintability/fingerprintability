from multiprocessing import cpu_count, Pool

# number of cores
NUM_PROCS = cpu_count()


def multiproc(f, it, procs=NUM_PROCS):
    p = Pool(NUM_PROCS)
    try:
        result = p.map(f, it)
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Quitting")
    p.close()
    p.join()
    return result


if __name__ == '__main__':
    pass
