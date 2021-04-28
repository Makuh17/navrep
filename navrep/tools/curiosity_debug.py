import time
import functools


def measure_duration(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs): 
        start_time = time.time()
        values = func(*args, **kwargs)
        end_time = time.time()    
        duration = end_time - start_time 
        print('Function {name}, took: {t}'.format(name=func.__name__, t=duration))
        return values
    return wrapper

@measure_duration
def test_function():
    for i in range(int(1e8)):
        pass

if __name__=='__main__':
   test_function()
