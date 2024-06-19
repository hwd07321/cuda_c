import torch
import reduce_cuda

def reduce(input_tensor):
    return reduce_cuda.reduce(input_tensor)


# using "python setup.py install" compile the code and the run this script

if __name__ == '__main__'{
    input_tensor = torch.rand(1024*1024, dtype=torch.float32).cuda()
    result = reduce(input_tensor)
    print(result)
}