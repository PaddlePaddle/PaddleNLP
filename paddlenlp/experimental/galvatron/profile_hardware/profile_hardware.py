from paddlenlp.experimental.galvatron.core.profiler import GalvatronHardwareProfiler
from paddlenlp.experimental.galvatron.core.arguments import initialize_galvatron
import os

if __name__ == '__main__':
    args = initialize_galvatron(mode='profile_hardware')
    print(args)
    profiler = GalvatronHardwareProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_path(path)
    
    # profile allreduce & p2p bandwidth
    profiler.profile_bandwidth(backend=args.backend)
    # profile allreduce & a2a bandwidth in different communication size
    profiler.profile_sp_bandwidth(backend=args.backend)
    # profile overlapping slowdown coefficient
    # profiler.profile_overlap()