import os
from typing import List, Tuple, Union
import numpy as np
from ..utils import read_json_config, write_json_config
from dataclasses import dataclass

@dataclass
class HardwareProfilerArgs:
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    max_pp_deg: int = 8
    max_tp_deg: int = 8
    start_mb: int = 1
    end_mb: int = 1024
    scale: int = 2
    mpi_path: str = "/usr/local/mpi"
    hostfile: str = "hostfile"
    avg_or_min_or_first: str = "avg"
    backend: str = 'nccl'
    
    def initialize(self, args_dict):
        self.num_nodes = int(args_dict.get("--num_nodes", self.num_nodes))
        self.num_gpus_per_node = int(args_dict.get("--num_gpus_per_node", self.num_gpus_per_node))
        self.max_pp_deg = int(args_dict.get("--max_pp_deg", self.max_pp_deg))
        self.start_mb = int(args_dict.get("--start_mb", self.start_mb))
        self.end_mb = int(args_dict.get("--end_mb", self.end_mb))
        self.scale = int(args_dict.get("--scale", self.scale))
        self.mpi_path = args_dict.get("--mpi_path", self.mpi_path)
        self.hostfile = args_dict.get("--hostfile", self.hostfile)
        self.avg_or_min_or_first = args_dict.get("--avg_or_min_or_first", self.avg_or_min_or_first)
        self.backend = args_dict.get('--backend', self.backend)
        self.max_tp_deg = int(args_dict.get('--max_tp_deg'))

class HardwareProfiler():
    """Hardware profiler for analyzing communication bandwidth and other hardware characteristics"""

    def __init__(self, args:HardwareProfilerArgs):
        self.args = args
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.execution_path = None
        
    def set_execution_path(self, path: str) -> None:
        self.execution_path = path

    def profile_bandwidth(self, backend: str = "nccl") -> None:
        """
            Profile communication bandwidth between devices
            This method profiles both allreduce and point-to-point communication bandwidth.
        """
        
        args = self.args
        world_size = args.num_nodes * args.num_gpus_per_node
        if backend != "nccl":
            raise NotImplementedError(
                "Only NCCL backend is supported for profiling bandwidth currently. "
                "Please set backend='nccl' to use NCCL tests."
            )

        # Create hardware config directory
        hardware_config_dir = os.path.join(self.execution_path, "./configs")
        os.makedirs(hardware_config_dir, exist_ok=True)

        # Profile allreduce bandwidth
        nccl_file = "build/all_reduce_perf"
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = f"allreduce_bandwidth_{args.num_nodes}nodes_{args.num_gpus_per_node}gpus_per_node.json" 
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        allreduce_size = world_size
        while allreduce_size > 1:
            print(f'============= allreduce_size: {allreduce_size} =============')
            allreduce_groups = self.generate_allreduce_groups(world_size, allreduce_size)
            bandwidth = self.launch_nccl_test(allreduce_groups, args.num_gpus_per_node, ARGS)
            key = f"allreduce_size_{int(allreduce_size)}"
            self.write_config(hardware_config_path, key, bandwidth)
            print("=" * 70, "\n")
            allreduce_size /= 2

        # Profile p2p bandwidth
        nccl_file = "build/sendrecv_perf"
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = f"p2p_bandwidth_{args.num_nodes}nodes_{args.num_gpus_per_node}gpus_per_node.json"
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        pp_deg = 2
        while pp_deg <= world_size and pp_deg <= args.max_pp_deg:
            print("============= pp_size: %d =============" % (pp_deg))
            p2p_groups = self.generate_p2p_groups(world_size, pp_deg)
            bandwidth = self.launch_nccl_test(p2p_groups, args.num_gpus_per_node, ARGS)
            key = "pp_size_%d" % pp_deg
            self.write_config(hardware_config_path, key, bandwidth)
            print("=" * 70, "\n")
            pp_deg *= 2

        os.system("rm -rf %s" % (os.path.join(self.execution_path, "nccl_test.log")))

    def write_config(self, hardware_config_path: str, key: str, bandwidth: float) -> None:
        config = read_json_config(hardware_config_path) if os.path.exists(hardware_config_path) else dict()
        config[key] = bandwidth
        write_json_config(hardware_config_path, config)
        print("Already written bandwidth/time %s into hardware config file %s!" % (key, hardware_config_path))

    def read_hostfile(self) -> List[str]:
        args = self.args
        hostfile = os.path.join(self.execution_path, args.hostfile)
        with open(hostfile, "r") as f:
            hostnames = f.readlines()
        hostnames = [hostname.strip() for hostname in hostnames if hostname.strip() != ""]

        return hostnames

    def prepare_nccltest_args(self, nccl_file="build/all_reduce_perf") -> str:
        """Prepare arguments for NCCL tests

        Args:
            nccl_file: Path to NCCL test executable relative to nccl_test_dir

        Returns:
            str: Command line arguments for NCCL test

        Note:
            Will build NCCL test if not already built
        """
        args = self.args
        nccl_test_dir = os.path.join(self.path, "./site_package/nccl-tests")
        nccl_file = os.path.join(nccl_test_dir, nccl_file)
        if not os.path.exists(nccl_file):
            print("Nccl test file %s does not exist!" % nccl_file)
            print("Building nccl-test...")
            if args.num_nodes == 1:
                os.system(
                    "USE_EXPORT_VARIABLE=1 MAKE_MPI=0 sh %s" % (os.path.join(self.path, "scripts/make_nccl_test.sh"))
                )
            else:
                os.system(
                    "USE_EXPORT_VARIABLE=1 MAKE_MPI=1 MPI_PATH=%s sh %s"
                    % (args.mpi_path, os.path.join(self.path, "scripts/make_nccl_test.sh"))
                )
            print("Nccl-test built succesfully!")
        ARGS = ""
        ARGS += "USE_EXPORT_VARIABLE=1 "
        ARGS += "START_MB=%d " % args.start_mb
        ARGS += "END_MB=%d " % args.end_mb
        ARGS += "SCALE=%d " % args.scale
        ARGS += "NCCLTEST_FILE=%s " % nccl_file
        ARGS += "OUTPUT_TO_LOG=1 "
        return ARGS

    def generate_allreduce_groups(self, world_size: int, allreduce_size: int) -> List[List[int]]:
        """Generate groups for allreduce communication

        Args:
            world_size: Total number of processes
            allreduce_size: Size of each allreduce group
            allreduce_consec: Whether to use consecutive GPU mapping

        Returns:
            List[List[int]]: List of process groups for allreduce
        """
        allreduce_size = int(allreduce_size)
        num_allreduce_groups = int(world_size // allreduce_size)
        allreduce_groups = []
        for i in range(num_allreduce_groups):
            ranks = list(range(i * allreduce_size, (i + 1) * allreduce_size))
            allreduce_groups.append(ranks)
        return allreduce_groups

    def generate_p2p_groups(self, world_size: int, pp_size: int) -> List[List[int]]:
        """Generate groups for point-to-point communication

        Args:
            world_size: Total number of processes
            pp_size: Size of each pipeline parallel group

        Returns:
            List[List[int]]: List of process groups for p2p communication
        """
        pp_size = int(pp_size)
        num_pp_groups = int(world_size // pp_size)
        pp_groups = []
        for i in range(num_pp_groups):
            ranks = list(range(i, world_size, num_pp_groups))
            pp_groups.append(ranks)
        return pp_groups

    def launch_nccl_test(
        self, groups: List[List[int]], num_gpus_per_node: int, ARGS: str, mode: str = "avg"
    ) -> Union[float, Tuple[List[int], List[float]]]:
        """Launch NCCL test for given process groups

        Args:
            groups: List of process groups to test
            num_gpus_per_node: Number of GPUs per node
            ARGS: Command line arguments for NCCL test
            mode: Test mode, either 'avg' for average bandwidth or 'detail' for detailed results

        Returns:
            Union[float, Tuple[List[int], List[float]]]:
                If mode=='avg': Average bandwidth in MB/s
                If mode=='detail': Tuple of (message sizes in MB, communication times in milliseconds)
        """
        hostnames = self.read_hostfile()
        bandwidths = []
        for group in groups:
            print("device group:", group)
            host_ids = sorted(list(set([rank // num_gpus_per_node for rank in group])))
            group_num_nodes = len(host_ids)
            group_num_gpus_per_node = len(group) // group_num_nodes
            cuda_visible_devices = sorted(list(set([rank % num_gpus_per_node for rank in group])))
            print(
                "num_nodes: %d, host_ids:" % group_num_nodes,
                host_ids,
                " num_gpus_per_node: %d, cuda_visible_devices:" % group_num_gpus_per_node,
                cuda_visible_devices,
            )
            hostname = ",".join([hostnames[i] for i in host_ids])
            DEVICE_ARGS = ""
            DEVICE_ARGS += "HOSTNAMES=%s " % hostname
            DEVICE_ARGS += "NUM_NODES=%d " % group_num_nodes
            DEVICE_ARGS += "NUM_GPUS_PER_NODE=%d " % group_num_gpus_per_node
            DEVICE_ARGS += 'DEVICES="CUDA_VISIBLE_DEVICES=%s" ' % (",".join([str(i) for i in cuda_visible_devices]))
            if mode == "detail":
                ARGS += "START_MB=1 "
                ARGS += "END_MB=1024 "
            print(DEVICE_ARGS + ARGS)
            os.system(DEVICE_ARGS + ARGS + "sh %s" % (os.path.join(self.path, "scripts/run_nccl_test.sh")))
            with open("nccl_log/1/rank.0/stdout", "r") as f:
                lines = f.readlines()
            if mode == "avg":
                for line in lines[::-1]:
                    if "Avg bus bandwidth" in line:
                        result = line
                        bandwidth = float(line.split()[-1])
                        break
                print(result)
                bandwidths.append(bandwidth)
                if self.args.avg_or_min_or_first == "first":
                    break
            else:
                sizes = []
                times = []
                for line in lines:
                    datas = line.split()
                    if len(datas) > 10 and datas[0].isdigit():
                        sizes.append(int(datas[0]) // 1024 // 1024)
                        times.append(float(datas[5]) / 1000)
                return sizes, times
        bandwidth = np.min(bandwidths) if self.args.avg_or_min_or_first == "min" else np.mean(bandwidths)
        print("Bandwidths:", bandwidths, "Average bandwidth:", bandwidth)
        print()
        return bandwidth

    # =============== For Launching Scripts for Profiling Overlap Slowdown Coefficient ===============
    def profile_overlap(self):
        print("Profiling overlap slowdown coefficient...")
        interpreter = os.getenv('INTERPRETER')
        CMD = f'{interpreter} -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir output/profile_overlap '
        script_path = os.path.join(self.path, "profile_overlap.py")
        CMD += f'{script_path} --output_dir "./output"'
        
        scripts_path = './scripts/profile_overlap.sh'
        with open(scripts_path, 'w') as f:
            NCCL_IB_HCA = os.getenv('NCCL_IB_HCA')
            NCCL_IB_DISABLE = os.getenv('NCCL_IB_DISABLE')
            f.write(f'export NCCL_IB_HCA={NCCL_IB_HCA}\n')
            f.write(f'export NCCL_IB_DISABLE={NCCL_IB_DISABLE}\n')
            f.write(f'echo "Running {CMD}"\n')
            f.write(f'{CMD}\n')
            f.write('sleep 1\n')
            
            f.write(f'rm -r ./profiler_log')
    
    def profile_allreduce(self):
        args = self.args
        print("Profiling allreduce bandwidth...")
        save_file_name = os.path.join(self.execution_path, f"configs/allreduce_bandwidth_{args.num_nodes}nodes_{args.num_gpus_per_node}gpus_per_node.json" )
        tp_limit = self.args.num_gpus_per_node
        CMD_LIST = []
        while tp_limit > 1:
            CMD = os.getenv('LAUNCHER')
            CMD += f' --log_dir output/profile_allreduce '
            script_path = os.path.join(self.path, "profile_allreduce.py ")
            CMD += f'{script_path} --output_dir "./output" '
            CMD += f'--profile_time 0 --tp_deg {tp_limit} --save_file_name {save_file_name} '
            CMD_LIST.append(CMD)
            tp_limit //= 2
        
        scripts_path = './scripts/profile_allreduce.sh'
        with open(scripts_path, 'w') as f:
            NCCL_IB_HCA = os.getenv('NCCL_IB_HCA')
            NCCL_IB_DISABLE = os.getenv('NCCL_IB_DISABLE')
            f.write(f'export NCCL_IB_HCA={NCCL_IB_HCA}\n')
            f.write(f'export NCCL_IB_DISABLE={NCCL_IB_DISABLE}\n')
            for cmd in CMD_LIST:
                f.write(f'echo "Running {cmd}"\n')
                f.write(f'{cmd}\n')
                f.write('sleep 1\n')
            
            f.write(f'rm -r ./profiler_log')
    
    def profile_p2p(self):
        args = self.args
        print("Profiling point-to-point bandwidth...")
        save_file_name = os.path.join(self.execution_path, f"configs/p2p_bandwidth_{args.num_nodes}nodes_{args.num_gpus_per_node}gpus_per_node.json")
        pp_deg = 2
        CMD_LIST = []
        while pp_deg <= args.max_pp_deg:
            CMD = os.getenv('LAUNCHER')
            CMD += ' --log_dir /output/profile_p2p '
            script_path = os.path.join(self.path, "profile_p2p.py")
            CMD += f'{script_path} --output_dir "./output" '
            CMD += f'--pp_deg {pp_deg} --save_file_name {save_file_name} '
            CMD_LIST.append(CMD)
            pp_deg *= 2
            
        scripts_path = './scripts/profile_p2p.sh'
        with open(scripts_path, 'w') as f:
            NCCL_IB_HCA = os.getenv('NCCL_IB_HCA')
            NCCL_IB_DISABLE = os.getenv('NCCL_IB_DISABLE')
            f.write(f'export NCCL_IB_HCA={NCCL_IB_HCA}\n')
            f.write(f'export NCCL_IB_DISABLE={NCCL_IB_DISABLE}\n')
            for cmd in CMD_LIST:
                f.write(f'echo "Running {cmd}"\n')
                f.write(f'{cmd}\n')
                f.write('sleep 1\n')

            f.write(f'rm -r ./profiler_log')
            
    def profile_sp(self):
        print("Profiling sp bandwidth...")
        args = self.args
        save_file_name = os.path.join(self.execution_path, f"configs/sp_time_{args.num_nodes}nodes_{args.num_gpus_per_node}gpus_per_node.json" )
        
        def allreduce_script(allreduce_size, buffer_size):
            CMD = os.getenv('LAUNCHER')
            CMD += ' --log_dir /output/profile_allreduce_sp '
            script_path = os.path.join(self.path, "profile_allreduce.py ")
            CMD += f'{script_path} --output_dir "./output" '
            CMD += f'--profile_time 1 --tp_deg {allreduce_size} --save_file_name {save_file_name} --local_batch_size {buffer_size}'
            return CMD
            
        scripts_path = './scripts/profile_allreduce_sp.sh'
        with open(scripts_path, 'w') as f:
            NCCL_IB_HCA = os.getenv('NCCL_IB_HCA')
            NCCL_IB_DISABLE = os.getenv('NCCL_IB_DISABLE')
            f.write(f'export NCCL_IB_HCA={NCCL_IB_HCA}\n')
            f.write(f'export NCCL_IB_DISABLE={NCCL_IB_DISABLE}\n')

            allreduce_size = min(self.args.num_nodes * self.args.num_gpus_per_node, self.args.max_tp_deg)
            while allreduce_size > 1:
                buffer_size = 1024
                while buffer_size >= 1:
                    script = allreduce_script(allreduce_size, buffer_size)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(f'{script}\n')
                    f.write("sleep 1\n")
                    buffer_size //= 2
                allreduce_size //= 2
            
            f.write(f'rm -r ./profiler_log')
                
        def all2all_script(all2all_size, buffer_size):
            CMD = os.getenv('LAUNCHER')
            CMD += ' --log_dir /output/profile_all2all '
            script_path = os.path.join(self.path, "profile_all2all.py ")
            CMD += f'{script_path} --output_dir "./output" '
            CMD += f'--tp_deg {all2all_size} --save_file_name {save_file_name} --local_batch_size {buffer_size}'
            return CMD
        
        scripts_path = './scripts/profile_all2all.sh'
        with open(scripts_path, "w") as f:
            NCCL_IB_HCA = os.getenv('NCCL_IB_HCA')
            NCCL_IB_DISABLE = os.getenv('NCCL_IB_DISABLE')
            f.write(f'export NCCL_IB_HCA={NCCL_IB_HCA}\n')
            f.write(f'export NCCL_IB_DISABLE={NCCL_IB_DISABLE}\n')
            
            all2all_size = min(self.args.num_nodes * self.args.num_gpus_per_node, self.args.max_tp_deg)
            while all2all_size > 1:
                buffer_size = 1024
                while buffer_size >= 1:
                    script = all2all_script(all2all_size, buffer_size)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(f'{script}\n')
                    f.write("sleep 1\n")
                    buffer_size //= 2
                all2all_size //= 2
        
                    
    # =============== remove some files after profiling ===============
    def remove_files(self):
        cmd_list = [
            "rm -r %s" % os.path.join(self.execution_path, "nccl_log"),
            "rm -r %s" % os.path.join(self.execution_path, "profiler_log"),
        ]
        
        for cmd in cmd_list:
            os.system(cmd)