def galvatron_profile_hardware_args(parser):
    group = parser.add_argument_group(title="Galvatron Profiling Hardware Arguments")
    
    group.add_argument(
        "--num_nodes", type=int, default=1, help="Number of Nodes.",
    )
    group.add_argument(
        "--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node.",
    )
    group.add_argument(
        "--master_addr", type=str, default='$MASTER_ADDR', help="Master address.",
    )
    group.add_argument(
        "--master_port", type=str, default='$MASTER_PORT', help="Master port.",
    )
    group.add_argument(
        "--node_rank", type=str, default='$RANK', help="Node rank.",
    )
    group.add_argument(
        "--max_tp_size", type=int, default=8, help="Maximum tensor parallel size.",
    )
    group.add_argument(
        '--envs', type=str, nargs='+', default=[], help='Additional environment variables in format KEY=VALUE',
    )
    group.add_argument(
        "--backend", type=str, default='nccl', help="Backend of nccl-tests.", choices=['nccl', 'paddle'],
    )
    group.add_argument(
        "--nccl_test_dir", type=str, default='nccl-tests', help="Directory of nccl-tests.",
    )
    group.add_argument(
        "--mpi_path", type=str, default='/usr/local/mpi/', help="MPI Path.",
    )
    group.add_argument(
        "--start_mb", type=int, default=16, help="Starting communication size in MB.",
    )
    group.add_argument(
        "--end_mb", type=int, default=512, help="Ending communication size in MB.",
    )
    group.add_argument(
        "--scale", type=int, default=2, help="Memory scale of nccl-tests.",
    )
    group.add_argument(
        "--hostfile", type=str, default='hostfile', help="Hostfile for nccl-tests.",
    )
    group.add_argument(
        "--avg_or_min_or_first", type=str, default='first', help="For a given group size, if 'first', only profile first group; if 'min', profile the group with minimum bandwidth; if 'avg', profile all groups and take the average bandwidth.", choices=['first', 'min', 'avg'],
    )
    group.add_argument(
        "--max_pp_deg", type=int, default=8, help="Maximum pipeline parallel degree to search."
    )
    group.add_argument(
        "--overlap_time_multiply", type=int, default=4, help='The multiple of communication time and computation time when overlapped.'
    )
    
    return parser