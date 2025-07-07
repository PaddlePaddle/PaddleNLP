
import paddle.distributed as dist

if __name__ == '__main__':
    mesh = dist.ProcessMesh([[0, 1, 2, 3]], dim_names=['dp', 'tp'])
    key = sorted(mesh.process_ids)
    print(f'key: {key}')
    print(f'type(key): {type(key)}')
    key_str = str(key)
    print(f'key_str: {key_str}')
    print(f'type(key_str): {type(key_str)}')
    string = f'_dp{mesh.shape[0]}_tp{mesh.shape[1]}'
    print(f'string: {string}')
    
    final_key = str(sorted(mesh.process_ids)) + f'_dp{mesh.shape[0]}_tp{mesh.shape[1]}'
    print(f'final_key: {final_key}')