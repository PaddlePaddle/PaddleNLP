if __name__ == '__main__':
    _linguangming_pre_prepared_rind_id = {}
    if not _linguangming_pre_prepared_rind_id:
        print(f'[linguangming] pre-prepared rind id is empty, preparing...')
        rind_id = 1
        world_size = 64
        numbers = list(range(world_size))
        def generate_all_permutations(len):
            from itertools import permutations
            return list(permutations(numbers, len))
        # for length in range(1, world_size + 1):
        #     permutations = generate_all_permutations(length)
        #     for permutation in permutations:
        #         _linguangming_pre_prepared_rind_id[tuple(permutation)] = rind_id
        #         rind_id += 1
        # print(f'[linguangming] pre-prepared rind id prepared, total: {len(_linguangming_pre_prepared_rind_id)}')
        def generate_ordered_combinations(numbers, length, reverse=False):
            if reverse:
                numbers = sorted(numbers, reverse=True)
            
            result = []
            stack = [(0, [])]
            
            while stack:
                start, path = stack.pop()
                if len(path) == length:
                    result.append(tuple(path))
                    continue
                for i in range(start, len(numbers)):
                    stack.append((i + 1, path + [numbers[i]]))
            
            return result
        all = []
        all_length = []
        i = 1
        while i <= world_size:
            all_length.append(i)
            i *= 2
            
        for length in all_length:
        # for length in range(1, world_size + 1):
            combinations = generate_ordered_combinations(numbers, length)
            for combination in combinations:
                all.append(combination)
                # all[combination] = _linguangming_pre_prepared_rind_id[combination]
            combinations = generate_ordered_combinations(numbers, length, reverse=True)
            for combination in combinations:
                all.append(combination)
                # all[combination] = _linguangming_pre_prepared_rind_id[combination]
        print(f'[linguangming] all combinations prepared, total: {len(all)}')
        for i, combination in enumerate(all):
            print(f'{i}: {combination}')