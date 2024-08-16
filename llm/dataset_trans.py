import json

data_dir = "piqa"

with open(os.path.join(data_dir, 'dev.json'), 'r') as f:
    with open(os.path.join(data_dir, 'new_dev.json'), 'r') as f:
        for line in f:
            line_data = json.load(line)
            new_line_data = {}
            sl = line_data.split()
            prefix, latter = sl[:-1], sl[-1]
            new_line_data['src'] = line_data['src'] + ' '.join(prefix)

