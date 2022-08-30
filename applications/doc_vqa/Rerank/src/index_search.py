import sys
import time
import faiss
import math
import numpy as np


def read_embed(file_name, dim=768, bs=3000):
    if file_name.endswith('npy'):
        i = 0
        emb_np = np.load(file_name)
        while (i < len(emb_np)):
            vec_list = emb_np[i:i + bs]
            i += bs
            yield vec_list
    else:
        vec_list = []
        with open(file_name) as inp:
            for line in inp:
                data = line.strip()
                vector = [float(item) for item in data.split(' ')]
                assert len(vector) == dim
                vec_list.append(vector)
                if len(vec_list) == bs:
                    yield vec_list
                    vec_list = []
            if vec_list:
                yield vec_list


def load_qid(file_name):
    qid_list = []
    with open(file_name) as inp:
        for line in inp:
            line = line.strip()
            qid = line.split('\t')[0]
            qid_list.append(qid)
    return qid_list


def search(index, emb_file, qid_list, outfile, top_k):
    q_idx = 0
    with open(outfile, 'w') as out:
        for batch_vec in read_embed(emb_file):
            q_emb_matrix = np.array(batch_vec)
            res_dist, res_p_id = index.search(q_emb_matrix.astype('float32'),
                                              top_k)
            for i in range(len(q_emb_matrix)):
                qid = qid_list[q_idx]
                for j in range(top_k):
                    pid = res_p_id[i][j]
                    score = res_dist[i][j]
                    out.write('%s\t%s\t%s\t%s\n' % (qid, pid, j + 1, score))
                q_idx += 1


def main():
    part = sys.argv[1]
    topk = int(sys.argv[2])
    q_text_file = sys.argv[3]
    outfile = 'output/res.top%s-part%s' % (topk, part)

    qid_list = load_qid(q_text_file)

    engine = faiss.read_index("output/para.index.part%s" % part)
    emb_file = 'output/query.emb.npy'
    search(engine, emb_file, qid_list, outfile, topk)


if __name__ == "__main__":
    main()
