import os
import re


def modify_doc_title_dir(abspath_rstfiles_dir):
    """
    rst文件中：有‘========’和‘----------’行的表示其行上一行的文字是标题，
    ‘=’和‘-’要大于等于标题的长度。
    使用sphinx-apidoc -o ./source/rst_files /home/myubuntu/pro/mypro命令将
    生成rst文件放在./source/rst_files目录下， 执行sphinx-quickstart命令生成的
    index.rst不用放到这个目录中。 或在source目录下新建
    rst_files目录然后将rst文件剪切到这个目录下，修改后再剪切出来
    生成rst文件后将rst_files/modules.rst文件中的标题去掉，并修改maxdepth字段。
    删除和修改使用sphinx-apidoc -o 命令的生成的rst文件中的标题
    :param abspath_rstfiles_dir: rst文件所在的文件夹的绝对路径
    :return:
    """
    rst_files = os.listdir(abspath_rstfiles_dir)
    # 要删除的节点(标题目录的节点)
    del_nodes = ['Submodules', 'Module contents', 'Subpackages']
    # 要删除的标题中的字符串
    del_str = [' module', ' package']
    # datasets需要的部分
    dataset_list = ['datasets', 'dataset']
    # 需要call方法
    add_call_files = [
        'data.collate', 'data.iterator', 'data.sampler', 'data.tokenizer',
        'data.vocab', 'tokenizer\_utils'
    ]
    # 删除inheritance
    del_inheritance = [
        'crf', 'tcn', 'distributed', 'dataset', 'paraller', 'decoder', 'rdrop',
        'decoding', 'fast\_transformer', 'Adamoptimizer', 'attention\_utils',
        'model\_utils', 'batch\_sampler', 'model'
    ]
    # 文档中空白的part，不显示
    del_rst = ['iterator', 'constant']
    for rst_file in rst_files:
        f = open(os.path.join(abspath_rstfiles_dir, rst_file), 'r')
        file_lines = f.readlines()
        f.close()
        write_con = []
        flag = 0
        first_line = file_lines[0]
        #去除不需要的datasets
        if 'datasets' in first_line:
            name = first_line.split()[0]
            length = len(name.split('.'))
            # paddlenlp.datasets 需要留下
            if length > 2:
                if 'datasets.dataset' not in first_line:
                    path = os.path.join(abspath_rstfiles_dir, rst_file)
                    print(path)
                    os.remove(path)
                    print(path)
                    continue
        # 去除文档中空白页面，目前是data.iterator, embeddings.constant部分
        del_rst_flag = 0
        for pattern in del_rst:
            if pattern in first_line:
                path = os.path.join(abspath_rstfiles_dir, rst_file)
                os.remove(path)
                del_rst_flag = 1
                break
        if del_rst_flag == 1:
            continue
        # 是否加入call
        add_call_files_flag = 0
        for i in add_call_files:
            if i in first_line:
                add_call_files_flag = 1
        # 是否删除inheritance
        del_inheritance_flag = 0
        for j in del_inheritance:
            if j in first_line:
                del_inheritance_flag = 1
        if 'modeling' in first_line:
            del_inheritance_flag = 0
        for file_line in file_lines:
            if file_line.strip() in del_nodes:
                flag = 1
                continue
            if flag:
                flag = 0
                continue
            if re.search(del_str[0], file_line):
                length = len(file_line.split('.'))
                if length > 2:
                    modify_line = file_line.split('.')[-1].replace(
                        del_str[0], '')
                else:
                    modify_line = file_line.replace(del_str[0], '')
                write_con.append(modify_line)
                continue
            if re.search(del_str[1], file_line):
                length = len(file_line.split('.'))
                if length > 2:
                    modify_line = file_line.split('.')[-1].replace(
                        del_str[1], '')
                else:
                    modify_line = file_line.replace(del_str[1], '')
                write_con.append(modify_line)
                continue
            if 'undoc-members' in file_line:
                if 'no-undoc-members' not in file_line:
                    file_line = file_line.replace('undoc-members',
                                                  'no-undoc-members')
            # 去除datasets中多余内容
            if 'paddlenlp.datasets' in file_line:
                last_name = file_line.split('.')[-1]
                if last_name.strip() not in dataset_list:
                    continue
            if 'show-inheritance' in file_line:
                if del_inheritance_flag == 0:
                    write_con.append(file_line)
            else:
                write_con.append(file_line)
        if add_call_files_flag == 1:
            write_con.append("   :special-members: __call__\n")
        f = open(os.path.join(abspath_rstfiles_dir, rst_file), 'w')
        f.writelines(write_con)
        f.close()


if __name__ == '__main__':
    modify_doc_title_dir('./source')
