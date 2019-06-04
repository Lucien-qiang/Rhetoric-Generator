'''
   Created by zhiqliu on 2018/12/18.
'''


import codecs
import os

def data_process(input1, input2, output):
    fw = codecs.open(output, 'w', 'utf-8')
    with codecs.open(input1, 'r', 'utf-8') as fr:
        lists = []
        for line in fr:
            lists.append(line.strip()[2:-2])
    with codecs.open(input2, 'r', 'utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line.split('\t')[0] not in lists:
                fw.write(line + '\n')

def c5_to_c3(input1, output):
    d = {'0':1,'1':1,'2':2,'3':2,'4':2,'5':0}
    fw = codecs.open(output, 'w', 'utf-8')
    with codecs.open(input1, 'r', 'utf-8') as fr:
        for line in fr:
            lines = line.strip().split('\t')
            lines[2] = str(d[lines[2]])
            fw.write('\t'.join(lines) + '\n')

#('../log/wrong_pred.txt', '../data2/trainset.txt', '../data2/trainset_new.txt')
c5_to_c3('../data2/testset_1023.txt', '../data2/testset_1128.txt')
