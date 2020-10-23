import numpy as np
import os
import sys
import json

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

makedirs('../data/pn/')
makedirs('../outputs/summary/')
makedirs('../outputs/ckpt/')
makedirs('../outputs/logits/')

# folder of training datasets
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = "../data/raw_data/"
# files to export data
if len(sys.argv) > 2:
    export_path = sys.argv[2]
else:
    export_path = "../data/"

fixlen = 120
maxlen = 100
labellen = 10

word2id = {}
relation2id = {}
hier1_relation2id = {}
hier2_relation2id = {}

word_size = 0
word_vec = None

def pos_embed(x):
    return max(0, min(x + maxlen, maxlen + maxlen + 1))


def find_index(x,y):
    for index, item in enumerate(y):
        if x == item:
            return index
    return -1

def assert_equal(x, y):
    assert x==y, 'ERROR: {} != {}'.format(x, y)


def init_word():
    # reading word embedding data...
    global word2id, word_size
    print('reading word embedding data...')
    f = open(data_path + 'vec.txt', "r")
    total, size = f.readline().strip().split()[:2]
    total = (int)(total)
    word_size = (int)(size)
    vec = np.ones((total, word_size), dtype = np.float32)
    for i in range(total):
        content = f.readline().strip().split()
        word2id[content[0]] = len(word2id)
        for j in range(word_size):
            vec[i][j] = (float)(content[j+1])
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    global word_vec
    word_vec = vec

def init_relation():
    global relation2id
    global hier1_relaiton2id
    global hier2_relation2id
    print('reading relation ids...')
    i_hier1 = 0
    i_hier2 = 0
    hier1_relation2id['NA'] = 0
    hier2_relation2id['NA'] = 0
    f = open(data_path + "relation2id.txt", "r")
    total = (int)(f.readline().strip())
    for i in range(total):
        content = f.readline().strip().split()
        relation2id[content[0]] = int(content[1])
        if content[0] != 'NA':
            relation_split = content[0].strip().split('/')
            first_relation = relation_split[1]
            second_relation = relation_split[2]
            temp = '/' + first_relation + '/' + second_relation
            if first_relation not in hier1_relation2id:
                i_hier1 += 1
                hier1_relation2id[first_relation] = i_hier1
            if temp not in hier2_relation2id:
                i_hier2 += 1
                hier2_relation2id[temp] = i_hier2
    f.close()


def sort_files(name):
    hash = {}
    f = open(data_path + name + '.txt','r')
    s = 0
    while True:
        content = f.readline()
        if content == '':
            break
        s = s + 1
        origin_data = content
        content = content.strip().split()
        en1_id = content[0]
        en2_id = content[1]
        rel_name = content[4]
        if rel_name in relation2id:
            relation = relation2id[rel_name]
        else:
            relation = relation2id['NA']
        id = str(en1_id)+"#"+str(en2_id)+"#"+str(relation)
        if not id in hash:
            hash[id] = []
        hash[id].append(origin_data)
    f.close()
    f = open(data_path + name + "_sort.txt", "w")
    f.write("%d\n"%(s))
    for i in hash:
        for j in hash[i]:
            f.write(j)
    f.close()

def sort_test_files(name):
    hash = {}
    f = open(data_path + name + '.txt','r')
    s = 0
    while True:
        content = f.readline()
        if content == '':
            break
        s = s + 1
        origin_data = content
        content = content.strip().split()
        en1_id = content[0]
        en2_id = content[1]
        rel_name = content[4]
        if rel_name in relation2id:
            relajtion = relation2id[rel_name]
        else:
            relation = relation2id['NA']
        id = str(en1_id)+"#"+str(en2_id)
        if not id in hash:
            hash[id] = []
        hash[id].append(origin_data)
    f.close()
    f = open(data_path + name + "_sort.txt", "w")
    f.write("%d\n"%(s))
    for i in hash:
        for j in hash[i]:
            f.write(j)
    f.close()

def init_train_files(name):
    print('reading ' + name + ' data...')
    f = open(data_path + name + ".txt", "r")
    total = (int)(f.readline().strip())
    sen_word = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype=np.int32)
    sen_mask = np.zeros((total, fixlen), dtype=np.int32)
    sen_len = np.zeros((total), dtype=np.int32)
    sen_label = np.zeros((total), dtype=np.int32)
    instance_scope = []
    instance_triple = []
    en1 = np.zeros((total), dtype=np.int32)
    en2 = np.zeros((total), dtype=np.int32)
    sen_hier1_label = np.zeros((total), dtype=np.int32)
    sen_hier2_label = np.zeros((total), dtype=np.int32)
    for s in range(total):
        content = f.readline().strip().split()
        sentence = content[5:-1]
        en1_id = content[0]
        en2_id = content[1]
        en1_name = content[2]
        en2_name = content[3]
        rel_name = content[4]
        if rel_name in relation2id:
            relation = relation2id[rel_name]
        else:
            relation = relation2id['NA']
            rel_name = 'NA'
        for i in range(len(sentence)):
            if sentence[i] == en1_name:
                en1pos = i
            if sentence[i] == en2_name:
                en2pos = i
        en_first = min(en1pos, en2pos)
        if en1_name in word2id:
            en1[s] = word2id[en1_name]
        else:
            en2[s] = word2id['UNK']
        en_second = en1pos + en2pos - en_first
        if en2_name in word2id:
            en2[s] = word2id[en2_name]
        else:
            en2[s] = word2id['UNK']
        for i in range(fixlen):
            sen_word[s][i] = word2id['BLANK']
            sen_pos1[s][i] = pos_embed(i - en1pos)
            sen_pos2[s][i] = pos_embed(i - en2pos)
            if i >= len(sentence):
                sen_mask[s][i] = 0
            elif i - en_first<=0:
                sen_mask[s][i] = 1
            elif i - en_second<=0:
                sen_mask[s][i] = 2
            else:
                sen_mask[s][i] = 3
        for i, word in enumerate(sentence):
            if i >= fixlen:
                break
            elif not word in word2id:
                sen_word[s][i] = word2id['UNK']
            else:
                sen_word[s][i] = word2id[word]
        sen_len[s] = min(fixlen, len(sentence))
        sen_label[s] = relation
        # split relation into hier1 and hier2
        if rel_name == 'NA':
            sen_hier1_label[s] = hier1_relation2id['NA']
            sen_hier2_label[s] = hier2_relation2id['NA']
        else:
            relation_split = rel_name.strip().split('/')
            first_relation = relation_split[1]
            second_relation = relation_split[2]
            sen_hier1_label[s] = hier1_relation2id[first_relation]
            temp_hier2_relation = '/' + first_relation + '/' + second_relation
            sen_hier2_label[s] = hier2_relation2id[temp_hier2_relation]
        tup = (en1_id, en2_id, relation)
        if instance_triple == [] or instance_triple[len(instance_triple) - 1] != tup:
            instance_triple.append(tup)
            instance_scope.append([s, s])
        instance_scope[len(instance_triple) - 1][1] = s
        if (s + 1) % 100 == 0:
            sys.stdout.write(str(s) + '\r')
            sys.stdout.flush()
    return np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask, en1, en2, sen_hier1_label, sen_hier2_label


def init_test_files(name):
    print("reading " + name + " data...")
    f = open(data_path + name + ".txt", "r")
    total = (int)(f.readline().strip())
    sen_word = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype=np.int32)
    sen_mask = np.zeros((total, fixlen), dtype=np.int32)
    sen_len = np.zeros((total), dtype=np.int32)
    sen_label = np.zeros((total), dtype=np.int32)
    entity_pair = []
    entity_scope = []
    en1 = np.zeros((total), dtype=np.int32)
    en2 = np.zeros((total), dtype=np.int32)
    sen_hier1_label = np.zeros((total), dtype=np.int32)
    sen_hier2_label = np.zeros((total), dtype=np.int32)
    for s in range(total):
        content = f.readline().strip().split()
        sentence = content[5:-1]
        en1_id = content[0]
        en2_id = content[1]
        en1_name = content[2]
        en2_name = content[3]
        rel_name = content[4]
        if rel_name in relation2id:
            relation = relation2id[rel_name]
        else:
            relation = relation2id['NA']
            rel_name = 'NA'
        for i in range(len(sentence)):
            if sentence[i] == en1_name:
                en1pos = i
            if sentence[i] == en2_name:
                en2pos = i
        en_first = min(en1pos, en2pos)
        if en1_name in word2id:
            en1[s] = word2id[en1_name]
        else:
            en1[s] = word2id['UNK']
        en_second = en1pos + en2pos - en_first
        if en2_name in word2id:
            en2[s] = word2id[en2_name]
        else:
            en2[s] = word2id['UNK']
        for i in range(fixlen):
            sen_word[s][i] = word2id['BLANK']
            sen_pos1[s][i] = pos_embed(i - en1pos)
            sen_pos2[s][i] = pos_embed(i - en2pos)
            if i >= len(sentence):
                sen_mask[s][i] = 0
            elif i - en_first<=0:
                sen_mask[s][i] = 1
            elif i - en_second<=0:
                sen_mask[s][i] = 2
            else:
                sen_mask[s][i] = 3
        for i, word in enumerate(sentence):
            if i >= fixlen:
                break
            elif not word in word2id:
                sen_word[s][i] = word2id['UNK']
            else:
                sen_word[s][i] = word2id[word]
        sen_len[s] = min(fixlen, len(sentence))
        sen_label[s] = relation
        # split relation into hier1 and hier2
        if rel_name == 'NA':
            sen_hier1_label[s] = hier1_relation2id['NA']
            sen_hier2_label[s] = hier2_relation2id['NA']
        else:
            relation_split = rel_name.strip().split('/')
            first_relation = relation_split[1]
            second_relation = relation_split[2]
            sen_hier1_label[s] = hier1_relation2id[first_relation]
            temp_hier2_relation = '/' + first_relation + '/' + second_relation
            sen_hier2_label[s] = hier2_relation2id[temp_hier2_relation]
        pair = (en1_id, en2_id)
        if entity_pair == [] or entity_pair[-1] != pair:
            entity_pair.append(pair)
            entity_scope.append([s, s])
        entity_scope[-1][1] = s
        if (s + 1) % 100 == 0:
            sys.stdout.write(str(s) + '\r')
            sys.stdout.flush()

    return np.array(entity_pair), np.array(entity_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask, en1, en2, sen_hier1_label, sen_hier2_label


def init_test_files_pn(name):
    print("reading " + name + " data...")
    f = open(data_path + name + ".txt", "r")
    total = (int)(f.readline().strip())
    sen_word = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype=np.int32)
    sen_mask = np.zeros((total, fixlen), dtype=np.int32)
    sen_len = np.zeros((total), dtype=np.int32)
    sen_label = np.zeros((total), dtype=np.int32)
    entity_pair = []
    entity_pair_pall = []
    entity_pair_palli = []
    entity_scope = []
    entity_scope_pall = []
    en1 = np.zeros((total), dtype=np.int32)
    en2 = np.zeros((total), dtype=np.int32)
    sen_hier1_label = np.zeros((total), dtype=np.int32)
    sen_hier2_label = np.zeros((total), dtype=np.int32)
    sall = 0
    for s in range(total):
        content = f.readline().strip().split()
        sentence = content[5:-1]
        en1_id = content[0]
        en2_id = content[1]
        en1_name = content[2]
        en2_name = content[3]
        rel_name = content[4]
        if rel_name in relation2id:
            relation = relation2id[rel_name]
        else:
            relation = relation2id['NA']
            rel_name = 'NA'
        for i in range(len(sentence)):
            if sentence[i] == en1_name:
                en1pos = i
            if sentence[i] == en2_name:
                en2pos = i
        en_first = min(en1pos,en2pos)
        if en1_name in word2id:
            en1[s] = word2id[en1_name]
        else:
            en1[s] = word2id['UNK']
        en_second = en1pos + en2pos - en_first
        if en2_name in word2id:
            en2[s] = word2id[en2_name]
        else:
            en2[s] = word2id['UNK']
        for i in range(fixlen):
            sen_word[s][i] = word2id['BLANK']
            sen_pos1[s][i] = pos_embed(i - en1pos)
            sen_pos2[s][i] = pos_embed(i - en2pos)
            if i >= len(sentence):
                sen_mask[s][i] = 0
            elif i - en_first<=0:
                sen_mask[s][i] = 1
            elif i - en_second<=0:
                sen_mask[s][i] = 2
            else:
                sen_mask[s][i] = 3
        for i, word in enumerate(sentence):
            if i >= fixlen:
                break
            elif not word in word2id:
                sen_word[s][i] = word2id['UNK']
            else:
                sen_word[s][i] = word2id[word]
        sen_len[s] = min(fixlen, len(sentence))
        sen_label[s] = relation
        if rel_name == 'NA':
            sen_hier1_label[s] = hier1_relation2id['NA']
            sen_hier2_label[s] = hier2_relation2id['NA']
        else:
            relation_split = rel_name.strip().split('/')
            first_relation = relation_split[1]
            second_relation = relation_split[2]
            sen_hier1_label[s] = hier1_relation2id[first_relation]
            temp_hier2_relation = '/' + first_relation + '/' + second_relation
            sen_hier2_label[s] = hier2_relation2id[temp_hier2_relation]
        pair = (en1_id, en2_id)
        if entity_pair == [] or entity_pair[-1] != pair:
            if len(entity_pair) > 0:
                first_t = entity_scope[-1][0]
                last_t = entity_scope[-1][1]
                if last_t - first_t > 0:
                    entity_pair_pall.append(entity_pair[-1])
                    entity_pair_palli.append(len(entity_pair) - 1)
                    entity_scope_pall.append([sall, sall + last_t - first_t])
                    sall = sall + 1 + last_t - first_t
            entity_pair.append(pair)
            entity_scope.append([s, s])
        entity_scope[-1][1] = s
        if (s + 1) % 100 == 0:
            sys.stdout.write(str(s) + '\r')
            sys.stdout.flush()
    f.close()
    first_t = entity_scope[-1][0]
    last_t = entity_scope[-1][1]
    if last_t - first_t > 0:
        entity_pair_pall.append(entity_pair[-1])
        entity_pair_palli.append(len(entity_pair) - 1)
        entity_scope_pall.append([sall, sall + last_t - first_t])
        sall = sall + 1 + last_t - first_t
    index_pall = np.hstack([np.arange(entity_scope[x][0], entity_scope[x][1] + 1) for x in entity_pair_palli])
    index_pone = np.hstack([np.random.randint(entity_scope[x][0], entity_scope[x][1] + 1) for x in entity_pair_palli])
    index_ptwo = np.hstack([np.random.choice(np.arange(entity_scope[x][0], entity_scope[x][1] + 1), 2, replace=False)
                            for x in entity_pair_palli])
    arrays = {}

    arrays['entity_pair'] = np.array(entity_pair)

    arrays['word'] = sen_word
    arrays['label'] = sen_label
    arrays['len'] = sen_len
    arrays['mask'] = sen_mask
    arrays['pos1'] = sen_pos1
    arrays['pos2'] = sen_pos2
    arrays['en1'] = en1
    arrays['en2'] = en2
    arrays['hier1'] = sen_hier1_label
    arrays['hier2'] = sen_hier2_label
    arrays['entity_scope'] = np.array(entity_scope)

    arrays['entity_pair_pn'] = np.array(entity_pair_pall)

    arrays['word_pall'] = sen_word[index_pall]
    arrays['label_pall'] = sen_label[index_pall]
    arrays['len_pall'] = sen_len[index_pall]
    arrays['mask_pall'] = sen_mask[index_pall]
    arrays['pos1_pall'] = sen_pos1[index_pall]
    arrays['pos2_pall'] = sen_pos2[index_pall]
    arrays['en1_pall'] = en1[index_pall]
    arrays['en2_pall'] = en2[index_pall]
    arrays['hier1_pall'] = sen_hier1_label[index_pall]
    arrays['hier2_pall'] = sen_hier2_label[index_pall]
    arrays['entity_scope_pall'] = np.array(entity_scope_pall)

    arrays['word_pone'] = sen_word[index_pone]
    arrays['label_pone'] = sen_label[index_pone]
    arrays['len_pone'] = sen_len[index_pone]
    arrays['mask_pone'] = sen_mask[index_pone]
    arrays['pos1_pone'] = sen_pos1[index_pone]
    arrays['pos2_pone'] = sen_pos2[index_pone]
    arrays['en1_pone'] = en1[index_pone]
    arrays['en2_pone'] = en2[index_pone]
    arrays['hier1_pone'] = sen_hier1_label[index_pone]
    arrays['hier2_pone'] = sen_hier2_label[index_pone]
    arrays['entity_scope_pone'] = np.tile(np.arange(arrays['word_pone'].shape[0]).reshape((-1, 1)), 2)

    arrays['word_ptwo'] = sen_word[index_ptwo]
    arrays['label_ptwo'] = sen_label[index_ptwo]
    arrays['len_ptwo'] = sen_len[index_ptwo]
    arrays['mask_ptwo'] = sen_mask[index_ptwo]
    arrays['pos1_ptwo'] = sen_pos1[index_ptwo]
    arrays['pos2_ptwo'] = sen_pos2[index_ptwo]
    arrays['en1_ptwo'] = en1[index_ptwo]
    arrays['en2_ptwo'] = en2[index_ptwo]
    arrays['hier1_ptwo'] = sen_hier1_label[index_ptwo]
    arrays['hier2_ptwo'] = sen_hier2_label[index_ptwo]
    arrays['entity_scope_ptwo'] = np.tile(2 * np.arange(arrays['word_pone'].shape[0]).reshape((-1, 1)), 2)
    arrays['entity_scope_ptwo'][:, 1] = arrays['entity_scope_ptwo'][:, 1] + 1

    fin = open(data_path + name + '.txt', 'r').readlines()[1:]
    fout = open(data_path + name + '_pall.txt', 'w')
    fout.write('{}\n'.format(sall))
    _ = [fout.write(fin[x]) for x in index_pall]
    assert_equal(len(_), sall)
    fout.close()

    fout = open(data_path + name + '_pone.txt', 'w')
    fout.write('{}\n'.format(arrays['word_pone'].shape[0]))
    _ = [fout.write(fin[x]) for x in index_pone]
    fout.close()

    fout = open(data_path + name + '_ptwo.txt', 'w')
    fout.write('{}\n'.format(arrays['word_ptwo'].shape[0]))
    _ = [fout.write(fin[x]) for x in index_pall]
    fout.close()

    return arrays



if __name__ == '__main__':
    init_word()
    init_relation()
    np.save(export_path + "vec", word_vec)
    json.dump({
        "word2id":word2id,
        "relation2id":relation2id,
        "hier1_relation2id":hier1_relation2id,
        "hier2_relation2id":hier2_relation2id,
        "word_size":word_size,
        "fixlen":fixlen,
        "maxlen":maxlen
    }, open(export_path+"config", "wt"))

    sort_files("train")
    instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask, train_en1, train_en2, train_sen_hier1, train_sen_hier2 = init_train_files("train_sort")
    np.save(export_path + "train_instance_triple", instance_triple)
    np.save(export_path + "train_instance_scope", instance_scope)
    np.save(export_path + "train_len", train_len)
    np.save(export_path + "train_label", train_label)
    np.save(export_path + "train_word", train_word)
    np.save(export_path + "train_pos1", train_pos1)
    np.save(export_path + "train_pos2", train_pos2)
    np.save(export_path + "train_mask", train_mask)
    np.save(export_path + "train_en1", train_en1)
    np.save(export_path + "train_en2", train_en2)
    np.save(export_path + "train_sen_hier1", train_sen_hier1)
    np.save(export_path + "train_sen_hier2", train_sen_hier2)

    sort_test_files("test")
    instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask, test_en1, test_en2, test_sen_hier1, test_sen_hier2 = init_test_files("test_sort")
    np.save(export_path + 'test_entity_pair', instance_triple)
    np.save(export_path + 'test_entity_scope', instance_scope)
    np.save(export_path + 'test_len', test_len)
    np.save(export_path + 'test_label', test_label)
    np.save(export_path + 'test_word', test_word)
    np.save(export_path + 'test_pos1', test_pos1)
    np.save(export_path + 'test_pos2', test_pos2)
    np.save(export_path + 'test_mask', test_mask)
    np.save(export_path + 'test_en1', test_en1)
    np.save(export_path + 'test_en2', test_en2)
    np.save(export_path + 'test_sen_hier1', test_sen_hier1)
    np.save(export_path + 'test_sen_hier2', test_sen_hier2)

    for name, data in init_test_files_pn("test_sort").items():
        np.save(export_path + 'pn/test_' + name + '.npy', data)

    # initialize bag label for test
    label = np.load("../data/test_label.npy")
    scope = np.load("../data/test_entity_scope.npy")
    all_true_label = np.zeros((scope.shape[0], 53))
    for pid in range(scope.shape[0]):
        all_true_label[pid][label[scope[pid][0]:scope[pid][1]+1]] = 1
    all_true_label = np.reshape(all_true_label[:, 1:], -1)
    np.save("../data/all_true_label.npy", all_true_label)

    label = np.load("../data/pn/test_label_pone.npy")
    scope = np.load("../data/pn/test_entity_scope_pone.npy")
    all_true_label = np.zeros((scope.shape[0], 53))
    for pid in range(scope.shape[0]):
        all_true_label[pid][label[scope[pid][0]:scope[pid][1]+1]] = 1
    all_true_label = np.reshape(all_true_label[:, 1:], -1)
    np.save("../data/pn/true_label.npy", all_true_label)
