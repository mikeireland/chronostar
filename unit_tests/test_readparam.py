"""Test basic functionality of readparams"""
import sys
sys.path.insert(0, '..')
from chronostar import readparam

def test_dummypars():
    dummypar_filename = 'temp_data/dummypar.par'
    dummydict = {
        'foo':10,
        'bar':20,
    }
    with open(dummypar_filename, 'w') as fp:
        for k,v in dummydict.items():
            fp.write('{}={}\n'.format(k,v))

    retrieved_dict = readparam.readParam(dummypar_filename)

    for key in dummydict:
        assert dummydict[key] == retrieved_dict[key]

def test_emptyfile():
    empty_filename = 'temp_data/empty.par'

    # Overwrite file to be empty?
    with open(empty_filename, 'w') as fp:
        pass
    empty_dict = readparam.readParam(empty_filename)
    assert len(empty_dict) == 0

def test_comments():
    comments_filename = 'temp_data/comments.par'
    dummydict = {
        'foo':10,
        'bar':20,
    }

    # Overwrite file to be comments?
    with open(comments_filename, 'w') as fp:
        fp.write('# This is a comment\n')
        for k,v in dummydict.items():
            fp.write('{}={}\n'.format(k,v))

    retrieved_dict = readparam.readParam(comments_filename)

    for key in dummydict:
        assert dummydict[key] == retrieved_dict[key]
    comments_dict = readparam.readParam(comments_filename)



