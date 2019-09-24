import os
import subprocess
from subprocess import Popen, PIPE
import sys

encoding = 'utf-8'

def test_mpi():
    print('test_mpi.py using python {}.{}'.format(
        *sys.version.split('.')[:2]
        ))
    command = 'mpirun -np 2 python mpi_script.py'.split()
    p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    print('\n**** mpi_script.py error: ****')
    print(err.decode(encoding))
    print(30*'*')
    print('*** mpi_scripty.py output: ***')
    print(output.decode(encoding)) 
    print(30*'*')
    assert len(err) == 0
#     try:
#         proc = subprocess.check_output(command, stderr=subprocess.STDOUT)
#     except subprocess.CalledProcessError:
#         assert False
    # subprocess.check_call(bash_command.split())
#    except subprocess.CalledProcessError:
#        assert False
#    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
#    output, error = process.communicate()
#    print(output)
#    assert error is None

if __name__=='__main__':
    test_mpi()

