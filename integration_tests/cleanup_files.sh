# remove subdirectories
DIRNAME=${PWD##*/}
EXPECTDIRNAME='integration_tests'
if [ $DIRNAME = $EXPECTDIRNAME ]
then
  echo 'Cleaning up'
  find . -mindepth 2 -maxdepth 2 -type d -exec rm -r '{}' \;
  find . -mindepth 2 ! -name '*.pyc' ! -name '*.py' ! -name '*.ini' ! -name '*.txt' -exec rm '{}' \;
else
  echo 'Careful! Calling an extensive "rm" in unknown directory!!'
  echo 'Nothing happened... this time.'
fi
