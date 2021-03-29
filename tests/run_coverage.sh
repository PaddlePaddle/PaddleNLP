MODULE=paddlenlp
if [ "$#" -gt 0 ];
then  MODULE=$1
fi
coverage run --source $MODULE -m unittest discover
coverage report -m
coverage html
