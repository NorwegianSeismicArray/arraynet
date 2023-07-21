#DEFINE THESE
USER=user
USER=andreask
PROJECTNAME=arraynet
# the script that starts the training of (different) models
SCRIPT_LOCATION=experiments.py
# the actual code
HELP_SCRIPT=train.py
REQUIREMENTS=requirements.txt
# directory with input (tf/data/)  and output data
LOCATION=./tf
# output for models
OUTPUT=tf/output
# working directory
WORK_DIR=`pwd`
# another working directory for faster training in case you are on a network disk (local disk, gpu, ...)
# Modify script if WORK_DIR and BASE_DIR are the same
BASE_DIR=/nobackup/$USER/$PROJECTNAME

test -d $OUTPUT || mkdir $OUTPUT
test -d $OUTPUT/models || mkdir $OUTPUT/models
test -d $BASE_DIR || mkdir $BASE_DIR
test -d $BASE_DIR/output || mkdir $BASE_DIR/output
test -d $BASE_DIR/output/models || mkdir $BASE_DIR/output/models

rsync -ahr --progress $LOCATION/* $BASE_DIR/

cp $SCRIPT_LOCATION $BASE_DIR/script.py
cp $HELP_SCRIPT $BASE_DIR/

cp $REQUIREMENTS $BASE_DIR/req.txt

cd $BASE_DIR
bash -c "pip install -r req.txt
         python script.py"

cd $WORK_DIR
#CLEAN UP
rsync -ahr --progress $BASE_DIR/output/*.npy $OUTPUT
rsync -ahr --progress $BASE_DIR/output/models/* $OUTPUT/models
