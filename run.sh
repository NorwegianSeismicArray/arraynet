
#DEFINE THESE
USER=erik
PROJECTNAME=csm
SCRIPT_LOCATION=experiments.py
HELP_SCRIPT=train.py
REQUIREMENTS=requirements.txt
LOCATION=/projects/active/MMON/Array_detection/ML_methods/csm_pattern_classification/tf/ 	#Will be mapped to /tf/data so make sure your scripts access data from /tf/data
OUTPUT=tf/output

#LOGIC (DO NOT CHANGE)
BASE_DIR=/nobackup/$USER/$PROJECTNAME
mkdir $BASE_DIR
rsync -ahr --progress $LOCATION/* $BASE_DIR/

cp $SCRIPT_LOCATION $BASE_DIR/script.py
cp $HELP_SCRIPT $BASE_DIR/

cp $REQUIREMENTS $BASE_DIR/req.txt
#rm -rf $BASE_DIR/nais
#git clone git@github.com:NorwegianSeismicArray/nais.git $BASE_DIR/nais

#"'device=0'" uses only one, "'device=0,1'" will use both
docker run -it --rm --gpus  '"device=1"' -v $BASE_DIR:/tf tensorflow/tensorflow:latest bash -c "pip install -r tf/req.txt
                                                                                                python tf/script.py"

#CLEAN UP
rsync -ahr --progress $BASE_DIR/output/*.npy $OUTPUT

