entity=`cat config.json | jq '.entity'`
echo $entity
if [ $entity = true ]
then
    python lstmIntent/IntentWithEntity.py
    python lstmIntent/EntityClassifier.py
fi
python lstmIntent/IntentWithoutEntity.py
