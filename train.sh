entity=`cat config.json | jq '.entity'`
echo $entity
if [ $entity = true ]
then
    python lstmIntent/EntityClassifier.py
    python lstmIntent/IntentWithEntity.py
fi
python lstmIntent/IntentWithoutEntity.py
