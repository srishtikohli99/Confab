entityExtractor=$(cat config.json | jq '.entityExtractor' | tr -d \")
entity=`cat config.json | jq '.entity'`
en="flair"
echo $entity
echo "$entityExtractor"
echo $en
if [ $entity = true ]
then
    if [[ $entityExtractor == $en ]]
    then
        # echo "here"
        python flairEntity/train.py
    fi
    # echo "here2"
    python lstmIntent/EntityClassifier.py
    python lstmIntent/IntentWithEntity.py
fi
# echo "here3"
python lstmIntent/IntentWithoutEntity.py
