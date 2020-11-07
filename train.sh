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
        python Morph/flairEntity/train.py
    fi
    # echo "here2"
    python bureau/EntityClassifier.py
    python bureau/IntentWithEntity.py
fi
# echo "here3"
python bureau/IntentWithoutEntity.py
